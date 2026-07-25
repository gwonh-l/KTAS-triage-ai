"""
main.py — KTAS 응급 중증도 분류 FastAPI 서빙
====================================================================
환자 증상 텍스트(자연어) → KTAS 1~5 등급 + (K4/K5) 진료과 추천을 반환하는 추론 서버.

실행:
    cd ktas_backend          # 반드시 이 디렉터리에서 (아래 상대경로 때문)
    uvicorn main:app --reload --port 8000

모델/설정 로드:
    - 가중치  : ./ktas_ai_model/  (model_loader.load_model 이 backbone + 2-head 로드)
    - 임계치 등: ./ktas_ai_model/threshold_config.json
                 (THRESHOLD_K4 / TEMPERATURE / ENTROPY_THRESHOLD / dept_classes)
      → 학습(ktas_model_train.py)이 calib split에서 산출해 저장한 값. 재학습 때마다 갱신됨.

핵심 설계 (임상 안전 우선):
    1) 보수적 상향(over-triage) — apply_conservative_threshold().
       '애매하면 더 위급한 쪽'. K4/K5 경증대 예측을 K4+K5 합산 확신이 부족하면 K3로 올림.
       학습(conservative_predict)·튜너와 '글자 그대로 동일' 해야 함(평가≠서빙 드리프트 방지).
    2) raw vs calibrated 확률 분리 — 등급 결정은 raw(=임계치와 같은 스케일),
       화면 표시·불확실도는 temperature 보정 확률. argmax는 T에 불변이라 등급 영향 없음.
    3) OOD/저신뢰는 reject 가 아니라 'uncertain 플래그 + 결과 유지'(고중증 침묵 방지).

엔드포인트:
    POST /predict  → PredictResponse (아래 스키마 참조)
    GET  /health   → 로드된 임계치/설정 점검
"""
import os
import json
import re
import hashlib
import logging
import numpy as np
import torch
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import load_model, KTASMultiTaskModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── threshold_config.json에서 동적 로드 ──────────────────────────────────────
# 임계치를 코드 상수가 아닌 외부 파일로 둔 이유: 재학습 없이 over/under 균형을 조절하고,
# 학습이 산출한 calib 보정값(temperature·entropy)을 서빙이 그대로 물려받기 위함.
_THRESHOLD_CONFIG_PATH = "./ktas_ai_model/threshold_config.json"  # cwd=ktas_backend 기준 상대경로

# config 로드 실패(파일 없음/키 누락) 시 사용할 진료과 클래스 폴백.
# ⚠️ 순서가 곧 dept_head 출력 인덱스 ↔ 클래스 매핑이다. 학습 때 DEPT_CLASSES와 100% 일치해야 함.
#    앞 9개는 가나다순, '외과'·'소아과'는 학습 코드에서 append 슬롯 → 절대 sorted() 금지.
_DEPT_CLASSES_FALLBACK = [
    '내과', '비뇨기과', '산부인과', '신경과', '안과', '이비인후과', '정신건강의학과', '정형외과', '피부과', '외과', '소아과'
]

def _load_config() -> tuple[float, list[str], float, Optional[float]]:
    """threshold_config.json → (THRESHOLD_K4, dept_classes, TEMPERATURE, ENTROPY_THRESHOLD).

    파일이 없으면 보수적 폴백으로 동작(서버는 죽지 않음). 각 키 의미는 아래 주석 참조.
    참고: 과거 THRESHOLD_K5 키가 있었으나 보수적 상향이 K4+K5 합산 단일 게이트로 바뀌며 폐기됨.
          기존 config에 THRESHOLD_K5가 남아 있어도 더는 읽지 않으므로 무해.
    """
    if os.path.exists(_THRESHOLD_CONFIG_PATH):
        with open(_THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # THRESHOLD_K4: 보수적 상향의 단일 임계치. K4+K5 합산 확신이 이 값 미만이면 K3로 상향.
        #               (값이 높을수록 더 많이 상향 = over-triage↑/under-triage↓)
        k4 = float(cfg.get("THRESHOLD_K4", 0.86))
        # dept_classes: dept_head 인덱스↔진료과명 매핑(순서 중요). 없으면 폴백 사용.
        dept_classes = cfg.get("dept_classes", _DEPT_CLASSES_FALLBACK)
        # TEMPERATURE: temperature scaling 계수(calib 적합). 표시 confidence·entropy 과신 완화용.
        #              없으면 1.0 = 스케일링 미적용. argmax 불변이라 '등급'에는 영향 없음.
        temperature = float(cfg.get("TEMPERATURE", 1.0))
        # ENTROPY_THRESHOLD: OOD/저신뢰 reject 임계치(calib 엔트로피 분포 상위 percentile).
        #                    없으면 None = 비활성화. 미보정 임계치로 정상 입력을 오거부하는 사고 방지.
        ent_raw = cfg.get("ENTROPY_THRESHOLD", None)
        entropy_threshold = float(ent_raw) if ent_raw is not None else None
        logger.info(
            f"config 로드: K4={k4} | dept {len(dept_classes)}개 | "
            f"T={temperature} | entropy_th={entropy_threshold}"
        )
        return k4, dept_classes, temperature, entropy_threshold
    # 파일 부재 시: K4=0.86(보수적), T=1.0(보정 없음), entropy=None(reject 끔)
    logger.warning("threshold_config.json 없음 → fallback 사용")
    return 0.86, _DEPT_CLASSES_FALLBACK, 1.0, None

# 모듈 로드 시 1회 실행 → 전역 상수로 고정(요청마다 재로딩 안 함).
# config를 바꾸면 서버 재시작 필요.
THRESHOLD_K4, DEPT_CLASSES, TEMPERATURE, ENTROPY_THRESHOLD = _load_config()
if ENTROPY_THRESHOLD is None:
    logger.warning("ENTROPY_THRESHOLD 미설정 → OOD reject 비활성화 (train.py 재학습으로 calib 보정 필요)")

KTAS_LABELS = ["KTAS1", "KTAS2", "KTAS3", "KTAS4", "KTAS5"]

# KTAS 공급자 매뉴얼(대한응급의학회, 2019) 약어 섹션 및 1차 고려사항 기준
# 수록 항목: 의식(GCS), 혈역학적 상태(BP, HR), 호흡(RR, SpO2), 통증(NRS),
#            소생술(CPR), 소아 의식 평가 대안(AVPU)
# → 해당 약어를 한국어로 치환하여 모델 입력 품질 및 한글 비율 계산 신뢰도 향상
ABBR_MAP = {
    'CPR':  '심폐소생술',
    'SpO2': '산소포화도',
    'GCS':  '의식수준',
    'NRS':  '통증점수',
    'BP':   '혈압',
    'HR':   '심박수',
    'RR':   '호흡수',
    'AVPU': '의식수준',
}
# 단어 경계(\b) 기반 치환, 대소문자 구분 없이 매칭
ABBR_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in ABBR_MAP) + r')\b',
    re.IGNORECASE
)

def _mask_for_log(text: str) -> str:
    """증상 원문은 건강정보(민감정보)이므로 로그에 평문 저장 금지.
    추적용으로 길이 + sha256 앞 8자리만 남긴다."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"len={len(text)} sha256={digest}"


# 앱 초기화 & 모델 로드
app = FastAPI(
    title="KTAS 응급 중증도 분류 API",
    description="환자 증상 텍스트 → KTAS 1~5 등급 분류 (임상 데이터 기반 보수적 예측)",
    version="1.0.0"
)

model, tokenizer, device = load_model()
logger.info(f"모델 로드 완료 | device: {device}")


# ── 요청 / 응답 스키마 ────────────────────────────────────────────────────────
class SymptomRequest(BaseModel):
    symptom_text: str          # 환자 증상 자연어 (예: "가슴이 심하게 아파요"). 1~200자, 한글 위주.


class PredictResponse(BaseModel):
    # ── 프런트엔드 연동 계약 ───────────────────────────────────────────────────
    # 이 필드 집합/타입이 곧 API 계약. 필드 추가는 안전하나, 이름·타입 변경/삭제는 프런트와 동기화 필요.
    # (참고: 과거 highlight_keywords 필드는 제거됨 — 프런트도 그 참조 제거 완료 가정)
    ktas_level:      int            # 최종 KTAS 등급 1~5 (보수적 상향 반영된 '표시용' 등급)
    is_emergency:    bool           # ktas_level <= 3 → True (권역응급센터 라우팅 신호)
    confidence:      float          # '최종 등급'의 softmax 확률(temperature 보정). 상향 시 K3 확률이라 낮게 보일 수 있음
    # ── 보수적 상향 투명성: adjusted=True일 때만 original_* 채워짐(아니면 None) ──
    original_level:       Optional[int]   = None   # 상향 전 모델 원예측 등급 (adjusted=True일 때만)
    original_confidence:  Optional[float] = None   # 상향 전 원예측 확률 (adjusted=True일 때만)
    adjusted:        bool           # 보수적 상향(K3로 올림)이 일어났는지
    adjusted_reason: Optional[str] = None          # 상향 사유 문자열 (None이면 미조정). 화면에 그대로 표시 가능
    # ── 진료과 추천: K4/K5(경증대)에서만 유효, K1~K3은 None ──
    dept:                 Optional[str]   = None   # 추천 진료과명 (DEPT_CLASSES 중 하나)
    dept_confidence:      Optional[float] = None   # dept softmax 확률
    probabilities:   dict           # {"KTAS1":..,"KTAS5":..} 전체 분포 (temperature 보정 후, 표시용)
    # ── OOD/저신뢰 안내(결과는 그대로 유지, 거부 아님) ──
    status:          str = "ok"                    # "ok" | "uncertain"
    uncertainty_notice: Optional[str] = None       # uncertain일 때 사용자 안내 문구
    entropy:         Optional[float] = None        # 예측 분포 엔트로피(보정 후). 클수록 불확실


# ── 입력 검증 & 정제 ──────────────────────────────────────────────────────────
def validate_and_clean(text: str) -> str:
    """증상 텍스트를 검증·정제해 모델 입력 문자열로 반환. 부적합 입력은 HTTP 422로 거부.

    ⚠️ 알려진 한계: 'ㅇㅇㅇ'(자모)는 완성형 한글 검사에서 막히지만, '아아아'·'가나' 같은
       완성형 무의미 입력은 통과한다(아래 2)의 반복 축소가 오히려 길이 게이트를 통과시킴).
       무의미/분포 외 입력의 근본 차단은 모델 엔트로피 reject(ENTROPY_THRESHOLD)의 몫이며,
       미보정 시 꺼져 있다.
    """
    text = text.strip()

    # 1) 빈 입력 / 과도한 길이 차단
    if not text:
        raise HTTPException(status_code=422, detail="증상 텍스트를 입력해 주세요.")
    if len(text) > 200:
        raise HTTPException(status_code=422, detail="200자 이내로 입력해 주세요.")

    # 2) 반복 문자 정제: 3회 이상 연속 → 2회로 축소. "아아아아파요" → "아아파요"
    #    과장 표현 정규화용. (부작용: 무의미 반복도 짧게 만들어 길이 게이트를 통과시킴 — docstring 참고)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 3) KTAS 가이드라인 약어 → 한국어 치환. 모델 입력 품질 + 아래 '한글 비율' 계산 정상화.
    #    예) "CPR 중입니다" → "심폐소생술 중입니다"  (ABBR_RE: 대소문자 무시, 단어경계 매칭)
    text = ABBR_RE.sub(lambda m: ABBR_MAP[m.group().upper()], text)

    hangul_chars = re.findall(r'[\uAC00-\uD7A3]', text)
    # 4) 완성형 한글(가-힣, U+AC00~U+D7A3) 최소 2자 요구.
    #    'ㅇㅇ'류 자모(U+3131~U+3163)는 이 범위 밖이라 0개로 잡혀 여기서 거부됨.
    if len(hangul_chars) < 2:
        raise HTTPException(status_code=422, detail="정확한 증상을 한국어로 입력해 주세요.")

    # 5) 한글 비율 검사: 공백·특수문자·자모단독 제거 후 '순수 글자' 중 한글이 절반 이상이어야 함.
    #    숫자/영문/이모지 범벅 등 비한국어 잡음 입력 차단. (ㄱ~ㅎ=자음, ㅏ~ㅣ=모음)
    # 순수 글자 추출 (공백 및 일반적인 특수문자, 자음/모음 단독 사용 등 제외)
    pure_text = re.sub(
        r'[\s\.,!\?~@#\$%\^&\*\(\)_\+\-\=\[\]\{\}\|;:\'\"<>/\u3131-\u314E\u314F-\u3163]',
        '', text
    )
    if len(pure_text) > 0:
        hangul_ratio = len(hangul_chars) / len(pure_text)
        if hangul_ratio < 0.5:
            raise HTTPException(status_code=422, detail="의미 없는 문자가 너무 많습니다. 한글 증상 위주로 정확히 입력해 주세요.")

    return text


# ── 보수적 상향(over-triage) 조정 로직 ────────────────────────────────────────
# 'K4+K5 합산 질량' 단일 게이트 방식(design B). 모델이 경증대(K4/K5)로 본 예측에만 적용.
#
# [반환] (pred_idx, adjusted: bool, reason: Optional[str])  ※ pred_idx는 0-based(0=K1..4=K5)
#
# [⚠️ 필수 불변식] 이 함수는 학습/튜너의 conservative_predict 와 '글자 그대로 동일' 해야 한다.
#   세 곳(main.py·ktas_model_train.py·ktas_hparam_tuner.py)이 어긋나면 '평가 수치 ≠ 서빙 동작'이
#   되어 under-triage<5% 안전 보증이 무의미해진다(과거 "tuner 좋고 train 다름" 재발 지점).
#
# [설계 근거] 구버전은 K4 단독 확신(probs[K4])만 보고 K3로 상향 → "손 베임/멍" 같은 명백한 경증이
#   K3(긴급)까지 과상향됐다. 불확실성이 'K3 vs K4'가 아니라 'K4 vs K5'(둘 다 경증)인데도 올린 것.
#   → K4+K5 합산(=P(비응급))을 보면 이 오상향을 걸러낼 수 있다.
def apply_conservative_threshold(pred: int, probs: np.ndarray):
    # 판단 기준 = K4+K5 합산(= 비응급 확신 질량). ※ probs는 raw softmax(THRESHOLD_K4와 동일 스케일).
    #   - 합산 <  THRESHOLD_K4 : K1~K3 쪽 질량이 의미있게 남음 → K3(idx 2) 상향(안전 방향)
    #   - 합산 >= THRESHOLD_K4 : 경증 확신 충분 → 모델 예측(K4/K5 중 높은 쪽=argmax) 그대로 유지
    #                            → 불확실성이 'K4 vs K5'일 뿐이면 K5도 K4로 끌어올리지 않고 살림
    # 예) "손 베임" K4+K5=0.78 → 유지 / "팔꿈치 멍" K4+K5=0.71 → K3 상향
    if pred in (3, 4):
        mild_mass = float(probs[3] + probs[4])
        if mild_mass < THRESHOLD_K4:
            reason = (
                f"비응급 확신도 K4+K5({mild_mass:.3f}) < 임계치({THRESHOLD_K4}) → K3 상향"
            )
            return 2, True, reason
        # 경증 확신 충분 → argmax(K4/K5) 그대로, 상향 없음 (K5 살림)
        return pred, False, None

    return pred, False, None

# ── 예측 엔드포인트 ───────────────────────────────────────────────────────────
# 처리 흐름: 입력검증 → 토큰화 → 모델 forward → (raw/cal 확률) → 보수적 상향 →
#            OOD 플래그 → 진료과(K4/K5) → 응답 조립.
@app.post("/predict", response_model=PredictResponse)
def predict(req: SymptomRequest):
    text = validate_and_clean(req.symptom_text)
    logger.info(f"입력 수신: {_mask_for_log(text)}")   # 증상 원문 비로깅(건강정보 보호)

    try:
        # ── 1) 토큰화 ── 학습과 동일 설정(max_len 180, pad/truncation)으로 드리프트 방지
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=180,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in encoding.items()}

        # ── 2) 모델 forward ── ktas_logits(5) + dept_logits(11). no_grad/detach로 추론 경량화
        with torch.no_grad():
            # forward 단일 경로 사용 (pooling 로직을 모델에 일원화 — 학습/서빙 드리프트 방지)
            ktas_logits, dept_logits = model(**inputs)
            ktas_logits = ktas_logits.detach()
            dept_logits = dept_logits.detach()

        # ── 3) 두 종류 확률 ── 용도가 달라 분리한다(핵심).
        # (a) raw_probs: 등급 '결정'용. THRESHOLD_K4가 보정된 바로 그 스케일.
        #     여기에 temperature를 먹이면 안전 임계치와 스케일이 어긋나 ACS-COT 보증이 깨진다.
        raw_probs = torch.softmax(ktas_logits, dim=-1).squeeze().cpu().numpy()
        # (b) cal_probs: '표시 confidence·불확실도'용. temperature scaling으로 과신 완화.
        #     argmax는 T에 불변이라 '등급'에는 영향 없음(표시 숫자만 부드러워짐).
        cal_probs = torch.softmax(ktas_logits / TEMPERATURE, dim=-1).squeeze().cpu().numpy()

        # ── 4) 보수적 상향 ── 원예측(argmax) → 안전 보정된 최종 등급.
        #     original_pred는 상향 전 등급(투명성 위해 응답에 함께 노출).
        original_pred = int(np.argmax(raw_probs))
        pred, adjusted, adjusted_reason = apply_conservative_threshold(original_pred, raw_probs)
        ktas_level   = pred + 1            # 0-based idx → 1-based KTAS 등급
        is_emergency = ktas_level <= 3     # K1~K3 = 응급(권역센터 라우팅)

        # ── 5) OOD/저신뢰 플래그 ── calibrated 확률 분포의 엔트로피(불확실도)로 판정.
        # 임계치 미보정(None) 시 비활성화. reject가 아니라 "결과 유지 + 사람 검토 권고"로,
        # 고중증 가능성을 침묵 처리하지 않는다(보수적 안전 철학 유지).
        entropy = float(-np.sum(cal_probs * np.log(cal_probs + 1e-12)))
        if ENTROPY_THRESHOLD is not None and entropy > ENTROPY_THRESHOLD:
            status = "uncertain"
            uncertainty_notice = (
                "입력 신뢰도가 낮습니다(분포 외/모호 가능). 결과는 참고용이며 "
                "반드시 의료진의 직접 확인이 필요합니다."
            )
        else:
            status = "ok"
            uncertainty_notice = None

        # ── 6) 진료과 추천 ── K4/K5(경증대)에서만 dept_head 사용, K1~K3은 None.
        #     dept_idx → DEPT_CLASSES 매핑(순서 의존). 범위 밖 방어로 None.
        if ktas_level >= 4:
            dept_probs = torch.softmax(dept_logits, dim=-1).squeeze().cpu().numpy()
            dept_idx   = int(np.argmax(dept_probs))
            dept_name  = DEPT_CLASSES[dept_idx] if dept_idx < len(DEPT_CLASSES) else None
            dept_conf  = round(float(dept_probs[dept_idx]), 4)
        else:
            dept_name = None
            dept_conf = None

        # ── 7) 응답 조립 ── 표시 확률은 cal_probs 사용.
        prob_dict = {KTAS_LABELS[i]: round(float(cal_probs[i]), 4) for i in range(5)}
        # original_*: 보수적 상향이 일어났을 때만 채움(상향 전 등급/확률 = 투명성). 아니면 None.
        original_level_out      = (original_pred + 1) if adjusted else None
        original_confidence_out = round(float(cal_probs[original_pred]), 4) if adjusted else None

        logger.info(
            f"결과: KTAS {ktas_level} | 응급:{is_emergency} | conf:{cal_probs[pred]:.3f} | "
            f"조정:{adjusted} | status:{status} | entropy:{entropy:.3f}"
            + (f" | 진료과:{dept_name}({dept_conf})" if dept_name else "")
        )

        return PredictResponse(
            ktas_level=ktas_level,
            is_emergency=is_emergency,
            confidence=round(float(cal_probs[pred]), 4),
            original_level=original_level_out,
            original_confidence=original_confidence_out,
            adjusted=adjusted,
            adjusted_reason=adjusted_reason,
            dept=dept_name,
            dept_confidence=dept_conf,
            probabilities=prob_dict,
            status=status,
            uncertainty_notice=uncertainty_notice,
            entropy=round(entropy, 4),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"예측 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

# ── 헬스체크 엔드포인트 ───────────────────────────────────────────────────────
# 배포 직후 점검용: 로드된 임계치/설정이 의도대로인지 확인.
#   - threshold_k4 가 재학습 산출값과 일치하는지 (옛 값이면 config 갱신 누락)
#   - dept_classes 가 11개·올바른 순서인지
#   - ood_reject_enabled 가 기대대로인지(entropy 미보정이면 false)
@app.get("/health")
def health():
    return {
        "status":            "ok",
        "model":             "klue/bert-base (KTAS multi-task fine-tuned)",
        "threshold_k4":      THRESHOLD_K4,        # 보수적 상향 단일 임계치(K4+K5 합산 게이트)
        "temperature":       TEMPERATURE,         # 표시 confidence 보정 계수
        "entropy_threshold": ENTROPY_THRESHOLD,   # OOD reject 임계치(None=비활성)
        "ood_reject_enabled": ENTROPY_THRESHOLD is not None,
        "dept_classes":      DEPT_CLASSES,        # 진료과 인덱스↔이름 매핑(순서 점검용)
    }