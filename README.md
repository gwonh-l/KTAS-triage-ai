# KTAS 응급 중증도 분류 + 진료과 추천 시스템

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

2026년 동국대학교 공개SW프로젝트 - 데이터 구축, 모델 설계·학습, 임계치 보정, API 서빙 구현

> 환자의 **자연어 증상 호소**를 입력받아 **KTAS 중증도(1~5등급)** 를 분류하고, 경증(K4·K5)에 대해 **진료과(11종)** 를 추천하는 멀티태스크 AI 서비스입니다.
> 응급실 **"뺑뺑이"(수용 거부 표류)** 완화를 위한 ETA 기반 환자–병원 매칭 파이프라인의 **분류 엔진**으로 설계되었습니다.

```
증상 텍스트 → [KTAS 분류 K1~K5] → (K4/K5) [진료과 추천] → ETA 병상 예측 → 병원 매칭
              └────────── 본 저장소의 범위 ───────────┘
```

> [!IMPORTANT]
> **임상 안전 우선 설계** — 중증 환자를 경증으로 오분류하는 **Under-triage를 5% 미만**으로 강제하는 hard gate를 학습·선택·서빙 전 과정에 적용합니다.

> [!WARNING]
> 본 시스템의 출력은 **의료 행위가 아닌 참고용 보조 정보**입니다. 정확한 진단과 처치는 반드시 의료 전문가의 판단을 따라야 합니다.

---

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [전체 파이프라인](#전체-파이프라인)
- [주요 특징](#주요-특징)
- [모델 아키텍처](#모델-아키텍처)
- [데이터셋](#데이터셋)
- [성능](#성능)
- [핵심 설계 — 임상 안전](#핵심-설계--임상-안전)
- [설치](#설치)
- [사용법](#사용법)
- [API 명세](#api-명세)
- [프로젝트 구조](#프로젝트-구조)
- [알려진 한계 / 향후 과제](#알려진-한계--향후-과제)
- [참고 문헌](#참고-문헌)
- [라이선스](#라이선스)

---

## 프로젝트 개요

응급실 "뺑뺑이"는 환자가 적절한 병원으로 즉시 이송되지 못하고 여러 병원을 표류하는 문제입니다. 이를 완화하려면 **환자의 중증도를 빠르고 안전하게 판정**하고, 그에 맞는 **진료과·병상 가용 병원으로 매칭**해야 합니다.

본 저장소는 그 파이프라인의 **AI 분류 엔진**을 담당합니다:

1. **KTAS 중증도 분류** — 환자/보호자가 입력한 자연어 증상을 KTAS 1~5등급으로 분류
2. **진료과 추천** — 경증(K4·K5)에 한해 11개 진료과 중 적합 과를 추천
3. **임상 안전 보정** — "애매하면 더 위급한 쪽"으로 보수적 상향하여 중증 누락(under-triage)을 억제

분류 결과는 후속 단계인 **ETA 병상 예측 → 병원 매칭**으로 전달됩니다.

---

## 전체 파이프라인

| 단계 | 구성요소 | 저장소 / 위치 | 설명 |
|---|---|---|---|
| ① 데이터 생성 | ER Data Generator | [ER-Data-Generator](https://github.com/gwonh-l/ER-Data-Generator) | KTAS 분류표 기반 가상 증상 텍스트를 GPT로 생성·정제·증강 |
| ② 모델 학습 | AI Modeling | `AI_modeling/` | Optuna 탐색 → 멀티태스크 BERT 학습 → test 평가 |
| ③ 서빙 | Backend API | `ktas_backend/` | FastAPI 추론 서버 (`/predict`, `/health`) |
| ④ 매칭 | (후속 시스템) | — | ETA 병상 예측 및 환자–병원 매칭 |

> 데이터셋은 GPT로 생성한 **가상 데이터**이며 실제 환자 정보(PHI)를 포함하지 않습니다. 생성 프롬프트·정제·증강 로직 전체는 [ER-Data-Generator](https://github.com/gwonh-l/ER-Data-Generator) 저장소에 공개되어 있습니다.

---

## 주요 특징

- **2-head 멀티태스크** — 단일 BERT backbone에서 중증도와 진료과를 동시 학습
- **임상 안전 게이트** — Under-triage < 5% (ACS-COT field triage 기준)를 **절대 제약**으로 강제
- **보수적 상향(over-triage) 보정** — `K4+K5 합산 확신`이 임계치 미만이면 K3로 상향해 중증 누락 방지
- **동적 임계치 보정** — calibration split에서 등급 임계치를 데이터로 역산 (모델 재학습 없이 over/under 조절)
- **표시 confidence 보정** — temperature scaling으로 과신 완화 (등급 결정에는 불변)
- **OOD/저신뢰 안내** — 분포 외·모호 입력은 거부가 아니라 "결과 유지 + 검토 권고"로 플래그
- **완전 재현 가능** — 시드·split·하이퍼파라미터 고정으로 탐색 결과가 학습에서 동일 재현

---

## 모델 아키텍처

```
                    ┌─────────────────────────┐
  증상 텍스트  ───▶  │   klue/bert-base (BERT)  │
                    └────────────┬────────────┘
                          [CLS] hidden state
                    ┌────────────┴────────────┐
                    ▼                          ▼
            ┌───────────────┐         ┌──────────────────┐
            │   ktas_head   │         │    dept_head     │
            │   Linear → 5  │         │   Linear → 11    │
            │ (K1~K5 중증도)│         │ (진료과, K4/K5만)│
            └───────────────┘         └──────────────────┘
```

- **Backbone**: `klue/bert-base`
- **Head 1 (`ktas_head`)** — 5-class KTAS 중증도 (K1~K5)
- **Head 2 (`dept_head`)** — 11-class 진료과. dept 라벨이 있는 K4/K5 샘플에서만 손실 계산, K1~K3·결측은 `ignore_index=-1`로 자동 제외
- **손실** — `total_loss = ktas_loss + dept_loss_weight · dept_loss`
  - `ktas_loss`: class-weighted CrossEntropy (+ label smoothing)
  - class weight = `base_weight · (√(N / (C·count)))^imbalance_lambda` — 빈도 역가중 + 난이도 보정 결합

### 진료과 클래스 (고정 순서, 학습·서빙 동일)
```
내과 · 비뇨기과 · 산부인과 · 신경과 · 안과 · 이비인후과 · 정신건강의학과 · 정형외과 · 피부과 · 외과 · 소아과
```
> 앞 9개는 가나다순, `외과`·`소아과`는 append 슬롯(향후 데이터 추가용). 인덱스 매핑이 깨지므로 **`sorted()` 재정렬 금지**.

---

## 데이터셋

| 항목 | 값 |
|---|---|
| 총 샘플 | **10,207건** (GPT 생성 가상 데이터) |
| KTAS 분포 | K1 582 / K2 3,522 / K3 2,712 / K4 2,728 / K5 663 |
| 진료과 분포 (K4/K5) | 내과 975 · 이비인후과 415 · 피부과 393 · 정형외과 376 · 비뇨기과 355 · 산부인과 220 · 외과 155 · 안과 152 · 신경과 137 · 정신건강의학과 117 · 소아과 96 |
| Split (seed 42, v4) | train 6,633 / dev 510 / calib 1,532 / test 1,532 |

- **train** — 모델 학습
- **dev** — epoch 선택
- **calib** — 등급 임계치 역산 (학습·선택에 미사용)
- **test** — 최종 평가 1회 (threshold 재보정 없음)

### 데이터 출처

본 저장소의 데이터는 **두 층**으로 구성됩니다.

| 층 | 파일 | 성격 |
|---|---|---|
| ① 분류 체계 | `중증도분류데이터(비정제)/`, `중증도분류데이터(정제)/` | 보건복지부 고시 별표를 표 추출·정제한 것 |
| ② 증상 텍스트 | `ktas_training_data_final.csv`의 `symptom_text` | ①을 프롬프트로 삼아 GPT로 생성한 가상 환자 발화 |

**① 분류 체계 — 원 출처**

> **「한국 응급환자 중증도 분류기준」** (보건복지부고시 제2023-287호)
> 발령 2023. 12. 28. / 시행 2024. 1. 1.
> 국가법령정보센터: https://www.law.go.kr/LSW//admRulInfoP.do?admRulSeq=2100000234560&chrClsCd=010201

`adult_raw.csv`(성인) / `pediatric_raw.csv`(소아)의 대분류·중분류·증상·KTAS 등급 매핑은 위 고시의 별표를 옮긴 것입니다. 이 고시는 국내 응급실 중증도 분류(KTAS, Korean Triage and Acuity Scale)의 법적 근거가 되는 문서입니다.

**가공 절차** (원본 → 학습 데이터)

1. 국가법령정보센터에서 고시 별표 PDF 내려받기 (`중증도분류데이터(정제)/adult_raw.pdf`, `pediatric_raw.pdf`)
2. `pdf_to_csv.py` — `tabula`로 표 추출 → `중증도분류데이터(비정제)/*.csv`
3. 수작업 정제 (병합 셀·줄바꿈 깨짐 보정) → `중증도분류데이터(정제)/*.csv`
4. `all_insert.py` — 성인/소아 병합 + `age_group` 부여 → `ktas_dataset.csv`
5. GPT로 각 증상에 대응하는 가상 환자 발화 생성·증강 → `ktas_training_data_final.csv`

**저작권**

해당 고시는 「저작권법」 제7조제2호(국가 또는 지방자치단체의 고시·공고·훈령 그 밖에 이와 유사한 것)에 따라 **저작권 보호 대상이 아닌 저작물**로, 자유롭게 이용·재배포할 수 있습니다. 다만 아래 두 가지는 이용자가 직접 확인하시기 바랍니다.

- **최신성** — 고시는 개정될 수 있습니다. 본 저장소의 데이터는 위 판본 기준이며, 실제 활용 시 국가법령정보센터에서 현행 고시를 확인하세요.
- **정확성** — 표 추출·정제 과정에서 오류가 있을 수 있습니다. 본 CSV는 **원본 고시를 대체하지 않습니다.** 법적·임상적 판단이 필요한 경우 반드시 원문을 기준으로 하세요.

---

## 성능

최종 모델 (test set, threshold 재보정 없음):

| 지표 | 값 | 목표 | 결과 |
|---|---|---|:--:|
| Macro F1 | **0.7498** | ≥ 0.68 | ✅ |
| K3 Under-triage | **3.44%** | < 5% (hard gate) | ✅ |
| K4 Over-triage | **32.03%** | 25~35% | ✅ |
| Dept Accuracy | **0.8821** | — | — |

**등급별 Recall** — K1 0.897 / K2 0.828 / K3 0.806 / K4 0.641 / K5 0.530

**Confusion Matrix** (행=정답, 열=예측):
```
        K1   K2   K3   K4   K5
 K1  [  78    8    1    0    0 ]
 K2  [  14  438   75    2    0 ]
 K3  [   1   64  328   13    1 ]
 K4  [   0    8  123  262   16 ]
 K5  [   0    1   18   28   53 ]
```

> 배포 설정값: `THRESHOLD_K4=0.88`, `TEMPERATURE=1.178`, `ENTROPY_THRESHOLD=1.2825`, 선택 epoch=6.
>
> ※ 위 수치는 **서빙과 동일한 보수적 상향 로직(K4+K5 합산 게이트, `THRESHOLD_K4=0.88`)** 으로 산출한 test 평가값입니다. 학습·튜너·서빙이 같은 로직을 공유하므로 "평가 수치 = 서빙 동작"이 보장됩니다. (튜너 Best score 751.60 ↔ 학습 Best epoch 6 score 751.60 일치로 재현성 확인)

---

## 핵심 설계 — 임상 안전

### 1) ACS-COT field triage 기준
- **Under-triage < 5%** — 임상 안전 hard gate. 모든 epoch/trial 선택의 **유일한 절대 제약**.
- **Over-triage 25~35%** — 목표 구간. 이탈 거리에 비례해 점수 감점.

> 응급 분류에서 over-triage(경증을 위급하게 봄)는 자원 낭비일 뿐이지만, under-triage(중증을 경증으로 봄)는 환자 안전에 직결됩니다. 그래서 비대칭적으로 under를 강하게 억제합니다.

### 2) 보수적 상향(over-triage) 보정 — K4+K5 합산 게이트

모델이 경증대(K4 또는 K5)로 예측한 경우, **K4+K5 합산 확률(= 비응급 확신)** 을 기준으로 판정합니다.

```
mild_mass = P(K4) + P(K5)
  • mild_mass <  THRESHOLD_K4  → K1~K3 쪽 질량이 남음     → K3로 보수적 상향 (안전)
  • mild_mass >= THRESHOLD_K4  → 경증임이 충분히 확실     → 모델 예측(K4/K5) 유지
```

이 방식은 "불확실성이 **K3 vs K4**(상향 필요)인지, **K4 vs K5**(둘 다 경증, 상향 불필요)인지"를 구분합니다.
예) *"손을 베였다"* → K4 0.51 / K5 0.28 → 합산 0.79 ≥ 임계치 → **K4 유지** (불필요한 K3 과상향 방지).

> 이 보수적 상향 규칙은 **서빙(`main.py`)·학습(`ktas_model_train.py`)·튜너(`ktas_hparam_tuner.py`)** 세 곳에서 동일하게 구현되어 "평가 수치 = 서빙 동작"을 보장합니다.

### 3) 동적 임계치 보정
- calibration split에서 K3 under-triage 목표를 만족하는 **최저 K4 임계치**를 역산
- 임계치 검색 상한 0.88로 과도한 강등 억제
- 임계치 조정은 모델 가중치를 바꾸지 않으므로(argmax 기반) F1에 거의 영향 없이 **under/over만 조정**

### 4) raw vs calibrated 확률 분리
- **raw softmax** — 등급 **결정**용. 안전 임계치가 보정된 바로 그 스케일.
- **calibrated softmax** (temperature scaling) — 화면 **표시 confidence·불확실도**용. argmax 불변이라 등급에는 영향 없음.

### 5) OOD / 저신뢰 처리
- calibration 엔트로피 분포 상위 percentile을 임계치로, 분포 외·모호 입력을 `uncertain`으로 플래그
- **거부가 아니라 결과 유지 + 의료진 검토 권고** — 고중증 가능성을 침묵 처리하지 않음

### 6) 재현성
- DataLoader shuffle 시드 고정 (`generator=torch.Generator().manual_seed(SEED)`)
- 모델 head 초기화 직전 `torch.manual_seed`, `cudnn.deterministic=True`
- tuner·train이 동일 선택 키·임계치 로직을 공유 → 탐색에서 고른 설정이 학습에서 그대로 재현

---

## 설치

```bash
# Python 3.10+ 권장
pip install -r requirements.txt
```

<details>
<summary>구성요소별 의존성 (수동 설치 시)</summary>

```bash
pip install torch transformers scikit-learn pandas numpy   # 공통 (학습·서빙)
pip install optuna                                         # 하이퍼파라미터 탐색 (AI_modeling)
pip install fastapi uvicorn pydantic                       # 서빙 (ktas_backend)
```
</details>

> GPU(CUDA)에서는 AMP(자동 혼합정밀)가 자동 활성화됩니다. CPU·Apple MPS에서도 동작합니다.

### 모델 가중치 준비

저장소에는 **backbone 가중치 `model.safetensors`(422MB)가 포함되어 있지 않습니다.** GitHub의 파일당 100MB 제한 때문입니다. 나머지 산출물(`task_heads.pt`, `tokenizer.json`, `threshold_config.json`, `final_test_metrics.json`)은 모두 포함되어 있습니다.

가중치 없이 `uvicorn main:app`을 실행하면 `model_loader.py`의 `AutoModel.from_pretrained()`에서 실패합니다. 아래 중 하나로 준비하세요.

**A. 직접 학습 (재현)**
```bash
cd AI_modeling
python3 ktas_model_train.py   # → ktas_ai_model/ 전체 재생성
```

**B. 배포된 가중치 내려받기**
Releases에 첨부된 `model.safetensors`를 `ktas_backend/ktas_ai_model/`에 두세요.

> A로 재학습한 경우 `threshold_config.json`·`final_test_metrics.json`도 함께 갱신됩니다. 저장소에 커밋된 값은 [성능](#성능) 표의 산출 근거이므로, 재현 결과와 비교하실 때 참고하세요.

---

## 사용법

### 1) 하이퍼파라미터 탐색 (선택)
```bash
cd AI_modeling
python3 ktas_hparam_tuner.py        # → optuna_results.json (Best 하이퍼파라미터)
```

### 2) 학습 + 평가
```bash
cd AI_modeling
# optuna_results.json의 Best 값을 ktas_model_train.py 상단 상수에 반영한 뒤 실행
python3 ktas_model_train.py         # → ktas_ai_model/ 에 모델·threshold·test 결과 저장
```

### 3) 서빙
```bash
cd ktas_backend                     # ⚠️ 반드시 이 디렉터리에서 (상대경로 ./ktas_ai_model/)
uvicorn main:app --reload --port 8000
# GET /health 로 진료과 11개·threshold 로드 확인
```

---

## API 명세

### `POST /predict`

**요청**
```json
{ "symptom_text": "가슴이 심하게 아파요" }
```

**응답** (`PredictResponse`) — *아래 값은 형식 설명용 예시이며 실제 추론 결과가 아닙니다.*
```json
{
  "ktas_level": 2,
  "is_emergency": true,
  "confidence": 0.817,
  "original_level": null,
  "original_confidence": null,
  "adjusted": false,
  "adjusted_reason": null,
  "dept": null,
  "dept_confidence": null,
  "probabilities": { "KTAS1": 0.02, "KTAS2": 0.82, "KTAS3": 0.13, "KTAS4": 0.02, "KTAS5": 0.01 },
  "status": "ok",
  "uncertainty_notice": null,
  "entropy": 0.61
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `ktas_level` | int | 최종 KTAS 등급 1~5 (보수적 상향 반영) |
| `is_emergency` | bool | `ktas_level <= 3` → 응급(권역센터 라우팅) |
| `confidence` | float | 최종 등급의 확률 (temperature 보정) |
| `original_level` / `original_confidence` | int / float \| null | 상향 전 원예측 (`adjusted=true`일 때만) |
| `adjusted` | bool | 보수적 상향 발생 여부 |
| `adjusted_reason` | str \| null | 상향 사유 문자열 |
| `dept` / `dept_confidence` | str / float \| null | 추천 진료과 (K4/K5에서만) |
| `probabilities` | object | KTAS1~5 전체 확률 분포 |
| `status` | str | `"ok"` \| `"uncertain"` (OOD/저신뢰) |
| `uncertainty_notice` | str \| null | uncertain일 때 안내 문구 |
| `entropy` | float | 예측 분포 엔트로피 (클수록 불확실) |

> 입력 검증: 빈 값/200자 초과/한글 2자 미만/한글 비율 50% 미만은 **HTTP 422**로 거부.

### `GET /health`
로드된 임계치·설정 점검 (`threshold_k4`, `temperature`, `entropy_threshold`, `dept_classes` 등).

---

## 프로젝트 구조

```
KTAS-triage-ai/
├── README.md                       # (본 문서) 전체 총정리
├── LICENSE                         # MIT
├── requirements.txt
├── ktas_training_data_final.csv    # 최종 학습 데이터 (10,207건)
│
├── 중증도분류데이터(비정제)/          # ① 고시 별표 → tabula 표 추출 직후
│   ├── adult_raw.csv
│   └── pediatric_raw.csv
│
├── 중증도분류데이터(정제)/            # ② 수작업 정제 + 병합
│   ├── adult_raw.pdf               #   원본: 보건복지부고시 제2023-287호 별표 (성인)
│   ├── pediatric_raw.pdf           #   원본: 〃 (소아)
│   ├── pdf_to_csv.py               #   PDF → CSV 표 추출
│   ├── adult_raw.csv
│   ├── pediatric_raw.csv
│   ├── all_insert.py               #   성인+소아 병합, age_group 부여
│   └── ktas_dataset.csv            #   병합 결과
│
├── AI_modeling/                    # 학습 파이프라인
│   ├── ktas_hparam_tuner.py        #   Optuna 탐색 (EPOCHS=6, N_TRIALS=20)
│   ├── ktas_model_train.py         #   학습 + 최종 test 평가
│   ├── optuna_results.json         #   탐색 Best 결과
│   └── README.md                   #   모델 상세 문서
│
└── ktas_backend/                   # 서빙
    ├── main.py                     #   FastAPI 엔드포인트 (/predict, /health)
    ├── model_loader.py             #   서빙용 모델 로더
    └── ktas_ai_model/              #   학습 산출물
        ├── model.safetensors       #   ⚠️ 미포함 (422MB) — 설치 > 모델 가중치 준비 참조
        ├── config.json             #   backbone 설정
        ├── tokenizer.json          #   토크나이저
        ├── task_heads.pt           #   ktas_head + dept_head 가중치
        ├── threshold_config.json   #   임계치·temperature·entropy
        └── final_test_metrics.json #   test 평가 결과
```

> 데이터 생성 파이프라인(GPT 증상 텍스트 생성)은 [ER-Data-Generator](https://github.com/gwonh-l/ER-Data-Generator) 저장소에 있으며, 본 저장소에는 최종 산출물 CSV만 포함되어 있습니다.
> `ktas_training_data_final.csv`는 `AI_modeling/`·`ktas_backend/`에도 동일 사본이 있습니다(각 스크립트가 상대경로로 참조). 수정 시 세 곳을 함께 갱신하세요.

---

## 알려진 한계 / 향후 과제

- **학습 데이터가 실제 환자 발화가 아님 (가장 근본적인 한계)** — 본 모델은 KTAS 분류기준표를 바탕으로 **GPT가 생성한 가상 증상 텍스트**로 학습되었습니다. 생성 텍스트는 기준표의 증상 서술을 비교적 충실히 따르므로, 증상과 등급의 대응이 실제 환자 발화보다 정제되어 있습니다. 실제 응급실 발화는 여러 증상이 뒤섞이고, 표현이 모호하며, 비의학적 어휘와 불필요한 정보가 함께 들어옵니다. 따라서 **위 성능 수치는 실제 임상 환경에서의 성능을 과대평가할 가능성이 높습니다.** 실제 응급실 데이터를 이용한 외부 검증(external validation)은 수행되지 않았습니다.
- **데이터 증강 leak 가능성** — 증강 샘플이 split 이전에 혼입되어 유사 문장이 train/test에 분산될 수 있습니다. 그룹 단위(원문 인덱스 기준) split 전환을 향후 과제로 둡니다.
- **K3/K4 경계 혼동** — KTAS 자체가 평가자 간 일치도가 가장 낮은 구간이며, 모델의 K3/K4 혼동도 이 본질적 난이도를 반영합니다.
- **K5 Recall (0.53)** — 최소 클래스(K5)의 recall이 상대적으로 낮습니다. 임상 안전(상향 방향)에는 무해하나 F1 손해 요인.
- **무의미 입력 방어** — 완성형 무의미 입력(예: "아아아")은 규칙 검증을 통과할 수 있어, 근본 차단은 엔트로피 기반 OOD reject에 의존합니다.

> 현재 test 지표는 학습·서빙 동일 로직으로 산출되어 서빙 동작과 일치하나, 데이터 증강 leak 가능성(위 참조)으로 인해 다소 낙관적으로 해석될 여지가 있습니다. 안전성에 대한 최종 결론은 **그룹 단위 split 재평가** 이후로 유보하는 것을 권장합니다.

---

## 참고 문헌
- Korean Triage and Acuity Scale (KTAS) 신뢰도 연구 (JKMS, 2019) — 평가자 간 일치도(weighted-κ 0.772) 및 K3/K4 불일치 경향
- ACS Committee on Trauma — field triage under-/over-triage 기준

---

## 라이선스

- **코드**: [MIT License](LICENSE)
- **데이터셋(`symptom_text`)**: GPT로 생성한 **가상 데이터**이며 실제 환자 정보(PHI)를 포함하지 않습니다.
- **중증도 분류 체계 데이터**: 「한국 응급환자 중증도 분류기준」(보건복지부고시 제2023-287호) 별표에서 파생. 「저작권법」 제7조제2호에 따라 저작권 보호 대상이 아닙니다. → [데이터 출처](#데이터-출처)
- **모델 가중치**: 용량 제한으로 `model.safetensors`(422MB)는 저장소에 포함되지 않습니다. → [모델 가중치 준비](#모델-가중치-준비)

---

*Built with PyTorch · Hugging Face Transformers · Optuna · FastAPI*
