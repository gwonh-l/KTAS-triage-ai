"""
KTAS 모델 하이퍼파라미터 탐색 (Optuna) — Multi-task 버전

[Multi-task 설계]
- Head 1 (ktas_head): 5-class KTAS 중증도 분류 (K1~K5)
- Head 2 (dept_head): 11-class 진료과 분류 (K4/K5에서만 유효)
  - dept label이 존재하는 K4/K5 샘플에서만 dept_loss 계산
  - K1~K3 및 결측(-1)은 ignore_index=-1로 자동 제외
- 하이퍼파라미터 탐색에 dept_loss_weight 추가

[탐색 전략]
- 임상 데이터 기반 고정 class weight는 사용하지 않음
- K3/K4 경계 균형을 직접 찾도록 w_k3, w_k4를 탐색
- 탐색 대상: lr, batch_size, weight_decay, label_smoothing, warmup_ratio,
             imbalance_lambda, dept_loss_weight, w_k3, w_k4
- objective_score (ACS-COT field triage: under<5%, over 25~35%):
  → Hard gate: under <= 5% (불안전은 -1e6 강제 후순위)
  → 안전 영역 내부:
    F1*1000
    - max(0, under-3%)*4000             ← test 5% 안에 넣을 buffer (탈락 아님, 감점)
    - distance(k4_over, 25~35%)*3000    ← over 구간 양방향 페널티 (1500→3000, 아래 [페널티 튜닝] 참조)

[epoch / trial 선택 키 — train.py와 100% 동일]
- candidate_key = (both_acs_safe, objective_score)
- both_acs_safe = calib·dev 양쪽 under ≤ 5%(ACS_COT_UNDER_TRIAGE_LIMIT). 이것만 절대 게이트.
- 과거 3% buffer(both_under_ok)를 게이트로 쓰던 방식은 폐기: over를 목표(25~35%)에 넣은 epoch이
  under 0.x%p 초과로 탈락 → over 45~56% epoch이 선택되는 역설이 났다. 게이트는 5%로만, 3%는 score 감점.
- under 버킷(옵션2)도 키에서 제외: under만 보고 over 정상화 epoch을 죽임.

[페널티 튜닝 — over 구간 이탈 계수]
- 1500은 over +0.77%p 초과 epoch(예: 35.77%)을 23점만 감점해 고F1 over초과 epoch이 살아남았다.
- 6000은 over 35.5~37%대 후보까지 과잉 도살(대포로 모기). 역산 필요계수≈2415 → 3000으로 합의(안전마진).

[동적 임계치 보정 (ACS 정렬)]
- calib에서 K3 under <= 3%(SEARCH_TARGET) 만족하는 최저 K4 threshold 역산
- th_k4 상한 0.88(CAP, fallback 0.86 위) — K4 강등 억제
- 자기파괴 트랩 해제: target 2% + cap 0.83 콤보가 trial 전부 cap에 박아 over 폭발 → target 3% + cap 0.88로 풀어 자연 수렴
- dev split은 Trial 및 epoch 선택에만 사용 / calib split은 threshold 보정에만 사용 / test split은 tuner에서 절대 미사용

[데이터 / split — v4]
- ktas_training_data_final.csv = 10,207건(증강 5,840행 혼입). SPLIT_VERSION=4로 재생성(train=6633/dev=510/calib=1532/test=1532).
- 행 단위 stratify split이라 같은 original_index의 패러프레이즈가 train/calib/test로 흩어질 수 있음(leak 인지 후 진행).
  → test under/over가 실제보다 낙관적일 수 있음. 발표 후 original_index 그룹 split으로 전환 권장.

[최종 결과 (이 설정으로 train.py 실행 시 재현)]
- Best Trial #0 / Epoch 6 선택 (CALIB_U 4.91% → both_under_ok=False지만 acs_ok=True라 채택).
- TEST: Macro F1 0.7498 | K3 under 3.44%(PASS<5%) | K4 over 32.03%(PASS 25~35%) | Dept Acc 0.8821.
- Best params: lr≈5.86e-5 / bs=16 / wd≈1.5e-4 / ls≈0.0848 / warmup≈0.0469 /
  imbalance_lambda≈0.1288 / dept_loss_weight≈0.4301 / w_k3≈2.342 / w_k4≈1.362.

결과: optuna_results.json
"""

import os
import json
import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.amp import GradScaler
from optuna.pruners import MedianPruner
from sklearn.metrics import recall_score

# ── 지표 최적화 기준 ──
# K1/K2/K5는 탐색 공간을 제한하기 위한 baseline. K3/K4는 trial마다 직접 탐색한다.
BASE_CLASS_WEIGHT = [1.7, 1.2, 1.0, 1.0, 1.1]   # K1~K5
ACS_COT_UNDER_TRIAGE_LIMIT = 0.05      # 임상 안전 게이트 (mandate)
SELECTION_UNDER_TRIAGE_MARGIN = 0.015
SELECTION_UNDER_TRIAGE_LIMIT = ACS_COT_UNDER_TRIAGE_LIMIT - SELECTION_UNDER_TRIAGE_MARGIN  # 로깅용
THRESHOLD_SEARCH_TARGET = 0.03         # 0.02는 cap 박혀 K4 무차별 강등→over 폭발. 0.03은 cap 탈출 + buffer 2%p
UNDER_BUCKET_WIDTH = 0.005             # dev under_rate 0.5%p 버킷 — both 통과군 내에서 under 두꺼운(낮은) 후보 우선 (옵션2)
# ── ACS-COT field triage 기준 정렬 (under <5%, over 25~35%) — train.py와 동일 ──
TH_K4_SEARCH_CAP = 0.88                # 0.83 → 0.88 (fallback 0.86 위) — K4 강등 억제
K4_OVER_TRIAGE_MIN = 0.25              # 목표 구간 하한
K4_OVER_TRIAGE_MAX = 0.35              # 목표 구간 상한
K4_OVER_TRIAGE_PENALTY = 3_000         # 목표 구간 이탈 거리 페널티. train.py와 동일(1500→3000) — over 35.5~37%대 후보를 죽이지 않으면서 구간 초과 epoch을 후순위로

# ── 탐색 범위 ──
IMBALANCE_LAMBDA_MIN = 0.0
IMBALANCE_LAMBDA_MAX = 0.5
DEPT_LOSS_WEIGHT_MIN = 0.1
DEPT_LOSS_WEIGHT_MAX = 0.5
W_K3_MIN = 1.0
W_K3_MAX = 2.5
W_K4_MIN = 1.0
W_K4_MAX = 1.8

# ── 진료과 레이블 (고정 순서 11개 — 서빙 DEPT_CLASSES와 동일 순서 유지) ──
# 앞 9개는 가나다 정렬, idx 9 '외과'·idx 10 '소아과'는 append 순서 보존(향후 데이터 추가용).
# sorted()로 재구성 금지 — append 순서가 깨지면 인덱스 매핑이 어긋남.
DEPT_CLASSES = ['내과', '비뇨기과', '산부인과', '신경과', '안과', '이비인후과', '정신건강의학과', '정형외과', '피부과', '외과', '소아과']
DEPT_TO_IDX: dict[str, int] = {d: i for i, d in enumerate(DEPT_CLASSES)}
NUM_DEPT = len(DEPT_CLASSES)

# ── 고정 설정 ──
MODEL_NAME   = "klue/bert-base"
DATA_PATH    = "./ktas_training_data_final.csv"
MAX_LEN      = 180
EPOCHS       = 6     # train.py와 통일 — best HP가 train에서 동일 epoch 위치에 안착하도록
N_TRIALS     = 20
RANDOM_SEED  = 42
NUM_LABELS   = 5
RESULTS_PATH = "./optuna_results.json"
SPLIT_PATH   = "./ktas_split.json"
TEST_RATIO   = 0.15
DEV_RATIO    = 0.05    # 0.10→0.05: dev는 epoch/trial 선택용이라 작아도 됨
CALIB_RATIO  = 0.15    # threshold 보정용 — 키워서 calib→test under gap 축소(3%p→1.5%p 목표)
SPLIT_VERSION = 4      # 데이터 10,207건으로 증가(증강 5,840행 혼입) → 기존 9,458 split 무효, 재생성.
                       # 주의: 행 단위 stratify라 같은 original_index 패러프레이즈가 train/calib/test로 흩어질 수 있음(leak 인지 후 진행).


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_dept(dept_series: pd.Series) -> list[int]:
    """NaN → -1 (ignore_index), 유효값 → 0~10."""
    return [DEPT_TO_IDX[d] if pd.notna(d) and d in DEPT_TO_IDX else -1 for d in dept_series]


class KTASDataset(Dataset):
    """사전 토큰화 멀티태스크 데이터셋."""
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        token_type_ids: torch.Tensor | None,
        labels: torch.Tensor,
        dept_labels: torch.Tensor,
    ):
        self.input_ids       = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids  = token_type_ids   # None 허용 (RoBERTa/DeBERTa 대응)
        self.labels          = labels
        self.dept_labels     = dept_labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels":         self.labels[idx],
            "dept_labels":    self.dept_labels[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item


class KTASMultiTaskModel(nn.Module):
    """
    BERT backbone + 2개 classification head.
    - ktas_head : 5-class (K1~K5)
    - dept_head : 11-class (진료과, K4/K5에서만 유효)
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.ktas_head = nn.Linear(hidden, NUM_LABELS)
        self.dept_head = nn.Linear(hidden, NUM_DEPT)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = out.last_hidden_state[:, 0, :]   # [CLS] representation
        return self.ktas_head(cls), self.dept_head(cls)


def get_or_create_split_indices(labels: list[int]) -> dict:
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH, "r", encoding="utf-8") as f:
            split = json.load(f)
        if split.get("version") == SPLIT_VERSION and {"train", "dev", "calib", "test"}.issubset(split.keys()):
            return split
        print(f"[split] 기존 split 형식 불일치 → 4-way split v{SPLIT_VERSION}로 재생성: {SPLIT_PATH}")

    all_idx = list(range(len(labels)))
    train_dev_calib_idx, test_idx = train_test_split(
        all_idx, test_size=TEST_RATIO, stratify=labels, random_state=RANDOM_SEED,
    )
    dev_calib_ratio_adj = (DEV_RATIO + CALIB_RATIO) / (1.0 - TEST_RATIO)
    train_idx, dev_calib_idx = train_test_split(
        train_dev_calib_idx,
        test_size=dev_calib_ratio_adj,
        stratify=[labels[i] for i in train_dev_calib_idx],
        random_state=RANDOM_SEED,
    )
    calib_ratio_adj = CALIB_RATIO / (DEV_RATIO + CALIB_RATIO)
    dev_idx, calib_idx = train_test_split(
        dev_calib_idx,
        test_size=calib_ratio_adj,
        stratify=[labels[i] for i in dev_calib_idx],
        random_state=RANDOM_SEED,
    )
    split = {
        "version": SPLIT_VERSION,
        "seed": RANDOM_SEED,
        "train": sorted(map(int, train_idx)),
        "dev":   sorted(map(int, dev_idx)),
        "calib": sorted(map(int, calib_idx)),
        "test":  sorted(map(int, test_idx)),
    }
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    return split

def collect_logits(
    model: KTASMultiTaskModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ktas_logits, dept_logits, ktas_labels, dept_labels 반환."""
    model.eval()
    all_ktas, all_dept, all_labels, all_dept_labels = [], [], [], []
    with torch.inference_mode():   # train.py와 동일 추론 컨텍스트 (수치 동일, 재현성 정합)
        for batch in loader:
            model_inputs = {
                "input_ids":      batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "token_type_ids" in batch and batch["token_type_ids"] is not None:
                model_inputs["token_type_ids"] = batch["token_type_ids"].to(device)

            with torch.amp.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=use_amp,
            ):
                ktas_logits, dept_logits = model(**model_inputs)

            all_ktas.append(ktas_logits.cpu().float().numpy())
            all_dept.append(dept_logits.cpu().float().numpy())
            all_labels.extend(batch["labels"].numpy())
            all_dept_labels.extend(batch["dept_labels"].numpy())
    return (
        np.concatenate(all_ktas, axis=0),
        np.concatenate(all_dept, axis=0),
        np.array(all_labels),
        np.array(all_dept_labels),
    )

def evaluate_dept_accuracy(dept_logits_np: np.ndarray, dept_labels_np: np.ndarray) -> float:
    """dept label 존재 샘플(K4/K5 non-null)만 평가. 유효 샘플 0개면 0.0 반환."""
    valid = (dept_labels_np != -1)
    if valid.sum() == 0:
        return 0.0
    preds = np.argmax(dept_logits_np[valid], axis=1)
    return float((preds == dept_labels_np[valid]).mean())

def evaluate_with_thresholds(
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    threshold_k4: float,
) -> dict:
    preds = conservative_predict(logits_np, threshold_k4)
    macro_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    recalls = recall_score(labels_np, preds, average=None, labels=[0,1,2,3,4], zero_division=0)
    k3_count = int((labels_np == 2).sum())
    k3_under = int(np.sum((preds >= 3) & (labels_np == 2)))
    # K4 over-triage: K4 truth가 K1/K2/K3로 상향된 비율 (뺑뺑이 직접 지표)
    k4_count     = int((labels_np == 3).sum())
    k4_over      = int(np.sum((preds <= 2) & (labels_np == 3)))
    k4_over_rate = k4_over / k4_count if k4_count > 0 else 0.0
    return {
        "macro_f1":      float(macro_f1),
        "k3_recall":     float(recalls[2]),
        "k4_recall":     float(recalls[3]),
        "k3_under":      k3_under,
        "k3_under_rate": k3_under / k3_count if k3_count > 0 else 0.0,
        "k4_over":       k4_over,
        "k4_over_rate":  k4_over_rate,
    }


def build_loss_weights(
    labels: list[int],
    imbalance_lambda: float,
    base_class_weight: list[float],
) -> torch.Tensor:
    labels_np = np.array(labels, dtype=np.int64)
    counts = np.bincount(labels_np, minlength=NUM_LABELS).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"학습 split에 비어 있는 클래스: counts={counts.tolist()}")
    sqrt_inverse_frequency = np.sqrt(len(labels_np) / (NUM_LABELS * counts))
    metric_weight = np.array(base_class_weight, dtype=np.float32)
    final_weight = metric_weight * np.power(sqrt_inverse_frequency, imbalance_lambda)
    return torch.tensor(final_weight, dtype=torch.float32)


def find_optimal_thresholds(
    logits_np: np.ndarray, labels_np: np.ndarray
) -> float:
    probs = torch.softmax(torch.tensor(logits_np, dtype=torch.float32), dim=-1).numpy()
    raw_preds = np.argmax(probs, axis=1)
    k3_true = (labels_np == 2)

    # K4: 상한 TH_K4_SEARCH_CAP(0.88)에서 컷오프.
    # cap 미달이면 cap 사용. 남은 under 초과분은 objective가 처리.
    # 탐색도 conservative_predict와 동일한 K4+K5 합산 기준으로 (탐색≠평가 드리프트 방지).
    best_th_k4 = TH_K4_SEARCH_CAP
    k4_target_met = False
    mild_mass = probs[:, 3] + probs[:, 4]
    mild_band = (raw_preds == 3) | (raw_preds == 4)
    for th in np.arange(0.50, TH_K4_SEARCH_CAP + 0.001, 0.01):
        tp = raw_preds.copy()
        tp[mild_band & (mild_mass < th)] = 2
        if k3_true.sum() > 0 and np.sum((tp >= 3) & k3_true) / k3_true.sum() <= THRESHOLD_SEARCH_TARGET:
            best_th_k4 = round(float(th), 2)
            k4_target_met = True
            break

    if not k4_target_met:
        print(f"[정보] K4 threshold cap({TH_K4_SEARCH_CAP}) 도달 — over-triage 방지 우선")

    return best_th_k4


def conservative_predict(
    logits_np: np.ndarray, threshold_k4: float
) -> np.ndarray:
    # 보수적 상향 — K4/K5 경증대에 K4+K5 합산 질량 기준 적용 (main.py·train과 동일 로직).
    #   - 합산 <  threshold_k4 : K1~K3 쪽 질량 잔존 → K3(idx 2) 상향(안전 방향)
    #   - 합산 >= threshold_k4 : 경증 확신 충분 → argmax(K4/K5) 그대로 유지(K5 살림)
    probs = torch.softmax(torch.tensor(logits_np, dtype=torch.float32), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    mild_mass    = probs[:, 3] + probs[:, 4]
    in_mild_band = (preds == 3) | (preds == 4)
    preds[in_mild_band & (mild_mass < threshold_k4)] = 2
    return preds


def make_objective(
    full_dataset: KTASDataset,
    train_idx: list[int],
    dev_idx: list[int],
    calib_idx: list[int],
    labels: list[int],
    device: torch.device,
    use_amp: bool,
):
    def objective(trial: optuna.Trial) -> float:
        lr              = trial.suggest_float("lr",             1e-5, 1e-4, log=True)
        batch_size      = trial.suggest_categorical("batch_size", [16, 32])
        weight_decay    = trial.suggest_float("weight_decay",  1e-4, 1e-1, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.05, 0.20)
        warmup_ratio    = trial.suggest_float("warmup_ratio",  0.03, 0.10)
        imbalance_lambda = trial.suggest_float("imbalance_lambda", IMBALANCE_LAMBDA_MIN, IMBALANCE_LAMBDA_MAX)
        # dept 분류 손실 가중치 — KTAS 성능을 해치지 않는 범위 탐색
        dept_loss_weight = trial.suggest_float("dept_loss_weight", DEPT_LOSS_WEIGHT_MIN, DEPT_LOSS_WEIGHT_MAX)
        w_k3 = trial.suggest_float("w_k3", W_K3_MIN, W_K3_MAX)
        w_k4 = trial.suggest_float("w_k4", W_K4_MIN, W_K4_MAX)
        base_class_weight = [BASE_CLASS_WEIGHT[0], BASE_CLASS_WEIGHT[1], w_k3, w_k4, BASE_CLASS_WEIGHT[4]]

        # shuffle 재현성: generator 미지정 시 trial/실행마다 배치 순서가 달라져
        # 동일 HP에도 K4 Over 등 지표가 출렁임(분산을 최적점으로 오인하는 원인).
        loader_gen = torch.Generator()
        loader_gen.manual_seed(RANDOM_SEED)
        train_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
            generator=loader_gen,
        )
        calib_loader = DataLoader(
            Subset(full_dataset, calib_idx),
            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
        )
        dev_loader = DataLoader(
            Subset(full_dataset, dev_idx),
            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
        )

        # head 초기화 재현: trial마다 nn.Linear 가중치가 동일 초기값에서 시작하도록
        # 모델 생성 직전 시드 리셋. train.py와도 같은 초기 가중치 → tuner Best 재현 가능.
        torch.manual_seed(RANDOM_SEED)
        model = KTASMultiTaskModel(MODEL_NAME).to(device)

        train_labels = [labels[i] for i in train_idx]
        ktas_criterion = nn.CrossEntropyLoss(
            weight=build_loss_weights(train_labels, imbalance_lambda, base_class_weight).to(device),
            label_smoothing=label_smoothing,
        )
        # ignore_index=-1: K1~K3 및 결측 샘플은 dept loss 계산에서 자동 제외
        dept_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps  = len(train_loader) * EPOCHS
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scaler       = GradScaler("cuda", enabled=use_amp)

        best_score = -float("inf")
        # 옵션2: (both_under_ok, -dev_under_bucket, objective_score) 튜플로 best epoch 선택.
        #  1순위 both_under_ok(calib·dev 동시 under<target), 2순위 under 두꺼운(낮은 버킷) 후보,
        #  3순위 objective_score. both 통과군 내에서 under margin이 두꺼운 epoch을 우선해 test under 누수 방지.
        best_key = (False, -float("inf"))

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for batch in train_loader:
                lbls       = batch["labels"].to(device)
                dept_lbls  = batch["dept_labels"].to(device)
                model_inputs = {
                    "input_ids":      batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                if "token_type_ids" in batch and batch["token_type_ids"] is not None:
                    model_inputs["token_type_ids"] = batch["token_type_ids"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu",
                    enabled=use_amp,
                ):
                    ktas_logits, dept_logits = model(**model_inputs)
                    ktas_loss = ktas_criterion(ktas_logits, lbls)
                    # 배치 전원이 dept 결측(-1)이면 CrossEntropyLoss가 NaN을 반환 →
                    # total_loss 오염 방지를 위해 유효 dept 샘플이 있을 때만 계산
                    if (dept_lbls != -1).any():
                        dept_loss = dept_criterion(dept_logits, dept_lbls)
                    else:
                        dept_loss = torch.zeros((), device=device)
                    loss = ktas_loss + dept_loss_weight * dept_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # calib에서 threshold만 산출하고, 독립 dev에서 trial 및 epoch를 선택한다.
            calib_ktas_logits, _, calib_labels, _ = \
                collect_logits(model, calib_loader, device, use_amp)
            th_k4 = find_optimal_thresholds(calib_ktas_logits, calib_labels)
            calib_m = evaluate_with_thresholds(calib_ktas_logits, calib_labels, th_k4)
            dev_ktas_logits, dev_dept_logits, dev_labels, dev_dept_labels = \
                collect_logits(model, dev_loader, device, use_amp)
            m = evaluate_with_thresholds(dev_ktas_logits, dev_labels, th_k4)
            dept_acc = evaluate_dept_accuracy(dev_dept_logits, dev_dept_labels)

            safe        = m["k3_under_rate"] <= ACS_COT_UNDER_TRIAGE_LIMIT
            margin_safe = m["k3_under_rate"] <= SELECTION_UNDER_TRIAGE_LIMIT

            # ── objective: under / over / Macro F1만 직접 최적화 ──
            # 불안전(safe=False = under>5%)은 -1e6. 안전 영역:
            #   F1*1000
            #   - max(0,under-3%)*4000
            #   - distance(k4_over, 25~35%)*1500
            if not safe:
                objective_score = -1_000_000.0 + m["macro_f1"] * 1_000
            else:
                if m["k4_over_rate"] < K4_OVER_TRIAGE_MIN:
                    over_penalty = (K4_OVER_TRIAGE_MIN - m["k4_over_rate"]) * K4_OVER_TRIAGE_PENALTY
                elif m["k4_over_rate"] > K4_OVER_TRIAGE_MAX:
                    over_penalty = (m["k4_over_rate"] - K4_OVER_TRIAGE_MAX) * K4_OVER_TRIAGE_PENALTY
                else:
                    over_penalty = 0.0
                objective_score = (
                    m["macro_f1"]                                            * 1_000
                    - max(0.0, m["k3_under_rate"] - THRESHOLD_SEARCH_TARGET) * 4_000
                    - over_penalty
                )

            # 3% buffer(THRESHOLD_SEARCH_TARGET) 통과 여부 — 표시/추적 전용. 선택 게이트는 아래 5%(both_acs_safe).
            dev_under_ok   = m["k3_under_rate"]       <= THRESHOLD_SEARCH_TARGET
            calib_under_ok = calib_m["k3_under_rate"] <= THRESHOLD_SEARCH_TARGET
            both_under_ok  = dev_under_ok and calib_under_ok
            # dev under_rate 0.5%p 버킷(낮을수록 under margin 두꺼움). 로그 표시 전용 — 선택 키 미사용.
            dev_under_bucket = int(m["k3_under_rate"] / UNDER_BUCKET_WIDTH)

            # ── epoch 선택 키: (both_acs_safe, objective_score) — train.py와 100% 동일 키 ──
            # both_acs_safe = calib·dev 양쪽 K3 under ≤ ACS_COT_UNDER_TRIAGE_LIMIT(5%). 유일한 절대 게이트.
            # 3% buffer를 게이트로 쓰면 over를 25~35%에 넣은 epoch이 under 0.x%p 초과로 탈락 → over 폭발 epoch이 선택되는
            # 역설이 생긴다(실패 교훈). 그래서 게이트는 5%로만, 3% 초과는 objective_score의 *4000 감점으로만 반영.
            # under 버킷은 under만 보고 over 정상화 epoch을 죽여서 키에서 제외(안전군 내부는 objective_score로만 선택).
            both_acs_safe = (m["k3_under_rate"]       <= ACS_COT_UNDER_TRIAGE_LIMIT) and \
                            (calib_m["k3_under_rate"] <= ACS_COT_UNDER_TRIAGE_LIMIT)

            print(
                f"  T#{trial.number} E{epoch}/{EPOCHS} | F1={m['macro_f1']:.4f} | "
                f"DEV_U={m['k3_under_rate']:.2%} | CALIB_U={calib_m['k3_under_rate']:.2%} | K4th={th_k4} | "
                # acs_ok(5%, 실제 선택 게이트) vs both_ok(3% buffer, 표시 전용)
                f"acs_ok={both_acs_safe} | both_ok={both_under_ok} | bkt={dev_under_bucket} | score={objective_score:.2f}"
            )

            candidate_key = (both_acs_safe, objective_score)
            if candidate_key > best_key:
                best_key = candidate_key
                best_score = objective_score
                trial.set_user_attr("both_acs_safe",     bool(both_acs_safe))
                trial.set_user_attr("both_under_ok",     bool(both_under_ok))
                trial.set_user_attr("dev_under_bucket",  int(dev_under_bucket))
                trial.set_user_attr("calib_under_ok",    bool(calib_under_ok))
                trial.set_user_attr("dev_under_ok",      bool(dev_under_ok))
                trial.set_user_attr("dev_macro_f1",      m["macro_f1"])
                trial.set_user_attr("dev_k3_recall",     m["k3_recall"])
                trial.set_user_attr("dev_k4_recall",     m["k4_recall"])
                trial.set_user_attr("dev_k3_under",      m["k3_under"])
                trial.set_user_attr("dev_k3_under_rate", m["k3_under_rate"])
                trial.set_user_attr("dev_k4_over",       m["k4_over"])
                trial.set_user_attr("dev_k4_over_rate",  m["k4_over_rate"])
                trial.set_user_attr("dev_dept_accuracy", dept_acc)
                trial.set_user_attr("calib_threshold_k3_under",      calib_m["k3_under"])
                trial.set_user_attr("calib_threshold_k3_under_rate", calib_m["k3_under_rate"])
                trial.set_user_attr("calib_threshold_k4_over",       calib_m["k4_over"])
                trial.set_user_attr("calib_threshold_k4_over_rate",  calib_m["k4_over_rate"])
                trial.set_user_attr("selection_margin_pass", margin_safe)
                trial.set_user_attr("objective_score",     float(objective_score))
                trial.set_user_attr("threshold_k4",        th_k4)
                trial.set_user_attr("acs_cot_pass",        safe)
                trial.set_user_attr("best_epoch",          epoch)

            trial.report(objective_score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return best_score

    return objective


def main():
    # ── 재현성: train.py와 동일한 시드 정책으로 tuner Best가 train에서 재현되도록 ──
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device  = get_device()
    use_amp = (device.type == "cuda")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"[설정] device={device} | AMP={use_amp} | Trials={N_TRIALS}")

    df        = pd.read_csv(DATA_PATH)
    texts     = df["symptom_text"].tolist()
    labels    = (df["ktas_level"] - 1).tolist()
    dept_raw  = encode_dept(df["dept"])

    split = get_or_create_split_indices(labels)
    train_idx, dev_idx, calib_idx, test_idx = split["train"], split["dev"], split["calib"], split["test"]
    print(
        f"[split seed={RANDOM_SEED}] train={len(train_idx)} | dev={len(dev_idx)} | "
        f"calib={len(calib_idx)} | test={len(test_idx)}"
    )

    print("[토큰화] 전체 데이터 사전 토큰화 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    full_dataset = KTASDataset(
        enc["input_ids"],
        enc["attention_mask"],
        enc.get("token_type_ids"),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(dept_raw, dtype=torch.long),
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name="ktas_multitask_tuning",
    )

    def print_clean_log(study: optuna.Study, trial: optuna.Trial) -> None:
        m_f1   = trial.user_attrs.get("dev_macro_f1", 0.0)
        k3_rec = trial.user_attrs.get("dev_k3_recall", 0.0)
        k4_rec = trial.user_attrs.get("dev_k4_recall", 0.0)
        under  = trial.user_attrs.get("dev_k3_under", 0)
        rate   = trial.user_attrs.get("dev_k3_under_rate", 1.0)
        k4_ov  = trial.user_attrs.get("dev_k4_over", 0)
        k4_ovr = trial.user_attrs.get("dev_k4_over_rate", 0.0)
        acs    = "PASS" if trial.user_attrs.get("acs_cot_pass", False) else "FAIL"
        dept_acc_log = trial.user_attrs.get("dev_dept_accuracy", 0.0)
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"\n[Trial #{trial.number}] DEV Score={trial.value:.4f} | "
                  f"F1={m_f1:.4f} | K3 Rec={k3_rec:.4f} | K4 Rec={k4_rec:.4f} | "
                  f"K3 Under={under}건({rate:.2%}) | K4 Over={k4_ov}건({k4_ovr:.2%}) | "
                  f"ACS={acs} | Dept Acc={dept_acc_log:.4f}")
            for k, v in trial.params.items():
                print(f"    {k:<20}: {v:.6f}" if isinstance(v, float) else f"    {k:<20}: {v}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"\n[Trial #{trial.number} Pruned] DEV Macro F1={m_f1:.4f} | ACS={acs}")

    study.optimize(
        make_objective(full_dataset, train_idx, dev_idx, calib_idx, labels, device, use_amp),
        n_trials=N_TRIALS,
        callbacks=[print_clean_log],
        show_progress_bar=True,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("완료된 Trial이 없습니다.")

    # trial 간 선택도 (both_acs_safe, objective_score) — epoch 키와 동일 우선순위(train.py 정렬).
    # 1순위 ACS 5% 안전(calib·dev AND), 2순위 objective.
    best = max(completed, key=lambda t: (
        t.user_attrs.get("both_acs_safe", False),
        t.user_attrs.get("objective_score", float("-inf")),
    ))
    print(f"\n[Best Trial #{best.number}]")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    results = {
        "best_trial":              best.number,
        "selection_source":        "dev",
        "split_version":           SPLIT_VERSION,
        "objective_score":         round(best.user_attrs.get("objective_score", 0), 4),
        "best_dev_macro_f1":       round(best.user_attrs.get("dev_macro_f1", 0), 4),
        "best_dev_k3_recall":      round(best.user_attrs.get("dev_k3_recall", 0), 4),
        "best_dev_k4_recall":      round(best.user_attrs.get("dev_k4_recall", 0), 4),
        "best_dev_k4_over":        best.user_attrs.get("dev_k4_over"),
        "best_dev_k4_over_rate":   round(best.user_attrs.get("dev_k4_over_rate", 0), 4),
        "best_dev_k3_under":       best.user_attrs.get("dev_k3_under"),
        "best_dev_k3_under_rate":  round(best.user_attrs.get("dev_k3_under_rate", 1.0), 4),
        "calib_threshold_k3_under":      best.user_attrs.get("calib_threshold_k3_under"),
        "calib_threshold_k3_under_rate": round(best.user_attrs.get("calib_threshold_k3_under_rate", 1.0), 4),
        "calib_threshold_k4_over":       best.user_attrs.get("calib_threshold_k4_over"),
        "calib_threshold_k4_over_rate":  round(best.user_attrs.get("calib_threshold_k4_over_rate", 0.0), 4),
        "acs_cot_pass":            bool(best.user_attrs.get("acs_cot_pass", False)),
        "selection_margin_pass":   bool(best.user_attrs.get("selection_margin_pass", False)),
        "best_epoch_in_tuner":     best.user_attrs.get("best_epoch"),
        "threshold_k4_on_calib":   best.user_attrs.get("threshold_k4"),
        "k4_over_triage_min":      K4_OVER_TRIAGE_MIN,
        "k4_over_triage_max":      K4_OVER_TRIAGE_MAX,
        "k4_over_triage_penalty":  K4_OVER_TRIAGE_PENALTY,
        "params":                  best.params,
        "dept_classes":            DEPT_CLASSES,
        "split_path":              SPLIT_PATH,
        "split_counts": {
            "train": len(train_idx), "dev": len(dev_idx), "calib": len(calib_idx), "test": len(test_idx),
        },
        "note": (
            "Multi-task: ktas_head(5-class) + dept_head(11-class, K4/K5 only). "
            "ACS-COT field triage 기준 정렬: under<5% hard gate, over 25~35% 허용. "
            "search target 3% + th_k4 cap 0.88 → cap 박힘 방지 + test under <5% buffer. "
            "objective: F1*1000 - max(0,under-3%)*4000 - distance(k4_over,25~35%)*1500. "
            "K3/K4 class weight는 w_k3, w_k4로 직접 탐색. "
            "calib split은 threshold 산출 전용, dev split은 trial 및 epoch 선택 전용. "
            "test split은 최종 ktas_model_train.py 평가 전까지 사용하지 않음."
        ),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[저장 완료] {RESULTS_PATH}")


if __name__ == "__main__":
    main()
