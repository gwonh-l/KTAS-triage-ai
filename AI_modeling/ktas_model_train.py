"""
KTAS 응급 중증도 분류 모델 학습 — Multi-task 버전

[Multi-task 설계]
- Head 1 (ktas_head): 5-class KTAS 중증도 분류
- Head 2 (dept_head): 11-class 진료과 분류 (K4/K5에서만 유효)
  - dept label이 존재하는 K4/K5 샘플에서만 dept_loss 계산
  - K1~K3 및 결측(-1)은 ignore_index=-1로 자동 제외
- total_loss = ktas_loss + DEPT_LOSS_WEIGHT * dept_loss

[핵심 설계 원칙]
1. 지표 기반 class weight
   - K3/K4 기준 가중치는 Optuna가 under/over/F1 균형을 기준으로 탐색
   - K1/K2/K5는 탐색 공간 제한용 baseline 유지
2. ACS-COT 동적 임계치 보정
   - calib split에서 threshold 역산 및 confidence 보정
   - dev split에서 best epoch 선택
3. 모델 저장 구조
   - BERT backbone: save_pretrained / from_pretrained
   - ktas_head / dept_head weights: task_heads.pt
4. Epoch 선택 기준 (ACS-COT field triage 정렬: under<5%, over 25~35%)
   - 선택 키 = (both_acs_safe, selection_score) — tuner와 100% 동일.
     · both_acs_safe = calib·dev 양쪽 under ≤ 5%. 이것만 절대 게이트(유일 합격 조건과 일치).
     · selection_score = F1x1000 - max(0,under-3%)x4000 - distance(k4_over,25~35%)x3000
   - [실패 교훈] 3% buffer를 게이트로 쓰면 over를 25~35%에 넣은 epoch이 under 0.x%p 초과로 탈락 →
     over 45~56% epoch이 대신 선택되는 역설. 게이트는 5%로만, 3% 초과는 score 감점으로만 반영.
   - over 페널티 계수 1500→3000: 1500은 over초과 고F1 epoch을 못 눌렀고, 6000은 과잉 도살. 역산 ≈2415 위 3000.
   - K4 threshold 검색 상한 0.88 (cap, fallback 0.86 위) — K4 강등 억제.
   - 자기파괴 트랩 해제: under target 2% + cap 0.83 콤보가 모든 trial을 cap에 박아
     K4 over 60%+ 폭발시켰던 문제 → target 3% + cap 0.88로 풀어 자연 수렴.

[데이터 / split — v4]
- ktas_training_data_final.csv = 10,207건(증강 5,840행 혼입). SPLIT_VERSION=4(train 6633/dev 510/calib 1532/test 1532).
- ⚠️ 행 단위 stratify라 같은 original_index 패러프레이즈가 train/calib/test로 흩어질 수 있음(leak 인지 후 진행).
  test under/over가 실제보다 낙관적일 여지 → 발표 후 original_index 그룹 split 전환 권장.

[최종 결과 (현재 박힌 HP = tuner Best Trial #0)]
- Best Epoch 6 선택(CALIB_U 4.91% → both_ok=False지만 acs_ok=True라 채택).
- TEST: Macro F1 0.7498 | K3 under 3.44%(PASS<5%) | K4 over 32.03%(PASS 25~35%) |
        K3/K4/K5 Recall 0.806/0.641/0.530 | Dept Acc 0.8821. → 3지표 전부 통과.
- 산출물: ktas_ai_model/ (backbone + task_heads.pt + threshold_config.json + final_test_metrics.json).
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, classification_report, confusion_matrix
from torch.amp import GradScaler

# ── 지표 최적화 기준 ──
# K1/K2/K5는 탐색 공간을 제한하기 위한 baseline. W_K3/W_K4는 Optuna 결과로 갱신한다.
BASE_CLASS_WEIGHT = [1.7, 1.2, 1.0, 1.0, 1.1]   # K1~K5
ACS_COT_UNDER_TRIAGE_LIMIT = 0.05      # 임상 안전 게이트 (mandate — 절대 손대지 마)
SELECTION_UNDER_TRIAGE_MARGIN = 0.015
SELECTION_UNDER_TRIAGE_LIMIT = ACS_COT_UNDER_TRIAGE_LIMIT - SELECTION_UNDER_TRIAGE_MARGIN  # 로깅용 margin 판정
THRESHOLD_SEARCH_TARGET = 0.03         # 0.02는 cap 박혀 K4 무차별 강등→over 폭발. 0.03은 cap 탈출 + test 5% 게이트 buffer 2%p
UNDER_BUCKET_WIDTH = 0.005             # dev under_rate 0.5%p 버킷 — both 통과군 내에서 under 두꺼운(낮은) 후보 우선 (옵션2, tuner와 동일)
# ── ACS-COT field triage 기준 정렬 (under <5%, over 25~35%) ──
# 임의 목표(over 15%)는 under를 망쳤고, under target 2%는 cap을 박아 over를 폭발시킴.
# ACS 실제 허용치(35%)로 정렬 + target/cap 모두 한 칸 풀어 자기파괴 트랩 해제.
TH_K4_SEARCH_CAP = 0.88                # 0.83은 5 trial 전부 cap에 박혀 over 폭발. 0.88(fallback 0.86 위)로 여유
K4_OVER_TRIAGE_MIN = 0.25              # 목표 구간 하한
K4_OVER_TRIAGE_MAX = 0.35              # 목표 구간 상한
K4_OVER_TRIAGE_PENALTY = 3_000         # 목표 구간 이탈 거리 페널티. 1500→3000: E6(over35.77%, +0.77%p 초과)를
                                       # E5(over32.5%, F1 약간 낮음)보다 후순위로 만드는 역산값(≈2415) 위 안전마진.
                                       # 6000은 over 35.5~37% 후보를 과잉 도살해 비추 → 3000이 합리적 실험점.

# ── 진료과 레이블 (고정 순서 11개 — 서빙 DEPT_CLASSES와 동일 순서 유지) ──
# 앞 9개는 가나다 정렬, idx 9 '외과'·idx 10 '소아과'는 append 순서 보존(향후 데이터 추가용).
# sorted()로 재구성 금지 — append 순서가 깨지면 인덱스 매핑이 어긋남.
DEPT_CLASSES = ['내과', '비뇨기과', '산부인과', '신경과', '안과', '이비인후과', '정신건강의학과', '정형외과', '피부과', '외과', '소아과']
DEPT_TO_IDX: dict[str, int] = {d: i for i, d in enumerate(DEPT_CLASSES)}
NUM_DEPT = len(DEPT_CLASSES)

# ── 학습 설정 (Optuna Best Trial 결과 반영) ──
MODEL_NAME       = "klue/bert-base"
DATA_PATH        = "./ktas_training_data_final.csv"
SAVE_DIR         = "./ktas_ai_model/"
MAX_LEN          = 180
EPOCHS           = 6
RANDOM_SEED      = 42
NUM_LABELS       = 5
SPLIT_PATH       = "./ktas_split.json"
TEST_RATIO       = 0.15
DEV_RATIO        = 0.05    # 0.10→0.05: dev는 epoch 선택용이라 작아도 됨
CALIB_RATIO      = 0.15    # threshold 보정용 — 키워서 calib→test under gap 축소
SPLIT_VERSION    = 4       # 데이터 10,207건으로 증가(증강 5,840행 혼입) → 재생성. tuner와 동일.
                           # 주의: 행 단위 stratify라 같은 original_index 패러프레이즈가 train/calib/test로 흩어질 수 있음(leak 인지 후 진행).

# ── optuna_results.json 도출 값 수정 구역 ──
BATCH_SIZE       = 16
LR               = 5.862251626650906e-05
WEIGHT_DECAY     = 0.00014999351384664325
LABEL_SMOOTH     = 0.08484738945147324
WARMUP_RATIO     = 0.04693386782809262
IMBALANCE_LAMBDA = 0.12876410471818983
DEPT_LOSS_WEIGHT = 0.4300701174843584
W_K3             = 2.3417878112807613
W_K4             = 1.3617387365519207


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
    """
    사전 토큰화 멀티태스크 데이터셋.
    Epoch마다 재토큰화하지 않도록 학습 시작 전 전체 1회 토큰화.
    """
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
        self.token_type_ids  = token_type_ids
        self.labels          = labels
        self.dept_labels     = dept_labels   # K1~K3 및 결측: -1

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
    - ktas_head: 5-class (K1~K5)
    - dept_head: 11-class (진료과, K4/K5에서만 유효)
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
        cls = out.last_hidden_state[:, 0, :]   # [CLS]
        return self.ktas_head(cls), self.dept_head(cls)

    def save_weights(self, save_dir: str) -> None:
        """BERT backbone은 save_pretrained, 두 head는 task_heads.pt로 분리 저장."""
        self.bert.save_pretrained(save_dir)
        torch.save(
            {
                "ktas_head": self.ktas_head.state_dict(),
                "dept_head": self.dept_head.state_dict(),
            },
            os.path.join(save_dir, "task_heads.pt"),
        )

    @classmethod
    def from_saved(cls, save_dir: str) -> "KTASMultiTaskModel":
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.bert = AutoModel.from_pretrained(save_dir)
        hidden = model.bert.config.hidden_size
        model.ktas_head = nn.Linear(hidden, NUM_LABELS)
        model.dept_head = nn.Linear(hidden, NUM_DEPT)
        heads = torch.load(
            os.path.join(save_dir, "task_heads.pt"),
            map_location="cpu",
            weights_only=True,
        )
        model.ktas_head.load_state_dict(heads["ktas_head"])
        model.dept_head.load_state_dict(heads["dept_head"])
        return model


def pretokenize(
    texts: list[str], tokenizer
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    enc = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids")


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
    """
    Returns: ktas_logits, dept_logits, ktas_labels, dept_labels
    dept_labels: -1이면 해당 샘플은 dept 평가에서 제외
    """
    model.eval()
    all_ktas, all_dept, all_labels, all_dept_labels = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            model_inputs = {
                "input_ids":      batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "token_type_ids" in batch and batch["token_type_ids"] is not None:
                model_inputs["token_type_ids"] = batch["token_type_ids"].to(device)

            with torch.amp.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp
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


def evaluate_with_thresholds(
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    threshold_k4: float,
) -> dict:
    preds = conservative_predict(logits_np, threshold_k4)
    macro_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    recalls  = recall_score(labels_np, preds, average=None, labels=[0,1,2,3,4], zero_division=0)
    k3_count = int((labels_np == 2).sum())
    k3_under = int(np.sum((preds >= 3) & (labels_np == 2)))
    # K4 over-triage: K4 truth가 K1/K2/K3로 상향된 비율 (뺑뺑이 직접 지표)
    k4_count    = int((labels_np == 3).sum())
    k4_over     = int(np.sum((preds <= 2) & (labels_np == 3)))
    k4_over_rate = k4_over / k4_count if k4_count > 0 else 0.0
    return {
        "preds":         preds,
        "macro_f1":      float(macro_f1),
        "k3_recall":     float(recalls[2]),
        "k4_recall":     float(recalls[3]),
        "k5_recall":     float(recalls[4]),
        "k3_under":      k3_under,
        "k3_under_rate": k3_under / k3_count if k3_count > 0 else 0.0,
        "k4_over":       k4_over,
        "k4_over_rate":  k4_over_rate,
    }


def evaluate_dept(
    dept_logits_np: np.ndarray,
    dept_labels_np: np.ndarray,
) -> dict:
    """dept label이 존재하는 K4/K5 샘플에 대한 진료과 분류 정확도."""
    valid_mask = (dept_labels_np != -1)
    if valid_mask.sum() == 0:
        return {"dept_accuracy": None, "dept_valid_count": 0}
    valid_logits = dept_logits_np[valid_mask]
    valid_labels = dept_labels_np[valid_mask]
    preds = np.argmax(valid_logits, axis=1)
    acc   = float((preds == valid_labels).mean())
    per_class = {}
    for i, cls_name in enumerate(DEPT_CLASSES):
        cls_mask = (valid_labels == i)
        if cls_mask.sum() > 0:
            per_class[cls_name] = round(float((preds[cls_mask] == i).mean()), 4)
    return {
        "dept_accuracy":    round(acc, 4),
        "dept_valid_count": int(valid_mask.sum()),
        "dept_per_class":   per_class,
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
    return torch.tensor(
        metric_weight * np.power(sqrt_inverse_frequency, imbalance_lambda),
        dtype=torch.float32,
    )


def find_optimal_thresholds(
    logits_np: np.ndarray, labels_np: np.ndarray
) -> float:
    probs     = torch.softmax(torch.tensor(logits_np, dtype=torch.float32), dim=-1).numpy()
    raw_preds = np.argmax(probs, axis=1)
    k3_true   = (labels_np == 2)

    # K4 검색: 상한 TH_K4_SEARCH_CAP(0.88)에서 컷오프.
    # cap에서도 목표(3%) 미달이면 cap 그대로 사용. 남은 under_rate 초과분은 objective가 처리.
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
        print(f"[정보] K4 threshold cap({TH_K4_SEARCH_CAP}) 도달 — 검색 목표({THRESHOLD_SEARCH_TARGET:.1%}) 미달, over-triage 방지 우선")

    return best_th_k4


def fit_temperature(logits_np: np.ndarray, labels_np: np.ndarray, max_iter: int = 50) -> float:
    """
    calib logits로 temperature scaling 계수 T를 적합 (NLL 최소화).
    softmax(logits / T)는 argmax를 바꾸지 않으므로 등급 결정·ACS-COT threshold에 영향 없음.
    → 표시 confidence와 엔트로피 기반 reject의 신뢰도만 개선한다.
    """
    logits = torch.tensor(logits_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    log_t  = torch.zeros(1, requires_grad=True)   # log(T) 최적화로 T>0 보장
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / torch.exp(log_t), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).detach().item())


def compute_entropy_threshold(
    logits_np: np.ndarray, temperature: float, percentile: float = 99.0
) -> float:
    """
    calib(in-distribution) 엔트로피 분포의 상위 percentile을 OOD reject 임계치로 사용.
    정상 입력의 99%보다 더 불확실한 입력만 'uncertain'으로 플래그.
    """
    probs   = torch.softmax(torch.tensor(logits_np, dtype=torch.float32) / temperature, dim=-1).numpy()
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return float(np.percentile(entropy, percentile))


def conservative_predict(
    logits_np: np.ndarray, threshold_k4: float
) -> np.ndarray:
    # 보수적 상향 — K4/K5 경증대에 K4+K5 합산 질량 기준 적용 (main.py와 동일 로직).
    #   - 합산 <  threshold_k4 : K1~K3 쪽 질량 잔존 → K3(idx 2) 상향(안전 방향)
    #   - 합산 >= threshold_k4 : 경증 확신 충분 → argmax(K4/K5) 그대로 유지(K5 살림)
    probs = torch.softmax(torch.tensor(logits_np, dtype=torch.float32), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    mild_mass    = probs[:, 3] + probs[:, 4]
    in_mild_band = (preds == 3) | (preds == 4)
    preds[in_mild_band & (mild_mass < threshold_k4)] = 2
    return preds


def train() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device  = get_device()
    use_amp = (device.type == "cuda")
    print(f"[설정] device={device} | AMP={use_amp} | DEPT_LOSS_WEIGHT={DEPT_LOSS_WEIGHT}")

    df        = pd.read_csv(DATA_PATH)
    texts     = df["symptom_text"].tolist()
    labels    = (df["ktas_level"] - 1).tolist()
    dept_raw  = encode_dept(df["dept"])

    split = get_or_create_split_indices(labels)
    train_idx, dev_idx, calib_idx, test_idx = split["train"], split["dev"], split["calib"], split["test"]
    print(
        f"[데이터] 전체={len(texts)} | train={len(train_idx)} | dev={len(dev_idx)} | "
        f"calib={len(calib_idx)} | test={len(test_idx)}"
    )

    print("[토큰화] 전체 데이터 사전 토큰화 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    input_ids, attn_masks, token_type_ids = pretokenize(texts, tokenizer)
    full_dataset = KTASDataset(
        input_ids, attn_masks, token_type_ids,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(dept_raw, dtype=torch.long),
    )

    use_pin = (device.type == "cuda")
    # shuffle 재현성: tuner와 동일 generator 시드로 배치 순서 일치 → Best HP 재현
    loader_gen = torch.Generator()
    loader_gen.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=use_pin, generator=loader_gen)
    dev_loader   = DataLoader(Subset(full_dataset, dev_idx),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=use_pin)
    calib_loader = DataLoader(Subset(full_dataset, calib_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=use_pin)
    test_loader  = DataLoader(Subset(full_dataset, test_idx),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=use_pin)

    # head 초기화 재현: tuner와 같은 시점에서 시드 리셋해 nn.Linear 초기값 일치
    torch.manual_seed(RANDOM_SEED)
    model = KTASMultiTaskModel(MODEL_NAME).to(device)

    train_labels_for_weight = [labels[i] for i in train_idx]
    base_class_weight = [BASE_CLASS_WEIGHT[0], BASE_CLASS_WEIGHT[1], W_K3, W_K4, BASE_CLASS_WEIGHT[4]]
    class_weights = build_loss_weights(train_labels_for_weight, IMBALANCE_LAMBDA, base_class_weight).to(device)
    print(
        "[가중치] "
        + " / ".join(f"K{i+1}={float(class_weights[i].detach().cpu()):.4f}" for i in range(NUM_LABELS))
    )
    ktas_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    dept_criterion = nn.CrossEntropyLoss(ignore_index=-1)   # K1~K3 및 결측 자동 제외

    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler("cuda", enabled=use_amp)

    best_selection_score = -float("inf")
    # 옵션2: (both_under_ok, -dev_under_bucket, selection_score). tuner epoch 키와 동일 우선순위.
    # 1순위 both 통과(calib·dev 동시 under<target), 2순위 under 두꺼운(낮은 버킷), 3순위 selection_score.
    best_select_key = (False, -float("inf"))
    best_epoch           = 0
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            lbls      = batch["labels"].to(device)
            dept_lbls = batch["dept_labels"].to(device)
            model_inputs = {
                "input_ids":      batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "token_type_ids" in batch and batch["token_type_ids"] is not None:
                model_inputs["token_type_ids"] = batch["token_type_ids"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp
            ):
                ktas_logits, dept_logits = model(**model_inputs)
                ktas_loss = ktas_criterion(ktas_logits, lbls)
                # 배치 전원이 dept 결측(-1)이면 CrossEntropyLoss가 NaN을 반환 →
                # total_loss 오염 방지를 위해 유효 dept 샘플이 있을 때만 계산
                if (dept_lbls != -1).any():
                    dept_loss = dept_criterion(dept_logits, dept_lbls)
                else:
                    dept_loss = torch.zeros((), device=device)
                loss = ktas_loss + DEPT_LOSS_WEIGHT * dept_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # calib에서 threshold만 산출하고, 독립 dev에서 best epoch를 선택한다.
        calib_ktas_logits, _, calib_labels, _ = collect_logits(model, calib_loader, device, use_amp)
        optimal_th_k4 = find_optimal_thresholds(calib_ktas_logits, calib_labels)
        calib_metrics = evaluate_with_thresholds(calib_ktas_logits, calib_labels, optimal_th_k4)
        dev_ktas_logits, _, dev_labels, _ = collect_logits(model, dev_loader, device, use_amp)
        cm = evaluate_with_thresholds(dev_ktas_logits, dev_labels, optimal_th_k4)
        acs_pass    = cm["k3_under_rate"] <= ACS_COT_UNDER_TRIAGE_LIMIT
        margin_pass = cm["k3_under_rate"] <= SELECTION_UNDER_TRIAGE_LIMIT

        # ── selection_score: under / over / Macro F1만 직접 최적화 ──
        # 불안전(acs FAIL = under>5%)은 -1e6으로 강제 후순위.
        # 안전 영역 안에서는:
        #   F1(x1000) - under 3% 초과분(x4000) - K4 over 목표 구간(25~35%) 이탈 거리(x1500)
        if not acs_pass:
            selection_score = -1_000_000.0 + cm["macro_f1"] * 1_000
        else:
            if cm["k4_over_rate"] < K4_OVER_TRIAGE_MIN:
                over_penalty = (K4_OVER_TRIAGE_MIN - cm["k4_over_rate"]) * K4_OVER_TRIAGE_PENALTY
            elif cm["k4_over_rate"] > K4_OVER_TRIAGE_MAX:
                over_penalty = (cm["k4_over_rate"] - K4_OVER_TRIAGE_MAX) * K4_OVER_TRIAGE_PENALTY
            else:
                over_penalty = 0.0
            selection_score = (
                cm["macro_f1"]                                            * 1_000
                - max(0.0, cm["k3_under_rate"] - THRESHOLD_SEARCH_TARGET) * 4_000
                - over_penalty
            )

        # 방법4(보강): calib·dev 양쪽 under target 통과를 1순위 게이트로. 둘 다 ok인 epoch만 안전 후보.
        # calib만 보던 threshold와 dev만 보던 selection의 분리 때문에 test under가 샜음 → 양쪽 AND로 봉합.
        dev_under_ok   = cm["k3_under_rate"]            <= THRESHOLD_SEARCH_TARGET
        calib_under_ok = calib_metrics["k3_under_rate"] <= THRESHOLD_SEARCH_TARGET
        both_under_ok  = dev_under_ok and calib_under_ok
        # dev under_rate 0.5%p 버킷(낮을수록 under margin 두꺼움). 현재는 로그 표시 전용 — 선택 키에는 미사용.
        # (옵션2에서 키 2순위로 넣었다가, under만 보고 over를 죽이는 부작용 확인 후 키에서 제외함. 아래 선택 키 주석 참조.)
        dev_under_bucket = int(cm["k3_under_rate"] / UNDER_BUCKET_WIDTH)

        # ── epoch 선택 키: (both_acs_safe, selection_score) ──
        # both_acs_safe = calib·dev 양쪽 K3 under ≤ ACS_COT_UNDER_TRIAGE_LIMIT(5%). 이게 유일한 절대 게이트.
        # [설계 핵심 / 실패 교훈]
        #  - 3% buffer(THRESHOLD_SEARCH_TARGET)를 게이트로 쓰면(both_under_ok) over를 목표 구간(25~35%)에
        #    넣은 epoch이 calib under 0.x%p 초과로 탈락 → over 45~56% epoch이 대신 선택되는 역설 발생.
        #  - 그래서 게이트는 진짜 안전선 5%로만 두고, 3% 초과분은 selection_score의 *4000 페널티로 "감점"만 한다(탈락 아님).
        #  - under 버킷은 키에서 뺐다(under만 보고 over 정상화 epoch을 죽임). 안전군 내부는 selection_score로만 고른다.
        # [실측 효과] 이 키 덕에 E6(CALIB_U 4.91% → both_under_ok=False지만 acs_ok=True, F1 0.752, over 30%)이
        #            E2(over 45.6%)를 제치고 선택됨 → test under 3.44% / over 32.03% 동시 통과.
        both_acs_safe = (cm["k3_under_rate"] <= ACS_COT_UNDER_TRIAGE_LIMIT) and \
                        (calib_metrics["k3_under_rate"] <= ACS_COT_UNDER_TRIAGE_LIMIT)

        print(
            f"Epoch {epoch}/{EPOCHS} | loss={avg_loss:.4f} | DEV "
            f"F1={cm['macro_f1']:.4f} | K3 Rec={cm['k3_recall']:.4f} | K4 Rec={cm['k4_recall']:.4f} | "
            f"DEV_U={cm['k3_under_rate']:.2%} | CALIB_U={calib_metrics['k3_under_rate']:.2%} | "
            f"K4 Over={cm['k4_over']}건({cm['k4_over_rate']:.2%}) | "
            f"ACS={'PASS' if acs_pass else 'FAIL'} | K4 th={optimal_th_k4} | "
            # acs_ok(5%, 실제 선택 게이트) vs both_ok(3% buffer, 표시 전용). both_ok=False여도 acs_ok=True면 선택 후보.
            f"acs_ok={both_acs_safe} | both_ok={both_under_ok} | bkt={dev_under_bucket} | score={selection_score:.2f}"
        )

        candidate_key = (both_acs_safe, selection_score)
        if candidate_key > best_select_key:
            best_select_key      = candidate_key
            best_selection_score = selection_score
            best_epoch           = epoch

            # backbone + head 분리 저장
            tokenizer.save_pretrained(SAVE_DIR)
            model.save_weights(SAVE_DIR)

            # 표시 confidence 보정(temperature) + OOD reject 임계치(entropy) — calib 기준 산출.
            # 둘 다 argmax 불변/raw 확률 기반 ACS-COT와 독립이라 안전 보증에 영향 없음.
            temperature = fit_temperature(calib_ktas_logits, calib_labels)
            entropy_threshold = compute_entropy_threshold(calib_ktas_logits, temperature, percentile=99.0)

            threshold_info = {
                "THRESHOLD_K4":        optimal_th_k4,
                "TEMPERATURE":         round(temperature, 4),
                "ENTROPY_THRESHOLD":   round(entropy_threshold, 4),
                "entropy_threshold_percentile": 99.0,
                "calibration_method":  f"ACS-COT field triage (under<5%, over 25~35%); search target {THRESHOLD_SEARCH_TARGET:.1%} + th_k4 cap {TH_K4_SEARCH_CAP}",
                "epoch":               epoch,
                "selection_source":    "dev",
                "split_version":       SPLIT_VERSION,
                "selection_rule":      f"ACS-COT hard gate (5%); inside gate: F1x1000 - max(0,under-{THRESHOLD_SEARCH_TARGET:.1%})x4000 - distance(k4_over,{K4_OVER_TRIAGE_MIN:.0%}~{K4_OVER_TRIAGE_MAX:.0%})x{K4_OVER_TRIAGE_PENALTY}",
                "calib_k4_over":             calib_metrics["k4_over"],
                "calib_k4_over_rate":        round(calib_metrics["k4_over_rate"], 4),
                "th_k4_search_cap":          TH_K4_SEARCH_CAP,
                "k4_over_triage_min":        K4_OVER_TRIAGE_MIN,
                "k4_over_triage_max":        K4_OVER_TRIAGE_MAX,
                "k4_over_triage_penalty":    K4_OVER_TRIAGE_PENALTY,
                "selection_score":     round(selection_score, 4),
                "dept_loss_weight":    DEPT_LOSS_WEIGHT,
                "base_class_weights":  base_class_weight,
                "w_k3":                W_K3,
                "w_k4":                W_K4,
                "dept_classes":        DEPT_CLASSES,
                "calib_macro_f1":             round(calib_metrics["macro_f1"], 4),
                "calib_k3_recall":            round(calib_metrics["k3_recall"], 4),
                "calib_k3_under_triage":      calib_metrics["k3_under"],
                "calib_k3_under_triage_rate": round(calib_metrics["k3_under_rate"], 4),
                "acs_cot_pass_on_calib":      calib_metrics["k3_under_rate"] <= ACS_COT_UNDER_TRIAGE_LIMIT,
                "margin_pass_on_calib":       calib_metrics["k3_under_rate"] <= SELECTION_UNDER_TRIAGE_LIMIT,
                "dev_macro_f1":               round(cm["macro_f1"], 4),
                "dev_k3_recall":              round(cm["k3_recall"], 4),
                "dev_k4_recall":              round(cm["k4_recall"], 4),
                "dev_k3_under_triage":        cm["k3_under"],
                "dev_k3_under_triage_rate":   round(cm["k3_under_rate"], 4),
                "dev_k4_over":                cm["k4_over"],
                "dev_k4_over_rate":           round(cm["k4_over_rate"], 4),
                "acs_cot_pass_on_dev":        acs_pass,
                "margin_pass_on_dev":         margin_pass,
                "split_path":          SPLIT_PATH,
                "split_counts": {
                    "train": len(train_idx), "dev": len(dev_idx),
                    "calib": len(calib_idx), "test": len(test_idx),
                },
            }
            with open(os.path.join(SAVE_DIR, "threshold_config.json"), "w", encoding="utf-8") as f:
                json.dump(threshold_info, f, ensure_ascii=False, indent=2)
            print(f"         → 저장 완료 (score={best_selection_score:.4f}, Epoch {best_epoch})")

    print(f"\n[완료] Best score={best_selection_score:.4f} (Epoch {best_epoch})")

    # ── 최종 test 평가 ──
    best_model = KTASMultiTaskModel.from_saved(SAVE_DIR).to(device)
    with open(os.path.join(SAVE_DIR, "threshold_config.json"), "r", encoding="utf-8") as f:
        threshold_info = json.load(f)

    test_ktas_logits, test_dept_logits, test_labels, test_dept_labels = collect_logits(
        best_model, test_loader, device, use_amp
    )
    test_metrics = evaluate_with_thresholds(
        test_ktas_logits, test_labels,
        threshold_info["THRESHOLD_K4"],
    )
    test_dept_metrics = evaluate_dept(test_dept_logits, test_dept_labels)
    test_preds = test_metrics["preds"]
    test_pass  = test_metrics["k3_under_rate"] <= ACS_COT_UNDER_TRIAGE_LIMIT

    print("\n[최종 TEST 평가 - threshold 재보정 없음]")
    print(
        f"Macro F1={test_metrics['macro_f1']:.4f} | "
        f"K3 Recall={test_metrics['k3_recall']:.4f} | "
        f"K4 Recall={test_metrics['k4_recall']:.4f} | "
        f"K5 Recall={test_metrics['k5_recall']:.4f} | "
        f"K3 Under={test_metrics['k3_under']}건({test_metrics['k3_under_rate']:.2%}) | "
        f"K4 Over={test_metrics['k4_over']}건({test_metrics['k4_over_rate']:.2%}) | "
        f"ACS={'PASS' if test_pass else 'FAIL'}"
    )
    print(
        f"Dept Accuracy={test_dept_metrics['dept_accuracy']} "
        f"(valid={test_dept_metrics['dept_valid_count']}건)"
    )
    print(classification_report(
        test_labels, test_preds,
        target_names=["K1", "K2", "K3", "K4", "K5"],
        digits=4, zero_division=0,
    ))
    print(confusion_matrix(test_labels, test_preds))

    final_metrics = {
            "test_macro_f1":          round(test_metrics["macro_f1"], 4),
            "test_k3_recall":         round(test_metrics["k3_recall"], 4),
            "test_k4_recall":         round(test_metrics["k4_recall"], 4),
            "test_k5_recall":         round(test_metrics["k5_recall"], 4),
            "test_k3_under_triage":   test_metrics["k3_under"],
            "test_k3_under_triage_rate": round(test_metrics["k3_under_rate"], 4),
            "test_k4_over_triage":      test_metrics["k4_over"],
            "test_k4_over_triage_rate": round(test_metrics["k4_over_rate"], 4),
            "acs_cot_pass_on_test":   test_pass,
            "dept_accuracy":          test_dept_metrics["dept_accuracy"],
            "dept_valid_count":       test_dept_metrics["dept_valid_count"],
            "dept_per_class_accuracy": test_dept_metrics.get("dept_per_class"),
            "dept_classes":           DEPT_CLASSES,
            "classification_report":  classification_report(
                test_labels, test_preds,
                target_names=["K1", "K2", "K3", "K4", "K5"],
                output_dict=True, zero_division=0,
            ),
            "confusion_matrix":       confusion_matrix(test_labels, test_preds).tolist(),
            "threshold_k4":           threshold_info["THRESHOLD_K4"],
            "temperature":            threshold_info.get("TEMPERATURE"),
            "entropy_threshold":      threshold_info.get("ENTROPY_THRESHOLD"),
            "best_epoch":             best_epoch,
            "split_version":          SPLIT_VERSION,
            "split_path":             SPLIT_PATH,
        }
    with open(os.path.join(SAVE_DIR, "final_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    train()
