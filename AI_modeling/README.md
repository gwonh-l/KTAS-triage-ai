# KTAS Multi-Task Classifier — 응급 중증도 분류 + 진료과 추천

환자의 자연어 증상 호소를 입력받아 **KTAS 중증도(1~5등급)** 를 분류하고, 경증(K4/K5)에 대해 **진료과(11종)** 를 함께 추천하는 멀티태스크 모델입니다. 응급실 "뺑뺑이"(수용 거부 표류) 완화를 위한 ETA 기반 환자–병원 매칭 파이프라인의 분류 엔진으로 설계되었습니다.

> **임상 안전 우선 설계** — 중증 환자를 경증으로 오분류하는 **Under-triage를 5% 미만**으로 강제하는 hard gate를 학습·선택 전 과정에 적용합니다.

```
증상 텍스트 → [KTAS 분류 K1~K5] → (K4/K5) [진료과 추천] → ETA 병상 예측 → 병원 매칭
```

---

## 목차
- [주요 특징](#주요-특징)
- [모델 아키텍처](#모델-아키텍처)
- [데이터셋](#데이터셋)
- [성능](#성능)
- [설치](#설치)
- [사용법](#사용법)
- [프로젝트 구조](#프로젝트-구조)
- [설계 노트](#설계-노트)
- [알려진 한계](#알려진-한계)
- [참고 문헌](#참고-문헌)

---

## 주요 특징
- **2-head 멀티태스크** — 단일 BERT backbone에서 중증도와 진료과를 동시 학습
- **임상 안전 게이트** — Under-triage < 5% (ACS-COT field triage 기준)를 절대 제약으로 강제
- **동적 임계치 보정** — calibration split에서 등급별 confidence threshold를 데이터로 역산
- **균형 잡힌 over-triage** — K4 과대평가를 25~35% 목표 구간으로 유도해 자원 낭비 억제
- **완전 재현 가능** — 시드·split·하이퍼파라미터를 고정해 탐색 결과가 학습에서 동일하게 재현

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
            │  ktas_head    │         │   dept_head      │
            │  Linear → 5   │         │   Linear → 11    │
            │ (K1~K5 중증도)│         │ (진료과, K4/K5만)│
            └───────────────┘         └──────────────────┘
```

- **Backbone**: `klue/bert-base`
- **Head 1 (`ktas_head`)** — 5-class KTAS 중증도 (K1~K5)
- **Head 2 (`dept_head`)** — 11-class 진료과. dept 라벨이 있는 K4/K5 샘플에서만 손실을 계산하고, K1~K3·결측은 `ignore_index=-1`로 자동 제외
- **손실** — `total_loss = ktas_loss + dept_loss_weight · dept_loss`
  - `ktas_loss`: class-weighted CrossEntropy (+ label smoothing)
  - class weight = `base_weight · (√(N / (C·count)))^imbalance_lambda` — 빈도 역가중에 난이도 보정을 결합

### 진료과 클래스 (고정 순서, 서빙과 동일)
```
내과 · 비뇨기과 · 산부인과 · 신경과 · 안과 · 이비인후과 · 정신건강의학과 · 정형외과 · 피부과 · 외과 · 소아과
```

---

## 데이터셋
| 항목 | 값 |
|---|---|
| 총 샘플 | 10,207건 |
| KTAS 분포 | K1 582 / K2 3,522 / K3 2,712 / K4 2,728 / K5 663 |
| 진료과 유효 | K4/K5의 dept 라벨 보유 샘플 (11클래스) |
| Split (seed 42) | train 6,633 / dev 510 / calib 1,532 / test 1,532 |

- **train** — 모델 학습
- **dev** — epoch 선택
- **calib** — 등급별 threshold 역산 (학습·선택에 미사용)
- **test** — 최종 평가 1회 (threshold 재보정 없음)

---

## 성능

최종 모델 (test set, threshold 재보정 없음):

| 지표 | 값 | 목표 | 결과 |
|---|---|---|:--:|
| Macro F1 | **0.7498** | ≥ 0.68 | ✅ |
| K3 Under-triage | **3.44%** | < 5% | ✅ |
| K4 Over-triage | **32.03%** | 25~35% | ✅ |
| Dept Accuracy | 0.8821 | — | — |

등급별 Recall — K1 0.897 / K2 0.828 / K3 0.806 / K4 0.641 / K5 0.530

**Confusion Matrix** (행=정답, 열=예측):
```
        K1   K2   K3   K4   K5
 K1  [  78    8    1    0    0 ]
 K2  [  14  438   75    2    0 ]
 K3  [   1   64  328   13    1 ]
 K4  [   0    8  123  262   16 ]
 K5  [   0    1   18   28   53 ]
```

---

## 설치

```bash
# Python 3.10+ 권장
pip install torch transformers optuna scikit-learn pandas numpy
# 서빙까지 사용 시
pip install fastapi uvicorn
```

> GPU(CUDA) 환경에서는 AMP(자동 혼합정밀)가 자동 활성화됩니다. CPU·MPS에서도 동작합니다.

---

## 사용법

### 1) 하이퍼파라미터 탐색 (선택)
```bash
python3 ktas_hparam_tuner.py
# → optuna_results.json 생성 (Best 하이퍼파라미터)
```

### 2) 학습 + 평가
```bash
python3 ktas_model_train.py
# → ktas_ai_model/ 에 모델·threshold·test 결과 저장
```
`optuna_results.json`의 Best 값을 `ktas_model_train.py` 상단 상수 구역(`LR`, `W_K3`, `W_K4` 등)에 반영한 뒤 실행합니다.

### 3) 서빙
```bash
uvicorn main:app --reload
# /health 로 진료과 11개 클래스·threshold 로드 확인
```

---

## 프로젝트 구조
```
.
├── ktas_hparam_tuner.py     # Optuna 하이퍼파라미터 탐색 (EPOCHS=6, N_TRIALS=20)
├── ktas_model_train.py      # 학습 + 최종 test 평가
├── model_loader.py          # 서빙용 모델 로더
├── main.py                  # FastAPI 서빙 엔드포인트
├── ktas_training_data_final.csv
├── ktas_split.json          # 4-way split (자동 생성, seed 42)
├── optuna_results.json      # 탐색 Best 결과
└── ktas_ai_model/           # 학습 산출물
    ├── (BERT backbone)
    ├── task_heads.pt        # ktas_head + dept_head 가중치
    ├── threshold_config.json
    └── final_test_metrics.json
```

---

## 설계 노트

### ACS-COT field triage 기준
- **Under-triage < 5%** — 임상 안전 hard gate. 모든 epoch/trial 선택의 유일한 절대 제약.
- **Over-triage 25~35%** — 목표 구간. 이탈 거리에 비례해 점수를 감점.

### Epoch / Trial 선택 키
```python
candidate_key = (both_acs_safe, score)
# both_acs_safe = (dev_under ≤ 5%) AND (calib_under ≤ 5%)     ← 절대 게이트
# score = F1·1000 − max(0, under−3%)·4000 − distance(over, 25~35%)·3000
```
- **게이트는 5%(임상 안전선)로만** 두고, 더 보수적인 3% 목표는 **점수 감점**으로만 반영합니다.
- 게이트를 3%로 조이면 over를 목표 구간에 잘 맞춘 epoch이 under 소수점 초과로 탈락하고, 오히려 over가 폭발한 epoch이 선택되는 역설이 생깁니다. 안전선과 buffer를 분리해 이를 해소했습니다.
- tuner와 train이 동일한 선택 키를 공유해, 탐색에서 고른 epoch이 학습에서 그대로 재현됩니다.

### 동적 임계치 보정
- calibration split에서 K3 under-triage ≤ 3%를 만족하는 **최저 K4 threshold**를 역산
- K4 threshold 검색 상한 0.88로 과도한 강등 억제
- threshold 조정은 모델 가중치를 바꾸지 않으므로(argmax 기반) F1에 거의 영향을 주지 않고 under/over만 조정

---

## 알려진 한계
- **데이터 증강 leak 가능성** — 증강 샘플이 split 이전에 혼입되어 유사 문장이 train/test에 분산될 수 있습니다. 그룹 단위 split 전환을 향후 과제로 둡니다.
- **K3/K4 경계 혼동** — KTAS 자체가 평가자 간 일치도가 가장 낮은 구간이며, 모델의 K3/K4 혼동도 이 본질적 난이도를 반영합니다.
- **K5 Recall (0.53)** — 최소 클래스(K5)의 recall이 상대적으로 낮습니다.

> 위 한계로 인해 현재 test 지표는 다소 낙관적으로 해석될 여지가 있으며, 안전성에 대한 최종 결론은 그룹 단위 split 재평가 이후로 유보하는 것을 권장합니다.

---

## 참고 문헌
- Korean Triage and Acuity Scale (KTAS) 신뢰도 연구 (JKMS, 2019) — 평가자 간 일치도(weighted-κ 0.772) 및 K3/K4 불일치 경향
- ACS Committee on Trauma — field triage under-/over-triage 기준

---

*Built with PyTorch · Hugging Face Transformers · Optuna · FastAPI*
