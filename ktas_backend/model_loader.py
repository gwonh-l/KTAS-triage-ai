import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "./ktas_ai_model/"
NUM_LABELS = 5
NUM_DEPT   = 11


class KTASMultiTaskModel(nn.Module):
    """
    BERT backbone + 2개 head.
    - ktas_head: 5-class (K1~K5)
    - dept_head: 11-class (진료과, K4/K5에서만 유효)

    model.bert(**inputs, output_attentions=True) 로 attention 직접 접근 가능.
    """
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.bert      = AutoModel.from_pretrained(model_name_or_path)
        hidden         = self.bert.config.hidden_size
        self.ktas_head = nn.Linear(hidden, NUM_LABELS)
        self.dept_head = nn.Linear(hidden, NUM_DEPT)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=return_attention,
        )
        cls = out.last_hidden_state[:, 0, :]   # pooling 단일 지점 — 학습/서빙 공유
        ktas_logits, dept_logits = self.ktas_head(cls), self.dept_head(cls)
        if return_attention:
            return ktas_logits, dept_logits, out.attentions
        return ktas_logits, dept_logits

    @classmethod
    def from_saved(cls, save_dir: str) -> "KTASMultiTaskModel":
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.bert      = AutoModel.from_pretrained(save_dir)
        hidden          = model.bert.config.hidden_size
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


def load_model() -> tuple[KTASMultiTaskModel, AutoTokenizer, torch.device]:
    """
    멀티태스크 KTAS 모델 로드. FastAPI 앱 시작 시 1회만 호출.
    attention 추출 시 model.bert(**inputs, output_attentions=True) 로 직접 접근.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = KTASMultiTaskModel.from_saved(MODEL_PATH)
    model.to(device)
    model.eval()

    return model, tokenizer, device