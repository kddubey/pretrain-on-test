from __future__ import annotations
from dataclasses import dataclass
from typing import Type

import torch
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)


@dataclass(frozen=True)
class Config:
    model_id: str
    model_class_pretrain: Type[PreTrainedModel]
    model_class_classification: Type[PreTrainedModel]
    mlm: bool | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mlm_probability: float | None = None
    model_path_pretrained: str = "_pretrained"
    model_path_classification: str = "_classifier"

    def __post_init__(self):
        if self.tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # Re-setting an attribute in a frozen object requires this call:
            object.__setattr__(self, "tokenizer", tokenizer)
        if self.mlm is None:
            mlm = issubclass(self.model_class_pretrain, BertForMaskedLM)
            object.__setattr__(self, "mlm", mlm)
