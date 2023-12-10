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
    """
    Configuration for a transformer-based pretraining + training classification
    procedure.
    """

    model_id: str
    model_class_pretrain: Type[PreTrainedModel]
    model_class_classification: Type[PreTrainedModel]
    mlm: bool | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    device: str | torch.device | None = None
    mlm_probability: float | None = None
    model_path_pretrained: str = "_pretrained"
    model_path_classification: str = "_classifier"
    max_length: int | None = None
    pretrain_per_device_train_batch_size: int = 16

    def __post_init__(self):
        if self.tokenizer is None:
            default_tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if default_tokenizer.pad_token is None:
                default_tokenizer.pad_token = default_tokenizer.eos_token
            # Re-setting an attribute in a frozen object requires this call:
            object.__setattr__(self, "tokenizer", default_tokenizer)
        if self.mlm is None:
            default_mlm = issubclass(self.model_class_pretrain, BertForMaskedLM)
            object.__setattr__(self, "mlm", default_mlm)
        if self.device is None:
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
            object.__setattr__(self, "device", default_device)
        if self.max_length is None:  # be explicit about the default
            default_max_length = self.tokenizer.model_max_length
            object.__setattr__(self, "max_length", default_max_length)
