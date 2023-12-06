from __future__ import annotations
from dataclasses import dataclass
from typing import Type

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel


@dataclass
class Config:
    model_class_pretrain: Type[PreTrainedModel]
    model_id_pretrain: str
    model_class_classification: Type[PreTrainedModel]
    tokenizer: PreTrainedTokenizerBase
    mlm: bool
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mlm_probability: float | None = None
    pretrained_model_path: str = "./pretrained"
