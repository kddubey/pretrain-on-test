"""
Pretrain on raw text or instructions with text.

Classify by adding a linear layer, or SFT, or zero-shot.

LoRA, QLoRA supported.

Repeatedley subsample a dataset to expose training and evaluation variance.
"""

from ._config import Config
from . import (
    classification,
    classification_sft,
    classification_zero_shot,
    data,
    experiment,
    pretrain,
    pretrain_for_sft,
)

__all__ = [
    "classification",
    "classification_sft",
    "classification_zero_shot",
    "Config",
    "data",
    "experiment",
    "pretrain",
    "pretrain_for_sft",
]
