"""
Pretrain on raw text or instructions with text.

Classify by adding a linear layer, or SFT, or zero-shot.

LoRA, QLoRA supported.

Repeatedley subsample a dataset to expose training and evaluation variance.
"""

# Topo order
from ._config import Config
from . import (
    data,
    protocols,
    classification,
    classification_sft,
    classification_zero_shot,
    pretrain,
    pretrain_for_sft,
    experiment,
)

__all__ = [
    "Config",
    "data",
    "protocols",
    "classification",
    "classification_sft",
    "classification_zero_shot",
    "pretrain",
    "pretrain_for_sft",
    "experiment",
]
