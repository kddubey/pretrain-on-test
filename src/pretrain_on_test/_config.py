from dataclasses import dataclass
import os
from typing import Literal

from huggingface_hub import login
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
    model_class_pretrain: type[PreTrainedModel]
    mlm: bool | None = None
    mlm_probability: float | None = None
    lora_pretrain: bool = False
    pretrain_method: Literal["raw-text", "instructions-with-text"] = "raw-text"
    classification_method: Literal["linear-layer", "sft", "zero-shot"] = "linear-layer"
    lora_classification: bool = False
    qlora: bool = False
    tokenizer: PreTrainedTokenizerBase | None = None
    device: str | torch.device | None = None
    per_device_train_batch_size_pretrain: int = 16
    per_device_train_batch_size_classification: int = 16
    per_device_eval_batch_size_classification: int = 64
    max_length: int | None = None
    num_train_epochs_classification: int = 3
    num_train_epochs_pretrain: int = 2
    model_path_pretrained: str = "_pretrained"
    model_path_classification: str = "_classifier"
    requires_hf_login: bool = False
    keep_models: bool = False
    pack: bool = False

    def __post_init__(self):
        if self.requires_hf_login:
            # Do this first so that the tokenizer can be downloaded
            login(token=os.environ["HF_TOKEN"])
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
        if self.pack and self.pretrain_method != "instructions-with-text":
            raise ValueError(
                "Currently, packing is only enabled for pretraining on instructions "
                "with text"
            )
