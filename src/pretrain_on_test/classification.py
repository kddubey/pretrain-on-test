from __future__ import annotations
from typing import Sequence

import numpy as np
import torch
from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from pretrain_on_test import Config


class _Dataset(torch.utils.data.Dataset):
    # taken from
    # https://huggingface.co/transformers/v3.2.0/custom_datasets.html#sequence-classification-with-imdb-reviews
    def __init__(self, encodings: BatchEncoding, labels: Sequence[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def _dataset(
    texts: list[str], labels: list[int], tokenizer: PreTrainedTokenizerBase
) -> _Dataset:
    encodings = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    labels = torch.tensor(labels)
    return _Dataset(encodings, labels)


def train(
    texts: list[str],
    labels: list[int],
    num_labels: int,
    config: Config,
    pretrained_model_name_or_path: str | None = None,
) -> Trainer:
    """
    Returns a model `Trainer` which was finetuned on classification data `texts,
    labels`. The model is saved in `config.model_path_classification`.

    If `pretrained_model_name_or_path is None`, then the model at
    `config.model_path_pretrained` is finetuned.
    """
    train_dataset = _dataset(texts, labels, config.tokenizer)
    classifier_args = TrainingArguments(
        output_dir=config.model_path_classification,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        optim="adamw_torch",
        disable_tqdm=False,
    )
    pretrained_model_name_or_path = (
        pretrained_model_name_or_path or config.model_path_pretrained
    )
    classifier_trainer = Trainer(
        model=(
            config.model_class_classification.from_pretrained(
                pretrained_model_name_or_path,
                num_labels=num_labels,
                output_attentions=False,
                output_hidden_states=False,
                ignore_mismatched_sizes=False,
            ).to(config.device)
        ),
        args=classifier_args,
        train_dataset=train_dataset,
    )
    classifier_trainer.train()
    return classifier_trainer


def predict_proba(
    texts: list[str], labels: list[int], trained_classifier: Trainer, config: Config
) -> np.ndarray:
    eval_dataset = _dataset(texts, labels, config.tokenizer)
    logits: np.ndarray = trained_classifier.predict(eval_dataset).predictions
    # I'm pretty sure that predictions are logits, not log-probs
    probs: torch.Tensor = torch.softmax(
        torch.tensor(logits, device=config.device), axis=-1
    )
    return probs.numpy(force=True)


# for class_idx, class_preds in enumerate(x.T):
#     df[f"pred_x_{class_idx}"] = class_preds
