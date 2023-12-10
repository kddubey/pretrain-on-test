"""
Train a pretrained LM using categorical cross entropy loss
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, Trainer, TrainingArguments

from pretrain_on_test import Config


class _Dataset(torch.utils.data.Dataset):
    def __init__(
        self, texts: list[str], labels: list[int], tokenizer: PreTrainedTokenizerBase
    ):
        self.encodings = tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        )
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


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
    train_dataset = _Dataset(texts, labels, config.tokenizer)
    classifier_args = TrainingArguments(
        output_dir=config.model_path_classification,
        per_device_train_batch_size=config.per_device_train_batch_size_classification,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        optim="adamw_torch",
        disable_tqdm=False,
    )
    pretrained_model_name_or_path = (
        pretrained_model_name_or_path or config.model_path_pretrained
    )
    model = config.model_class_classification.from_pretrained(
        pretrained_model_name_or_path,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
        ignore_mismatched_sizes=False,
    ).to(config.device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    classifier_trainer = Trainer(
        model=model,
        args=classifier_args,
        train_dataset=train_dataset,
    )
    classifier_trainer.train()
    return classifier_trainer


def predict_proba(
    texts: list[str], labels: list[int], trained_classifier: Trainer, config: Config
) -> np.ndarray:
    eval_dataset = _Dataset(texts, labels, config.tokenizer)
    logits: np.ndarray = trained_classifier.predict(eval_dataset).predictions
    # I'm pretty sure that predictions are logits, not log-probs
    probs: torch.Tensor = torch.softmax(
        torch.tensor(logits, device=config.device), axis=-1
    )
    return probs.numpy(force=True)
