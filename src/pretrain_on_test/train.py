from __future__ import annotations
from typing import Sequence

import torch
from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from pretrain_on_test import Config


class _TextClassificationDataset(torch.utils.data.Dataset):
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


def _classification_dataset(
    texts: list[str], labels: list[int], tokenizer: PreTrainedTokenizerBase
) -> _TextClassificationDataset:
    encodings = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    labels = torch.tensor(labels)
    return _TextClassificationDataset(encodings, labels)


def classification(
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
    train_dataset = _classification_dataset(texts, labels)
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
                pretrained_model_name_or_path, num_labels=num_labels
            ).to("cuda")
        ),
        args=classifier_args,
        train_dataset=train_dataset,
    )
    classifier_trainer.train()
    return classifier_trainer


def accuracy(texts: list[str], labels: list[int], trained_classifier: Trainer) -> float:
    """
    Returns the accuracy of `trained_classifier` on `texts` by comparing its
    predictions to `labels`.
    """
    eval_dataset = _classification_dataset(texts, labels)
    pred_out = trained_classifier.predict(eval_dataset)
    preds = pred_out.predictions.argmax(axis=1)
    return torch.mean(preds == labels)
