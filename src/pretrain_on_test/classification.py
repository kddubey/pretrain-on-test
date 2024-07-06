"""
Train a pretrained LM to do classification by adding a linear layer transforming a
representative token embedding ([CLS] for BERT, or the last token for autoregressive
LMs) to a distribution over classes.
"""

from typing import cast
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
import torch
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from pretrain_on_test import Config


class _Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        texts: list[str],
        labels: list[int] | None = None,
    ):
        self.encodings = tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        )
        self._num_texts = self.encodings["input_ids"].shape[0]
        if labels is not None:
            self.labels = torch.tensor(labels)
        else:  # just doing inference
            self.labels = None

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return self._num_texts


def train(
    texts: list[str],
    labels: list[int],
    num_labels: int,
    config: Config,
    pretrained_model_name_or_path: str | None = None,
    is_pretrained_fresh: bool = False,  # TODO: will need for pretrained LoRA
) -> Trainer:
    """
    Returns a model `Trainer` which was finetuned on classification data `texts,
    labels`. The model is saved in `config.model_path_classification`.

    If `pretrained_model_name_or_path is None`, then the model at
    `config.model_path_pretrained` is finetuned.
    """
    train_dataset = _Dataset(config.tokenizer, texts, labels)
    pretrained_model_name_or_path = (
        pretrained_model_name_or_path or config.model_path_pretrained
    )

    # Load in the pretrained model and add a linear layer to it for performing
    # classification
    device_map = config.device if "bert" in config.model_id.lower() else "auto"
    # BertForSequenceClassification does not support `device_map='auto'`
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
        ignore_mismatched_sizes=False,
        device_map=device_map,
    )
    model = cast(PreTrainedModel, model)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    # Maybe set up LoRA
    if config.lora_classification:
        lora_config = LoraConfig(  # TODO: check Raschka recommendations
            task_type=TaskType.SEQ_CLS,
            r=4,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Train and save to config.model_path_classification
    classifier_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=config.model_path_classification,
            per_device_train_batch_size=config.per_device_train_batch_size_classification,
            per_device_eval_batch_size=config.per_device_eval_batch_size_classification,
            num_train_epochs=config.num_train_epochs_classification,
            weight_decay=0.01,
            optim="adamw_torch",
            disable_tqdm=False,
        ),
        train_dataset=train_dataset,
        tokenizer=config.tokenizer,
    )
    classifier_trainer.train()
    return classifier_trainer


def predict_proba(texts: list[str], trained_classifier: Trainer) -> np.ndarray:
    eval_dataset = _Dataset(trained_classifier.tokenizer, texts)
    logits: np.ndarray = trained_classifier.predict(eval_dataset).predictions
    # predictions are logits, not log-probs. (I checked that some are positive)
    probs: torch.Tensor = torch.softmax(
        torch.tensor(logits, device=trained_classifier.model.device), axis=-1
    )
    return probs.numpy(force=True)
