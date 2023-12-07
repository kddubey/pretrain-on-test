"""
Train a freshly loaded pretrained LM using its canonical loss function: MLM for BERT,
CLM for GPT-2.
"""
from __future__ import annotations

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from pretrain_on_test import Config


class _Dataset(torch.utils.data.Dataset):
    def __init__(
        self, sentences, tokenizer: PreTrainedTokenizerBase, max_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length

    def __getitem__(self, item):
        return self.tokenizer(
            self.sentences[item],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )

    def __len__(self):
        return len(self.sentences)


def train(
    texts: list[str],
    config: Config,
    max_length: int = 128,
):
    """
    Saves a fresh model which is pretrained on `texts` to
    `config.model_path_pretrained`.

    It can be loaded for other tasks using, for example::

        from transformers import GPT2ForSequenceClassification
        model = GPT2ForSequenceClassification.from_pretrained(
            config.model_path_pretrained
        )
    """
    # Set up data
    train_dataset = _Dataset(texts, config.tokenizer, max_length=max_length)
    data_collator = DataCollatorForLanguageModeling(
        config.tokenizer,
        mlm=config.mlm,
        mlm_probability=config.mlm_probability,
    )
    # Set up Trainer
    training_args = TrainingArguments(
        output_dir=config.model_path_pretrained,
        overwrite_output_dir=True,
        learning_rate=1e-4,
        num_train_epochs=2,
        per_device_train_batch_size=64,
        save_strategy="no",
        optim="adamw_torch",
        prediction_loss_only=True,
        disable_tqdm=False,
    )
    # Trainer will modify the model, so need to re-load a fresh one every time this
    # function is called
    model = config.model_class_pretrain.from_pretrained(config.model_id).to(
        config.device
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model()
