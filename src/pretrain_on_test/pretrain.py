"""
Train a freshly loaded pretrained LM using its original loss function: MLM for BERT, CLM
for GPT-2
"""

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
        self,
        sentences,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length

    def __getitem__(self, item):
        return self.tokenizer(
            self.sentences[item],
            max_length=self.max_length,
            add_special_tokens=True,
            truncation=True,
            return_special_tokens_mask=True,
        )

    def __len__(self):
        return len(self.sentences)


def train(texts: list[str], config: Config):
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
    train_dataset = _Dataset(texts, config.tokenizer, config.max_length)
    data_collator = DataCollatorForLanguageModeling(
        config.tokenizer,
        mlm=config.mlm,
        mlm_probability=config.mlm_probability,
    )
    # Set up Trainer
    training_args = TrainingArguments(
        output_dir=config.model_path_pretrained,
        per_device_train_batch_size=config.per_device_train_batch_size_pretrain,
        num_train_epochs=config.num_train_epochs_pretrain,
        overwrite_output_dir=True,
        learning_rate=1e-4,
        save_strategy="no",
        optim="adamw_torch",
        prediction_loss_only=True,
        disable_tqdm=False,
    )
    # Trainer will modify the model, so need to re-load a fresh one every time this
    # function is called
    model = config.model_class_pretrain.from_pretrained(
        config.model_id,
        output_attentions=False,
        output_hidden_states=False,
    ).to(config.device)
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
