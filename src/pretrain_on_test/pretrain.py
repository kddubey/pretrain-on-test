"""
Train a freshly loaded pretrained LM using its original loss function: MLM for BERT, CLM
for everything else.
"""

from peft import get_peft_model, LoraConfig, TaskType
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import TrainOutput

from pretrain_on_test import Config
from pretrain_on_test.data import ClassificationDatasetInfo


class _Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        texts: list[str],
        max_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __getitem__(self, item):
        return self.tokenizer(
            self.texts[item],
            max_length=self.max_length,
            add_special_tokens=True,
            truncation=True,
            return_special_tokens_mask=True,
        )

    def __len__(self):
        return len(self.texts)


def train(
    texts: list[str],
    config: Config,
    classification_dataset_info: ClassificationDatasetInfo,
) -> TrainOutput:
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
    train_dataset = _Dataset(config.tokenizer, texts, config.max_length)
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

    # Maybe set up LoRA
    if config.lora_pretrain:
        # TODO: Raschka recommends enabling for more layers:
        # https://magazine.sebastianraschka.com/i/138081202/enable-lora-for-more-layers
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Train and save to config.model_path_pretrained
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    train_output = trainer.train()
    if config.lora_pretrain:
        # Just save the adapter weights. We'll merge them into the base model before
        # classification training
        model.save_pretrained(config.model_path_pretrained)
    else:
        trainer.save_model()
    return train_output
