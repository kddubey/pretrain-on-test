"""
Train a freshly loaded pretrained autoregressive LM on SFT-looking data w/o answers.
"""

from functools import partial
from transformers import DataCollatorForLanguageModeling
from transformers.trainer_utils import TrainOutput

from pretrain_on_test import Config
from pretrain_on_test.data import ClassificationDatasetInfo
from . import _dum


def train(
    texts: list[str],
    config: Config,
    classification_dataset_info: ClassificationDatasetInfo,
) -> TrainOutput:
    """
    Returns a finetuned model and its tokenizer.

    The freshly pretrained model is finetuned on SFT-looking data w/o answers.
    """
    class_names = [""] * len(texts)
    data_collator = DataCollatorForLanguageModeling(
        config.tokenizer,
        mlm=config.mlm,
        mlm_probability=config.mlm_probability,
    )
    _, train_output = _dum.train(
        texts,
        class_names,
        classification_dataset_info.class_names,
        classification_dataset_info.task_description,
        data_collator,
        config.tokenizer,
        from_pretrained_lora=False,
        pretrained_model_name_or_path=config.model_id,
        output_dir=config.model_path_pretrained,
        per_device_train_batch_size=config.per_device_train_batch_size_pretrain,
        num_train_epochs=config.num_train_epochs_pretrain,
        max_length=config.max_length,
        lora=config.lora_pretrain,
        qlora=config.qlora,
        is_pretrained_fresh=True,
        device_map=config.device,
        chat_text_post_processor=partial(
            _dum.chat_text_post_processor, config.tokenizer
        ),
    )
    return train_output
