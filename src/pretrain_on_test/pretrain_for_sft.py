"""
Train a freshly loaded pretrained autoregressive LM on SFT-looking data w/o answers.
"""

from functools import partial
from transformers import DataCollatorForLanguageModeling

from pretrain_on_test import Config
from . import _dum


def chat_text_post_processor(eos_token: str | None, chat_text: str) -> str:
    # W/o this, training data would look like:
    # blah blah blah ### Answer:</s>
    # Pretty sure that's a problem b/c the model will allocate a ton of probability to
    # the EOS token after the : token, which might throw it off when we need
    # classification answers.
    if eos_token is not None:
        return chat_text.removesuffix(eos_token)
    else:
        return chat_text


def train(
    texts: list[str],
    class_names_unique: tuple[str, ...],
    task_description: str,
    config: Config,
):
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
    return _dum.train(
        texts,
        class_names,
        class_names_unique,
        task_description,
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
            chat_text_post_processor, config.tokenizer.eos_token
        ),
    )
