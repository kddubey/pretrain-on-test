"""
Train a freshly loaded pretrained autoregressive LM on SFT-looking data w/o answers.
"""

from transformers import DataCollatorForLanguageModeling

from pretrain_on_test import Config
from . import _dum


def chat_text_post_processor(eos_token: str, chat_text: str) -> str:
    # W/o this, training data would look like:
    # blah blah blah ### Answer:</s>
    # Pretty sure that's a problem b/c the model will allocate a ton of probability to
    # the EOS token after the : token, which might throw it off when we need
    # classification answers.
    return chat_text.removesuffix(eos_token)


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
    from_pretrained_lora = False
    return _dum.train(
        texts,
        class_names,
        class_names_unique,
        task_description,
        data_collator,
        config.model_class_pretrain,
        config.tokenizer,
        from_pretrained_lora,
        config.model_id,
        config.model_path_pretrained,
        config.per_device_train_batch_size_pretrain,
        config.num_train_epochs_pretrain,
        config.max_length,
        config.lora_pretrain,
        config.qlora,
        is_pretrained_fresh=not from_pretrained_lora,
        device_map=config.device,
        chat_text_post_processor=chat_text_post_processor,
    )
