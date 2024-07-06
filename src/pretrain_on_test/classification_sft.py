"""
Train a (finetuned) autoregressive LM to do classification using SFT.
"""

from trl import DataCollatorForCompletionOnlyLM

from pretrain_on_test import Config
from . import _dum


def train(
    texts: list[str],
    labels: list[int],
    class_names_unique: tuple[str, ...],
    task_description: str,
    config: Config,
    pretrained_model_name_or_path: str | None = None,
    is_pretrained_fresh: bool = False,
):
    """
    Returns a finetuned model and its tokenizer.

    The model is finetuned via SFT to do classification.

    If `pretrained_model_name_or_path is None`, then the model at
    `config.model_path_pretrained` is finetuned.
    """
    class_names = [class_names_unique[label] for label in labels]
    response_template_ids = config.tokenizer.encode(
        _dum.RESPONSE_TEMPLATE, add_special_tokens=False
    )[1:]  # is correct b/c RESPONSE_TEMPLATE starts with a whitespace
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=config.tokenizer
    )
    return _dum.train(
        texts,
        class_names,
        class_names_unique,
        task_description,
        data_collator,
        config.tokenizer,
        config.lora_pretrain,
        pretrained_model_name_or_path=(
            pretrained_model_name_or_path or config.model_path_pretrained
        ),
        output_dir=config.model_path_classification,
        per_device_train_batch_size=config.per_device_train_batch_size_classification,
        num_train_epochs=config.num_train_epochs_classification,
        max_length=config.max_length,
        lora=config.lora_classification,
        qlora=config.qlora,
        is_pretrained_fresh=is_pretrained_fresh,
        device_map=config.device,
        chat_text_post_processor=None,
    )


predict_proba = _dum.predict_proba
