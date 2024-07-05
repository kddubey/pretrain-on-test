"""
Train a (finetuned) autoregressive LM to do classification using SFT.
"""

from os.path import commonprefix

import cappr
from transformers import PreTrainedModel, PreTrainedTokenizerBase
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
        config.model_class_classification,
        config.tokenizer,
        config.lora_pretrain,
        pretrained_model_name_or_path or config.model_path_pretrained,
        config.model_path_classification,
        config.per_device_train_batch_size_classification,
        config.num_train_epochs_classification,
        config.max_length,
        config.lora_classification,
        config.qlora,
        is_pretrained_fresh=is_pretrained_fresh,
        device_map=config.device,
        chat_text_post_processor=None,
    )


def predict_proba(
    texts: list[str],
    model_and_tokenizer: tuple[PreTrainedModel, PreTrainedTokenizerBase],
    class_names_unique: tuple[str, ...],
    task_description: str,
    batch_size: int = 2,
    batch_size_completions: int | None = None,
):
    _, tokenizer = model_and_tokenizer
    chats_without_answers = _dum._create_chats(
        texts,
        class_names=[""] * len(texts),
        class_names_unique=class_names_unique,
        task_description=task_description,
        system_role=_dum._system_role(tokenizer),
    )
    prompts = _dum._formatter(chats_without_answers, tokenizer)
    instruction = commonprefix(prompts)
    instruction = instruction[: instruction.index(_dum.QUERY_TEMPLATE)]
    prompts = [
        prompt.removeprefix(instruction).removesuffix(tokenizer.eos_token)
        for prompt in prompts
    ]

    with cappr.huggingface.classify.cache(
        model_and_tokenizer, prefixes=instruction, logits_all=False
    ) as cached:
        return cappr.huggingface.classify.predict_proba(
            prompts,
            completions=class_names_unique,
            model_and_tokenizer=cached,
            batch_size=batch_size,
            batch_size_completions=batch_size_completions,
        )
