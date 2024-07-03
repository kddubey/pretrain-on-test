"""
Train a pretrained autoregressive LM to do classification using SFT.
"""

from functools import partial
from typing import cast
import warnings

import cappr
from datasets import Dataset
import numpy as np
from peft import (
    get_peft_model,
    AutoPeftModelForCausalLM,
    LoraConfig,
    PeftMixedModel,
    TaskType,
)
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
except AssertionError as exception:
    if not str(exception).startswith("Torch not compiled with CUDA enabled"):
        raise exception
    print("Not importing unsloth")
    FastLanguageModel = type("Dummy", (object,), {})
    is_bfloat16_supported = lambda: False

from pretrain_on_test import Config


RESPONSE_TEMPLATE = " ### Answer:"
# That extra space---^---is necessary in SFT for correct parsing of the
# completion/response for BPE tokenizers


def _instruction_formatter(
    class_names_unique: tuple[str, ...], task_description: str
) -> str:
    class_names_unique_string = "\n".join(class_names_unique)
    return (
        "Your task is to classify a given text as one of these categories:\n"
        f"{class_names_unique_string}\n\n"
        f"{task_description}\n\n"
    )


def _body_formatter(texts: list[str], class_names: list[str]) -> list[str]:
    return [
        (
            f"Text: {text}\n"
            f"{RESPONSE_TEMPLATE} {class_name}".rstrip()  # empty class_name = inference
        )
        for text, class_name in zip(texts, class_names, strict=True)
    ]


def _prompt_completion_formatter(
    class_names_unique: tuple[str, ...],
    task_description: str,
    texts: list[str],
    class_names: list[str],
) -> list[str]:
    texts = [texts] if isinstance(texts, str) else texts
    class_names = [class_names] if isinstance(class_names, str) else class_names
    instruction = _instruction_formatter(class_names_unique, task_description)
    return [instruction + body for body in _body_formatter(texts, class_names)]


def _sft_trainer_formatting_func(
    class_names_unique: tuple[str, ...],
    task_description: str,
    batch: dict[str, list[str]],
):
    # The SFTTrainer requires this type of function. See:
    # https://huggingface.co/docs/trl/en/sft_trainer#train-on-completions-only
    return _prompt_completion_formatter(
        class_names_unique, task_description, batch["text"], batch["class_name"]
    )


def train(
    texts: list[str],
    labels: list[int],
    class_names_unique: tuple[str, ...],
    task_description: str,
    config: Config,
    pretrained_model_name_or_path: str | None = None,
    is_pretrained_fresh: bool = False,
    max_length_chars: int = 1_000,
) -> Trainer:
    """
    Returns a model `Trainer` which was finetuned on classification data `texts,
    labels`. The model is saved in `config.model_path_classification`.

    If `pretrained_model_name_or_path is None`, then the model at
    `config.model_path_pretrained` is finetuned.
    """
    dataset = Dataset.from_dict(
        {
            "text": [text[:max_length_chars] for text in texts],
            "class_name": [class_names_unique[label] for label in labels],
        }
    )

    # Load in the pretrained model
    pretrained_model_name_or_path = (
        pretrained_model_name_or_path or config.model_path_pretrained
    )
    if config.lora_pretrain and not is_pretrained_fresh:
        state_dict = (
            cast(
                PeftMixedModel,
                AutoPeftModelForCausalLM.from_pretrained(pretrained_model_name_or_path),
            )
            # Load in the pretrained LM (w/ the original/fresh pretrained weights) and
            # (separately) the adapter weights stored at pretrained_model_name_or_path
            .merge_and_unload()  # merge in the adapter weights
            .state_dict()
        )
    else:
        state_dict = None

    is_unsloth = issubclass(config.model_class_classification, FastLanguageModel)
    if is_unsloth:
        model, tokenizer = config.model_class_classification.from_pretrained(
            pretrained_model_name_or_path,
            state_dict=state_dict,
            load_in_4bit=True,
        )
        object.__setattr__(config, "tokenizer", tokenizer)
    else:
        model = config.model_class_classification.from_pretrained(
            pretrained_model_name_or_path, state_dict=state_dict
        ).to(config.device)

    # Maybe set up LoRA
    if config.lora_classification:
        if is_unsloth:
            model = FastLanguageModel.get_peft_model(model)
        else:
            # HPs from https://huggingface.co/docs/trl/en/sft_trainer#training-adapters
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=config.model_path_classification,
        per_device_train_batch_size=config.per_device_train_batch_size_classification,
        per_device_eval_batch_size=config.per_device_eval_batch_size_classification,
        num_train_epochs=config.num_train_epochs_classification,
        save_strategy="no",
        optim="adamw_torch" if not is_unsloth else "adamw_8bit",
        fp16=is_unsloth and (not is_bfloat16_supported()),
        bf16=is_unsloth and is_bfloat16_supported(),
        prediction_loss_only=True,
        disable_tqdm=False,
    )

    if config.tokenizer.pad_token_id is None:
        config.tokenizer.pad_token_id = config.tokenizer.eos_token_id
    # To enable batching. It's ok to do this b/c we never use the EOS token / we're
    # never generating and terminating.
    config.tokenizer.padding_side = "right"

    trainer = SFTTrainer(
        model=model,
        tokenizer=config.tokenizer,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=config.max_length,
        formatting_func=partial(
            _sft_trainer_formatting_func, class_names_unique, task_description
        ),
        data_collator=DataCollatorForCompletionOnlyLM(
            config.tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)[1:],
            tokenizer=config.tokenizer,
        ),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="error", category=UserWarning, message="Could not find response key"
        )
        # This warning indicates that RESPONSE_TEMPLATE was not found, which is likely
        # b/c of a whitespace tokenization issue. For now, I hardcoded data_collator to
        # work for Llama-like tokenizers which add a BOS token. TODO: check that it
        # works for BPE/GPT-2-like tokenizers
        trainer.train()
        # For non-LoRA, train modifies the model object itself. What about for LoRA?
        # TODO: check if I need to merge and unload
    return trainer


def predict_proba(
    texts: list[str],
    trained_classifier: SFTTrainer,
    class_names_unique: tuple[str, ...],
    task_description: str,
) -> np.ndarray:
    instruction = _instruction_formatter(class_names_unique, task_description)
    prompts = _body_formatter(texts, class_names=[""] * len(texts))

    try:
        FastLanguageModel.for_inference(trained_classifier.model)
    except Exception:
        pass

    with cappr.huggingface.classify.cache(
        model_and_tokenizer=(trained_classifier.model, trained_classifier.tokenizer),
        prefixes=instruction,
        logits_all=False,
    ) as cached:
        return cappr.huggingface.classify.predict_proba(
            prompts, completions=class_names_unique, model_and_tokenizer=cached
        )
