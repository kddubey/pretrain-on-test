"""
Train a pretrained autoregressive LM to do classification using SFT.
"""

from functools import partial
from os.path import commonprefix
from typing import Callable, TypedDict, cast
import warnings

import cappr
from datasets import Dataset
import numpy as np
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
    LoraConfig,
    PeftMixedModel,
    TaskType,
)
import torch
from transformers import BitsAndBytesConfig, PreTrainedTokenizerBase, Trainer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

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


def _query_formatter(text: str, max_length_chars: int = 1_000) -> str:
    return f"Text: {text[:max_length_chars]}\n"


def _answer_formatter(class_name: str) -> str:
    return f"{RESPONSE_TEMPLATE} {class_name}".rstrip()


class _Message(TypedDict):
    role: str
    content: str


def _get_apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[list[_Message]], str]:
    # This is slightly incorrect for transfomers < 4.43. Maybe some tokenizers for
    # instruction-trained models use the default_chat_template, and leave chat_template
    # as None. This is the warning you get if tokenizer.chat_template is None:
    #
    # No chat template is set for this tokenizer, falling back to a default class-level
    # template. This is very error-prone, because models are often trained with
    # templates different from the class default! Default chat templates are a legacy
    # feature and will be removed in Transformers v4.43, at which point any code
    # depending on them will stop working.
    if tokenizer.chat_template is not None:
        return partial(tokenizer.apply_chat_template, tokenize=False)
    else:

        def join_content(messages: list[_Message]) -> str:
            return "".join(message["content"] for message in messages)

        return join_content


def _create_chats(
    texts: list[str],
    class_names: list[str],
    class_names_unique: tuple[str, ...],
    task_description: str,
    system_role: str | None = "system",
) -> list[list[_Message]]:
    instruction = _instruction_formatter(class_names_unique, task_description)

    def instruction_and_query(text: str) -> list[_Message]:
        if system_role is None:
            return [{"role": "user", "content": instruction + _query_formatter(text)}]
        else:
            return [
                {"role": system_role, "content": instruction},
                {"role": "user", "content": _query_formatter(text)},
            ]

    return [
        instruction_and_query(text)
        + [{"role": "assistant", "content": _answer_formatter(class_name)}]
        for text, class_name in zip(texts, class_names, strict=True)
    ]


def _formatter(
    tokenizer: PreTrainedTokenizerBase, chats: list[list[_Message]]
) -> list[str]:
    apply_chat_template = _get_apply_chat_template(tokenizer)
    res = [apply_chat_template(chat) for chat in chats]
    return res


def _system_role(tokenizer: PreTrainedTokenizerBase) -> str | None:
    return "system" if "'system'" in (tokenizer.chat_template or "") else None


def train(
    texts: list[str],
    labels: list[int],
    class_names_unique: tuple[str, ...],
    task_description: str,
    config: Config,
    pretrained_model_name_or_path: str | None = None,
    is_pretrained_fresh: bool = False,
) -> Trainer:
    """
    Returns a model `Trainer` which was finetuned on classification data `texts,
    labels`. The model is saved in `config.model_path_classification`.

    If `pretrained_model_name_or_path is None`, then the model at
    `config.model_path_pretrained` is finetuned.
    """
    dataset = Dataset.from_dict(
        {
            "chat": _create_chats(
                texts,
                [class_names_unique[label] for label in labels],
                class_names_unique,
                task_description,
                system_role=_system_role(config.tokenizer),
            )
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

    model = config.model_class_classification.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        state_dict=state_dict,
        device_map="auto" if config.sft_qlora else config.device,
        # TODO: always use auto. Annoying b/c auto results in mps on my macbook, which
        # doesn't work b/c of missing attn kernels. Should instead use CPU.
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if config.sft_qlora
            else None
        ),
    )
    if config.sft_qlora:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

    # Maybe set up LoRA
    if config.lora_classification:
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

    if config.tokenizer.pad_token_id is None:
        # To enable batching. It's ok to do this b/c we never use the EOS token / we're
        # never generating and terminating.
        config.tokenizer.pad_token_id = config.tokenizer.eos_token_id
    config.tokenizer.padding_side = "right"

    # model.config.use_cache = False
    trainer = SFTTrainer(
        model=model,
        tokenizer=config.tokenizer,
        args=SFTConfig(
            output_dir=config.model_path_classification,
            per_device_train_batch_size=config.per_device_train_batch_size_classification,
            per_device_eval_batch_size=config.per_device_eval_batch_size_classification,
            num_train_epochs=config.num_train_epochs_classification,
            max_seq_length=config.max_length,
            save_strategy="no",
            optim="paged_adamw_8bit" if config.sft_qlora else "adamw_torch",
            learning_rate=2e-4 if config.sft_qlora else 5e-5,
            fp16=config.sft_qlora,
            disable_tqdm=False,
        ),
        train_dataset=dataset,
        formatting_func=lambda batch: _formatter(config.tokenizer, batch["chat"]),
        data_collator=DataCollatorForCompletionOnlyLM(
            config.tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)[1:],
            tokenizer=config.tokenizer,
        ),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="error", category=UserWarning, message="Could not find response key"
        )
        # This warning indicates something went quite wrong. Opting to raise it as an
        # error instead. It indicates that RESPONSE_TEMPLATE was not found, which is
        # likely b/c of a whitespace tokenization issue. For now, I hardcoded
        # data_collator to work for Llama-like tokenizers which add a BOS token. TODO:
        # check that it works for BPE/GPT-2-like tokenizers
        trainer.train()  # train modifies the model object itself.
    return trainer


def predict_proba(
    texts: list[str],
    trained_classifier: SFTTrainer,
    class_names_unique: tuple[str, ...],
    task_description: str,
    batch_size: int = 2,
    batch_size_completions: int | None = None,
) -> np.ndarray:
    chats_without_answers = _create_chats(
        texts,
        class_names=[""] * len(texts),
        class_names_unique=class_names_unique,
        task_description=task_description,
        system_role=_system_role(trained_classifier.tokenizer),
    )
    prompts = _formatter(trained_classifier.tokenizer, chats_without_answers)
    instruction = commonprefix(prompts)
    prompts = [prompt.removeprefix(instruction) for prompt in prompts]

    with cappr.huggingface.classify.cache(
        model_and_tokenizer=(trained_classifier.model, trained_classifier.tokenizer),
        prefixes=instruction.rstrip(),
        logits_all=False,
    ) as cached:
        return cappr.huggingface.classify.predict_proba(
            prompts,
            completions=class_names_unique,
            model_and_tokenizer=cached,
            batch_size=batch_size,
            batch_size_completions=batch_size_completions,
        )
