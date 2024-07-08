"""
Train a pretrained autoregressive LM, optionally with (Q)LoRA.
"""

from functools import partial
from os.path import commonprefix
from typing import Callable, TypedDict, cast
import warnings

import cappr
from datasets import Dataset
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
    LoraConfig,
    PeftMixedModel,
    TaskType,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer_utils import TrainOutput
from trl import SFTConfig, SFTTrainer

from pretrain_on_test import Config
from pretrain_on_test.data import ClassificationDatasetInfo, NUM_CHARACTERS_MAX


QUERY_TEMPLATE = "### Text:"

RESPONSE_TEMPLATE = " ### Answer:"
"""
That extra space-----^-is necessary in SFT for correct parsing of the 
completion/response for BPE tokenizers
"""

_MODELS_WHICH_DONT_DO_WELL_WITH_SYSTEM_PROMPT = {
    "phi-3",  # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/51#66328cab9292069aed6a425b
}
"""
These models might have 'system' in their chat_template but apparently they shouldn't be
used
"""

_IS_BF16_SUPPORTED = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _instruction_formatter(
    class_names_unique: tuple[str, ...], task_description: str
) -> str:
    class_names_unique_string = "\n".join(class_names_unique)
    return (
        "Your task is to classify a given text as one of these categories:\n"
        f"{class_names_unique_string}\n\n"
        f"{task_description}\n\n"
    )


def _query_formatter(text: str) -> str:
    return f"{QUERY_TEMPLATE} {text[:NUM_CHARACTERS_MAX]}\n"


def _answer_formatter(class_name: str) -> str:
    return f"{RESPONSE_TEMPLATE} {class_name}".rstrip()


class _Message(TypedDict):
    role: str
    content: str


def _get_apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[list[_Message]], str]:
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


def chat_text_post_processor(tokenizer: PreTrainedTokenizerBase, chat_text: str) -> str:
    # W/o this, training data would look like, e.g.,
    #
    # blah blah blah ### Answer:</s>
    #
    # That's prolly a problem b/c the model will allocate a ton of probability to the
    # EOS token after the : token, which might throw it off when we need classification
    # answers.
    model_name = _model_name(tokenizer)
    if "phi-3" in model_name.lower():
        # phi-3 doesn't record this in tokenizer.special_tokens_map or anywhere besides
        # tokenizer.chat_template, which I'm not sure how to automatically parse
        suffix_token = "<|end|>\n"
    else:
        suffix_token = tokenizer.eos_token
    if suffix_token is not None:
        return chat_text.removesuffix(suffix_token)
    else:
        return chat_text


def _formatter(
    chats: list[list[_Message]],
    tokenizer: PreTrainedTokenizerBase,
    chat_text_post_processor: Callable[[str], str] | None = None,
) -> list[str]:
    apply_chat_template = _get_apply_chat_template(tokenizer)
    chat_texts = [apply_chat_template(chat) for chat in chats]
    if chat_text_post_processor is not None:
        chat_texts = [chat_text_post_processor(chat_text) for chat_text in chat_texts]
    return chat_texts


def _model_name(tokenizer: PreTrainedTokenizerBase) -> str:
    model_name = cast(str, tokenizer.name_or_path)
    if model_name.startswith("_"):
        raise ValueError(
            f"Not sure where this tokenizer came from: {model_name}. May not be able "
            "to correctly determine its name."
        )
    return model_name


def _system_role(tokenizer: PreTrainedTokenizerBase) -> str | None:
    # TODO: what's a good way to do this? Current way prolly inaccurate. rn I'm just
    # checking the tokenizer manually before running it
    model_name = _model_name(tokenizer)
    if any(
        name in model_name.lower()
        for name in _MODELS_WHICH_DONT_DO_WELL_WITH_SYSTEM_PROMPT
    ):
        return None
    else:
        return "system" if "'system'" in (tokenizer.chat_template or "") else None


def load_model(
    from_pretrained_lora: bool,
    pretrained_model_name_or_path: str,
    qlora: bool,
    is_pretrained_fresh: bool = False,
    device_map: str = "auto",
) -> PreTrainedModel:
    if qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None
    loading_kwargs = dict(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device_map="auto" if qlora else device_map,
        quantization_config=quantization_config,
        torch_dtype="auto" if quantization_config is None else None,
    )
    if from_pretrained_lora and not is_pretrained_fresh:
        model = AutoPeftModelForCausalLM.from_pretrained(**loading_kwargs)
        return cast(PeftMixedModel, model).merge_and_unload()
    else:
        return AutoModelForCausalLM.from_pretrained(**loading_kwargs)


def train(
    texts: list[str],
    class_names: list[str],
    class_names_unique: tuple[str, ...],
    task_description: str,
    # bleh
    data_collator: DataCollatorForLanguageModeling,
    tokenizer: PreTrainedTokenizerBase,
    from_pretrained_lora: bool,
    pretrained_model_name_or_path: str,
    output_dir: str,
    per_device_train_batch_size: int,
    num_train_epochs: int,
    max_length: int,
    lora: bool,
    qlora: bool,
    is_pretrained_fresh: bool = False,
    device_map: str = "auto",
    chat_text_post_processor: Callable[[str], str] | None = None,
) -> tuple[tuple[PreTrainedModel, PreTrainedTokenizerBase], TrainOutput]:
    """
    Returns a finetuned model and its tokenizer.
    """
    dataset = Dataset.from_dict(
        {
            "chat": _create_chats(
                texts,
                class_names,
                class_names_unique,
                task_description,
                system_role=_system_role(tokenizer),
            )
        }
    )

    # Set up model
    model = load_model(
        from_pretrained_lora,
        pretrained_model_name_or_path,
        qlora,
        is_pretrained_fresh=is_pretrained_fresh,
        device_map=device_map,
    )
    if qlora:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    if lora or qlora:
        parameters_names = "\n".join(model.state_dict().keys())
        target_modules = (
            ["qkv_proj"] if "qkv_proj" in parameters_names else ["q_proj", "v_proj"]
        )
        # HPs from https://huggingface.co/docs/trl/en/sft_trainer#training-adapters
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Set up the tokenizer for batching
    if tokenizer.pad_token_id is None:
        # To enable batching. It's ok to do this b/c we never use the EOS token / we're
        # never generating and terminating.
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Set up trainer. The data_collator defines the objective.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            max_seq_length=max_length,
            save_strategy="no",
            optim="paged_adamw_8bit" if qlora else "adamw_torch",
            learning_rate=2e-4 if qlora else 5e-5,
            fp16=qlora and not _IS_BF16_SUPPORTED,
            bf16=qlora and _IS_BF16_SUPPORTED,
            disable_tqdm=False,
        ),
        train_dataset=dataset,
        formatting_func=lambda batch: _formatter(
            batch["chat"],
            tokenizer,
            chat_text_post_processor=chat_text_post_processor,
        ),
        data_collator=data_collator,
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
        train_output = trainer.train()  # train modifies the model object itself.
    trainer.model.save_pretrained(output_dir)  # just save LoRA's weights
    return (trainer.model, trainer.tokenizer), train_output


def _instruction_and_prompts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    class_names_unique: tuple[str, ...],
    task_description: str,
) -> tuple[str, list[str]]:
    """
    Factors out the common instruction from `texts`. This is useful b/c CAPPr caches it.
    """
    chats_without_answers = _create_chats(
        texts,
        class_names=[""] * len(texts),
        class_names_unique=class_names_unique,
        task_description=task_description,
        system_role=_system_role(tokenizer),
    )
    prompts = _formatter(
        chats_without_answers,
        tokenizer,
        chat_text_post_processor=partial(chat_text_post_processor, tokenizer),
    )
    instruction = commonprefix(prompts)
    instruction = instruction[: instruction.index(QUERY_TEMPLATE)]
    prompts = [prompt.removeprefix(instruction) for prompt in prompts]
    return instruction, prompts


def predict_proba(
    texts: list[str],
    model_and_tokenizer: tuple[PreTrainedModel, PreTrainedTokenizerBase],
    config: Config,
    classification_dataset_info: ClassificationDatasetInfo,
):
    class_names_unique = classification_dataset_info.class_names
    instruction, prompts = _instruction_and_prompts(
        texts,
        model_and_tokenizer[1],
        class_names_unique,
        classification_dataset_info.task_description,
    )
    with cappr.huggingface.classify.cache(
        model_and_tokenizer, prefixes=instruction, logits_all=False
    ) as cached:
        return cappr.huggingface.classify.predict_proba(
            prompts,
            completions=class_names_unique,
            model_and_tokenizer=cached,
            batch_size=config.per_device_eval_batch_size_classification,
            batch_size_completions=None,  # run all completions at once
        )
