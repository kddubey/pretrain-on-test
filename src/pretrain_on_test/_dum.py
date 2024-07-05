"""
Train a pretrained autoregressive LM, optionally with (Q)LoRA.
"""

from functools import partial
from typing import Callable, TypedDict, cast
import warnings

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
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import SFTConfig, SFTTrainer


QUERY_TEMPLATE = "### Text:"
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
    return f"{QUERY_TEMPLATE} {text[:max_length_chars]}\n"


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


def _formatter(
    chats: list[list[_Message]],
    tokenizer: PreTrainedTokenizerBase,
    chat_text_post_processor: Callable[[str], str] | None = None,
) -> list[str]:
    apply_chat_template = _get_apply_chat_template(tokenizer)
    chat_texts = [apply_chat_template(chat) for chat in chats]
    if chat_text_post_processor is None:
        return chat_texts
    else:
        return [chat_text_post_processor(chat_text) for chat_text in chat_texts]


def _system_role(tokenizer: PreTrainedTokenizerBase) -> str | None:
    return "system" if "'system'" in (tokenizer.chat_template or "") else None


def train(
    texts: list[str],
    class_names: list[str],
    class_names_unique: tuple[str, ...],
    task_description: str,
    # bleh
    data_collator: DataCollatorForLanguageModeling,
    model_class: type[PreTrainedModel],
    tokenizer: PreTrainedTokenizerBase,
    from_pretrained_lora: bool,
    pretrained_model_name_or_path: str,
    output_dir: str,
    per_device_train_batch_size_classification: int,
    num_train_epochs_classification: int,
    max_length: int,
    lora: bool,
    qlora: bool,
    is_pretrained_fresh: bool = False,
    device_map: str = "auto",
    chat_text_post_processor: Callable[[str], str] | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
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

    # Load in the pretrained model
    if from_pretrained_lora and not is_pretrained_fresh:
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

    model = model_class.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        state_dict=state_dict,
        device_map="auto" if qlora else device_map,
        # TODO: always use auto. Annoying b/c auto results in mps on my macbook for
        # transformers >= 4.42, which doesn't work b/c of missing attn kernels. Should
        # instead use CPU.
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if qlora
            else None
        ),
    )
    if qlora:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

    # Maybe set up LoRA
    if lora or qlora:
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
            per_device_train_batch_size=per_device_train_batch_size_classification,
            num_train_epochs=num_train_epochs_classification,
            max_seq_length=max_length,
            save_strategy="no",
            optim="paged_adamw_8bit" if qlora else "adamw_torch",
            learning_rate=2e-4 if qlora else 5e-5,
            fp16=qlora,
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
        trainer.train()  # train modifies the model object itself.
    return (trainer.model, trainer.tokenizer)
