"""
Main script to run the experiment.
"""

import json
import os
from datetime import datetime
from functools import partial
from typing import Any, Callable, Collection, Literal

from pydantic import BaseModel, ConfigDict, Field
from tap import tapify
import torch
from transformers import AutoModelForCausalLM, BertForMaskedLM, GPT2LMHeadModel

import pretrain_on_test
import cloud
from cloud import do_nothing


try:
    from IPython.display import clear_output
except ModuleNotFoundError:
    clear_output = do_nothing


_field_for_config = partial(Field, json_schema_extra={"is_for_config": True})


LmType = Literal[
    "bert",
    "gpt2",
    "mistral-lora-sft",
    "mistral-qlora-sft",
    "mistral-instruct-qlora-sft",
    "mistral-instruct-qlora-zero-shot",
    "Phi-3-mini-4k-instruct-qlora-zero-shot",
    # For quick CPU tests
    "bert-tiny",
    "gpt2-tiny",
    "mistral-lora-sft-tiny",
    "mistral-instruct-lora-sft-tiny",
    "Phi-3-mini-128k-instruct-lora-sft-tiny",
    "mistral-instruct-lora-zero-shot-tiny",
    "Phi-3-mini-128k-instruct-lora-zero-shot-tiny",
]


class Experiment(BaseModel):
    """
    Experiment configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    # Pydantic stuff: extra attributes are not allowed, and the object is immutable

    lm_type: LmType = Field(
        description=(
            "Type of language model. *-tiny models have random weights and should only "
            "be used for testing."
        )
    )
    run_name: str = Field(
        default="",
        description=(
            "Name of the run, in case it helps you remember what changed. If supplied, "
            "this name gets appended to the run ID string: run-{timestamp}-{run_name}"
        ),
    )
    dataset_names: list[str] | None = Field(
        default=None,
        description=(
            "Space-separated list of HuggingFace datasets, e.g., "
            "ag_news dair-ai/emotion SetFit/enron_spam. "
            "By default, all datasets from the paper are used"
        ),
    )
    num_subsamples: int = Field(
        default=50, description="Number of subsamples to draw from the dataset"
    )
    num_train: int = Field(
        default=100, description="Number of observations for classification training"
    )
    num_test: int = Field(
        default=200,
        description="Number of observations for pretraining and for evaluation",
    )
    max_length: int | None = Field(
        default=256,
        description=(
            "Number of context tokens for pretraining. Set to None to use the model's "
            "default"
        ),
    )
    # Model-independent arguments which are passed to the config
    per_device_train_batch_size_pretrain: int = _field_for_config(
        default=16, description="Batch size for pretraining"
    )
    per_device_train_batch_size_classification: int = _field_for_config(
        default=16, description="Batch size for classification training"
    )
    per_device_eval_batch_size_classification: int = _field_for_config(
        default=64, description="Batch size for classification evaluation"
    )
    num_train_epochs_classification: int = _field_for_config(
        default=3, description="Number of epochs for classification training"
    )
    num_train_epochs_pretrain: int = _field_for_config(
        default=2, description="Number of epochs for pretraining"
    )


lm_type_to_config_creator: dict[str, Callable[[Any], pretrain_on_test.Config]] = {
    "bert": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="bert-base-uncased",
        model_class_pretrain=BertForMaskedLM,
        mlm=True,
        mlm_probability=0.15,
        pretrain_method="raw-text",
        lora_pretrain=False,
        classification_method="linear-layer",
        lora_classification=False,
        max_length=256,
        **model_independent_kwargs,
    ),
    "gpt2": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="gpt2",
        model_class_pretrain=GPT2LMHeadModel,
        pretrain_method="raw-text",
        lora_pretrain=False,
        classification_method="linear-layer",
        lora_classification=False,
        max_length=256,
        **model_independent_kwargs,
    ),
    "mistral-qlora-sft": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="mistralai/Mistral-7B-v0.3",
        model_class_pretrain=AutoModelForCausalLM,
        pretrain_method="instructions-with-text",
        lora_pretrain=True,
        classification_method="sft",
        lora_classification=True,
        qlora=True,
        max_length=512,
        **model_independent_kwargs,
    ),
    "mistral-instruct-qlora-sft": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_class_pretrain=AutoModelForCausalLM,
        pretrain_method="instructions-with-text",
        lora_pretrain=True,
        classification_method="sft",
        lora_classification=True,
        qlora=True,
        max_length=512,
        **model_independent_kwargs,
    ),
    "mistral-instruct-qlora-zero-shot": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_class_pretrain=AutoModelForCausalLM,
        pretrain_method="instructions-with-text",
        lora_pretrain=True,
        qlora=True,
        classification_method="zero-shot",
        max_length=512,
        **model_independent_kwargs,
    ),
    "Phi-3-mini-4k-instruct-qlora-zero-shot": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        model_class_pretrain=AutoModelForCausalLM,
        pretrain_method="instructions-with-text",
        lora_pretrain=True,
        qlora=True,
        classification_method="zero-shot",
        max_length=512,
        **model_independent_kwargs,
    ),
    # For quick CPU tests. These are useful for prototyping new LM types
    "bert-tiny": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="hf-internal-testing/tiny-random-BertModel",
        model_class_pretrain=BertForMaskedLM,
        mlm=True,
        mlm_probability=0.15,
        pretrain_method="raw-text",
        lora_pretrain=False,
        classification_method="linear-layer",
        lora_classification=False,
        max_length=256,
        **model_independent_kwargs,
    ),
    "gpt2-tiny": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="hf-internal-testing/tiny-random-gpt2",
        model_class_pretrain=GPT2LMHeadModel,
        pretrain_method="raw-text",
        lora_pretrain=False,
        classification_method="linear-layer",
        lora_classification=False,
        max_length=256,
        **model_independent_kwargs,
    ),
    "mistral-lora-sft-tiny": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="hf-internal-testing/tiny-random-AutoModelForCausalLM",
        model_class_pretrain=AutoModelForCausalLM,
        pretrain_method="instructions-with-text",
        lora_pretrain=True,
        classification_method="sft",
        lora_classification=True,
        max_length=512,
        **model_independent_kwargs,
    ),
    "mistral-instruct-lora-sft-tiny": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="ml6team/tiny-random-mistral-instruct",
        model_class_pretrain=AutoModelForCausalLM,
        pretrain_method="instructions-with-text",
        lora_pretrain=True,
        classification_method="sft",
        lora_classification=True,
        max_length=512,
        **model_independent_kwargs,
    ),
    "Phi-3-mini-128k-instruct-lora-sft-tiny": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="yujiepan/phi-3-tiny-random",
        model_class_pretrain=AutoModelForCausalLM,
        pretrain_method="instructions-with-text",
        lora_pretrain=True,
        classification_method="sft",
        lora_classification=True,
        max_length=512,
        **model_independent_kwargs,
    ),
    "mistral-instruct-lora-zero-shot-tiny": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="ml6team/tiny-random-mistral-instruct",
        model_class_pretrain=AutoModelForCausalLM,
        pretrain_method="instructions-with-text",
        lora_pretrain=True,
        classification_method="zero-shot",
        max_length=512,
        **model_independent_kwargs,
    ),
    "Phi-3-mini-128k-instruct-lora-zero-shot-tiny": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="yujiepan/phi-3-tiny-random",
        model_class_pretrain=AutoModelForCausalLM,
        pretrain_method="instructions-with-text",
        lora_pretrain=True,
        classification_method="zero-shot",
        max_length=512,
        **model_independent_kwargs,
    ),
}


def _check_dataset_names(dataset_names: Collection[str] | None) -> list[str]:
    if dataset_names is None:
        dataset_names = list(
            pretrain_on_test.data.hf_dataset_name_to_classification_dataset_info.keys()
        )

    def remove_owner(dataset_name: str) -> str:
        return dataset_name.split("/")[-1]

    dataset_names_without_owners = [
        remove_owner(dataset_name) for dataset_name in dataset_names
    ]
    if len(set(dataset_names_without_owners)) < len(dataset_names_without_owners):
        raise ValueError(
            "Some datasets have the same name. (They may have different owners. "
            "But that's still not allowed.)"
        )
    return sorted(dataset_names, key=remove_owner)


def run(
    experiment: Experiment,
    create_logger: cloud.CreateLogger = cloud.create_logger_local,
    upload_directory: cloud.UploadDirectory = do_nothing,
) -> str:
    """
    Run the experiment.

    Parameters
    ----------
    experiment : Experiment
        configuration for the experiment
    create_logger : cloud.CreateLogger, optional
        Callable which takes as input a single argument for the name of the log
        group/label/tag, and outputs a `logging.Logger` object. By default, a logger is
        created which only logs to stdout.
    upload_directory : cloud.UploadDirectory, optional
        Callable which takes as input `directory` and `logger` arguments and uploads all
        local content in `directory` somewhere else, e.g., S3. By default, nothing is
        uploaded.

    Returns
    -------
    str
        run ID
    """
    # Meta info
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = (
        f"run-{current_time}{'-' + experiment.run_name if experiment.run_name else ''}"
    )

    # Create logger
    logger = create_logger(run_id)
    logger.info(f"ID of the run: {run_id}")
    logger.info(experiment)

    try:
        if torch.cuda.is_available():
            logger.info("GPU detected.")
        else:
            logger.info("No GPU detected.")
            if "cpu-test" not in experiment.run_name:
                raise ValueError(
                    "No GPU was detected. If this is intentional, please include "
                    "'cpu-test' somewhere in the run_name argument."
                )

        # Create results_dir using core settings from the experiment: n and the LM
        results_dir = os.path.join(
            run_id,
            "accuracies",
            f"m{experiment.num_train}",
            f"n{experiment.num_test}",
            experiment.lm_type,
        )

        # Upload experiment settings
        if not os.path.exists(run_id):
            os.makedirs(run_id)
        with open(os.path.join(run_id, "experiment.json"), "w") as json_file:
            experiment_as_dict = experiment.model_dump()
            json.dump(experiment_as_dict, json_file, indent=4)
        upload_directory(directory=run_id, logger=logger)

        # Create config from experiment
        model_independent_attributes = [
            field_name
            for field_name, field_info in Experiment.model_fields.items()
            if (getattr(field_info, "json_schema_extra") or {}).get(
                "is_for_config", False
            )
        ]
        model_independent_kwargs = {
            attr: getattr(experiment, attr) for attr in model_independent_attributes
        }
        config = lm_type_to_config_creator[experiment.lm_type](
            **model_independent_kwargs
        )

        # Check that the dataset names don't conflict w/ each other
        dataset_names = _check_dataset_names(experiment.dataset_names)

        # Run experiment on each dataset
        _ = torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        for dataset_name in dataset_names:
            classification_dataset_info = (
                pretrain_on_test.data.hf_dataset_name_to_classification_dataset_info[
                    dataset_name
                ]
            )
            df = pretrain_on_test.data.load_classification_data(
                classification_dataset_info
            )
            clear_output(wait=True)
            dataset_dir = pretrain_on_test.experiment.replicate(
                df,
                classification_dataset_info,
                dataset_name,
                results_dir,
                config,
                logger,
                num_subsamples=experiment.num_subsamples,
                num_train=experiment.num_train,
                num_test=experiment.num_test,
            )
            # Sync w/ cloud
            upload_directory(directory=dataset_dir, logger=logger)
    except Exception as exception:
        try:
            msg = f"Encountered an error with dataset {dataset_name}: "
        except UnboundLocalError:
            msg = ""
        logger.error(f"{msg}{exception}", exc_info=True)
        raise

    return run_id


if __name__ == "__main__":
    experiment = tapify(Experiment)
    cloud_provider = os.environ.get("PRETRAIN_ON_TEST_CLOUD_PROVIDER")
    # Env var b/c it's reasonable to run this script many times in one session. So just
    # need to specify the env var once
    create_data_handlers = cloud.cloud_provider_to_create_data_handlers[cloud_provider]
    data_handlers = create_data_handlers()
    run(
        experiment,
        create_logger=data_handlers.create_logger,
        upload_directory=data_handlers.upload_directory,
    )
