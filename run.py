"""
Main script to run the experiment.
"""

import os
import dataclasses
from datetime import datetime
from functools import partial
from typing import Collection, get_args, Literal

import pydantic
from pydantic import Field
from tap import tapify
import torch
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
)

import pretrain_on_test
import cloud

do_nothing = lambda *args, **kwargs: None

try:
    from IPython.display import clear_output
except ModuleNotFoundError:
    clear_output = do_nothing


_field_for_config = partial(Field, json_schema_extra={"is_for_config": True})


@pydantic.dataclasses.dataclass(frozen=True, config=dict(extra="forbid"))
class Experiment:
    """
    Experiment configuration.
    """

    lm_type: Literal["bert", "gpt2"] = Field(description="Type of language model")
    run_name: str = Field(
        default="",
        description="Name of the run, in case it helps you remember what changed",
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
    max_length: int | None = _field_for_config(
        default=256,
        description=(
            "Number of context tokens for pretraining. Set to None to use the model's "
            "default"
        ),
    )
    num_train_epochs_classification: int = _field_for_config(
        default=3, description="Number of epochs for classification training"
    )
    num_train_epochs_pretrain: int = _field_for_config(
        default=2, description="Number of epochs for pretraining"
    )


lm_type_to_config_creator = {
    "bert": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="bert-base-uncased",
        model_class_pretrain=BertForMaskedLM,
        model_class_classification=BertForSequenceClassification,
        mlm=True,
        mlm_probability=0.15,
        **model_independent_kwargs,
    ),
    "gpt2": lambda **model_independent_kwargs: pretrain_on_test.Config(
        model_id="gpt2",
        model_class_pretrain=GPT2LMHeadModel,
        model_class_classification=GPT2ForSequenceClassification,
        **model_independent_kwargs,
    ),
}


def _check_dataset_names(dataset_names: Collection[str] | None) -> list[str]:
    if dataset_names is None:
        dataset_names = list(get_args(pretrain_on_test.HuggingFaceDatasetNames))

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
):
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
        # Create config from experiment
        model_independent_attributes = [
            field.name
            for field in dataclasses.fields(Experiment)
            if (getattr(field.default, "json_schema_extra") or {}).get(
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

        # Create results_dir using core settings from the experiment: n and the LM
        results_dir = os.path.join(
            run_id, "accuracies", f"num_test_{experiment.num_test}", experiment.lm_type
        )

        # Run experiment on each dataset
        _ = torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        for dataset_name in dataset_names:
            df = pretrain_on_test.load_classification_data_from_hf(dataset_name)
            clear_output(wait=True)
            dataset_dir = pretrain_on_test.experiment.replicate(
                df,
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
        logger.error(exception, exc_info=True)
        raise


cloud_provider_to_create_data_handlers = {
    None: lambda: dict(
        create_logger=cloud.create_logger_local,
        upload_directory=do_nothing,
    ),
    # They're lambdas so that evaluation is delayed; cloud-specific modules aren't
    # imported and cloud-specific env vars aren't checked until needed
    "gcp": lambda: dict(
        create_logger=cloud.create_logger_gcp,
        upload_directory=cloud.UploadGCP(
            bucket_name=os.environ["PRETRAIN_ON_TEST_BUCKET_NAME"]
        ).upload_directory,
    ),
}


if __name__ == "__main__":
    experiment = tapify(Experiment)
    cloud_provider = os.environ.get("PRETRAIN_ON_TEST_CLOUD_PROVIDER")
    # Env var b/c it's reasonable to run this script many times in one session. So just
    # need to specify the env var once
    create_data_handlers = cloud_provider_to_create_data_handlers[cloud_provider]
    data_handlers = create_data_handlers()
    run(experiment, **data_handlers)
