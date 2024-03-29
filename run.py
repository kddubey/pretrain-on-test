"""
Main script to run the experiment
"""

import dataclasses
from functools import partial
from typing import Collection, get_args, Literal

import pydantic
from pydantic import Field
import torch
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
)

try:
    from IPython.display import clear_output
except ModuleNotFoundError:
    clear_output = lambda *args, **kwargs: None

import pretrain_on_test
from _to_tap import tap_class_from_data_model


_field_for_config = partial(Field, json_schema_extra={"is_for_config": True})


@pydantic.dataclasses.dataclass(frozen=True, config=dict(extra="forbid"))
class Experiment:
    """
    Experiment configuration.
    """

    lm_type: Literal["bert", "gpt2"] = Field(description="Type of language model")
    results_dir: str = Field(
        default="accuracies", description="Directory to store experiment results"
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


_lm_type_to_config_creator = {
    "bert": lambda **model_indepedent_kwargs: pretrain_on_test.Config(
        model_id="bert-base-uncased",
        model_class_pretrain=BertForMaskedLM,
        model_class_classification=BertForSequenceClassification,
        mlm=True,
        mlm_probability=0.15,
        **model_indepedent_kwargs,
    ),
    "gpt2": lambda **model_indepedent_kwargs: pretrain_on_test.Config(
        model_id="gpt2",
        model_class_pretrain=GPT2LMHeadModel,
        model_class_classification=GPT2ForSequenceClassification,
        **model_indepedent_kwargs,
    ),
}


def _check_dataset_names(dataset_names: Collection[str] | None) -> list[str]:
    if dataset_names is None:
        dataset_names = list(get_args(pretrain_on_test.HuggingFaceDatasetNames))

    def remove_owner(dataset_name: str) -> str:
        return dataset_name.split("/")[-1]

    dataset_names = sorted(set(dataset_names), key=remove_owner)
    dataset_names_without_owners = [
        remove_owner(dataset_name) for dataset_name in dataset_names
    ]
    if len(set(dataset_names_without_owners)) < len(dataset_names):
        raise ValueError(
            "Some datasets have the same name. (They may have different owners. "
            "But that's still not allowed.)"
        )
    return dataset_names


# bleh
_get_json_schema_extra = (
    lambda field_info: getattr(field_info, "json_schema_extra") or {}
)
_model_independent_attributes = [
    field.name
    for field in dataclasses.fields(Experiment)
    if hasattr(field.default, "json_schema_extra")
    and _get_json_schema_extra(field.default).get("is_for_config", False)
]


def run(experiment: Experiment):
    """
    Main function to run the experiment.
    """
    _ = torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    config = _lm_type_to_config_creator[experiment.lm_type](
        **{attr: getattr(experiment, attr) for attr in _model_independent_attributes}
    )
    dataset_names = _check_dataset_names(experiment.dataset_names)
    for dataset_name in dataset_names:
        df = pretrain_on_test.load_classification_data_from_hf(dataset_name)
        clear_output(wait=True)
        pretrain_on_test.experiment.replicate(
            df,
            dataset_name,
            experiment.results_dir,
            config,
            num_subsamples=experiment.num_subsamples,
            num_train=experiment.num_train,
            num_test=experiment.num_test,
        )


if __name__ == "__main__":
    ExperimentArgParser = tap_class_from_data_model(Experiment)
    args = ExperimentArgParser(description=__doc__).parse_args()
    experiment = Experiment(**args.as_dict())
    run(experiment)
