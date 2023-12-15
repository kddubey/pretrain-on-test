"""
Main script to run the experiment
"""
from typing import Collection, get_args, Literal, Type

from pydantic import BaseModel, ConfigDict, Field
from tap import Tap
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


class Experiment(BaseModel):
    """
    Experiment configuration.
    """

    model_config = ConfigDict(extra="forbid")
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
    # TODO: figure out how to re-use the defaults. Maybe put this class in pretrain
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
    per_device_train_batch_size_pretrain: int = Field(
        default=16, description="Batch size for pretraining"
    )
    per_device_train_batch_size_classification: int = Field(
        default=16, description="Batch size for classification training"
    )
    per_device_eval_batch_size_classification: int = Field(
        default=64, description="Batch size for classification evaluation"
    )
    max_length: int | None = Field(
        default=256,
        description=(
            "Number of context tokens for pretraining. Set to None to use the model's "
            "default"
        ),
    )
    num_train_epochs_classification: int = Field(
        default=3, description="Number of epochs for classification training"
    )
    num_train_epochs_pretrain: int = Field(
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


def run(experiment: Experiment):
    """
    Main function to run the experiment.
    """
    model_independent_attributes = [
        "per_device_train_batch_size_pretrain",
        "per_device_train_batch_size_classification",
        "per_device_eval_batch_size_classification",
        "max_length",
        "num_train_epochs_classification",
        "num_train_epochs_pretrain",
    ]
    config = _lm_type_to_config_creator[experiment.lm_type](
        **{attr: getattr(experiment, attr) for attr in model_independent_attributes}
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


def _tap_from_pydantic_model(model: Type[BaseModel]) -> Type[Tap]:
    class ArgParser(Tap):
        def configure(self):
            for name, field in model.model_fields.items():
                self._annotations[name] = field.annotation
                self.class_variables[name] = {"comment": field.description or ""}
                if field.is_required():
                    kwargs = {}
                else:
                    kwargs = dict(required=False, default=field.default)
                self.add_argument(f"--{name}", **kwargs)

    return ArgParser


if __name__ == "__main__":
    _ExperimentArgParser = _tap_from_pydantic_model(Experiment)
    args = _ExperimentArgParser(description=__doc__).parse_args()
    experiment = Experiment(**args.as_dict())
    run(experiment)
