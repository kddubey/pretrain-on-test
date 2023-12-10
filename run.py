"""
Main script to run the experiment

Help::

    python run.py -h
"""
from typing import Collection, get_args, Literal

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


# The values are lambdas so that evaluation (downloading the tokenizer) is done only
# when requested
model_type_to_config = {
    "bert": lambda **kwargs: pretrain_on_test.Config(
        model_id="bert-base-uncased",
        model_class_pretrain=BertForMaskedLM,
        model_class_classification=BertForSequenceClassification,
        mlm=True,
        mlm_probability=0.15,
        **kwargs,
    ),
    "gpt2": lambda **kwargs: pretrain_on_test.Config(
        model_id="gpt2",
        model_class_pretrain=GPT2LMHeadModel,
        model_class_classification=GPT2ForSequenceClassification,
        **kwargs,
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


def run(
    model_type: Literal["bert", "gpt2"],
    dataset_names: Collection[str] | None,
    results_dir: str = "accuracies",
    num_subsamples: int = 50,
    num_train: int = 100,
    num_test: int = 200,
    per_device_train_batch_size_pretrain: int = 16,
    per_device_train_batch_size_classification: int = 16,
    per_device_eval_batch_size_classification: int = 64,
    max_length: int | None = 256,
    num_train_epochs_classification: int = 3,
):
    """
    Main function to run the experiment.
    """
    config = model_type_to_config[model_type](
        per_device_train_batch_size_pretrain=per_device_train_batch_size_pretrain,
        per_device_train_batch_size_classification=per_device_train_batch_size_classification,
        per_device_eval_batch_size_classification=per_device_eval_batch_size_classification,
        max_length=max_length,
        num_train_epochs_classification=num_train_epochs_classification,
    )
    dataset_names = _check_dataset_names(dataset_names)
    for dataset_name in dataset_names:
        df = pretrain_on_test.load_classification_data_from_hf(dataset_name)
        clear_output(wait=True)
        pretrain_on_test.experiment.replicate(
            df,
            dataset_name,
            results_dir,
            config,
            num_subsamples=num_subsamples,
            num_train=num_train,
            num_test=num_test,
        )


class ExperimentArgParser(Tap):
    # TODO: figure out how to share this between here and the run signature
    model_type: Literal["bert", "gpt2"]

    results_dir: str = "accuracies"
    "Directory to store experiment results"

    dataset_names: list[str] | None = None
    """
    Space-separated list of HuggingFace datasets, e.g.,
    ag_news dair-ai/emotion SetFit/enron_spam. By default, 14 datasets are used
    """

    num_subsamples: int = 50
    "Number of subsamples to draw from the dataset"

    num_train: int = 100
    "Number of classification training observations"

    num_test: int = 200
    "Number of observations for pretraining and for evaluation"

    per_device_train_batch_size_pretrain: int = 16
    "Batch size for pretraining"

    per_device_train_batch_size_classification: int = 16
    "Batch size for classification training"

    per_device_eval_batch_size_classification: int = 64
    "Batch size for classification evaluation"

    max_length: int | None = 256
    "Number of context tokens for pretraining. Set to None to use the model's default"

    num_train_epochs_classification: int = 3
    "Number of epochs for classification training"


if __name__ == "__main__":
    args = ExperimentArgParser().parse_args()
    run(
        args.model_type,
        dataset_names=args.dataset_names,
        results_dir=args.results_dir,
        num_subsamples=args.num_subsamples,
        num_train=args.num_train,
        num_test=args.num_test,
    )
