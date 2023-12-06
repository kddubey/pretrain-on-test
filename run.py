"""
Main script to run experiments.

Help::

    python run.py -h
"""
import os
from typing import get_args, Literal, Collection

from tap import Tap
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
)

import pretrain_on_test


class ExperimentArgParser(Tap):
    model_type: Literal["bert", "gpt2"]
    results_dir: str = "accuracies"
    "Directory to store experiment results"
    dataset_names: str | None = None
    """
    Comma-separated list of HuggingFace datasets, e.g.,
    "ag_news,dair-ai/emotion,SetFit/enron_spam". By default, all datasets are used
    """
    num_replications: int = 50
    "Number of subsamples to draw from the dataset"
    num_train: int = 100
    "Number of classification training observations"
    num_test: int = 200
    "Number of observations for pretraining and classification evaluation"

    def process_args(self):
        if isinstance(self.dataset_names, str):
            delim = ", " if " " in self.dataset_names else ","
            self.dataset_names = sorted(set(self.dataset_names.split(delim)))
            dataset_names_without_owners = [
                dataset_name.split("/")[-1] for dataset_name in self.dataset_names
            ]
            if len(set(dataset_names_without_owners)) < len(self.dataset_names):
                raise ValueError(
                    "Some datasets have different owners but the same name. "
                    "This currently isn't allowed."
                )


model_type_to_config = {
    "bert": lambda: pretrain_on_test.Config(
        model_id="bert-base-uncased",
        model_class_pretrain=BertForMaskedLM,
        model_class_classification=BertForSequenceClassification,
        mlm=True,
        mlm_probability=0.15,
    ),
    "gpt2": lambda: pretrain_on_test.Config(
        model_id="gpt2",
        model_class_pretrain=GPT2LMHeadModel,
        model_class_classification=GPT2ForSequenceClassification,
    ),
}


def run(
    model_type: Literal["bert", "gpt2"],
    results_dir: str = "accuracies",
    dataset_names: str | Collection[str] | None = None,
    num_replications: int = 50,
    num_train: int = 100,
    num_test: int = 200,
):
    if dataset_names is None:
        dataset_names: tuple[str] = get_args(pretrain_on_test.HuggingFaceDatasetNames)
    if isinstance(dataset_names, str):
        dataset_names = (dataset_names,)
    config = model_type_to_config[model_type]()
    for dataset_name in dataset_names:
        df = pretrain_on_test.load_data(dataset_name)
        dataset_name_no_owner = dataset_name.split("/")[-1]
        file_path = os.path.join(results_dir, f"{dataset_name_no_owner}.csv")
        pretrain_on_test.experiment.replicate(
            df,
            file_path,
            config,
            num_replications=num_replications,
            num_train=num_train,
            num_test=num_test,
        )


if __name__ == "__main__":
    args = ExperimentArgParser().parse_args()
    run(
        args.model_type,
        results_dir=args.results_dir,
        dataset_names=args.dataset_names,
        num_replications=args.num_replications,
        num_train=args.num_train,
        num_test=args.num_test,
    )
