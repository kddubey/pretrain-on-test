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
    dataset_names: list[str] | None = None
    """
    Space-separated list of HuggingFace datasets, e.g.,
    "ag_news dair-ai/emotion SetFit/enron_spam". By default, all datasets are used
    """
    num_replications: int = 50
    "Number of subsamples to draw from the dataset"
    num_train: int = 100
    "Number of classification training observations"
    num_test: int = 200
    "Number of observations for pretraining and classification evaluation"


# The values are lambdas so that evaluation (downloading the tokenizer) is done only
# when requested
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
    dataset_names: Collection[str] | None = None,
    num_replications: int = 50,
    num_train: int = 100,
    num_test: int = 200,
):
    config = model_type_to_config[model_type]()
    for dataset_name in dataset_names:
        df = pretrain_on_test.load_classification_data_from_hf(dataset_name)
        pretrain_on_test.experiment.replicate(
            df,
            dataset_name,
            results_dir,
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
