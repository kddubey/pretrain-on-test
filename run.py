"""
Main script to run experiments.

Help::

    python run.py -h
"""
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
    dataset_names: str | None = None
    """
    Comma-separated list of HuggingFace datasets, e.g.,
    "ag_news,dair-ai/emotion,SetFit/enron_spam". By default, all datasets are used.
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
            self.dataset_names = self.dataset_names.split(delim)


model_type_to_config = {
    "bert": lambda: pretrain_on_test.Config(
        model_class_pretrain=BertForMaskedLM,
        model_class_classification=BertForSequenceClassification,
        model_id="bert-base-uncased",
        mlm=True,
        mlm_probability=0.15,
    ),
    "gpt2": lambda: pretrain_on_test.Config(
        model_class_pretrain=GPT2LMHeadModel,
        model_class_classification=GPT2ForSequenceClassification,
        model_id="gpt2",
    ),
}


def run(
    model_type: Literal["bert", "gpt2"],
    dataset_names: str | Collection[str] | None = None,
    num_replications: int = 50,
    num_train: int = 100,
    num_test: int = 200,
):
    if dataset_names is None:
        dataset_names: tuple[str] = get_args(pretrain_on_test.HuggingFaceDatasetNames)
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    config = model_type_to_config[model_type]()
    for dataset_name in dataset_names:
        df = pretrain_on_test.load_data(dataset_name)
        pretrain_on_test.experiment.replicate(
            df,
            dataset_name,
            config,
            num_replications=num_replications,
            num_train=num_train,
            num_test=num_test,
        )


if __name__ == "__main__":
    args = ExperimentArgParser().parse_args()
    run(
        args.model_type,
        dataset_names=args.dataset_names,
        num_replications=args.num_replications,
        num_train=args.num_train,
        num_test=args.num_test,
    )
