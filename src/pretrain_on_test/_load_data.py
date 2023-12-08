from __future__ import annotations
from typing import Any, Callable, Literal

from datasets import load_dataset
import numpy as np
import pandas as pd


HuggingFaceDatasetNames = Literal[
    "ag_news",
    "SetFit/amazon_counterfactual_en",
    "app_reviews",
    "christinacdl/clickbait_notclickbait_dataset",
    "aladar/craigslist_bargains",
    "dair-ai/emotion",
    "SetFit/enron_spam",
    "ethos",
    "financial_phrasebank",
    "mteb/mtop_domain",
    "rotten_tomatoes",
    "trec",
    "yahoo_answers_topics",
    "yelp_review_full",
]
"Default set of classification datasets."


_ProcessDataFrame = Callable[[pd.DataFrame], pd.DataFrame]
"""
Returns a dataframe with canonical "text" and "label" columns:
- "text": the text to classify
- "label": the 0-indexed class which the text belongs to
"""


_dataset_to_processor: dict[str, _ProcessDataFrame] = {
    "app_reviews": lambda df: df.assign(text=df["review"], label=df["star"] - 1),
    "financial_phrasebank": lambda df: df.assign(text=df["sentence"]),
    "trec": lambda df: df.assign(label=df["coarse_label"]),
    "yahoo_answers_topics": lambda df: df.assign(
        text=df["question_title"].str.cat(df["question_content"], sep="\n"),
        label=df["topic"],
    ),
}


_dataset_to_loading_args: dict[str, tuple[Any]] = {
    "ethos": "binary",
    "financial_phrasebank": "sentences_allagree",
}


def load_classification_data_from_hf(
    huggingface_dataset_name: str | HuggingFaceDatasetNames,
) -> pd.DataFrame:
    """
    Returns a canonical classification dataset from the HuggingFace datasets hub:

    https://huggingface.co/datasets/{huggingface_dataset_name}
    """
    loading_args = _dataset_to_loading_args.get(huggingface_dataset_name, ())
    df = pd.DataFrame(
        load_dataset(huggingface_dataset_name, split="train", *loading_args)
    )
    process = _dataset_to_processor.get(huggingface_dataset_name, lambda df: df)
    df = process(df)
    df["text"] = df["text"].fillna("")

    if len(set(df.index)) != len(df):
        raise ValueError("The dataframe has non-unique indices")
    if "text" not in df.columns:
        raise ValueError('The dataframe is missing a "text" column')
    if "label" not in df.columns:
        raise ValueError('The dataframe is missing a "label" column')

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    missing_labels = set(np.arange(df["label"].max())) - set(np.unique(df["label"]))
    if missing_labels:
        raise ValueError(
            f"The dataset is missing the following labels: {sorted(missing_labels)}"
        )

    return df
