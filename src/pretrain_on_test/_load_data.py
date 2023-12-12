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
    "emo",
    "dair-ai/emotion",
    "SetFit/enron_spam",
    "ethos",
    "financial_phrasebank",
    "hyperpartisan_news_detection",
    "AmazonScience/massive",
    "movie_rationales",
    "mteb/mtop_domain",
    "poem_sentiment",
    "rotten_tomatoes",
    "silicone",
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


# Looks like this is also referred to as the "subset"
_dataset_to_config_name: dict[str, str] = {
    "ethos": "binary",
    "financial_phrasebank": "sentences_allagree",
    "hyperpartisan_news_detection": "bypublisher",
    "AmazonScience/massive": "en-US",
    "silicone": "dyda_da",
}


_dataset_to_processor: dict[str, _ProcessDataFrame] = {
    "app_reviews": lambda df: df.assign(text=df["review"], label=df["star"] - 1),
    "financial_phrasebank": lambda df: df.assign(text=df["sentence"]),
    "hyperpartisan_news_detection": lambda df: df.assign(
        text=df["title"], label=df["hyperpartisan"]
    ),
    "AmazonScience/massive": lambda df: df.assign(text=df["utt"], label=df["scenario"]),
    "movie_rationales": lambda df: df.assign(text=df["review"]),
    "poem_sentiment": lambda df: df.assign(text=df["verse_text"]),
    "silicone": lambda df: df.assign(text=df["Utterance"], label=df["Label"]),
    "trec": lambda df: df.assign(label=df["coarse_label"]),
    "yahoo_answers_topics": lambda df: df.assign(
        text=df["question_title"].str.cat(df["question_content"], sep="\n"),
        label=df["topic"],
    ),
}


def load_classification_data_from_hf(
    huggingface_dataset_name: str | HuggingFaceDatasetNames,
) -> pd.DataFrame:
    """
    Returns a canonical classification dataset from the HuggingFace datasets hub:

    https://huggingface.co/datasets/{huggingface_dataset_name}
    """
    config_name = _dataset_to_config_name.get(huggingface_dataset_name)
    df = pd.DataFrame(
        load_dataset(huggingface_dataset_name, config_name, split="train")
    )
    process = _dataset_to_processor.get(huggingface_dataset_name, lambda df: df)
    df = process(df)
    df["text"] = df["text"].fillna("")

    if len(set(df.index)) != len(df):
        raise ValueError("The dataframe has non-unique indices")
    required_format_message = (
        'It must have a "text" (str) and "label" (0-indexed integers) column'
    )
    if "text" not in df.columns:
        raise ValueError(
            'The dataframe is missing a "text" column. ' + required_format_message
        )
    if "label" not in df.columns:
        raise ValueError(
            'The dataframe is missing a "label" column. ' + required_format_message
        )

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    missing_labels = set(np.arange(df["label"].max())) - set(np.unique(df["label"]))
    if missing_labels:
        raise ValueError(
            f"The dataset is missing the following labels: {sorted(missing_labels)}"
        )

    return df
