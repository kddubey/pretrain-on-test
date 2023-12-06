from __future__ import annotations
from typing import Literal

from datasets import load_dataset
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


def load_data(huggingface_dataset_name: str | HuggingFaceDatasetNames) -> pd.DataFrame:
    df = pd.DataFrame(load_dataset(huggingface_dataset_name, split="train"))

    if huggingface_dataset_name == "yahoo_answers_topics":
        df["text"] = df["question_title"].str.cat(df["question_content"], sep="\n")
        df["label"] = df["topic"]
    elif huggingface_dataset_name == "trec":
        df["label"] = df["coarse_label"]
    elif huggingface_dataset_name == "financial_phrasebank":
        df["text"] = df["sentence"]
    elif huggingface_dataset_name == "app_reviews":
        df["text"] = df["review"]
        df["label"] = df["star"] - 1

    df["text"] = df["text"].fillna("")

    assert len(set(df.index)) == len(df)
    assert "text" in df.columns
    assert "label" in df.columns

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df
