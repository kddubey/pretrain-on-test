"""
Load classification data. Also records task-specific info for each of the 25 HF datasets
in the paper.
"""

from functools import partial
from typing import Annotated, Callable

from datasets import load_dataset
from pydantic import Field, BaseModel, ConfigDict
from pydantic.functional_validators import AfterValidator
import numpy as np
import pandas as pd


NUM_CHARACTERS_MAX = 1_000
"""
Sometimes need to filter out ginormous texts, so only keep texts w/ < 1_000 characters.
"""


def _loader_hf(
    hf_dataset_name: str,
    config_name: str | None = None,
    trust_remote_code: bool | None = False,
    **kwargs,
) -> pd.DataFrame:
    load_dataset_kwargs = dict(
        path=hf_dataset_name,
        name=config_name,
        split="train",
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    try:
        dataset = load_dataset(**load_dataset_kwargs)
    except ValueError as exception:
        attempted_split = load_dataset_kwargs["split"]
        if not str(exception).startswith(f'Unknown split "{attempted_split}"'):
            raise exception
        load_dataset_kwargs["split"] = "test"
        dataset = load_dataset(**load_dataset_kwargs)
    return pd.DataFrame(dataset)


def _check_nonempty(obj):
    assert obj, "Cannot be empty."
    return obj


_NonEmptyString = Annotated[str, AfterValidator(_check_nonempty)]


class ClassificationDatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_location: _NonEmptyString = Field(
        description=(
            "Where the dataset is and what it's called. This string gets passed "
            "to `loader`."
        )
    )
    class_names: tuple[_NonEmptyString, ...] = Field(
        description=(
            "`class_names[label_idx]` is the task-specific name corresponding to the "
            "integer-label `label_idx`."
        )
    )
    task_description: str = Field(
        description=(
            "Instructions for performing the task. These instructions are intended "
            "for prompting an LLM. Use language like "
            "'The text is {blank}. Answer with {blank}'."
        )
    )
    loader: Callable[[str], pd.DataFrame] = Field(
        default=partial(_loader_hf, config_name=None, trust_remote_code=False),
        description="Returns a dataframe after inputting `dataset_location`.",
    )
    processor: Callable[[pd.DataFrame], pd.DataFrame] = Field(
        default=lambda df: df,
        description=(
            'Returns a dataframe with canonical "text" and "label" columns: '
            '"text": the text to classify, '
            '"label": the 0-indexed class which the text belongs to.'
        ),
    )


hf_dataset_name_to_classification_dataset_info: dict[str, ClassificationDatasetInfo] = {
    "ag_news": ClassificationDatasetInfo(
        dataset_location="ag_news",
        class_names=("World", "Sports", "Business", "Sci/Tech"),
        task_description="The text is a news article. Answer with its topic.",
    ),
    "SetFit/amazon_counterfactual_en": ClassificationDatasetInfo(
        dataset_location="SetFit/amazon_counterfactual_en",
        class_names=("not counterfactual", "counterfactual"),
        task_description=(
            "The text is an Amazon product review. Answer with whether or not the text "
            "contains a counterfactual statement, which describes an event that did "
            "not or cannot take place."
        ),
    ),
    "app_reviews": ClassificationDatasetInfo(
        dataset_location="app_reviews",
        class_names=("1", "2", "3", "4", "5"),
        task_description=(
            "The text is an app review. Answer with a rating from 1-5, where 1 "
            "indicates that the review says the app is bad, and 5 stars indicates "
            "that the review says the app is great."
        ),
        processor=lambda df: df.assign(text=df["review"], label=df["star"] - 1),
    ),
    "blog_authorship_corpus": ClassificationDatasetInfo(
        dataset_location="blog_authorship_corpus",
        class_names=("female", "male"),
        task_description=(
            "The text is a blog post. Answer with your best guess for the gender of "
            "the post's author."
        ),
        loader=partial(_loader_hf, trust_remote_code=True),
        processor=lambda df: df.assign(
            label=df["gender"].map({"female": 0, "male": 1})
        ).copy()[df["text"].str.len() < NUM_CHARACTERS_MAX],
    ),
    "christinacdl/clickbait_notclickbait_dataset": ClassificationDatasetInfo(
        dataset_location="christinacdl/clickbait_notclickbait_dataset",
        class_names=("not clickbait", "clickbait"),
        task_description=(
            "The text is the headline of an article. Answer with whether or not the "
            "headline is clickbait."
        ),
    ),
    "climate_fever": ClassificationDatasetInfo(
        dataset_location="climate_fever",
        class_names=("supported", "refuted", "not enough info", "disputed"),
        task_description=(  # NOTE: this task is a modification of the original one
            "The text is a claim about climate change. Answer with how scientifically "
            "evidenced the claim is."
        ),
        processor=lambda df: df.assign(text=df["claim"], label=df["claim_label"]),
    ),
    "aladar/craigslist_bargains": ClassificationDatasetInfo(
        dataset_location="aladar/craigslist_bargains",
        class_names=(
            "bike",
            "car",
            "electronics",
            "furniture",
            "housing",
            "phone",
        ),
        task_description=(
            "The text is a dialogue between a buyer and seller on Craigslist. Answer "
            "with the category of the product they are discussing."
        ),
        processor=lambda df: df.copy()[df["text"].notna()],
    ),
    "disaster_response_messages": ClassificationDatasetInfo(
        dataset_location="disaster_response_messages",
        class_names=(
            "direct message",
            "news story",
            "social media post",
        ),
        task_description=(
            "The text is a disaster response message. Answer with the most likely "
            "source of the message"
        ),
        loader=partial(_loader_hf, trust_remote_code=True),
        processor=lambda df: df.assign(
            text=df["message"],
            label=df["genre"].map({"direct": 0, "news": 1, "social": 2}),
        ).copy()[
            (df["message"].str.len() < NUM_CHARACTERS_MAX)
            & (df["message"].fillna("").str.len() > 0)
        ],
    ),
    "aladar/emo": ClassificationDatasetInfo(
        dataset_location="aladar/emo",
        class_names=("others", "happy", "sad", "angry"),
        task_description=(
            "The text is an utterance from a dialogue. Answer with the underlying "
            "emotion of the utterance."
        ),
    ),
    "dair-ai/emotion": ClassificationDatasetInfo(
        dataset_location="dair-ai/emotion",
        class_names=("sadness", "joy", "love", "anger", "fear", "surprise"),
        task_description=(
            "The text is a Twitter message. Answer with its underlying emotion."
        ),
        loader=partial(_loader_hf, trust_remote_code=True),
    ),
    "SetFit/enron_spam": ClassificationDatasetInfo(
        dataset_location="SetFit/enron_spam",
        class_names=("ham", "spam"),
        task_description=(
            "The text is the subject and message of an email. Answer with whether or "
            "not the email is spam or ham (i.e., not spam)."
        ),
        processor=lambda df: df.copy()[df["text"].fillna("").str.len() > 0],
    ),
    "financial_phrasebank": ClassificationDatasetInfo(
        dataset_location="financial_phrasebank",
        class_names=("negative", "neutral", "positive"),
        task_description=(
            "The text is a sentence from financial news. Answer with the sentiment of "
            "this sentence."
        ),
        loader=partial(
            _loader_hf, config_name="sentences_allagree", trust_remote_code=True
        ),
        processor=lambda df: df.assign(text=df["sentence"]),
    ),
    "classla/FRENK-hate-en": ClassificationDatasetInfo(
        dataset_location="classla/FRENK-hate-en",
        class_names=("LGBT", "migrants"),
        task_description=(
            "The text is a Facebook comment on a news article. Answer with the topic "
            "of the news article."
        ),
        loader=partial(_loader_hf, trust_remote_code=True),
        processor=lambda df: df.assign(
            label=df["topic"].map({"lgbt": 0, "migrants": 1})
        ).copy()[df["text"].str.len() < NUM_CHARACTERS_MAX],
    ),
    "hyperpartisan_news_detection": ClassificationDatasetInfo(
        dataset_location="hyperpartisan_news_detection",
        class_names=("not hyperpartisan", "hyperpartisan"),
        task_description=(
            "The text is the headline of a news article. Answer with whether or not it "
            "is hyperpartisan."
        ),
        loader=partial(_loader_hf, config_name="bypublisher", trust_remote_code=True),
        processor=lambda df: df.copy()[df["title"].fillna("").str.len() > 0].assign(
            text=df["title"], label=df["hyperpartisan"]
        ),
    ),
    "limit": ClassificationDatasetInfo(
        dataset_location="limit",
        class_names=("no motion", "motion"),
        task_description=(
            "Answer with whether the text describes the movement of a physical entity "
            "(motion) or not (no motion)."
        ),
        loader=partial(_loader_hf, trust_remote_code=True),
        processor=lambda df: df.assign(
            text=df["sentence"], label=df["motion"].map({"no": 0, "yes": 1})
        ),
    ),
    "AmazonScience/massive": ClassificationDatasetInfo(
        dataset_location="AmazonScience/massive",
        class_names=(
            "social",
            "transport",
            "calendar",
            "play",
            "news",
            "datetime",
            "recommendation",
            "email",
            "iot",
            "general",
            "audio",
            "lists",
            "qa",
            "cooking",
            "takeaway",
            "music",
            "alarm",
            "weather",
        ),
        task_description=(
            "The text is an utterance from a person to a voice assistant. Answer with "
            "the domain of the utterance."
        ),
        loader=partial(_loader_hf, config_name="en-US", trust_remote_code=True),
        processor=lambda df: df.assign(text=df["utt"], label=df["scenario"]),
    ),
    "movie_rationales": ClassificationDatasetInfo(
        dataset_location="movie_rationales",
        class_names=("negative", "positive"),
        task_description="The text is a movie review. Answer with its sentiment.",
        loader=partial(_loader_hf, trust_remote_code=True),
        processor=lambda df: df.assign(text=df["review"]),
    ),
    "mteb/mtop_domain": ClassificationDatasetInfo(
        dataset_location="mteb/mtop_domain",
        class_names=(
            "messaging",
            "calling",
            "event",
            "timer",
            "music",
            "weather",
            "alarm",
            "people",
            "reminder",
            "recipes",
            "news",
        ),
        task_description=(
            "The text is an utterance from a person to a voice assistant. Answer with "
            "the domain of the utterance."
        ),
        loader=partial(_loader_hf, config_name="en", trust_remote_code=True),
    ),
    "ccdv/patent-classification": ClassificationDatasetInfo(
        dataset_location="ccdv/patent-classification",
        class_names=(
            "Human Necessities",
            "Performing Operations; Transporting",
            "Chemistry; Metallurgy",
            "Textiles; Paper",
            "Fixed Constructions",
            "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
            "Physics",
            "Electricity",
            "General tagging of new or cross-sectional technology",
        ),
        task_description=(
            "The text is the title of a patent and its abstract. Answer with its "
            "domain."
        ),
        loader=partial(_loader_hf, config_name="abstract", trust_remote_code=True),
        processor=lambda df: df.copy()[df["text"].str.len() < NUM_CHARACTERS_MAX],
    ),
    "rotten_tomatoes": ClassificationDatasetInfo(
        dataset_location="rotten_tomatoes",
        class_names=("negative", "positive"),
        task_description="The text is a movie review. Answer with its sentiment.",
    ),
    "silicone": ClassificationDatasetInfo(
        dataset_location="silicone",
        class_names=("commissive", "directive", "inform", "question"),
        task_description="The text is an utterance. Answer with its broader intention.",
        loader=partial(_loader_hf, config_name="dyda_da", trust_remote_code=True),
        processor=lambda df: df.assign(text=df["Utterance"], label=df["Label"]),
    ),
    "trec": ClassificationDatasetInfo(
        dataset_location="trec",
        class_names=(
            "Abbreviation",
            "Entity",
            "Description and abstract concept",
            "Human being",
            "Location",
            "Numeric value",
        ),
        task_description=(
            "The text is a question. Answer with the broader category it belongs to."
        ),
        loader=partial(_loader_hf, trust_remote_code=True),
        processor=lambda df: df.assign(label=df["coarse_label"]),
    ),
    "tweets_hate_speech_detection": ClassificationDatasetInfo(
        dataset_location="tweets_hate_speech_detection",
        class_names=("not hate speech", "hate speech"),
        task_description=(
            "The text is a tweet. Answer with whether or not it may be associated with "
            "hate speech, i.e., it has a racist or sexist sentiment associated with it."
        ),
        loader=partial(_loader_hf, trust_remote_code=True),
        processor=lambda df: df.assign(text=df["tweet"]),
    ),
    "yahoo_answers_topics": ClassificationDatasetInfo(
        dataset_location="yahoo_answers_topics",
        class_names=(
            "Society & Culture",
            "Science & Mathematics",
            "Health",
            "Education & Reference",
            "Computers & Internet",
            "Sports",
            "Business & Finance",
            "Entertainment & Music",
            "Family & Relationships",
            "Politics & Government",
        ),
        task_description=(
            "The text is a question title and body from Yahoo Answers. Answer with its "
            "topic."
        ),
        loader=partial(_loader_hf, trust_remote_code=True),
        processor=lambda df: df.assign(
            text=df["question_title"].str.cat(df["question_content"], sep="\n"),
            label=df["topic"],
        ),
    ),
    "yelp_review_full": ClassificationDatasetInfo(
        dataset_location="yelp_review_full",
        class_names=("1", "2", "3", "4", "5"),
        task_description=(
            "The text is a Yelp review. Answer with a rating from 1-5, where 1 "
            "indicates that the review says the business is bad, and 5 stars indicates "
            "that the review says the business is great."
        ),
    ),
}


def load_classification_data(
    classification_dataset_info: ClassificationDatasetInfo,
) -> pd.DataFrame:
    """
    Returns a canonical classification dataframe.

    `classification_dataset_info` describes how to load and process the dataset.
    """
    df = classification_dataset_info.loader(
        classification_dataset_info.dataset_location
    )
    df = classification_dataset_info.processor(df)

    # Before returning df, ensure the text and label columns pass some basic checks
    required_format_message = (
        'It must have a "text" (str) and "label" (0-indexed consecutive integers) '
        "columns."
    )
    if "text" not in df.columns:
        raise ValueError(
            f'The dataframe is missing a "text" column. {required_format_message}'
        )
    if "label" not in df.columns:
        raise ValueError(
            f'The dataframe is missing a "label" column. {required_format_message}'
        )

    if df["text"].isna().sum() > 0:
        raise ValueError("NA texts are not allowed.")
    if (df["text"].str.len() == 0).sum() > 0:
        raise ValueError("Empty texts are not allowed.")

    if len(set(df.index)) != len(df):
        raise ValueError("The dataframe has non-unique indices")

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    labels_set = set(df["label"])
    missing_labels = set(np.arange(df["label"].max())) - labels_set
    if missing_labels:
        raise ValueError(
            f"The dataset is missing the following labels: {sorted(missing_labels)}"
        )

    if (num_class_names := len(classification_dataset_info.class_names)) != (
        num_labels_data := len(labels_set)
    ):
        dont_match_message = (
            f"The pre-specified number of class names, {num_class_names}, does not "
            f"match the number of labels in the data, {num_labels_data}."
        )
        if num_class_names < num_labels_data:
            raise ValueError(
                f"{dont_match_message} Please ensure that all labels are present in "
                "the data."
            )
        else:
            raise ValueError(
                f"{dont_match_message} Please ensure that the pre-specified class "
                "names, classification_dataset_info.class_names, is accurate."
            )

    return df
