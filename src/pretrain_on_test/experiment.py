from __future__ import annotations
import os
import shutil

from IPython.display import clear_output
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import logging

from pretrain_on_test import Config, train
from pretrain_on_test.pretrain import pretrain


logging.set_verbosity_error()
# Ignore the HF warning about untrained weights


def _stratified_sample(
    df: pd.DataFrame, sample_size: int, random_state: int = None
) -> pd.DataFrame:
    # Let's not worry about not exactly returning a df w/ size sample_size for
    # now. It's nbd for this experiment
    num_labels = len(set(df["label"]))
    num_obs_per_label = int(sample_size / num_labels)

    def label_sampler(df_label: pd.DataFrame) -> pd.DataFrame:
        return df_label.sample(num_obs_per_label, random_state=random_state)

    return df.groupby("label", group_keys=False).apply(label_sampler)


def _split(
    df: pd.DataFrame,
    num_train: int = 100,
    num_test: int = 200,
    random_state: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns 3 (non-overlapping) dataframes which are randomly subsampled from
    `df`. The first has `num_train` rows, and the last two have `num_test` rows
    each.
    """
    df_train = _stratified_sample(df, num_train, random_state=random_state)
    random_state = None if random_state is None else random_state + 1
    df_extra, df_test = train_test_split(
        df.drop(df_train.index),
        train_size=num_test,
        test_size=num_test,
        random_state=random_state,
    )
    return df_train, df_extra, df_test


def _experiment(
    df: pd.DataFrame,
    config: Config,
    num_train: int = 100,
    num_test: int = 200,
    random_state: int = None,
) -> dict[str, float]:
    df_train, df_extra, df_test = _split(
        df, num_train=num_train, num_test=num_test, random_state=random_state
    )
    num_labels = len(set(df["label"]))  # configure output dim of linear layer

    # Run the methodology which does no pretraining. We'll compare to this data
    # to demonstrate that pretraining/domain adaptation helps, so that there's an effect
    # to detect
    print("Base - training classifier")
    trained_classifier = train.classification(
        df_train["text"].tolist(),
        df_train["label"].tolist(),
        num_labels=num_labels,
        config=config,
        pretrained_model_name_or_path=config.model_id,
    )
    print("Base - testing")
    base_accuracy = train.accuracy(
        df_test["text"].tolist(), df_test["label"].tolist(), trained_classifier
    )

    # Run the fair pretraining methodology
    print("Extra - pretraining")
    pretrain(df_extra["text"].tolist(), config)  # saved in config.model_path_pretrained
    print("Extra - training")
    trained_classifier = train.classification(
        df_train["text"].tolist(),
        df_train["label"].tolist(),
        num_labels=num_labels,
        config=config,
        pretrained_model_name_or_path=config.model_path_pretrained,
    )
    shutil.rmtree(config.model_path_pretrained)
    shutil.rmtree(config.model_path_classification)
    print("Extra - testing")
    extra_accuracy = train.accuracy(
        df_test["text"].tolist(), df_test["label"].tolist(), trained_classifier
    )

    # Run the (presumably) unfair pretraining methodology
    print("Test - pretraining")
    pretrain(df_test["text"].tolist(), config)  # saved in config.model_path_pretrained
    print("Test - training")
    trained_classifier = train.classification(
        df_train["text"].tolist(),
        df_train["label"].tolist(),
        num_labels=num_labels,
        config=config,
        pretrained_model_name_or_path=config.model_path_pretrained,
    )
    shutil.rmtree(config.model_path_pretrained)
    shutil.rmtree(config.model_path_classification)
    print("Test - testing")
    test_accuracy = train.accuracy(
        df_test["text"].tolist(), df_test["label"].tolist(), trained_classifier
    )

    # Paired data
    return {
        "base": base_accuracy,
        "extra": extra_accuracy,
        "test": test_accuracy,
        "majority": df_test["label"].value_counts(normalize=True).max(),
    }


def replicate(
    df: pd.DataFrame,
    file_path: str,
    config: Config,
    num_replications: int = 50,
    num_train: int = 100,
    num_test: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    accuracies: list[dict[str, float]] = []
    for i in range(num_replications):
        clear_output(wait=True)
        print(f"Running trial {i+1} of {num_replications}\n")
        accuracies_replication = _experiment(
            df,
            config,
            num_train=num_train,
            num_test=num_test,
            random_state=random_state + i,
        )
        accuracies.append(accuracies_replication)
    accuracies_df = pd.DataFrame(accuracies)
    # Add some useful metadata. This is static across all replications/subsamples
    accuracies_df["num_classes"] = len(df["label"].unique())
    accuracies_df["majority_all"] = df["label"].value_counts(normalize=True).max()
    # Save to local as CSV
    os.makedirs(file_path, exist_ok=True)
    accuracies_df.to_csv(file_path, index=False)
    return accuracies_df
