from __future__ import annotations
import os
import shutil

from IPython.display import clear_output
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import logging

from pretrain_on_test import classification, Config, pretrain


# logging.set_verbosity_error()
# Ignore the HF warning about untrained weights. We always train them


torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


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
    df_extra, df_test = train_test_split(
        df.drop(df_train.index),
        train_size=num_test,
        test_size=num_test,
        random_state=random_state,
    )
    return df_train, df_extra, df_test


def _add_pred_probs(
    df: pd.DataFrame, model_type_to_pred_probs: dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Returns a new dataframe with a column of predicted probabilities for each class and
    each model type.
    """
    df = df.copy()
    for model_type, pred_probs in model_type_to_pred_probs.items():
        if pred_probs.ndim != 2:
            raise ValueError(
                f"Expected 2-D predicted probabilities. Got a {pred_probs.ndim}-D "
                f"array for {model_type}"
            )
        for class_idx, class_pred_probs in enumerate(pred_probs.T):
            df[f"pred_prob_{class_idx}_{model_type}"] = class_pred_probs
    return df


def _experiment(
    df: pd.DataFrame,
    config: Config,
    num_train: int = 100,
    num_test: int = 200,
    random_state_subsamples: int = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    df_train, df_extra, df_test = _split(
        df, num_train=num_train, num_test=num_test, random_state=random_state_subsamples
    )
    num_labels = len(set(df["label"]))  # configure output dimension of linear layer

    model_type_to_test_probs: dict[str, np.ndarray] = {}

    # Run the methodology which does no pretraining. We'll compare to this data
    # to demonstrate that pretraining/domain adaptation helps, so that there's an effect
    # to detect
    print("Base - training classifier")
    trained_classifier = classification.train(
        df_train["text"].tolist(),
        df_train["label"].tolist(),
        num_labels=num_labels,
        config=config,
        pretrained_model_name_or_path=config.model_id,
    )
    print("Base - testing")
    model_type_to_test_probs["base"] = classification.predict_proba(
        df_test["text"].tolist(), df_test["label"].tolist(), trained_classifier, config
    )

    # Run the fair pretraining methodology
    print("Extra - pretraining")
    pretrain.train(
        df_extra["text"].tolist(), config
    )  # saved pretrained model in config.model_path_pretrained
    print("Extra - training")
    trained_classifier = classification.train(
        df_train["text"].tolist(),
        df_train["label"].tolist(),
        num_labels=num_labels,
        config=config,
        pretrained_model_name_or_path=config.model_path_pretrained,
    )
    shutil.rmtree(config.model_path_pretrained)
    shutil.rmtree(config.model_path_classification)
    print("Extra - testing")
    model_type_to_test_probs["extra"] = classification.predict_proba(
        df_test["text"].tolist(), df_test["label"].tolist(), trained_classifier, config
    )

    # Run the (presumably) unfair pretraining methodology
    print("Test - pretraining")
    pretrain.train(
        df_test["text"].tolist(), config
    )  # saved pretrained model in config.model_path_pretrained
    print("Test - training")
    trained_classifier = classification.train(
        df_train["text"].tolist(),
        df_train["label"].tolist(),
        num_labels=num_labels,
        config=config,
        pretrained_model_name_or_path=config.model_path_pretrained,
    )
    shutil.rmtree(config.model_path_pretrained)
    shutil.rmtree(config.model_path_classification)
    print("Test - testing")
    model_type_to_test_probs["test"] = classification.predict_proba(
        df_test["text"].tolist(), df_test["label"].tolist(), trained_classifier, config
    )

    # Compute accuracies on test
    accuracy = lambda test_probs: np.mean(
        df_test["label"] == np.argmax(test_probs, axis=1)
    )
    model_type_to_accuracy: dict[str, float] = {
        model_type: accuracy(test_probs)
        for model_type, test_probs in model_type_to_test_probs.items()
    }
    model_type_to_accuracy["majority"] = (
        df_test["label"].value_counts(normalize=True).max()
    )
    return (
        _add_pred_probs(df_test, model_type_to_test_probs),
        model_type_to_accuracy,
    )


def replicate(
    df: pd.DataFrame,
    dataset_name: str,
    results_dir: str,
    config: Config,
    num_replications: int = 50,
    num_train: int = 100,
    num_test: int = 200,
    random_state_subsamples: int = 42,
) -> pd.DataFrame:
    """
    Runs the main experiment, and saves results as CSVs locally.
    """
    dataset_dir = os.path.join(results_dir, dataset_name.split("/")[-1])  # drop owner
    if os.path.exists(dataset_dir):
        raise ValueError(
            f"Results for this dataset already exist in {dataset_dir}. Please move it"
        )
    else:
        os.makedirs(dataset_dir)

    # Repeat experiment on num_replications random subsamples of df
    accuracy_records: list[dict[str, float]] = []
    for replication in range(1, num_replications + 1):  # 1-indexed
        clear_output(wait=True)
        print(f"Running replication {replication} of {num_replications}\n")
        df_test_with_pred_probs, accuracies_replication = _experiment(
            df,
            config,
            num_train=num_train,
            num_test=num_test,
            random_state_subsamples=random_state_subsamples + replication,
        )
        accuracy_records.append(accuracies_replication)
        # Save df_test_with_pred_probs
        file_path_replication = os.path.join(
            dataset_dir, f"subsample_test{replication}.csv"
        )
        df_test_with_pred_probs.to_csv(file_path_replication, index=True)

    # Save accuracies for each replication
    accuracies_df = pd.DataFrame(accuracy_records)
    # Add some useful metadata. This is static across all replications/subsamples
    accuracies_df["num_classes"] = len(df["label"].unique())
    accuracies_df["majority_all"] = df["label"].value_counts(normalize=True).max()
    file_path_accuracies = os.path.join(dataset_dir, "accuracies.csv")
    accuracies_df.to_csv(file_path_accuracies, index=False)
