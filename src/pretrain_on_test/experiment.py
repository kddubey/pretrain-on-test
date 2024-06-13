"""
Run the experiment on random subsamples of a classification dataset
"""

import logging
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import logging as hf_logging

try:
    from IPython.display import clear_output
except ModuleNotFoundError:
    clear_output = lambda *args, **kwargs: None

from pretrain_on_test import classification, Config, pretrain


hf_logging.set_verbosity_error()
# Ignore the HF warning about untrained weights. We always train them


def _stratified_sample(
    df: pd.DataFrame, sample_size: int, random_state: int = None
) -> pd.DataFrame:
    # Let's not worry about not exactly returning a df w/ size sample_size. It's
    # consistent across subsamples b/c the discrepancy only depends on the number of
    # possible labels
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
    logger: logging.Logger,
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
    logger.info("Base - training")
    trained_classifier = classification.train(
        df_train["text"].tolist(),
        df_train["label"].tolist(),
        num_labels=num_labels,
        config=config,
        pretrained_model_name_or_path=config.model_id,
    )
    logger.info("Base - testing")
    model_type_to_test_probs["base"] = classification.predict_proba(
        df_test["text"].tolist(), trained_classifier, config
    )

    # Run the fair pretraining methodology
    logger.info("Extra - pretraining")
    pretrain.train(
        df_extra["text"].tolist(), config
    )  # saved pretrained model in config.model_path_pretrained
    logger.info("Extra - training")
    try:
        trained_classifier = classification.train(
            df_train["text"].tolist(),
            df_train["label"].tolist(),
            num_labels=num_labels,
            config=config,
            pretrained_model_name_or_path=config.model_path_pretrained,
        )
    finally:
        shutil.rmtree(config.model_path_pretrained)
        shutil.rmtree(config.model_path_classification)
    logger.info("Extra - testing")
    model_type_to_test_probs["extra"] = classification.predict_proba(
        df_test["text"].tolist(), trained_classifier, config
    )

    # Run the (presumably) unfair pretraining methodology
    logger.info("Test - pretraining")
    pretrain.train(
        df_test["text"].tolist(), config
    )  # saved pretrained model in config.model_path_pretrained
    logger.info("Test - training")
    try:
        trained_classifier = classification.train(
            df_train["text"].tolist(),
            df_train["label"].tolist(),
            num_labels=num_labels,
            config=config,
            pretrained_model_name_or_path=config.model_path_pretrained,
        )
    finally:
        shutil.rmtree(config.model_path_pretrained)
        shutil.rmtree(config.model_path_classification)
    logger.info("Test - testing")
    model_type_to_test_probs["test"] = classification.predict_proba(
        df_test["text"].tolist(), trained_classifier, config
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
    logger: logging.Logger,
    num_subsamples: int = 50,
    num_train: int = 100,
    num_test: int = 200,
    random_state_subsamples: int = 42,
) -> str:
    """
    Runs the main experiment, and saves results as CSVs locally.
    """
    dataset_dir = os.path.join(results_dir, dataset_name.split("/")[-1])  # drop owner
    if os.path.exists(dataset_dir):
        raise ValueError(
            f"Results for this dataset already exist in {dataset_dir}. Please move it"
        )

    # Repeat experiment on num_subsamples random subsamples of df
    accuracy_records: list[dict[str, float]] = []
    progress_bar = tqdm(range(1, num_subsamples + 1), desc=f"{dataset_name}")
    n_digits = len(str(num_subsamples + 1))
    for subsample_idx in progress_bar:
        clear_output(wait=True)
        print(progress_bar)
        logger.info(
            f"Dataset - {dataset_name}; "
            f"Subsample - {subsample_idx} of {num_subsamples}"
        )
        df_test_with_pred_probs, accuracies_subsample = _experiment(
            df,
            config,
            logger,
            num_train=num_train,
            num_test=num_test,
            random_state_subsamples=random_state_subsamples + subsample_idx,
        )
        accuracy_records.append(accuracies_subsample)
        # Save df_test_with_pred_probs
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        file_path_subsample = os.path.join(
            dataset_dir, f"subsample_test_{str(subsample_idx).zfill(n_digits)}.csv"
        )
        logger.info(f"Writing to {file_path_subsample}")
        df_test_with_pred_probs.to_csv(
            file_path_subsample, index=True, index_label="index_from_full_split"
        )

    # Save accuracies for each subsample
    accuracies_df = pd.DataFrame(accuracy_records)
    # Add some useful metadata. This is static across all subsamples
    accuracies_df["num_classes"] = len(df["label"].unique())
    accuracies_df["majority_all"] = df["label"].value_counts(normalize=True).max()
    file_path_accuracies = os.path.join(dataset_dir, "accuracies.csv")
    logger.info(f"Writing to {file_path_accuracies}")
    accuracies_df.to_csv(file_path_accuracies, index=False)
    return dataset_dir
