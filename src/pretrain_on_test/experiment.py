"""
Run the experiment on random subsamples of a classification dataset
"""

import logging
import os
import shutil
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import logging as hf_logging
from transformers.trainer_utils import TrainOutput

try:
    from IPython.display import clear_output
except ModuleNotFoundError:
    clear_output = lambda *args, **kwargs: None

import pretrain_on_test


hf_logging.set_verbosity_error()
# Ignore the HF warning about untrained weights for sequence classifier / linear layer. We always train them


_pretrain_method_to_module = {
    "raw-text": pretrain_on_test.pretrain,
    "instructions-with-text": pretrain_on_test.pretrain_for_sft,
}
_classification_method_to_module = {
    "linear-layer": pretrain_on_test.classification,
    "sft": pretrain_on_test.classification_sft,
    "zero-shot": pretrain_on_test.classification_zero_shot,
}


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
    df: pd.DataFrame, split_to_pred_probs: dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Returns a new dataframe with a column of predicted probabilities for each class and
    each model type.
    """
    df = df.copy()
    for split, pred_probs in split_to_pred_probs.items():
        if pred_probs.ndim != 2:
            raise ValueError(
                f"Expected 2-D predicted probabilities. Got a {pred_probs.ndim}-D "
                f"array for {split}"
            )
        for class_idx, class_pred_probs in enumerate(pred_probs.T):
            df[f"pred_prob_{class_idx}_{split}"] = class_pred_probs
    return df


def _maybe_rmtree(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def _experiment(
    df: pd.DataFrame,
    classification_dataset_info: pretrain_on_test.data.ClassificationDatasetInfo,
    config: pretrain_on_test.Config,
    logger: logging.Logger,
    num_train: int = 100,
    num_test: int = 200,
    random_state_subsamples: int = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    # Get the pretraining and classification modules
    train_type_to_module: dict[
        str,
        pretrain_on_test.protocols.Pretrain | pretrain_on_test.protocols.Classification,
    ] = {
        "pretrain": _pretrain_method_to_module[config.pretrain_method],
        "classification": _classification_method_to_module[
            config.classification_method
        ],
    }

    # Arguments passed to each train_type x split pair
    config_and_classification_dataset_info = dict(
        config=config, classification_dataset_info=classification_dataset_info
    )
    train_type_to_split_to_kwargs = {
        "pretrain": {
            "extra": config_and_classification_dataset_info,
            "test": config_and_classification_dataset_info,
        },
        "classification": {
            "base": dict(
                **config_and_classification_dataset_info,
                # Load a freshly pretrained model
                pretrained_model_name_or_path=config.model_id,
                is_pretrained_fresh=True,
            ),
            "extra": dict(
                **config_and_classification_dataset_info,
                # Load the model that was just pretrained
                pretrained_model_name_or_path=config.model_path_pretrained,
                is_pretrained_fresh=False,
            ),
            "test": dict(
                **config_and_classification_dataset_info,
                # Load the model that was just pretrained
                pretrained_model_name_or_path=config.model_path_pretrained,
                is_pretrained_fresh=False,
            ),
        },
    }

    # Define data passed to each train_type x split combo
    df_train, df_extra, df_test = _split(
        df, num_train=num_train, num_test=num_test, random_state=random_state_subsamples
    )
    _classification_data: tuple[list[str], list[int]] = (
        df_train["text"].tolist(),
        df_train["label"].tolist(),
    )
    test_data: list[str] = df_test["text"].tolist()
    train_type_to_split_to_data: dict[str, dict[str, tuple[list, ...]]] = {
        "pretrain": {
            "extra": (df_extra["text"].tolist(),),
            "test": (df_test["text"].tolist(),),
        },
        "classification": {
            "base": _classification_data,
            "extra": _classification_data,
            "test": _classification_data,
        },
    }

    # Output data we'll update
    train_type_split_train_output: list[dict[str, str, TrainOutput]] = []
    split_to_test_probs: dict[
        Literal["base", "extra", "test", "majority"], np.ndarray
    ] = {}
    split_to_accuracy: dict[Literal["base", "extra", "test", "majority"], float] = {}
    split_to_accuracy["majority"] = df_test["label"].value_counts(normalize=True).max()
    accuracy = lambda test_probs: np.mean(
        df_test["label"] == np.argmax(test_probs, axis=1)
    )

    def train(
        split: Literal["base", "extra", "test"],
        train_type: Literal["pretrain", "classification"],
    ):
        module = train_type_to_module[train_type]
        kwargs = train_type_to_split_to_kwargs[train_type][split]
        data = train_type_to_split_to_data[train_type][split]

        logger.info(f"{split.upper()} - {train_type}")
        if train_type == "pretrain":
            train_output = module.train(*data, **kwargs)
        else:
            trained_model, train_output = module.train(*data, **kwargs)
        train_type_split_train_output.append(
            dict(train_type=train_type, split=split, train_output=train_output)
        )

        if train_type == "classification":
            logger.info(f"{split.upper()} - evaluating on test")
            pred_probs = module.predict_proba(
                test_data, trained_model, **config_and_classification_dataset_info
            )
            split_to_test_probs[split] = pred_probs
            split_to_accuracy[split] = accuracy(pred_probs)

    # Run the methodology which does no pretraining. We'll compare to this data
    # to demonstrate that pretraining/domain adaptation helps, so that there's an effect
    # to detect
    try:
        train(split="base", train_type="classification")
    finally:
        _maybe_rmtree(config.model_path_classification)

    # Run the fair pretraining methodology
    try:
        train(split="extra", train_type="pretrain")
        train(split="extra", train_type="classification")
    finally:
        _maybe_rmtree(config.model_path_pretrained)
        _maybe_rmtree(config.model_path_classification)

    # Run the (presumably) unfair pretraining methodology
    try:
        train(split="test", train_type="pretrain")
        train(split="test", train_type="classification")
    finally:
        _maybe_rmtree(config.model_path_pretrained)
        _maybe_rmtree(config.model_path_classification)

    # Updated output data
    df_test_with_pred_probs = _add_pred_probs(df_test, split_to_test_probs)
    return df_test_with_pred_probs, split_to_accuracy


def replicate(
    df: pd.DataFrame,
    classification_dataset_info: pretrain_on_test.data.ClassificationDatasetInfo,
    dataset_name: str,
    results_dir: str,
    config: pretrain_on_test.Config,
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
        df_test_with_pred_probs, split_to_accuracy = _experiment(
            df,
            classification_dataset_info,
            config,
            logger,
            num_train=num_train,
            num_test=num_test,
            random_state_subsamples=random_state_subsamples + subsample_idx,
        )
        accuracy_records.append(split_to_accuracy)
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
    # Add some useful metapretrain_on_test.data. This is static across all subsamples
    accuracies_df["num_classes"] = len(df["label"].unique())
    accuracies_df["majority_all"] = df["label"].value_counts(normalize=True).max()
    file_path_accuracies = os.path.join(dataset_dir, "accuracies.csv")
    logger.info(f"Writing to {file_path_accuracies}")
    accuracies_df.to_csv(file_path_accuracies, index=False)
    return dataset_dir
