"""
Protocols for pretraining and classification modules. Please carefully read the
docstrings to correctly implement the protocol. Avoid using `**kwargs` just to make
code run. That could result in silently wrong behavior.
"""

from typing import Protocol, TypeVar

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer
from transformers.trainer_utils import TrainOutput

from pretrain_on_test import Config
from pretrain_on_test.data import ClassificationDatasetInfo


class Pretrain(Protocol):
    @staticmethod
    def train(
        texts: list[str],
        config: Config,
        classification_dataset_info: ClassificationDatasetInfo,
    ) -> TrainOutput:
        """
        Pretrains the model at `config.model_id` on `texts`, and saves it to
        `config.model_path_pretrained`.

        Parameters
        ----------
        texts : list[str]
            The raw (unlabeled) texsts to pretrain on.
        config : Config
            Contains basic pretraining hyperparameters.

        Returns
        -------
        TrainOutput
            Contains training loss and other metrics.
        """


TrainedClassifier = TypeVar(
    "TrainedClassifier",
    Trainer,  # can be convenient b/c it'll handle batching and tokenization
    tuple[PreTrainedModel, PreTrainedTokenizerBase],  # all that's needed for CAPPr
)


class Classification(Protocol):
    @staticmethod
    def train(
        texts: list[str],
        labels: list[int],
        config: Config,
        classification_dataset_info: ClassificationDatasetInfo,
        pretrained_model_name_or_path: str | None,
        is_pretrained_fresh: bool,
    ) -> tuple[TrainedClassifier, TrainOutput]:
        """
        Trains a model on `(text, label)` pairs.

        If `pretrained_model_name_or_path is None`, then the model at
        `config.model_path_pretrained` should be finetuned.

        Parameters
        ----------
        texts : list[str]
            The raw texts to train on.
        labels : list[int]
            The corresponding 0-indexed integer-label for each text in `texts`.
        config : Config
            Contains basic classification training hyperparameters.
        pretrained_model_name_or_path : str | None
            The location of the model (in HF or local) which will be finetuned to do
            classifcation.
        is_pretrained_fresh : bool
            TODO: I need to get rid of this.

        Returns
        -------
        tuple[TrainedClassifier, TrainOutput]
            The TrainedClassifier is passed to `predict_proba` for inference.
            TrainOutput contains training loss and other metrics.
        """

    @staticmethod
    def predict_proba(
        texts: list[str],
        trained_classifier: TrainedClassifier,
        config: Config,
        classification_dataset_info: ClassificationDatasetInfo,
    ) -> np.ndarray:
        """
        Predict the probability of each class for each text.

        Parameters
        ----------
        texts : list[str]
            The raw texts to classify.
        trained_classifier : TrainedClassifier
            The classifier.

        Returns
        -------
        np.ndarray
            Array of probabilities with shape `(len(texts), # classes)`.
        """
