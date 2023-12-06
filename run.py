from __future__ import annotations
from typing import Literal

from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
)

import pretrain_on_test


model_type_to_config = {
    "bert": pretrain_on_test.Config(
        model_class_pretrain=BertForMaskedLM,
        model_class_classification=BertForSequenceClassification,
        model_id="bert-case-uncased",
        mlm=True,
        mlm_probability=0.15,
    ),
    "gpt2": pretrain_on_test.Config(
        model_class_pretrain=GPT2LMHeadModel,
        model_class_classification=GPT2ForSequenceClassification,
        model_id="gpt2",
    ),
}


def run(model_type: Literal["bert", "gpt2"]):
    config = model_type_to_config


if __name__ == "__main__":
    run()
