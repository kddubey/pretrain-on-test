"""
Don't train anything. Just give the model instructions in its context.
"""

from pretrain_on_test import Config
from . import _dum


def train(
    texts: list[str],
    labels: list[int],
    config: Config,
    pretrained_model_name_or_path: str | None = None,
    is_pretrained_fresh: bool = False,
):
    """
    Returns back the pretrained model and its tokenizer.

    Doesn't do any training. So `texts` and `labels` are unused.

    If `pretrained_model_name_or_path is None`, then the model at
    `config.model_path_pretrained` is returned.
    """
    model = _dum.load_model(
        config.model_class_classification,
        config.lora_pretrain,
        pretrained_model_name_or_path or config.model_path_pretrained,
        config.qlora,
        is_pretrained_fresh=is_pretrained_fresh,
        device_map=config.device,
    )
    return model, config.tokenizer


predict_proba = _dum.predict_proba
