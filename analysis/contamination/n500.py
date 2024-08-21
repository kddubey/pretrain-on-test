import os

import numpy as np
import polars as pl
from scipy.stats import permutation_test

paths = [
    "run-2024-08-13_19-06-44-n500_contamination",
    "run-2024-08-21_00-45-59-n500_contamination",
]
path_acc = os.path.join(
    "accuracies",
    "m100",
    "n500",
    "mistral-qlora-zero-shot-packing",
    "ag_news",
    "accuracies.csv",
)

df = pl.concat([pl.read_csv(os.path.join(path, path_acc)) for path in paths])

print(df.describe())

p_value: float = permutation_test(
    data=(df["test"], df["extra"]),
    statistic=lambda x, y: np.mean(x - y),
    alternative="greater",  # acc_test > acc_extra
    permutation_type="samples",  # paired observations
    n_resamples=10_000,
).pvalue

mean_eval_bias = df.select(pl.col("test") - pl.col("extra")).mean().item()
print(f"{mean_eval_bias=}, {p_value=}")  # 0.25
