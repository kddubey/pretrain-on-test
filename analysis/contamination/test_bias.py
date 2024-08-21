import os

import numpy as np
import polars as pl
from scipy.stats import permutation_test

path_acc = os.path.join(
    "run-2024-08-13_19-06-44-n500_contamination",
    "accuracies",
    "m100",
    "n500",
    "mistral-qlora-zero-shot-packing",
    "ag_news",
    "accuracies.csv",
)

df = pl.read_csv(path_acc)
num_subsamples = len(df)

# There was a training run that super underperformed. Excluding it causes:
# mean_eval_bias=0.0032, p_value=0.4057

# acc_thresh = 0.43
# mask = (pl.col("extra") > acc_thresh).and_((pl.col("test") > acc_thresh))
# df = df.filter(mask)
# assert len(df) == num_subsamples - 1

# print(df.describe())

p_value: float = permutation_test(
    data=(df["test"], df["extra"]),
    statistic=lambda x, y: np.mean(x - y),
    alternative="greater",  # acc_test > acc_extra
    permutation_type="samples",  # paired observations
    n_resamples=100_000,
).pvalue

mean_eval_bias = df.select(pl.col("test") - pl.col("extra")).mean().item()
print(f"{mean_eval_bias=:.4f}, {p_value=:.4f}")
# mean_eval_bias=0.0105, p_value=0.2409
