import os

import numpy as np
import polars as pl
from scipy.stats import permutation_test


df = pl.read_csv(
    os.path.join(
        "run-2024-08-13_19-06-44-n500_contamination",
        "accuracies",
        "m100",
        "n500",
        "mistral-qlora-zero-shot",
        "ag_news",
        "accuracies.csv",
    )
)
p_value: float = permutation_test(
    data=(df["test"], df["extra"]),
    statistic=lambda x, y: np.mean(x - y),
    alternative="greater",  # acc_test > acc_extra
    permutation_type="samples",  # paired observations
    n_resamples=10_000,
).pvalue
print(p_value)  # 0.2
