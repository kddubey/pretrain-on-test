import os
import sys

import polars as pl
from tap import tapify

# sys hack to import from parent
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import utils


def summarize(accuracies_home_dir: str = os.path.join("..", "accuracies_from_paper")):
    """
    Summarize all the results in two tables showing the sample means and SEs. Takes 1
    second.

    Parameters
    ----------
    accuracies_home_dir : str, optional
        Where the accuracy data is stored, relative to the analysis/results dir, by
        default os.path.join("..", "accuracies_from_paper")
    """
    pl.Config.set_tbl_hide_dataframe_shape(True)

    num_trains = [
        int(num_train_dir.lstrip("m"))
        for num_train_dir in os.listdir(accuracies_home_dir)
    ]

    for num_train in num_trains:
        print("\n" + ("-" * os.get_terminal_size().columns) + "\n")
        print(f"m (# training observations) = {num_train}")
        print()
        num_tests = [
            int(num_test_dir.lstrip("n"))
            for num_test_dir in os.listdir(
                os.path.join(accuracies_home_dir, f"m{num_train}")
            )
        ]
        dfs: list[pl.DataFrame] = []
        for num_test in num_tests:
            df = utils.load_all_accuracies(
                os.path.join(accuracies_home_dir, f"m{num_train}"), num_test
            ).with_columns(num_test=pl.lit(num_test))
            df = df.select("num_test").with_columns(df.select(pl.exclude("num_test")))
            dfs.append(df)
        accuracy_df = pl.concat(dfs).sort(["num_test", "lm_type", "dataset"])

        print("extra - base (pretraining boost)")
        print(
            utils._summarize_differences(
                accuracy_df.with_columns(diff=pl.col("extra") - pl.col("base")),
                groups=("num_test", "lm_type"),
            ).rename({"num_test": "n (# test observations)"})
        )
        print()

        print("test - extra (evaluation bias)")
        print(
            utils._summarize_differences(
                accuracy_df.with_columns(diff=pl.col("test") - pl.col("extra")),
                groups=("num_test", "lm_type"),
            ).rename({"num_test": "n (# test observations)"})
        )


if __name__ == "__main__":
    tapify(summarize)
