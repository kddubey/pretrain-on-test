import os
from typing import Literal

from tap import tapify

import utils


def analysis(num_test: Literal[100, 200, 500], cores: int = 4) -> None:
    """
    Fit the hierarchical model, and save its inference data.

    Parameters
    ----------
    num_test : Literal[100, 200, 500]
        The dataset of accuracies which you'd like to analyze.
    cores : int, optional
        The number of CPU cores for running the analysis in parallel, by default 4
    """
    accuracies_home_dir = "accuracies_from_paper"
    num_correct_df = utils.load_all_num_correct(accuracies_home_dir, num_test)

    equation = "p(num_correct, num_test) ~ method + lm_type + (1|dataset/pair)"
    id_vars = ["num_test", "pair", "lm_type", "dataset"]

    print("Analyzing extra - base\n")
    model_control, summary_control, az_summary_control = utils.stat_model(
        num_correct_df,
        treatment="extra",
        control="base",
        equation=equation,
        id_vars=id_vars,
        cores=cores,
        plot=False,
    )
    print("Writing control inference data")
    summary_control.to_netcdf(filename=f"main_{num_test}_control.nc")
    del model_control
    del summary_control

    print("\n" + ("#" * os.get_terminal_size().columns) + "\n")
    print("Analyzing test - extra\n")
    model_bias, summary_bias, az_summary_bias = utils.stat_model(
        num_correct_df,
        treatment="test",
        control="extra",
        equation=equation,
        id_vars=id_vars,
        cores=cores,
        plot=False,
    )
    print("Writing treatment inference data")
    summary_bias.to_netcdf(filename=f"main_{num_test}_treatment.nc")


if __name__ == "__main__":
    tapify(analysis)
