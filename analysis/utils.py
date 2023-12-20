"""
Analyze data
"""
import os
from typing import Sequence

import arviz as az
import bambi as bmb
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
import torch


ACCURACY_COLUMNS = ["base", "extra", "test", "majority", "majority_all"]


lm_type_to_name = {
    "bert": "BERT",
    "gpt2": "GPT-2",
}


def load_accuracies(accuracies_dir: str) -> pl.DataFrame:
    """
    Load a DataFrame of accuracies from a directory structured like so::

        {accuracies_dir}
        ├── {dataset1}
            ├── accuracies.csv
            └── ...
        ├── {dataset2}
            ├── accuracies.csv
            └── ...
    """
    dfs: list[pl.DataFrame] = []
    for dataset in sorted(os.listdir(accuracies_dir)):
        df = pl.read_csv(os.path.join(accuracies_dir, dataset, "accuracies.csv"))
        df = df.with_columns(pl.lit(dataset).alias("dataset"))
        dfs.append(df)
    accuracy_df: pl.DataFrame = pl.concat(dfs)
    return accuracy_df.select(["dataset", "num_classes"] + ACCURACY_COLUMNS)


def accuracies_to_num_correct(accuracy_df: pl.DataFrame, num_test: int) -> pl.DataFrame:
    """
    Convert accuracies to the number of correct predictions
    """
    return (
        (accuracy_df.select(ACCURACY_COLUMNS) * num_test)
        .cast(pl.Int64)
        .with_columns(accuracy_df.select(pl.exclude(ACCURACY_COLUMNS)))
        .select(accuracy_df.columns)
    )


def load_num_correct(accuracies_dir: str, num_test: int) -> pl.DataFrame:
    """
    Load a DataFrame for the number of correct predictions from a directory structured
    like so::

        {accuracies_dir}
        ├── {dataset1}
            ├── accuracies.csv
            └── ...
        ├── {dataset2}
            ├── accuracies.csv
            └── ...
    """
    accuracy_df = load_accuracies(accuracies_dir)
    return accuracies_to_num_correct(accuracy_df, num_test)


def violin_plot(
    accuracy_df: pd.DataFrame,
    title: str,
    color,
    **ylabel_kwargs,
):
    """
    Violin plot of `diff` column for each dataset.
    """
    _, axes = plt.subplots(figsize=(16, 2))
    axes: plt.Axes
    sns.violinplot(data=accuracy_df, x="dataset", y="diff", ax=axes, color=color)
    plt.axhline(0, linestyle="dashed", color="gray")
    axes.set_title(title)
    axes.yaxis.grid(True)
    axes.set_xlabel("Text classification dataset")
    axes.set_ylabel(**ylabel_kwargs)
    plt.xticks(rotation=45, ha="right")
    plt.show()


def _summarize_differences(accuracy_df: pl.DataFrame) -> pl.DataFrame:
    """
    Mean and standard deviation of `diff` column for each dataset.
    """
    return (
        accuracy_df.group_by("dataset")
        .agg(mean=pl.col("diff").mean(), std=pl.col("diff").std())
        .sort(by="dataset")
    )


def eda(
    accuracy_df: pl.DataFrame,
    treatment: str,
    control: str,
    title: str,
    color,
    ylabel: str = "",
    **ylabel_kwargs,
) -> pl.DataFrame:
    """
    Returns the mean and standard deviation of the difference between `treatment` and
    `control` columns.

    Displays a violin plot of the differences, and prints the overall mean and standard
    deviation of the difference.
    """
    accuracy_df = accuracy_df.with_columns(diff=pl.col(treatment) - pl.col(control))

    diffco_texa = (
        f"$\\text{{acc}}_\\text{{{treatment}}}$ - $\\text{{acc}}_\\text{{{control}}}$"
    )
    ylabel = f"{ylabel}\n({diffco_texa})"
    violin_plot(accuracy_df.to_pandas(), title, color, ylabel=ylabel, **ylabel_kwargs)

    summary = _summarize_differences(accuracy_df)
    print("Overall difference:")
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    print(summary.select(["mean", "std"]).mean())
    return summary


def stat_model(
    num_correct_df: pl.DataFrame,
    treatment: str,
    control: str,
    equation: str = "num_correct ~ method + (1|dataset/pair)",
    id_vars: Sequence[str] = ("pair", "dataset"),
    chains: int = 4,
    cores: int = 1,
    random_seed: int = 123,
) -> tuple[bmb.Model, az.InferenceData, pl.DataFrame]:
    """
    See the README for the specification of the model in `model.ipynb`.
    """
    id_vars = list(id_vars)
    df = (
        num_correct_df.with_columns(pl.Series("pair", range(len(num_correct_df))))
        # pair indexes the subsample
        .select(id_vars + [control, treatment])
        .melt(id_vars=id_vars, variable_name="method", value_name="num_correct")
        .sort("pair")
        .to_pandas()
    )
    # Pairs/subsamples were formed from the dataset. So it's nested, not crossed.
    # Technically, crossed notation—(1|dataset) + (1|pair)—would still result in a
    # nested inference b/c pair is uniquely coded across datasets
    model = bmb.Model(equation, family="poisson", data=df)
    inference_method = "mcmc" if not torch.cuda.is_available() else "nuts_numpyro"
    fit_summary: az.InferenceData = model.fit(
        inference_method=inference_method,
        chains=chains,
        cores=cores,
        random_seed=random_seed,
    )
    az_summary: pd.DataFrame = az.summary(fit_summary)
    display(
        az_summary.loc[
            [index_name for index_name in az_summary.index if "method" in index_name]
        ]
    )

    az.plot_trace(fit_summary, compact=False, var_names="method", filter_vars="like")
    az_summary = pl.from_pandas(az_summary, include_index=True).rename(
        {"None": "effect"}
    )
    return model, fit_summary, az_summary
