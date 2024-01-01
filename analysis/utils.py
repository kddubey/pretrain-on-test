"""
Load and analyze data
"""
import os
from typing import Any, Callable, Sequence

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
        ├── ...
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
    Convert accuracies to the number of correct predictions.
    """
    return (
        (accuracy_df.select(ACCURACY_COLUMNS) * num_test)
        .cast(pl.Int64)
        .with_columns(accuracy_df.select(pl.exclude(ACCURACY_COLUMNS)))
        .select(accuracy_df.columns)
        .with_columns(pl.lit(num_test).alias("num_test"))
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
        ├── ...
    """
    accuracy_df = load_accuracies(accuracies_dir)
    return accuracies_to_num_correct(accuracy_df, num_test)


def _load_all(
    loader: Callable[[str, Any], pl.DataFrame],
    accuracies_home_dir: str,
    num_test: int,
    *args,
    **kwargs,
) -> pl.DataFrame:
    dfs: list[pl.DataFrame] = []
    for lm_type in lm_type_to_name.keys():
        accuracies_dir = os.path.join(accuracies_home_dir, str(num_test), lm_type)
        df = loader(accuracies_dir, *args, **kwargs).with_columns(
            lm_type=pl.lit(lm_type)
        )
        df = df.select("lm_type").with_columns(df.select(pl.exclude("lm_type")))
        dfs.append(df)
    return pl.concat(dfs).sort(["lm_type", "dataset"])


def load_all_num_correct(accuracies_home_dir: str, num_test: int) -> pl.DataFrame:
    """
    Load a DataFrame for the number of correct predictions (across LM types) from a
    directory structured like so::

        {accuracies_home_dir}
        ├── bert
            ├── {dataset1}
                ├── accuracies.csv
                └── ...
            ├── {dataset2}
                ├── accuracies.csv
                └── ...
            ├── ...
        ├── gpt2
            ├── {dataset1}
                ├── accuracies.csv
                └── ...
            ├── {dataset2}
                ├── accuracies.csv
                └── ...
            ├── ...
    """
    return _load_all(load_num_correct, accuracies_home_dir, num_test, num_test)


def load_all_accuracies(accuracies_home_dir: str, num_test: int) -> pl.DataFrame:
    """
    Load a DataFrame for the accuracies (across LM types) from a directory structured
    like so::

        {accuracies_home_dir}
        ├── bert
            ├── {dataset1}
                ├── accuracies.csv
                └── ...
            ├── {dataset2}
                ├── accuracies.csv
                └── ...
            ├── ...
        ├── gpt2
            ├── {dataset1}
                ├── accuracies.csv
                └── ...
            ├── {dataset2}
                ├── accuracies.csv
                └── ...
            ├── ...
    """
    return _load_all(load_accuracies, accuracies_home_dir, num_test)


def violin_plot(
    accuracy_df: pl.DataFrame, title: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    ```
        {title}

    dataset 1   👄

    dataset 2   🫦

    ...

    dataset N   👄
    ```
    """
    data_for_plot = (
        accuracy_df.with_columns(
            extra_base=pl.col("extra") - pl.col("base"),
            test_extra=pl.col("test") - pl.col("extra"),
        )
        .select(["dataset", "extra_base", "test_extra"])
        .melt(id_vars="dataset", variable_name="distribution", value_name="diff")
        .with_columns(
            pl.col("distribution").replace(
                {
                    "extra_base": f'{diffco_texa("extra", "base")} (pretraining boost)',
                    "test_extra": f'{diffco_texa("test", "extra")} (evaluation bias)',
                }
            )
        )
    )
    is_ax_none = ax is None
    if is_ax_none:
        _, ax = plt.subplots(figsize=(4, 16))
    ax = sns.violinplot(
        data_for_plot.sort("dataset"),
        y="dataset",
        x="diff",
        hue="distribution",
        ax=ax,
        orient="h",
        split=True,
        inner=None,
        cut=0,
        fill=False,
    )
    if not is_ax_none:
        ax.legend().set_visible(False)
    ax.axvline(0, linestyle="dashed", color="gray")
    ax.set_title(title)
    ax.yaxis.grid(True)
    ax.set_ylabel("Text classification dataset")
    ax.set_xlabel("Accuracy difference")
    if is_ax_none:
        plt.show()
    return ax


def _summarize_differences(accuracy_df: pl.DataFrame) -> pl.DataFrame:
    """
    Mean and standard deviation of `diff` column for each dataset.
    """
    return (
        accuracy_df.group_by("dataset")
        .agg(mean=pl.col("diff").mean(), std=pl.col("diff").std())
        .sort(by="dataset")
    )


def diffco_texa(treatment: str, control: str) -> str:
    return (
        f"$\\text{{acc}}_\\text{{{treatment}}}$ - $\\text{{acc}}_\\text{{{control}}}$"
    )


def eda(
    accuracy_df: pl.DataFrame, treatment: str, control: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Returns the mean and standard deviation of the raw and relative difference between
    `treatment` and `control` columns. Prints the overall mean and standard deviation of
    the raw and relative difference.
    """
    summary = _summarize_differences(
        accuracy_df.with_columns(diff=pl.col(treatment) - pl.col(control))
    )
    print("Overall difference:")
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    print(summary.select(["mean", "std"]).mean())

    print("Overall difference (relative):")
    summary_relative = _summarize_differences(
        accuracy_df.with_columns(
            diff=(pl.col(treatment) - pl.col(control)) / pl.col(control)
        )
    )
    print(summary_relative.select(["mean", "std"]).mean())  # math not right but idc
    return summary, summary_relative


def melt_num_correct(
    num_correct_df: pl.DataFrame,
    treatment: str,
    control: str,
    id_vars: Sequence[str] = ("num_test", "pair", "dataset"),
) -> pd.DataFrame:
    """
    Melt `num_correct_df` into a format which is compatible with Wilkinson notation.
    A pandas (not polars) dataframe is returned because `bambi` (and others) assume a
    pandas dataframe.
    """
    id_vars = list(id_vars)
    if "num_test" not in id_vars:
        id_vars = ["num_test"] + id_vars
    return (
        num_correct_df.with_columns(pl.Series("pair", range(len(num_correct_df))))
        # pair indexes the subsample
        .select(id_vars + [control, treatment])
        .melt(id_vars=id_vars, variable_name="method", value_name="num_correct")
        .sort("pair")
        .to_pandas()
    )


def create_model(
    num_correct_df_melted: pd.DataFrame,
    equation: str,
    treatment_effect_prior_mean_std: tuple[float, float],
) -> bmb.Model:
    # Fit model
    # Default sigma = 3.5355 results in really wide priors after prior predictive checks
    method_mu, method_sigma = treatment_effect_prior_mean_std
    priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
        "method": bmb.Prior("Normal", mu=method_mu, sigma=method_sigma),
    }
    if "dataset" in num_correct_df_melted.columns:
        priors["1|dataset"] = bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
        )
        priors["1|dataset:pair"] = bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
        )
    else:  # the meta-analysis only keeps one subsample per dataset
        priors["1|pair"] = bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
        )
    return bmb.Model(
        equation, family="binomial", data=num_correct_df_melted, priors=priors
    )


def stat_model(
    num_correct_df: pl.DataFrame,
    treatment: str,
    control: str,
    equation: str,
    id_vars: Sequence[str] = ("num_test", "pair", "dataset"),
    plot: bool = True,
    treatment_effect_prior_mean_std: tuple[float, float] = (0, 1),
    chains: int = 4,
    cores: int = 1,
    random_seed: int = 123,
) -> tuple[bmb.Model, az.InferenceData, pl.DataFrame]:
    """
    See the README for the specification of the model.

    Note about `equation`: Pairs/subsamples were formed from the dataset. So it's
    nested, not crossed. Technically, crossed notation—(1|dataset) + (1|pair)—would
    still result in a # nested inference b/c pair is uniquely coded across datasets
    """
    num_correct_df_melted = melt_num_correct(
        num_correct_df, treatment, control, id_vars=id_vars
    )
    # Fit model
    model = create_model(
        num_correct_df_melted,
        equation,
        treatment_effect_prior_mean_std=treatment_effect_prior_mean_std,
    )
    inference_method = "mcmc" if not torch.cuda.is_available() else "nuts_numpyro"
    fit_summary: az.InferenceData = model.fit(
        inference_method=inference_method,
        chains=chains,
        cores=cores,
        random_seed=random_seed,
        tune=500,
    )
    # Analyze model
    az_summary: pd.DataFrame = az.summary(fit_summary)
    display(az_summary.loc[az_summary.index.str.contains("method")])
    if plot:
        az.plot_trace(
            fit_summary, compact=False, var_names="method", filter_vars="like"
        )
    az_summary = pl.from_pandas(az_summary, include_index=True).rename(
        {"None": "effect"}
    )
    return model, fit_summary, az_summary
