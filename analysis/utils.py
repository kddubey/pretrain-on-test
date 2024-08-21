"""
Load and analyze data
"""

import os
from typing import Any, Callable, Sequence

import arviz as az
import bambi as bmb
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import polars as pl
import seaborn as sns
import torch
from tqdm.auto import tqdm
import xarray as xr


ACCURACY_COLUMNS = ["base", "extra", "test", "majority", "majority_all"]


lm_type_to_name = {
    "bert": "BERT",
    "gpt2": "GPT-2",
    "gpt2-epochs-2": "GPT-2 (2 epochs)",
    "mistral-qlora-zero-shot": "Mistral-7B, zero-shot",
    "mistral-qlora-zero-shot-packing": "Mistral-7B, zero-shot with packing",
}


effect_to_name: dict[str, str] = {
    "boost": "pretraining boost",
    "bias": "evaluation bias",
}
effect_to_treatment_and_control: dict[str, tuple[str, str]] = {
    "boost": ("extra", "base"),
    "bias": ("test", "extra"),
}


subsample_group = ("num_train", "num_test", "lm_type", "dataset")
"Maximal set of columns which define a group of replicated subsample scores"


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
    path_num_test = os.path.join(accuracies_home_dir, f"n{num_test}")
    dfs: list[pl.DataFrame] = []
    lm_types = [
        lm_type
        for lm_type in os.listdir(path_num_test)
        if os.path.isdir(os.path.join(path_num_test, lm_type))
    ]
    for lm_type in lm_types:
        accuracies_dir = os.path.join(path_num_test, lm_type)
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
        ├── n{num_test}
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
        ├── n{num_test}
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


def texalo(acc_type: str):
    return f"$\\text{{acc}}_\\text{{{acc_type}}}$"


def diffco_texa(treatment: str, control: str) -> str:
    return f"{texalo(treatment)} - {texalo(control)}"


def violin_plot(
    accuracy_df: pl.DataFrame, title: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    ```
          {title}

    dataset 1   👄

    dataset 2   🫦

       ...      ...

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
    ax.set_xlabel("accuracy difference")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    if is_ax_none:
        plt.show()
    return ax


def violin_plot_multiple_lms(accuracy_df: pl.DataFrame, num_test: int, num_train: int):
    """
    ```
    [ legend ]  Accuracy difference distributions
        (m = {num_train}, n = {num_test} test observations)

               lm_type1   lm_type2  ...  lm_type M

    dataset 1    👄         🫦      ...     👄

    dataset 2    🫦         👄      ...     👄

       ...      ...         ...     ...     ...

    dataset N    👄         🫦      ...     🫦
    ```
    """
    num_lm_types = accuracy_df["lm_type"].unique().len()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=accuracy_df["lm_type"].unique().len(),
        figsize=(4 * num_lm_types, 16),
    )
    xlim = (-0.5, 0.5)
    axes: list[plt.Axes] = [axes] if isinstance(axes, plt.Axes) else axes
    for subplot_idx, ((lm_type,), accuracy_df_lm) in enumerate(
        accuracy_df.partition_by("lm_type", maintain_order=True, as_dict=True).items()
    ):
        ax = axes[subplot_idx]
        ax.set_xlim(xlim)
        _ = violin_plot(accuracy_df_lm, title=lm_type_to_name[lm_type], ax=ax)
        if subplot_idx > 0:
            ax.set_yticklabels([])
            ax.set_ylabel(" ")
    fig.suptitle(
        (
            f"Accuracy difference distributions\n"
            f"$m = {num_train}$ train, $n = {num_test}$ test observations"
        ),
        y=1,
        x=0.65,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left")
    fig.tight_layout()


def _num_subsamples(df: pl.DataFrame) -> int:
    group = [column for column in subsample_group if column in df.columns]
    num = df.group_by(group).count().select("count").unique()
    if num.shape[0] > 1:
        raise ValueError("There are multiple sets of data aggregated here.")
    return num.head(n=1).item()


def _summarize_differences(
    accuracy_df: pl.DataFrame, groups: tuple[str] = ("dataset",)
) -> pl.DataFrame:
    """
    Mean and standard error of `diff` column for each group.
    """
    num_subsamples = _num_subsamples(accuracy_df)
    num_subsamples_sqrt: float = num_subsamples**0.5
    return (
        accuracy_df.group_by(*groups)
        .agg(mean=pl.col("diff").mean(), se=pl.col("diff").std() / num_subsamples_sqrt)
        .sort(by=groups)
    )


def eda(
    accuracy_df: pl.DataFrame,
    treatment: str,
    control: str,
    groups: tuple[str] = ("dataset",),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Returns the mean and standard error of the raw and relative difference between
    `treatment` and `control` columns for each dataset.

    Prints the overall mean and standard error of the raw and relative difference. The
    standard error is over the average accuracy for each dataset, because dataset
    subsamples are technical replicates.
    """
    summary = _summarize_differences(
        accuracy_df.with_columns(diff=pl.col(treatment) - pl.col(control)),
        groups=groups,
    )
    num_datasets = accuracy_df["dataset"].unique().len()
    num_datasets_sqrt: float = num_datasets**0.5
    print("Overall difference:")
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    print(
        summary.select(
            pl.col("mean").mean(), pl.col("mean").std().alias("se") / num_datasets_sqrt
        )
    )

    print("Overall difference (relative):")
    summary_relative = _summarize_differences(
        accuracy_df.with_columns(
            diff=(pl.col(treatment) - pl.col(control)) / pl.col(control)
        ),
        groups=groups,
    )
    print(
        summary_relative.select(
            pl.col("mean").mean(), pl.col("mean").std().alias("se") / num_datasets_sqrt
        )
    )
    return summary, summary_relative


def melt_num_correct(
    num_correct_df: pl.DataFrame,
    treatment: str,
    control: str,
    id_vars: Sequence[str] = ("num_test", "pair", "lm_type", "dataset"),
) -> pl.DataFrame:
    """
    Melt `num_correct_df` into a format which is compatible with Wilkinson notation.
    A pandas (not polars) dataframe is returned because `bambi` (and others) assume a
    pandas dataframe.
    """
    id_vars = list(id_vars)
    if "lm_type" not in id_vars:
        id_vars = ["lm_type"] + id_vars
    if "num_test" not in id_vars:
        id_vars = ["num_test"] + id_vars
    return (
        num_correct_df.group_by("lm_type", maintain_order=True)
        .map_groups(lambda df: df.with_columns(pl.Series("pair", range(len(df)))))
        # pair indexes the subsample. It's common across LM types b/c seeds were set
        # when subsampling.
        .select(id_vars + [control, treatment])
        .melt(id_vars=id_vars, variable_name="method", value_name="num_correct")
        .sort("lm_type", "pair")
    )


def create_model(
    num_correct_df_melted: pd.DataFrame,
    equation: str,
    treatment_effect_prior_mean_std: tuple[float, float],
) -> bmb.Model:
    # Default sigma=3.5355 results in really wide priors after prior predictive checks
    # sigma=1 results in more reasonable priors. See test.ipynb
    method_mu, method_sigma = treatment_effect_prior_mean_std
    priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
        "method": bmb.Prior("Normal", mu=method_mu, sigma=method_sigma),
        # SkewNormal w/ alpha=5 doesn't change posteriors but is arguably a more
        # reasonable prior b/c it's way less likely for their to be negative effects
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


def fit_model(
    num_correct_df: pl.DataFrame,
    treatment: str,
    control: str,
    equation: str,
    id_vars: Sequence[str] = ("num_test", "pair", "dataset"),
    plot: bool = True,
    sample_posterior_predictive: bool = True,
    treatment_effect_prior_mean_std: tuple[float, float] = (0, 1),
    chains: int = 4,
    cores: int = 1,
    random_seed: int = 123,
    draws: int = 1000,
    tune: int = 500,
) -> tuple[bmb.Model, az.InferenceData, pl.DataFrame]:
    """
    Note about `equation`: pairs/subsamples were formed from the dataset. So it's
    nested, not crossed. Technically, crossed notation—(1|dataset) + (1|pair)—would
    still result in a # nested inference b/c pair is uniquely coded across datasets.
    """
    num_correct_df_melted = melt_num_correct(
        num_correct_df, treatment, control, id_vars=id_vars
    ).to_pandas()

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
        draws=draws,
        tune=tune,
    )
    if sample_posterior_predictive:
        print(
            "Sampling posterior predictive. This will take at least 30 min. "
            "This issue tracks a progress bar feature: "
            "https://github.com/bambinos/bambi/issues/818"
        )
        model.predict(fit_summary, kind="pps")

    # Analyze model
    az_summary: pd.DataFrame = az.summary(fit_summary, hdi_prob=0.89, round_to=5)
    display(az_summary.loc[az_summary.index.str.contains("method")])
    if plot:
        az.plot_trace(
            fit_summary, compact=False, var_names="method", filter_vars="like"
        )
    az_summary = pl.from_pandas(az_summary, include_index=True).rename(
        {"None": "effect"}
    )
    return model, fit_summary, az_summary


def num_correct_df_from_predicions(
    num_correct_df: pl.DataFrame,
    predictions: Sequence[float],
    treatment: str,
    control: str,
    id_vars: Sequence[str] = ("num_test", "pair", "lm_type", "dataset"),
) -> pl.DataFrame:
    """
    Returns a dataframe which looks like `num_correct_df` where observations are filled
    by `predictions`.

    `predictions` is assumed to come from an `InferenceData` xarray. So `predictions` is
    from the melted version of `num_correct_df`.
    """
    return (
        melt_num_correct(num_correct_df, treatment, control, id_vars=id_vars)
        .with_columns(pl.Series(predictions).alias("num_correct"))
        .pivot(
            values="num_correct",
            index=id_vars,
            on="method",
        )
        .drop("pair")
    )


def _marginal_mean_diffs(
    num_correct_df: pl.DataFrame,
    predictions: xr.DataArray,
    treatment: str,
    control: str,
) -> list[float]:
    """
    Distribution for the expected accuracy difference between the treatment and control.
    """
    # This can be vectorized, but I don't want to risk incorrectness
    num_test: int = num_correct_df.select("num_test")[0].item()
    mean_diffs = []
    for draw in tqdm(
        range(predictions.shape[1]), desc=f"Marginalizing each draw (n = {num_test})"
    ):
        num_correct_df_simulated = num_correct_df_from_predicions(
            num_correct_df, predictions[:, draw].to_numpy(), treatment, control
        )
        mean_diffs.append(
            num_correct_df_simulated.select(
                (pl.col(treatment) - pl.col(control)) / num_test
            )
            .mean()
            .item()
        )
    return mean_diffs


def posterior_marginal_mean_diffs(
    summary: az.InferenceData,
    num_correct_df: pl.DataFrame,
    treatment: str,
    control: str,
) -> list[float]:
    posterior_predictive = az.extract(summary, group="posterior_predictive")
    mean_diffs = _marginal_mean_diffs(
        num_correct_df,
        posterior_predictive["p(num_correct, num_test)"],
        treatment,
        control,
    )
    return mean_diffs
