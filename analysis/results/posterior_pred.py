import os
import sys
from typing import Literal

import arviz as az
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tap import tapify
from tqdm.auto import tqdm

# sys hack to import from parent
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import utils


sns.set_theme(style="darkgrid")


effect_to_name: dict[str, str] = {
    "control": "pretraining boost",
    "treatment": "evaluation bias",
}


def posterior_pred(
    num_train: Literal[50, 100],
    num_tests: tuple[int] = (50, 100, 200, 500),
    accuracies_home_dir: str = os.path.join("..", "accuracies_from_paper"),
    netcdfs_path: str = os.path.join("..", "netcdfs"),
):
    """
    Saves posterior predictions to posterior_pred_m{num_train}_n{num_test}.png

    Takes ~2 minutes
    """
    num_test_to_diffs_df: dict[int, pl.DataFrame] = {}
    for num_test in tqdm(num_tests, "Processing treatment and control datasets"):
        num_correct_df = utils.load_all_num_correct(
            os.path.join(accuracies_home_dir, f"m{num_train}"), num_test
        )
        effect_to_diffs_df = {}
        for effect in effect_to_name:
            az_summary = az.from_netcdf(
                os.path.join(
                    netcdfs_path,
                    f"m{num_train}",
                    f"main_m{num_train}_n{num_test}_{effect}.nc",
                )
            )
            diffs_df = pl.DataFrame(
                utils.posterior_marginal_mean_diffs(az_summary, num_correct_df)
            )
            effect_to_diffs_df[effect] = diffs_df
        num_test_to_diffs_df[num_test] = pl.DataFrame(effect_to_diffs_df)
    del az_summary  # do notebooks garbage collect this?

    fig, axes = plt.subplots(nrows=len(num_tests), ncols=1)
    fig.set_size_inches(7, (2 * len(num_tests)))
    xlim = (-0.01, 0.06)
    axes: list[plt.Axes]
    for subplot_idx, (num_test, diffs_df) in enumerate(num_test_to_diffs_df.items()):
        ax = axes[subplot_idx]
        _ = ax.set_xlim(xlim)
        _ = sns.kdeplot(
            ax=ax, data=diffs_df.rename(effect_to_name).to_pandas(), fill=True
        )
        ax.axvline(0, linestyle="dashed", color="k")
        _ = ax.set_xlabel("average accuracy difference")
        _ = ax.set_ylabel("posterior density")
        _ = ax.set_yticks([])
        _ = ax.set_title(
            f"$m={{{num_train}}}$ training observations, "
            f"$n={{{num_test}}}$ test observations"
        )

    # fig.suptitle(f"Average accuracy differences\n", y=1.01)
    fig.legend(
        handles=reversed(ax.legend_.legend_handles),
        labels=reversed([t.get_text() for t in ax.legend_.get_texts()]),
        ncol=2,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.55, 1.07),
    )

    for ax in axes:
        ax.legend().set_visible(False)

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.01)

    fig.savefig(f"posterior_pred_m{num_train}.png", bbox_inches="tight")


if __name__ == "__main__":
    tapify(posterior_pred)
