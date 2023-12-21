"""
Run a meta-analysis to assess the importance of replicating/subsampling. Will save
posterior means to ./meta_means_{num_test}_{comparison}.csv.

Runs stuff in parallel, which makes logs messy. But whatever.
"""

from functools import partial
from itertools import islice
import multiprocessing
from typing import Sequence, Literal

import polars as pl
from tap import Tap
from tqdm.auto import tqdm

import utils


def _down_sample(
    num_correct_df: pl.DataFrame,
    sample_size: int = 1,
    seed: int | None = None,
    group: tuple[str] = ("lm_type", "dataset"),
) -> pl.DataFrame:
    return num_correct_df.filter(
        pl.int_range(0, pl.count()).shuffle(seed=seed).over(*group) < sample_size
    )


def _batched(iterable, n):
    # https://docs.python.org/3/library/itertools.html#itertools.batched
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def _sample_posterior_mean(
    num_correct_df: pl.DataFrame,
    treatment: str,
    control: str,
    chains: int,
    seeds: Sequence[int],
) -> list[float]:
    equation = "num_correct ~ method + lm_type + (1|pair)"
    # We'll drop dataset b/c it's redunant w/ pair
    id_vars = ["pair", "lm_type"]
    posterior_means: list[float] = []

    for seed in tqdm(seeds, total=len(seeds), desc=f"Samples {seeds[0]} - {seeds[-1]}"):
        num_correct_df_sample = _down_sample(num_correct_df, seed=seed).drop("dataset")
        _, _, az_summary = utils.stat_model(
            num_correct_df_sample,
            treatment,
            control,
            equation=equation,
            id_vars=id_vars,
            chains=chains,
            plot=False,
        )
        effect_of_method = az_summary.filter(
            az_summary["effect"].str.contains("method")
        )["mean"]
        assert len(effect_of_method) == 1  # always True b/c we pick treatment, control
        posterior_means.append(effect_of_method[0])
    return posterior_means


def sample_posterior_mean(
    num_correct_df: pl.DataFrame,
    treatment: str,
    control: str,
    num_samples: int = 500,
    chains: int = 1,
    cores: int | None = None,
) -> list[float]:
    single_core_sample = partial(
        _sample_posterior_mean, num_correct_df, treatment, control, chains
    )
    cores = multiprocessing.cpu_count() - 2 if cores is None else cores
    batch_size = (num_samples // cores) + 1
    batches = list(_batched(range(num_samples), batch_size))
    with multiprocessing.Pool(cores) as pool:
        means = pool.map(single_core_sample, batches)
    return [mean for means_batch in means for mean in means_batch]


class ArgParser(Tap):
    comparison: Literal["control", "treatment"]
    "control: acc_extra - acc_base. treatment: acc_test - acc_extra"

    num_test: Literal[200, 500]
    "The number of test observations in the accuracy data."

    num_samples: int = 200
    "Number of subsamples to draw from accuracy data, i.e., number of means to compute"

    accuracies_home_dir: str = "accuracies_from_paper"
    "Directory containing accuracies for multiple LM types."


if __name__ == "__main__":
    args = ArgParser(__doc__).parse_args()
    num_correct_df = utils.load_num_correct_all(args.accuracies_home_dir, args.num_test)

    if args.comparison == "control":
        treatment, control = "extra", "base"
    else:
        treatment, control = "test", "extra"
    posterior_means = sample_posterior_mean(
        num_correct_df, treatment, control, num_samples=args.num_samples
    )
    pl.DataFrame({"mean": posterior_means}).write_csv(
        f"meta_means_{args.num_test}_{args.comparison}.csv"
    )
