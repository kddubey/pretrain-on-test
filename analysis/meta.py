"""
Run a meta-analysis to assess the importance of replicating/subsampling. Will save
posterior means to ./meta_means_{num_test}_{comparison}.csv.

Runs stuff in parallel, which makes logs messy.
"""

from functools import partial
from itertools import islice
import multiprocessing
from typing import Generator, Iterable, Literal, Sequence, TypeVar

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
    """
    Randomly select a single subsample from each `group`. Returns a realization of an
    experiment which does not replicate within each dataset.
    """
    return num_correct_df.filter(
        pl.int_range(0, pl.count()).shuffle(seed=seed).over(*group) < sample_size
    )


_T = TypeVar("_T")


def _batched(
    iterable: Iterable[_T], batch_size: int
) -> Generator[list[_T], None, None]:
    # Adapted from:
    # https://docs.python.org/3/library/itertools.html#itertools.batched
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch


def _sample_posterior_mean(
    num_correct_df: pl.DataFrame,
    treatment: str,
    control: str,
    chains: int,
    seeds: Sequence[int],
) -> list[dict[str, float]]:
    """
    Down-sample `num_correct_df` and compute the posterior mean of the `treatment -
    control` effect.
    """
    equation = "p(num_correct, num_test) ~ method + lm_type + (1|pair)"
    # We'll drop dataset b/c it's redunant w/ pair
    id_vars = ["num_test", "pair", "lm_type"]
    summaries: list[dict[str, float]] = []
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
            sample_posterior_predictive=False,
        )
        effect_summary = (
            az_summary.filter(az_summary["effect"].str.contains("method"))
            .select(pl.exclude(["effect", "ess_bulk", "ess_tail", "r_hat"]))
            .to_dicts()
        )
        assert len(effect_summary) == 1  # always True b/c we pick treatment, control
        summaries.append(effect_summary[0])
    return summaries


def sample_posterior_mean(
    num_correct_df: pl.DataFrame,
    treatment: str,
    control: str,
    num_samples: int = 500,
    chains: int = 1,
    cores: int | None = None,
) -> list[float]:
    """
    Paralleliza
    """
    single_core_sample = partial(
        _sample_posterior_mean, num_correct_df, treatment, control, chains
    )
    cores = multiprocessing.cpu_count() - 2 if cores is None else cores
    batch_size = (num_samples // cores) + 1
    batches = list(_batched(range(num_samples), batch_size))
    with multiprocessing.Pool(cores) as pool:
        summaries = pool.map(single_core_sample, batches)
    return [summary for summaries_batch in summaries for summary in summaries_batch]


class ArgParser(Tap):
    num_test: Literal[200, 500] = 200
    "The number of test observations in the accuracy data."

    comparison: Literal["control", "treatment"] = "treatment"
    "control: acc_extra - acc_base. treatment: acc_test - acc_extra"

    num_samples: int = 500
    """
    Number of subsamples to draw from accuracy data, i.e., number of means to compute in
    this simulation study
    """

    accuracies_home_dir: str = "accuracies_from_paper"
    "Directory containing accuracies for multiple LM types."


if __name__ == "__main__":
    args = ArgParser(__doc__).parse_args()
    num_correct_df = utils.load_all_num_correct(args.accuracies_home_dir, args.num_test)
    if args.comparison == "control":
        treatment, control = "extra", "base"
    else:
        treatment, control = "test", "extra"
    summaries = sample_posterior_mean(
        num_correct_df, treatment, control, num_samples=args.num_samples
    )
    pl.DataFrame(summaries).write_csv(f"meta_{args.num_test}_{args.comparison}.csv")
