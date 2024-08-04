"""
Run a meta-analysis to assess the importance of replicating/subsampling. Will save
posterior means to ./meta_m{num_train}_n{num_test}_{comparison}.csv.

Runs stuff in parallel, which makes logs messy.

Took my mac around 15 minutes to complete.
"""

from functools import partial
from itertools import islice
import multiprocessing
import os
import sys
from typing import Generator, Iterable, Literal, Sequence, TypeVar

import polars as pl
from tap import Tap
from tqdm.auto import tqdm

# sys hack to import from parent
sys.path.insert(1, os.path.join(sys.path[0], ".."))
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
        pl.int_range(0, pl.len()).shuffle(seed=seed).over(*group) < sample_size
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
    id_vars = ("num_test", "pair", "lm_type")
    if num_correct_df["lm_type"].unique().len() == 1:
        equation = "p(num_correct, num_test) ~ method + (1|pair)"
    else:
        equation = "p(num_correct, num_test) ~ method + lm_type + (1|pair)"
        # We'll drop dataset b/c it's redunant w/ pair

    summaries: list[dict[str, float]] = []
    for seed in tqdm(seeds, total=len(seeds), desc=f"Samples {seeds[0]} - {seeds[-1]}"):
        # TODO: parallelization is messing up the seeds somehow. For now, not setting it
        num_correct_df_sample = _down_sample(num_correct_df, seed=None).drop("dataset")
        _, _, az_summary = utils.fit_model(
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
    num_train: Literal[50, 100] = 100
    "The number of training observations in the accuracy data."

    num_test: Literal[50, 100, 200, 500] = 500
    "The number of test observations in the accuracy data."

    comparison: Literal["boost", "bias"] = "bias"
    "boost: acc_extra - acc_base. bias: acc_test - acc_extra"

    num_samples: int = 500
    """
    Number of subsamples to draw from accuracy data, i.e., number of means to compute in
    this simulation study
    """

    accuracies_home_dir: str = os.path.join("..", "accuracies_from_paper")
    "Directory containing accuracies for multiple LM types."


if __name__ == "__main__":
    args = ArgParser(__doc__).parse_args()
    num_correct_df = utils.load_all_num_correct(
        os.path.join(args.accuracies_home_dir, f"m{args.num_train}"), args.num_test
    )
    if args.comparison == "boost":
        treatment, control = "extra", "base"
    else:
        treatment, control = "test", "extra"
    summaries = sample_posterior_mean(
        num_correct_df, treatment, control, num_samples=args.num_samples
    )
    if "zero_shot" in args.accuracies_home_dir:
        path = (
            f"meta_zero_shot_m{args.num_train}_n{args.num_test}_{args.comparison}.csv"
        )
    else:
        path = f"meta_m{args.num_train}_n{args.num_test}_{args.comparison}.csv"
    pl.DataFrame(summaries).write_csv(path)
