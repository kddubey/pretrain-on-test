# analysis

Notebooks which analyze the accuracy scores in `./accuracies_*/`.

## Setup

At the repo root, install these dependencies (in a virtual environment):

```bash
python -m pip install ".[stat]"
```


## Notebooks

[`dataset.ipynb`](./dataset.ipynb) visualizes
$\text{acc}\_\text{base}$,
$\text{acc}\_\text{extra}$, and
$\text{acc}\_\text{test}$, and tests that
$\text{E}[\text{acc}\_\text{test} - \text{acc}\_\text{extra}] = 0$ for each dataset.

[`./fit_posteriors/`](./fit_posteriors/) fits the hierarchical models.

[`./meta/`](./meta/) assesses the importance of repeated subsampling of each dataset.

[`./results/`](./results/) visualizes effects of interest. This is the main result.

[`./contamination/`](./contamination/) sees if a particular contamination test raises an
arguably false alarm.

[`test.ipynb`](./test.ipynb) tests that the inference code statistically works.

<details>
<summary>Why is m 100 in the zero-shot experiments?</summary>

The `./accuracies_zero_shot*/` data were run from
[experiments](../cloud_scripts/gcp/experiments/) with `--num_train 100` (the default
value), but the training dataset is
[unused](https://github.com/kddubey/pretrain-on-test/blob/32c06e3278e6206410966e607f3d7637841f2775/src/pretrain_on_test/classification_zero_shot.py#L13).
`--num_train 100` is supplied to keep every subsample's test split identical across
few-shot and zero-shot experiments, in case we ever want to compare them. The
[subsampling
code](https://github.com/kddubey/pretrain-on-test/blob/32c06e3278e6206410966e607f3d7637841f2775/src/pretrain_on_test/experiment.py#L62)
works by stratify-sampling (by the label) $m$ train observations, and then it randomly
draws $n$ extra and $n$ test observations from the rest of the data. The seed is the
subsample number, which is universal across experiment configurations. Changing $m$
would cause the test split to be different, which makes comparisons less controlled.

</details>


<details>
<summary>Why do this fancy stuff?</summary>

A small evaluation bias is noteworthy. We need to say how confident we are in our
measurement. The paper explains why naively computing standard errors is not great for
the confidence part. (It usually is great and sufficient.)

The paper also motivates the estimation of task-level effects. When picking lemons or
cherries from a big tree, the magnitude of effects are overestimated due to selection
bias. Priors shift them towards 0 to improve estimation.

All in all, I want to expose and communicate the considerable variance involved in this
research, and reduce estimation bias.

</details>
