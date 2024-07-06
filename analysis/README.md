# analysis

Notebooks which do dataset-level, model/LM-level, and overall analysis of the accuracy
scores in `./accuracies_from_paper`.

## Setup

At the repo root, install these dependencies (in a virtual environment):

```bash
python -m pip install ".[stat]"
```


## Contents

[`dataset.ipynb`](./dataset.ipynb) visualizes $\text{acc}\_\text{base},
\text{acc}\_\text{extra}$ and $\text{acc}\_\text{test}$, and tests that
$\text{E}[\text{acc}\_\text{test} - \text{acc}\_\text{extra}] = 0$ for each dataset.

[`./fit_posteriors/`](./fit_posteriors/) contains the posterior distribution of $\beta$
when stratifying by the LM type.

[`./results/`](./results/) visualizes the marginal effects of interest. This is the main
result.

[`./meta/`](./meta/) assesses the importance of subsampling / replicating within
each dataset.

[`test.ipynb`](./test.ipynb) tests that the inference code statistically works.
