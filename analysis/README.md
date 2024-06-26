# analysis

Notebooks which do dataset-level, model/LM-level, and overall analysis of the accuracy
scores in `./accuracies_from_paper`.

## Setup

At the repo root, install these dependencies (in a virtual environment):

```bash
python -m pip install ".[stat]"
```


## Model

<details>
<summary>Equation</summary>

The model is a multilevel one which stratifies by the type of LM:

$$
\begin{align*}
Y_{ijkl} \sim \text{Binomial}(n, \lambda_{ijkl}) && \text{number of correct predictions} \\
\text{logit}(\lambda_{ijkl}) = \mu + \alpha z_i + U_j + V_{jk} + \beta x_{ijkl} && \text{additive effects} \\
\mu \sim \text{Normal}(0, 1) && \text{prior for intercept} \\
\alpha \sim \text{Normal}(0, 5) && \text{prior for LM type effect} \\
U_j \sim \text{Normal}(0, \sigma_{U}) && \text{effect of dataset} \\
V_{jk} \sim \text{Normal}(0, \sigma_{V}) && \text{(nested) effect of dataset subsample} \\
\beta \sim \text{Normal}(0, 1) && \text{prior for treatment effect} \\
\sigma_{U}, \sigma_{V} \sim \text{HalfNormal}(0, 1) && \text{prior for standard deviations}.
\end{align*}
$$

$n = 50, 100, 200$ or $500$ depending on the dataset of scores you want to analyze.

$i = 1, 2$ for BERT and GPT-2, respectively.

$z_i = 0$ if $i = 1$ else it's $1$.

$j = 1, 2, \dots, 25$ for the dataset.

$k = 1, 2, \dots, $ ($100$ for $n = 50$ and $100$, $50$ for $n = 200$, $20$ for $n =
500$) for the subsample of dataset $j$.

$l = 1, 2$ for control and treatment, respectively.

$x_{ijkl} = 0$ if $l = 0$ else it's $1$. The model is fit via MCMC.

</details>


<details>
<summary>Why?</summary>

The model might be fancy-looking and fancily-estimated. Here is justification for that.

Reporting means is not enough, especially when studying few-shot learning. The first
figure in
[`./fit_posteriors/m100/main_m100_n500.ipynb`](./fit_posteriors/m100/main_m100_n500.ipynb)
demonstrates that there is considerable variance, despite pairing the accuracy
estimators. (One source of variance is intentionally introduced: the subsamples/splits.
The other source of variance is inherent: the added linear layer to perform
classification is initialized with random weights.) While these visualizations tell us
about how raw accuracy differences vary, they do not tell us how the mean accuracy
difference varies. We seek a neat answer to the core question: on our benchmark of 25
classification tasks, how much does the average benchmark accuracy differ between two
modeling techniques, and how much does this average difference vary?

One way to communicate the variance is to estimate the standard error of the mean
difference across classification tasks. But the standard error statistic can be
difficult to interpret ([Morey et al.,
2016](https://pubmed.ncbi.nlm.nih.gov/26450628/)). Furthermore, its computation is not
completely trivial due to the data's hierarchical dependency structure: each triple,
($\text{acc}\_\text{extra}, \text{acc}\_\text{test}, \text{acc}\_\text{base}$), is drawn
from (`train`, `test`), which is itself drawn from the given classification dataset.

This analysis does not aim to estimate standard errors. Instead, posterior predictive
distributions will be estimated by fitting and sampling from a hierarchical model,
specified below. [Slide
14](https://docs.google.com/presentation/d/1WiaTOMplciOHM3qp6FTu5BYRlDdlrRI5A5ayOLaBEUA/edit#slide=id.g2689f42eff3_0_108)
contains a figure connecting the data generation process to the model.

</details>


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
