# analysis

Notebooks which do dataset-level, model/LM-level, and overall analysis of the accuracy
scores in `./accuracies_from_paper`.


## Model

$$
\begin{align*}
Y_{ijk} \sim \text{Binomial}(n, \lambda_{ijk}) && \text{number of correct predictions} \\
\text{logit}(\lambda_{ijk}) = \mu + U_i + V_{ij} + \beta x_{ijk} && \text{additive effects} \\
\mu \sim \text{Normal}(0, 1) && \text{prior for intercept} \\
U_i \sim \text{Normal}(0, \sigma_{D}) && \text{effect of dataset} \\
V_{ij} \sim \text{Normal}(0, \sigma_{V}) && \text{(nested) effect of dataset subsample} \\
\beta \sim \text{Normal}(0, 1) && \text{prior for main effect} \\
\sigma_{D}, \sigma_{V} \sim \text{HalfNormal}(0, 1) && \text{prior for standard deviations}.
\end{align*}
$$

$n = 200$ or $n = 500$ depending on the dataset of scores you want to analyze.

$i = 1, 2, \dots, 20$ for the dataset.

$j = 1, 2, \dots, 50$ (or $20$ for $n = 500$) for the subsample of dataset $i$.

$k = 1, 2$ for control and treatment.

$x_{ijk} = 0$ if $k = 1$ else it's $1$. Inference on $\beta$ is performed via MCMC.

Note: I'm still learning how to do this type of analysis.


## Notebooks

`describe_datasets.ipynb` summarizes each text classification dataset.

`dataset.ipynb` visualizes
$\text{acc}_\text{base}, \text{acc}_\text{extra}$ and $\text{acc}_\text{test}$, and
tests that
$\text{E}[\text{acc}_\text{test} - \text{acc}_\text{extra}] = 0$
for each dataset.

`main.ipynb` contains the posterior distribution of $\beta$ when stratifying by the LM
type. This analysis is the main conclusion.

`model.ipynb` contains the posterior distribution of $\beta$ for each LM type—BERT and
GPT-2.

`meta.ipynb` assesses the importance of subsampling / replicating within each dataset.

`test.ipynb` tests that the inference code works.
