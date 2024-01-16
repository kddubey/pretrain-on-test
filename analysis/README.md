# analysis

Notebooks which do dataset-level, model/LM-level, and overall analysis of the accuracy
scores in `./accuracies_from_paper`.


## Model

The model is a multilevel one which stratifies by the type of LM:

$$
\begin{align*}
Y_{ijkl} \sim \text{Binomial}(n, \lambda_{ijkl}) && \text{number of correct predictions} \\
\text{logit}(\lambda_{ijkl}) = \mu + \alpha z_i + U_j + V_{jk} + \beta x_{ijkl} && \text{additive effects} \\
\mu \sim \text{Normal}(0, 1) && \text{prior for intercept} \\
\alpha \sim \text{Normal}(0, 5) && \text{prior for LM type effect} \\
U_j \sim \text{Normal}(0, \sigma_{D}) && \text{effect of dataset} \\
V_{jk} \sim \text{Normal}(0, \sigma_{V}) && \text{(nested) effect of dataset subsample} \\
\beta \sim \text{Normal}(0, 1) && \text{prior for treatment effect} \\
\sigma_{D}, \sigma_{V} \sim \text{HalfNormal}(0, 1) && \text{prior for standard deviations}.
\end{align*}
$$

$n = 200$ or $n = 500$ depending on the dataset of scores you want to analyze.

$i = 1, 2$ for BERT and GPT-2, respectively.

$z_i = 0$ if $i = 1$ else it's $1$.

$j = 1, 2, \dots, 20$ for the dataset.

$k = 1, 2, \dots, 50$ (or $20$ for $n = 500$) for the subsample of dataset $j$.

$l = 1, 2$ for control and treatment, respectively.

$x_{ijkl} = 0$ if $l = 0$ else it's $1$. Model is fit via MCMC.

Note: I'm still learning how to do this type of analysis.


## Notebooks

`describe_datasets.ipynb` summarizes each text classification dataset.

`dataset.ipynb` visualizes $\text{acc}\_\text{base}, \text{acc}\_\text{extra}$ and
$\text{acc}\_\text{test}$, and tests that $\text{E}[\text{acc}\_\text{test} -
\text{acc}\_\text{extra}] = 0$ for each dataset.

`main_200.ipynb` contains the posterior distribution of $\beta$ when stratifying by the
LM type for $n = 200$.

`main_500.ipynb` contains the posterior distribution of $\beta$ when stratifying by the
LM type for $n = 500$.

`posterior_pred.ipynb` contains the main marginal effects of interest. It's the main
figure.

`model.ipynb` contains the posterior distribution of $\beta$ for each LM type—BERT and
GPT-2.

`meta.ipynb` assesses the importance of subsampling / replicating within each dataset.

`test.ipynb` tests that the inference code works.
