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
<summary>Why?</summary>

The model might be fancy-looking and fancily-estimated. Here is justification for that.

Reporting means is not enough, especially when studying few-shot learning. The two long
figures in [`main_200.ipynb`](./main_200.ipynb) and [`main_500.ipynb`](./main_500.ipynb)
demonstrate that there is considerable variance, despite pairing the accuracy
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

$n = 200$ or $n = 500$ depending on the dataset of scores you want to analyze.

$i = 1, 2$ for BERT and GPT-2, respectively.

$z_i = 0$ if $i = 1$ else it's $1$.

$j = 1, 2, \dots, 20$ for the dataset.

$k = 1, 2, \dots, 50$ (or $20$ for $n = 500$) for the subsample of dataset $j$.

$l = 1, 2$ for control and treatment, respectively.

$x_{ijkl} = 0$ if $l = 0$ else it's $1$. Model is fit via MCMC.

Note: I'm still learning how to do this type of analysis.


## Notebooks

[`dataset.ipynb`](./dataset.ipynb) visualizes $\text{acc}\_\text{base},
\text{acc}\_\text{extra}$ and $\text{acc}\_\text{test}$, and tests that
$\text{E}[\text{acc}\_\text{test} - \text{acc}\_\text{extra}] = 0$ for each dataset.

[`main_200.ipynb`](./main_200.ipynb) contains the posterior distribution of $\beta$ when
stratifying by the LM type for $n = 200$.

[`main_500.ipynb`](./main_500.ipynb) contains the posterior distribution of $\beta$ when
stratifying by the LM type for $n = 500$.

[`posterior_pred.ipynb`](./posterior_pred.ipynb) visualizes the marginal effects of
interest. <span style="background-color: #FFFF00"><b>This is the main result</b></span>.

[`meta.ipynb`](./meta.ipynb) assesses the importance of subsampling / replicating within
each dataset.

[`test.ipynb`](./test.ipynb) tests that the inference code statistically works.

[`model.ipynb`](./model.ipynb) contains the posterior distribution of $\beta$ for each
LM type—BERT and GPT-2. I don't think there's a great reason to be really interested in
this analysis b/c [`posterior_pred.ipynb`](./posterior_pred.ipynb) can be edited to
stratify posterior predictions.


## Usage

All of the analyses can be run locally, but I was hitting performance issues for $n =
50$ and $100$ b/c the number of subsamples for each dataset is $100$. Also, multicore
isn't working locally. I ran it in the cloud instead.

Here are instructions for Google Cloud (currently extremely manual b/c this project is
just me doing stupid stuff rn hopefully once or twice):

1. cd to the cloud scripts dir:

   ```bash
   cd ../cloud_scripts/gcp
   ```

2. Launch a high-memory CPU instance:

   ```bash
   python launch.py \
       --experiment_type cpu \
       --instance_name_prefix instance-analysis \
       --dont_run_experiment
   ```

3. SSH into it using the command printed by step 2

4. You are now in the instance. Indicate your cloud provider:

   ```bash
   export PRETRAIN_ON_TEST_CLOUD_PROVIDER="gcp"
   ```

5. Just in case:

   ```bash
   sudo apt-get update
   sudo apt-get install -y git
   ```

6. Clone this repo:

   ```bash
   git clone https://github.com/kddubey/pretrain-on-test.git
   ```

7. cd to the repo:

   ```bash
   cd pretrain-on-test
   ```

8. Move the setup script here:

   ```bash
   mv cloud_scripts/_setup_python_env_analysis.sh .
   ```

9. Run it:

   ```bash
   ./_setup_python_env_analysis.sh
   ```

10. You should now be in a Python virtual environment called `pretrain-env`. You're now
    ready to run the analysis. Consider opening a new `screen` for it (ctrl+a + d to exit
    it):

    ```bash
    screen -S analysis

    activate_pretrain_env
    ```

11. cd to `analysis`:

    ```bash
    cd analysis
    ```

12. Run it for your desired set of scores, e.g.,

    ```bash
    export NUM_TEST=100
    ```

    ```bash
    python run.py --num_test $NUM_TEST
    ```

13. Move the newly created netcdf files to a new dir:

    ```bash
    export TIMESTAMP=$(python -c 'from datetime import datetime; print(datetime.now().strftime("%Y%m%d%H%M%S"))')
    ```

    ```bash
    export NETCDF_DIR="n${NUM_TEST}_${TIMESTAMP}"
    ```

    ```bash
    mkdir $NETCDF_DIR
    ```

    ```bash
    mv main_${NUM_TEST}_control.nc $NETCDF_DIR
    ```

    ```bash
    mv main_${NUM_TEST}_treatment.nc $NETCDF_DIR
    ```

14. cd to the repo root, and then upload the netcdf files to the bucket:

    ```bash
    cd ..
    ```

    ```bash
    python
    ```

    ```python
    import logging
    import os

    import cloud

    logger = logging.getLogger(__name__)

    netcdf_dir = os.path.join("analysis", os.environ["NETCDF_DIR"])

    cloud.UploadGCP(os.environ["PRETRAIN_ON_TEST_BUCKET_NAME"]).upload_directory(
        netcdf_dir, logger
    )
    ```
