# Is pretraining on test set texts (without labels) ok?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&style=for-the-badge)](https://www.python.org/downloads/release/python-3100/)

A partial, empirical answer to my [question on Cross
Validated](https://stats.stackexchange.com/questions/611877/is-pretraining-on-test-set-texts-without-labels-ok).

[Slides](https://docs.google.com/presentation/d/1WiaTOMplciOHM3qp6FTu5BYRlDdlrRI5A5ayOLaBEUA/edit?usp=sharing)
(please play the slideshow instead of scrolling through slides).

**This research is a work in progress.** Currently, there are results for pretraining on
50, 100, 200, and 500 unlabeled texts and training on 100 classification examples. See
the plot in [`analysis/posterior_pred.ipynb`](analysis/posterior_pred.ipynb).


## Question

<details>
<summary>In researcher terms</summary>

There's a new, hot, few-shot, NLP benchmark on the block. Alice submits her model to the
leaderboard and gets SOTA accuracy $x$. Bob submits a model which he pretrained on
*unlabeled* text from the test set, and gets accuracy $x + \epsilon$. Bob gets all the
glory. Alice disputes his score. She says he used test set data, a big no-no. Alice
argues that had Bob pretrained on text which is statistically independent of the test
data, his score would be lower. Bob counters that he didn't use test set labels, so his
score is valid. Who is right, Alice or Bob?

</details>

<details>
<summary>In engineer terms</summary>

> Andy: Hey team, I'm lookin at the notebook for our new model by @Barbie, and I see:

```python
    test_set_accuracy = (
        llm
        .pretrain(df_test["text"])
        .train(df_train["text"], df_train["label"])
        .evaluate(df_test["text"], df_test["label"])
    )
```

> Barbie: it should be fine bc i didnt do:

```python
    llm.train(df_test["text"], df_test["label"])
```

> Andy: Interesting. I'm not sure if it's ok to pretrain on unlabeled test set
> text like that. Could `test_set_accuracy` be higher than what we'll see in production?

> Barbie: 🤔

</details>

## Setup

1. Clone repo

   ```bash
   git clone https://github.com/kddubey/pretrain-on-test.git
   ```

2. cd to the repo

   ```bash
   cd pretrain-on-test
   ```

3. Install dependencies (in a virtual environment)

   ```bash
   python -m pip install .
   ```


## Usage

Reproduce the experiment results by running [`./experiment.sh`](./experiment.sh) on a T4
GPU, which will take roughly 50 hours to finish. Batch sizes were set to almost-maximize
GPU utilization on a single T4 GPU.

I ran experiments in parallel using [Google
Cloud](https://github.com/kddubey/pretrain-on-test/tree/main/cloud_scripts/gcp).

To analyze the accuracy data, see [`analysis/`](./analysis/).


<details>
<summary>Terminal (local)</summary>

```bash
python run.py --help
```

For a quick, CPU-friendly, local run:

```bash
./experiment_mini.sh
```

</details>


<details>
<summary>Notebook (local)</summary>

The terminal output is quite verbose. For minimal but sufficient info, run this
in a notebook.

```python
from run import run, Experiment

experiment = Experiment(lm_type="bert", dataset_names=...)
run(experiment)
```

For a quick, CPU-friendly, local run:

```python
from run import run, Experiment

experiment = Experiment(
    lm_type="bert-tiny",
    dataset_names=["ag_news", "SetFit/amazon_counterfactual_en"],
    num_subsamples=1,
    num_train=10,
    num_test=10,
    num_train_epochs_classification=1,
    num_train_epochs_pretrain=1,
    per_device_train_batch_size_pretrain=4,
    per_device_train_batch_size_classification=4,
    per_device_eval_batch_size_classification=4,
)

run(experiment)
```

</details>


<details>
<summary>Google Cloud Platform</summary>

[`cloud_scripts/gcp`](./cloud_scripts/gcp)

</details>


<details>
<summary>Other cloud providers</summary>

Other cloud providers are not yet supported, sorry.

To support them, implement logging and file uploading functionality. See
[`cloud.py`](./cloud.py).

You'll probably find
[`./cloud_scripts/_setup_python_env.sh`](./cloud_scripts/_setup_python_env.sh) useful
for cloud runs. Note that it assumes that the bucket name is
`pretrain-on-test-accuracies`, and that the GPU image you're using already has Python
3.10+, pip, and venv/conda on it.

</details>
