# Can I pretrain on unlabeled test data?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&style=for-the-badge)](https://www.python.org/downloads/release/python-3100/)

There are more training runs than there are [jelly beans in this vat right
here](https://www.youtube.com/watch?v=CCvVEszRiDI&t=225s).

[Paper](https://arxiv.org/abs/2410.00179).

[Slides](https://docs.google.com/presentation/d/1WiaTOMplciOHM3qp6FTu5BYRlDdlrRI5A5ayOLaBEUA/edit?usp=sharing)
(please play the slideshow instead of scrolling through slides).

[Poster](https://docs.google.com/presentation/d/1RKS0scvabb92GZngEpJ3KQoNbhch6kj_uv2p87kxyZU/edit?usp=sharing).


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

If I deploy `llm`, will `test_set_accuracy` be higher than what we'll see in production?

```python
    test_set_accuracy = (
        llm
        .pretrain(df_test["text"])
        .train(df_train["text"], df_train["label"])
        .evaluate(df_test["text"], df_test["label"])
    )
```

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


## Experiments

<details>
<summary>BERT and GPT-2</summary>

Section 7 in the paper.

Reproduce the experiment results by running [`./experiment.sh`](./experiment.sh) on a T4
GPU. Batch sizes were set to safely avoid OOMs across the many pretraining and
finetuning runs that will occur. But they were not set too low; GPU utilization hovers
from 50-80%. The experiment will take ~5 days to finish. I ran experiments in parallel
through [Google
Cloud](https://github.com/kddubey/pretrain-on-test/tree/main/cloud_scripts/gcp).

The set of accuracy data used in the paper, including observation-level per-class
probability scores, can be downloaded at [this Google Drive
link](https://drive.google.com/file/d/1n7N4uTKgcUJZ7hjAbZYGTpGEPoYxQVGx/view?usp=sharing)
(3.29 GB unzipped, just a bunch of CSVs).

</details>


<details>
<summary>Overtraining</summary>

Section 8 in the paper.

Experiment files are in
[`./cloud_scripts/gcp/experiments/gpt2-epochs-2/`](./cloud_scripts/gcp/experiments/gpt2-epochs-2/).
Run on a T4 GPU. Takes around 6 hours.

The set of accuracy data used in the paper, including observation-level per-class
probability scores, can be downloaded at [this Google Drive
link](https://drive.google.com/file/d/1Lk1SA10eS1T-dtptQff2qD9EfHLr2cDK/view?usp=sharing)
(0.941 GB unzipped, just a bunch of CSVs).

</details>


<details>
<summary>QLoRA + zero-shot Mistral 7B</summary>

Section 9 in the paper.

Experiment files are in
[`./cloud_scripts/gcp/experiments/zero-shot/`](./cloud_scripts/gcp/experiments/zero-shot/).
Run on an L4 GPU. Takes around 10 hours. Batch sizes can be reduced to run experiments
on a T4 GPU, but it'll take much longer.

The set of accuracy data used in the paper, including observation-level per-class
probability scores, can be downloaded at [this Google Drive
link](https://drive.google.com/file/d/1tccOFyQHJgqmFeRoulQiaJgItEXzN1Xx/view?usp=sharing)
(53.4 MB unzipped, just a bunch of CSVs).

</details>


<details>
<summary>QLoRA + zero-shot Mistral 7B + packing</summary>

Section 9.1 in the paper.

Experiment files are in
[`./cloud_scripts/gcp/experiments/zero-shot-packing/`](./cloud_scripts/gcp/experiments/zero-shot-packing/).
Run on an L4 GPU. Takes around 10 hours. Batch sizes can be reduced to run experiments
on a T4 GPU, but it'll take much longer.

The set of accuracy data used in the paper, including observation-level per-class
probability scores, can be downloaded at [this Google Drive
link](https://drive.google.com/file/d/13BLC7wjlkiuK7tG_s9Ilvj4x8grhTBM0/view?usp=sharing)
(53.3 MB unzipped, just a bunch of CSVs).

</details>


## Analysis

After finishing the experiment, follow the [instructions
here](https://github.com/kddubey/pretrain-on-test/tree/main/cloud_scripts/gcp#merge-data).

To analyze the accuracy data, see [`analysis/`](./analysis/).


## Usage

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


## Numbers

<details>
<summary>81,000 models evaluated</summary>

```
[
    3 models evaluated (base, extra, test) per LM type per task per repeat x
    2 LM types x
    25 tasks x
    (
        100 repeats for n=50 +
        100 repeats for n=100 +
        50 repeats for n=200 +
        20 repeats for n=500
    )
] x 2 (for m = 50, 100) = 81,000 models evaluated
```

</details>


<details>
<summary>135,000 training runs</summary>

```
[
    (
        (1 classification training for base) +
        (1 pretraining + 1 classification training for extra) +
        (1 pretraining + 1 classification training for test) +
    ) training runs per LM type per task per repeat x
    2 LM types x
    25 tasks x
    (
        100 repeats for n=50 +
        100 repeats for n=100 +
        50 repeats for n=200 +
        20 repeats for n=500
    )
] x 2 (for m = 50, 100) = 135,000 training runs
```

</details>


## Related work

A complement to this paper are the results around **text contamination** in [*Does Data
Contamination Make a Difference? Insights from Intentionally Contaminating Pre-training
Data For Language
Models*](https://openreview.net/forum?id=wSpwj7xab9&referrer=%5Bthe%20profile%20of%20Sanmi%20Koyejo%5D(%2Fprofile%3Fid%3D~Sanmi_Koyejo1)).
This paper directly addresses the limitation in my paper that I don't study the initial
pretraining stage of an LM. (Unfortunately I didn't see this paper until a few weeks
after I presented the poster. So it's not cited in my paper or the poster.)
