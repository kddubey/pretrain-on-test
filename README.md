# Is pretraining on test set texts (without labels) ok?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&style=for-the-badge)](https://www.python.org/downloads/release/python-3100/)


## Setup

1. Clone repo to local

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

   To also run the analyses:

   ```bash
   python -m pip install ".[stat]"
   ```


## Usage

Reproduce the experiment results. By default, results are saved to `./accuracies`.

Default batch sizes are set for a single T4 GPU.

<details>
<summary>Terminal</summary>

```bash
python run.py --lm_type bert | tee run.log
```

For quick, CPU-friendly, local tests:

```bash
python run.py \
--lm_type bert \
--dataset_names ag_news \
--num_subsamples 1 \
--num_train 10 \
--num_test 10 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1 \
--per_device_train_batch_size_pretrain 4 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 4
```

</details>


<details>
<summary>Notebook</summary>

The stdout for terminal runs is quite verbose. For minimal but sufficient info, run this
in a notebook.

```python
from run import run, Experiment

experiment = Experiment(lm_type="bert")
run(experiment)
```

For quick, CPU-friendly, local tests:

```python
from run import run, Experiment

experiment = Experiment(
    lm_type="bert",
    dataset_names=["ag_news"],
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
<summary>Google Colab</summary>

Run [this
notebook](https://github.com/kddubey/pretrain-on-test/blob/main/google_colab.ipynb) on a
T4 GPU for free in Google Colab.

</details>
