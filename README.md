# Is pretraining on test set texts (without labels) ok?

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?logo=python&style=for-the-badge)](https://www.python.org/downloads/release/python-390/)


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

   To also run the meta-analysis:

   ```bash
   python -m pip install ".[meta]"
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
--num_train_epochs_pretrain 1
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
)

run(experiment)
```

</details>


<details>
<summary>Google Colab</summary>

See [this
notebook](https://github.com/kddubey/pretrain-on-test/blob/main/google_colab.ipynb).

</details>
