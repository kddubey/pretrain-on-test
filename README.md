# Is pretraining on test set texts (without labels) ok?

See work in progress
[here](https://stats.stackexchange.com/questions/611877/is-pretraining-on-test-set-texts-without-labels-ok).


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

```bash
python run.py --model_type bert
```

For quick local tests:

```bash
python run.py \
--model_type bert \
--dataset_names ag_news \
--num_replications 1 \
--num_train 10 \
--num_test 10
```

Batch sizes are currently hardcoded for running on a single T4 GPU.
TODO: input batch sizes as args for larger GPUs.
