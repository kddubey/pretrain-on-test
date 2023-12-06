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

On a T4 GPU:

```
python run.py --model_type bert
```

TODO: input batch sizes as args for larger GPUs.
