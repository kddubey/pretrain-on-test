#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name n500_gpt2_3 \
--dataset_names \
    classla/FRENK-hate-en \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2
