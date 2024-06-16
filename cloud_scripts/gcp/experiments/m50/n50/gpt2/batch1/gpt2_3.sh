#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n50_gpt2_3 \
--dataset_names \
    classla/FRENK-hate-en \
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1
