#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n200_gpt2_6 \
--dataset_names \
    app_reviews \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2
