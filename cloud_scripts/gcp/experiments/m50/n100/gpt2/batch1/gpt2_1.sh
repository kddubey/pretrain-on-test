#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n100_gpt2_1 \
--dataset_names \
    yahoo_answers_topics \
--num_train 50 \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 16 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2
