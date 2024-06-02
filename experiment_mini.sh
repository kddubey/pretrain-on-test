#!/bin/bash
python run.py \
--lm_type bert-tiny \
--run_name cpu-test-bert \
--dataset_names ag_news SetFit/amazon_counterfactual_en \
--num_subsamples 2 \
--num_train 10 \
--num_test 10 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1 \
--per_device_train_batch_size_pretrain 4 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 4

python run.py \
--lm_type gpt2-tiny \
--run_name cpu-test-gpt2 \
--dataset_names ag_news SetFit/amazon_counterfactual_en \
--num_subsamples 2 \
--num_train 10 \
--num_test 10 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1 \
--per_device_train_batch_size_pretrain 4 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 4
