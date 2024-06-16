#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n200_bert_4 \
--dataset_names \
    aladar/craigslist_bargains \
    movie_rationales \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2
