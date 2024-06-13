#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n100_bert_3 \
--dataset_names \
    classla/FRENK-hate-en \
    blog_authorship_corpus \
--num_train 50 \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2
