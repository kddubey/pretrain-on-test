#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name n500_bert_1 \
--dataset_names \
    yahoo_answers_topics \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2
