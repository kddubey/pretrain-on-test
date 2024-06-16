#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n200_bert_6 \
--dataset_names \
    hyperpartisan_news_detection \
    limit \
    AmazonScience/massive \
    mteb/mtop_domain \
    ccdv/patent-classification \
    rotten_tomatoes \
    silicone \
    trec \
    tweets_hate_speech_detection \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2
