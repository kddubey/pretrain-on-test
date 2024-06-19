#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n100_gpt2_5 \
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
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1
