#!/bin/bash

# # run-2024-06-03_05-48-48
# python run.py \
# --lm_type bert \
# --dataset_names \
#     yahoo_answers_topics \
# --num_test 500 \
# --num_subsamples 20 \
# --per_device_train_batch_size_pretrain 8 \
# --num_train_epochs_classification 1 \
# --num_train_epochs_pretrain 2

# # run-2024-06-03_07-07-16
# python run.py \
# --lm_type bert \
# --dataset_names \
#     SetFit/enron_spam \
#     yelp_review_full \
# --num_test 500 \
# --num_subsamples 20 \
# --per_device_train_batch_size_pretrain 8 \
# --per_device_train_batch_size_classification 4 \
# --per_device_eval_batch_size_classification 32 \
# --num_train_epochs_classification 1 \
# --num_train_epochs_pretrain 2

# # run-2024-06-04_00-44-44
# python run.py \
# --lm_type bert \
# --dataset_names \
#     classla/FRENK-hate-en \
#     blog_authorship_corpus \
# --num_test 500 \
# --num_subsamples 20 \
# --per_device_train_batch_size_pretrain 8 \
# --per_device_train_batch_size_classification 8 \
# --per_device_eval_batch_size_classification 32 \
# --num_train_epochs_classification 1 \
# --num_train_epochs_pretrain 2

python run.py \
--lm_type bert \
--dataset_names \
    aladar/craigslist_bargains \
    movie_rationales \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2
