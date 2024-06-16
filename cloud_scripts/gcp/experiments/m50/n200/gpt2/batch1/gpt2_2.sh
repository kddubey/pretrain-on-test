#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n200_gpt2_2 \
--dataset_names \
    blog_authorship_corpus \
    aladar/craigslist_bargains \
    SetFit/enron_spam \
    movie_rationales \
    yelp_review_full \
--num_train 50 \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2
