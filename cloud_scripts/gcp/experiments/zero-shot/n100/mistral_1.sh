#!/bin/bash
TQDM_DISABLE=1 python run.py \
    --lm_type mistral-qlora-zero-shot \
    --run_name n100_mistral-qlora-zero-shot_1 \
    --dataset_names \
        classla/FRENK-hate-en \
        blog_authorship_corpus \
        aladar/craigslist_bargains \
        SetFit/enron_spam \
        movie_rationales \
        yahoo_answers_topics \
        yelp_review_full \
    --num_test 100 \
    --num_subsamples 20 \
    --per_device_train_batch_size_pretrain 8 \
    --per_device_eval_batch_size_classification 8 \
    --num_train_epochs_pretrain 1
