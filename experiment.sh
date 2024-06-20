# experiments/m100/n100/bert/bert_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n100_bert_1 \
--dataset_names \
    yahoo_answers_topics \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n100/bert/bert_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n100_bert_2 \
--dataset_names \
    SetFit/enron_spam \
    yelp_review_full \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n100/bert/bert_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n100_bert_3 \
--dataset_names \
    classla/FRENK-hate-en \
    blog_authorship_corpus \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n100/bert/bert_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n100_bert_4 \
--dataset_names \
    aladar/craigslist_bargains \
    movie_rationales \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n100/bert/bert_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n100_bert_5 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    app_reviews \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m100/n100/bert/bert_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n100_bert_6 \
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
--num_train_epochs_pretrain 2


# experiments/m100/n100/gpt2/gpt2_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n100_gpt2_1 \
--dataset_names \
    yahoo_answers_topics \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 16 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m100/n100/gpt2/gpt2_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n100_gpt2_2 \
--dataset_names \
    blog_authorship_corpus \
    aladar/craigslist_bargains \
    SetFit/enron_spam \
    movie_rationales \
    yelp_review_full \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m100/n100/gpt2/gpt2_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n100_gpt2_3 \
--dataset_names \
    classla/FRENK-hate-en \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m100/n100/gpt2/gpt2_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n100_gpt2_4 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m100/n100/gpt2/gpt2_5.sh#!/bin/bash
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


# experiments/m100/n100/gpt2/gpt2_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n100_gpt2_6 \
--dataset_names \
    app_reviews \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m100/n200/bert/bert_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n200_bert_1 \
--dataset_names \
    yahoo_answers_topics \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n200/bert/bert_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n200_bert_2 \
--dataset_names \
    SetFit/enron_spam \
    yelp_review_full \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n200/bert/bert_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n200_bert_3 \
--dataset_names \
    classla/FRENK-hate-en \
    blog_authorship_corpus \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n200/bert/bert_4.sh#!/bin/bash
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


# experiments/m100/n200/bert/bert_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n200_bert_5 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    app_reviews \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m100/n200/bert/bert_6.sh#!/bin/bash
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


# experiments/m100/n200/gpt2/gpt2_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n200_gpt2_1 \
--dataset_names \
    yahoo_answers_topics \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 16 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m100/n200/gpt2/gpt2_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n200_gpt2_2 \
--dataset_names \
    blog_authorship_corpus \
    aladar/craigslist_bargains \
    SetFit/enron_spam \
    movie_rationales \
    yelp_review_full \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m100/n200/gpt2/gpt2_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n200_gpt2_3 \
--dataset_names \
    classla/FRENK-hate-en \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m100/n200/gpt2/gpt2_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n200_gpt2_4 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m100/n200/gpt2/gpt2_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n200_gpt2_5 \
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
--num_train_epochs_pretrain 1


# experiments/m100/n200/gpt2/gpt2_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n200_gpt2_6 \
--dataset_names \
    app_reviews \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m100/n50/bert/bert_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n50_bert_1 \
--dataset_names \
    yahoo_answers_topics \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n50/bert/bert_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n50_bert_2 \
--dataset_names \
    SetFit/enron_spam \
    yelp_review_full \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n50/bert/bert_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n50_bert_3 \
--dataset_names \
    classla/FRENK-hate-en \
    blog_authorship_corpus \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n50/bert/bert_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n50_bert_4 \
--dataset_names \
    aladar/craigslist_bargains \
    movie_rationales \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n50/bert/bert_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n50_bert_5 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    app_reviews \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m100/n50/bert/bert_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n50_bert_6 \
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
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m100/n50/gpt2/gpt2_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n50_gpt2_1 \
--dataset_names \
    yahoo_answers_topics \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 16 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m100/n50/gpt2/gpt2_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n50_gpt2_2 \
--dataset_names \
    blog_authorship_corpus \
    aladar/craigslist_bargains \
    SetFit/enron_spam \
    movie_rationales \
    yelp_review_full \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m100/n50/gpt2/gpt2_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n50_gpt2_3 \
--dataset_names \
    classla/FRENK-hate-en \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m100/n50/gpt2/gpt2_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n50_gpt2_4 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m100/n50/gpt2/gpt2_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n50_gpt2_5 \
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
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m100/n50/gpt2/gpt2_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n50_gpt2_6 \
--dataset_names \
    app_reviews \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m100/n500/bert/bert_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n500_bert_1 \
--dataset_names \
    yahoo_answers_topics \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n500/bert/bert_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n500_bert_2 \
--dataset_names \
    SetFit/enron_spam \
    yelp_review_full \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n500/bert/bert_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n500_bert_3 \
--dataset_names \
    classla/FRENK-hate-en \
    blog_authorship_corpus \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m100/n500/bert/bert_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n500_bert_4 \
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


# experiments/m100/n500/bert/bert_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n500_bert_5 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    app_reviews \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m100/n500/bert/bert_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m100_n500_bert_6 \
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
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m100/n500/gpt2/gpt2_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n500_gpt2_1 \
--dataset_names \
    yahoo_answers_topics \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 16 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m100/n500/gpt2/gpt2_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n500_gpt2_2 \
--dataset_names \
    blog_authorship_corpus \
    aladar/craigslist_bargains \
    SetFit/enron_spam \
    movie_rationales \
    yelp_review_full \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m100/n500/gpt2/gpt2_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n500_gpt2_3 \
--dataset_names \
    classla/FRENK-hate-en \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m100/n500/gpt2/gpt2_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n500_gpt2_4 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m100/n500/gpt2/gpt2_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n500_gpt2_5 \
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
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m100/n500/gpt2/gpt2_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m100_n500_gpt2_6 \
--dataset_names \
    app_reviews \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m50/n100/bert/bert_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n100_bert_1 \
--dataset_names \
    yahoo_answers_topics \
--num_train 50 \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n100/bert/bert_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n100_bert_2 \
--dataset_names \
    SetFit/enron_spam \
    yelp_review_full \
--num_train 50 \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n100/bert/bert_3.sh#!/bin/bash
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


# experiments/m50/n100/bert/bert_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n100_bert_4 \
--dataset_names \
    aladar/craigslist_bargains \
    movie_rationales \
--num_train 50 \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n100/bert/bert_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n100_bert_5 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    app_reviews \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_train 50 \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m50/n100/bert/bert_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n100_bert_6 \
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
--num_train 50 \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m50/n100/gpt2/gpt2_1.sh#!/bin/bash
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
--num_train_epochs_pretrain 1


# experiments/m50/n100/gpt2/gpt2_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n100_gpt2_2 \
--dataset_names \
    blog_authorship_corpus \
    aladar/craigslist_bargains \
    SetFit/enron_spam \
    movie_rationales \
    yelp_review_full \
--num_train 50 \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m50/n100/gpt2/gpt2_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n100_gpt2_3 \
--dataset_names \
    classla/FRENK-hate-en \
--num_train 50 \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m50/n100/gpt2/gpt2_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n100_gpt2_4 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_train 50 \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m50/n100/gpt2/gpt2_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n100_gpt2_5 \
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
--num_train 50 \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m50/n100/gpt2/gpt2_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n100_gpt2_6 \
--dataset_names \
    app_reviews \
--num_train 50 \
--num_test 100 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m50/n200/bert/bert_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n200_bert_1 \
--dataset_names \
    yahoo_answers_topics \
--num_train 50 \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n200/bert/bert_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n200_bert_2 \
--dataset_names \
    SetFit/enron_spam \
    yelp_review_full \
--num_train 50 \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n200/bert/bert_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n200_bert_3 \
--dataset_names \
    classla/FRENK-hate-en \
    blog_authorship_corpus \
--num_train 50 \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n200/bert/bert_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n200_bert_4 \
--dataset_names \
    aladar/craigslist_bargains \
    movie_rationales \
--num_train 50 \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n200/bert/bert_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n200_bert_5 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    app_reviews \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_train 50 \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m50/n200/bert/bert_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n200_bert_6 \
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
--num_train 50 \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m50/n200/gpt2/gpt2_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n200_gpt2_1 \
--dataset_names \
    yahoo_answers_topics \
--num_train 50 \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 16 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m50/n200/gpt2/gpt2_2.sh#!/bin/bash
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
--num_train_epochs_pretrain 1


# experiments/m50/n200/gpt2/gpt2_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n200_gpt2_3 \
--dataset_names \
    classla/FRENK-hate-en \
--num_train 50 \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m50/n200/gpt2/gpt2_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n200_gpt2_4 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_train 50 \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m50/n200/gpt2/gpt2_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n200_gpt2_5 \
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
--num_train 50 \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m50/n200/gpt2/gpt2_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n200_gpt2_6 \
--dataset_names \
    app_reviews \
--num_train 50 \
--num_test 200 \
--num_subsamples 50 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m50/n50/bert/bert_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n50_bert_1 \
--dataset_names \
    yahoo_answers_topics \
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n50/bert/bert_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n50_bert_2 \
--dataset_names \
    SetFit/enron_spam \
    yelp_review_full \
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n50/bert/bert_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n50_bert_3 \
--dataset_names \
    classla/FRENK-hate-en \
    blog_authorship_corpus \
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n50/bert/bert_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n50_bert_4 \
--dataset_names \
    aladar/craigslist_bargains \
    movie_rationales \
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n50/bert/bert_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n50_bert_5 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    app_reviews \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m50/n50/bert/bert_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n50_bert_6 \
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
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m50/n50/gpt2/gpt2_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n50_gpt2_1 \
--dataset_names \
    yahoo_answers_topics \
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 16 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m50/n50/gpt2/gpt2_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n50_gpt2_2 \
--dataset_names \
    blog_authorship_corpus \
    aladar/craigslist_bargains \
    SetFit/enron_spam \
    movie_rationales \
    yelp_review_full \
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m50/n50/gpt2/gpt2_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n50_gpt2_3 \
--dataset_names \
    classla/FRENK-hate-en \
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m50/n50/gpt2/gpt2_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n50_gpt2_4 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m50/n50/gpt2/gpt2_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n50_gpt2_5 \
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
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m50/n50/gpt2/gpt2_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n50_gpt2_6 \
--dataset_names \
    app_reviews \
--num_train 50 \
--num_test 50 \
--num_subsamples 100 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m50/n500/bert/bert_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n500_bert_1 \
--dataset_names \
    yahoo_answers_topics \
--num_train 50 \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n500/bert/bert_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n500_bert_2 \
--dataset_names \
    SetFit/enron_spam \
    yelp_review_full \
--num_train 50 \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n500/bert/bert_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n500_bert_3 \
--dataset_names \
    classla/FRENK-hate-en \
    blog_authorship_corpus \
--num_train 50 \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n500/bert/bert_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n500_bert_4 \
--dataset_names \
    aladar/craigslist_bargains \
    movie_rationales \
--num_train 50 \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


# experiments/m50/n500/bert/bert_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n500_bert_5 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    app_reviews \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_train 50 \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m50/n500/bert/bert_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name m50_n500_bert_6 \
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
--num_train 50 \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2


# experiments/m50/n500/gpt2/gpt2_1.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n500_gpt2_1 \
--dataset_names \
    yahoo_answers_topics \
--num_train 50 \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 16 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m50/n500/gpt2/gpt2_2.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n500_gpt2_2 \
--dataset_names \
    blog_authorship_corpus \
    aladar/craigslist_bargains \
    SetFit/enron_spam \
    movie_rationales \
    yelp_review_full \
--num_train 50 \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m50/n500/gpt2/gpt2_3.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n500_gpt2_3 \
--dataset_names \
    classla/FRENK-hate-en \
--num_train 50 \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


# experiments/m50/n500/gpt2/gpt2_4.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n500_gpt2_4 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    aladar/emo \
    dair-ai/emotion \
    financial_phrasebank \
--num_train 50 \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m50/n500/gpt2/gpt2_5.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n500_gpt2_5 \
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
--num_train 50 \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


# experiments/m50/n500/gpt2/gpt2_6.sh#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type gpt2 \
--run_name m50_n500_gpt2_6 \
--dataset_names \
    app_reviews \
--num_train 50 \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 1


