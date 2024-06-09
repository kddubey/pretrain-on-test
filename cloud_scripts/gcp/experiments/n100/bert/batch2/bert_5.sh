#!/bin/bash
TQDM_DISABLE=1 python run.py \
--lm_type bert \
--run_name n100_bert_5 \
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
