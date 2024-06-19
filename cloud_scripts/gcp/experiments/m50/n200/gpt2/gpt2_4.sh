#!/bin/bash
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
