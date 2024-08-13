#!/bin/bash
TQDM_DISABLE=1 python run.py \
    --lm_type mistral-qlora-zero-shot \
    --run_name n100_mistral-qlora-zero-shot-packing_6 \
    --dataset_names \
        ccdv/patent-classification \
        rotten_tomatoes \
        silicone \
        trec \
        tweets_hate_speech_detection \
    --num_test 100 \
    --num_subsamples 20 \
    --per_device_train_batch_size_pretrain 8 \
    --per_device_eval_batch_size_classification 16 \
    --num_train_epochs_pretrain 1
