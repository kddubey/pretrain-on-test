# This will write to a new directory: ./accuracies
# Hyperparameters are set to run on a single T4 GPU (15 GB VRAM)
# They vary by dataset b/c some datasets have much longer text than others
# They vary by the LM type b/c different architectures cause differences in peak memory
# For n=500, the number of pretraining epochs is reduced from 2 to 1 to save time


########################################################################################


# n=200, BERT
python run.py \
--results_dir "accuracies/200/bert" \
--lm_type bert \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    app_reviews \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    emo \
    dair-ai/emotion \
    financial_phrasebank \
    hyperpartisan_news_detection \
    limit \
    AmazonScience/massive \
    mteb/mtop_domain \
    ccdv/patent-classification \
    rotten_tomatoes \
    silicone \
    trec \
    tweets_hate_speech_detection \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2

python run.py \
--results_dir "accuracies/200/bert" \
--lm_type bert \
--dataset_names \
    blog_authorship_corpus \
    craigslist_bargains \
    classla/FRENK-hate-en \
    movie_rationales \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2

python run.py \
--results_dir "accuracies/200/bert" \
--lm_type bert \
--dataset_names \
    enron_spam \
    yelp_review_full \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2

python run.py \
--results_dir "accuracies/200/bert" \
--lm_type bert \
--dataset_names \
    yahoo_answers_topics \
--per_device_train_batch_size_pretrain 8 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


########################################################################################


# n=200, GPT-2
python run.py \
--results_dir "accuracies/200/gpt2" \
--lm_type gpt2 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    app_reviews \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    emo \
    dair-ai/emotion \
    financial_phrasebank \
    hyperpartisan_news_detection \
    limit \
    AmazonScience/massive \
    mteb/mtop_domain \
    ccdv/patent-classification \
    rotten_tomatoes \
    silicone \
    trec \
    tweets_hate_speech_detection \
--per_device_train_batch_size_pretrain 16 \
--per_device_train_batch_size_classification 16 \
--per_device_eval_batch_size_classification 64 \
--num_train_epochs_classification 3 \
--num_train_epochs_pretrain 2

python run.py \
--results_dir "accuracies/200/gpt2" \
--lm_type gpt2 \
--dataset_names \
    blog_authorship_corpus \
    craigslist_bargains \
    classla/FRENK-hate-en \
    movie_rationales \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2

python run.py \
--results_dir "accuracies/200/gpt2" \
--lm_type gpt2 \
--dataset_names \
    enron_spam \
    yelp_review_full \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2

python run.py \
--results_dir "accuracies/200/gpt2" \
--lm_type gpt2 \
--dataset_names \
    yahoo_answers_topics \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 16 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


########################################################################################


# n=500, BERT
python run.py \
--results_dir "accuracies/500/bert" \
--lm_type bert \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    app_reviews \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    emo \
    dair-ai/emotion \
    financial_phrasebank \
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

python run.py \
--results_dir "accuracies/500/bert" \
--lm_type bert \
--dataset_names \
    blog_authorship_corpus \
    craigslist_bargains \
    classla/FRENK-hate-en \
    movie_rationales \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1

python run.py \
--results_dir "accuracies/500/bert" \
--lm_type bert \
--dataset_names \
    enron_spam \
    yelp_review_full \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1

python run.py \
--results_dir "accuracies/500/bert" \
--lm_type bert \
--dataset_names \
    yahoo_answers_topics \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1


########################################################################################


# n=500, GPT-2
python run.py \
--results_dir "accuracies/500/gpt2" \
--lm_type gpt2 \
--dataset_names \
    ag_news \
    SetFit/amazon_counterfactual_en \
    app_reviews \
    christinacdl/clickbait_notclickbait_dataset \
    climate_fever \
    disaster_response_messages \
    emo \
    dair-ai/emotion \
    financial_phrasebank \
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

python run.py \
--results_dir "accuracies/500/gpt2" \
--lm_type gpt2 \
--dataset_names \
    blog_authorship_corpus \
    craigslist_bargains \
    classla/FRENK-hate-en \
    movie_rationales \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1

python run.py \
--results_dir "accuracies/500/gpt2" \
--lm_type gpt2 \
--dataset_names \
    enron_spam \
    yelp_review_full \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 4 \
--per_device_eval_batch_size_classification 32 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1

python run.py \
--results_dir "accuracies/500/gpt2" \
--lm_type gpt2 \
--dataset_names \
    yahoo_answers_topics \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 16 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 1
