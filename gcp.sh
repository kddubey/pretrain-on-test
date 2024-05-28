# Set up venv
python3 -m venv pretrain

source pretrain/bin/activate

git clone https://github.com/kddubey/pretrain-on-test.git

cd pretrain-on-test

python -m pip install .


# Set up for cloud
python -m pip install ".[gcp]"
export PRETRAIN_ON_TEST_CLOUD_PROVIDER="gcp"
export PRETRAIN_ON_TEST_BUCKET_NAME="pretrain-on-test-accuracies"


########################################################################################


# n=500, BERT
python run.py \
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
--num_train_epochs_pretrain 2

python run.py \
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
--num_train_epochs_pretrain 2

python run.py \
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
--num_train_epochs_pretrain 2

python run.py \
--lm_type bert \
--dataset_names \
    yahoo_answers_topics \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


########################################################################################


# n=500, GPT-2
python run.py \
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
--num_train_epochs_pretrain 2

python run.py \
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
--num_train_epochs_pretrain 2

python run.py \
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
--num_train_epochs_pretrain 2

python run.py \
--lm_type gpt2 \
--dataset_names \
    yahoo_answers_topics \
--num_test 500 \
--num_subsamples 20 \
--per_device_train_batch_size_pretrain 8 \
--per_device_train_batch_size_classification 8 \
--per_device_eval_batch_size_classification 16 \
--num_train_epochs_classification 1 \
--num_train_epochs_pretrain 2


########################################################################################


shutdown -h now
