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


# Run experiment
./experiment.sh


# Shutdown regardless of success or failure
shutdown -h now
