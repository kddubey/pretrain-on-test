#!/bin/bash
# May need
sudo apt update
sudo apt install -y python3-pip git python3.11-venv


# Set up venv
python3 -m venv pretrain-env
source pretrain-env/bin/activate
git clone https://github.com/kddubey/pretrain-on-test.git
cd pretrain-on-test
# For CPU, don't install a bunch of nvidia packages:
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install .


# Set up for cloud
python -m pip install ".[gcp]"
export PRETRAIN_ON_TEST_CLOUD_PROVIDER="gcp"
export PRETRAIN_ON_TEST_BUCKET_NAME="pretrain-on-test-accuracies"


# Run experiment
./experiment_mini.sh


# Shut down regardless of success or failure
sudo shutdown -h now
