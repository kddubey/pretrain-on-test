#!/bin/bash
# May need
sudo apt update
sudo apt install -y python3-pip git python3.11-venv


# Set up venv
python3 -m venv pretrain-env
source pretrain-env/bin/activate
python -m pip install wheel
git clone https://github.com/kddubey/pretrain-on-test.git
cd pretrain-on-test


# Don't waste time installing nvidia packages if there's no GPU
no_gpu_detected() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi -L &> /dev/null; then
            return 1  # exit status 1 is Falsy; GPU detected
        fi
    fi
    return 0  # No GPU detected
}

if no_gpu_detected; then
    echo "No GPU detected. Installing CPU version of PyTorch."
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "GPU detected. Installing default version."
    python -m pip install .
fi


# Set up for cloud
python -m pip install ".[gcp]"
export PRETRAIN_ON_TEST_CLOUD_PROVIDER="gcp"
export PRETRAIN_ON_TEST_BUCKET_NAME="pretrain-on-test-accuracies"


# Run experiment
./experiment_mini.sh


# Shut down regardless of success or failure
sudo shutdown -h now
