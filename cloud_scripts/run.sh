#!/bin/bash
# This startup / user data script can be run on any Linux instance.
# It sets up the Python env, runs the experiment, and shuts down the instance.


set -uox pipefail
# https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
# No e b/c want to shut down regardless of success or failure


no_gpu_detected() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi -L &> /dev/null; then
            return 1  # exit status 1 is Falsy; GPU detected
        fi
    fi
    return 0  # No GPU detected
}


if no_gpu_detected; then
    echo "No GPU detected."
else
    echo "GPU detected."
fi


sudo apt-get update
sudo apt-get install -y git


# Set up env. GCP's GPU image can't support venv easily, only conda.
if command -v conda &> /dev/null; then
    echo "Creating a new conda environment"
    conda deactivate
    conda create -y -n pretrain-env python=3.10
    conda activate pretrain-env
else
    sudo apt-get install -y python3-pip python3.11-venv
    echo "Creating a new Python virtual environment"
    python3 -m venv pretrain-env
    source pretrain-env/bin/activate
    python -m pip install wheel
fi

git clone https://github.com/kddubey/pretrain-on-test.git
cd pretrain-on-test

# Don't install torch's nvidia deps if there's no GPU
if no_gpu_detected; then
    echo "No GPU detected. Installing CPU version of PyTorch."
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "GPU detected."
fi

python -m pip install .

# Set up for cloud
python -m pip install ".[$PRETRAIN_ON_TEST_CLOUD_PROVIDER]"
export PRETRAIN_ON_TEST_BUCKET_NAME="pretrain-on-test-accuracies"


# Run experiment
if no_gpu_detected; then
    echo "Running experiment_mini.sh"
    screen -dmS experiment ./experiment_mini.sh
else
    echo "Running experiment.sh"
    screen -dmS experiment ./experiment.sh
fi


# Shut down regardless of success or failure
sudo shutdown -h now
