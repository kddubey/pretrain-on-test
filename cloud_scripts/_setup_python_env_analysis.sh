#!/bin/bash
# This script sets up the Python environment required to run the analysis.
# It can be run on any Debian instance with the env var PRETRAIN_ON_TEST_CLOUD_PROVIDER


set -uox pipefail
# https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
# No e b/c want to shut down regardless of success or failure


echo "Cloud provider: $PRETRAIN_ON_TEST_CLOUD_PROVIDER"  # must be set


check_if_no_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi -L &> /dev/null; then
            return 1  # exit status 1 is Falsy; GPU detected
        fi
    fi
    return 0  # No GPU detected
}


if check_if_no_gpu; then
    echo "No GPU detected."
else
    echo "GPU detected."
fi


# Set up Python env. GCP's GPU image can't support venv easily, only conda.
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

# Don't install torch's nvidia deps if there's no GPU
if check_if_no_gpu; then
    echo "No GPU detected. Installing CPU version of PyTorch."
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "GPU detected."
fi

python -m pip install ".[stat]"

# Set up for cloud
python -m pip install ".[$PRETRAIN_ON_TEST_CLOUD_PROVIDER]"


# Utility to activate env
activate_pretrain_env() {
    if command -v conda &> /dev/null; then
        conda activate pretrain-env
    else
        source ~/pretrain-on-test/pretrain-env/bin/activate
    fi
}

export -f activate_pretrain_env
