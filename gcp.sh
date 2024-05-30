#! /bin/bash
sudo apt update
sudo apt -y install apache2
cat <<EOF > /var/www/html/index.html
<html><body><p>Linux startup script from a local file.</p></body></html>
EOF
# pre-amble taken from
# https://cloud.google.com/compute/docs/instances/startup-scripts/linux#passing-local


# TODO: check that this returns 1 on the GPU instance
no_gpu_detected() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi -L &> /dev/null; then
            return 1  # exit status 1 is Falsy; GPU detected
        fi
    fi
    return 0  # No GPU detected
}


# TODO: check if ok for GPU image
sudo apt install -y python3-pip git python3.11-venv


# Set up venv
python3 -m venv pretrain-env
source pretrain-env/bin/activate
python -m pip install wheel
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
python -m pip install ".[gcp]"
export PRETRAIN_ON_TEST_CLOUD_PROVIDER="gcp"
export PRETRAIN_ON_TEST_BUCKET_NAME="pretrain-on-test-accuracies"


# Run experiment
if no_gpu_detected; then
    echo "Running experiment_mini.sh"
    ./experiment_mini.sh
else
    echo "Running experiment.sh"
    ./experiment.sh
fi


# Shut down regardless of success or failure
sudo shutdown -h now
