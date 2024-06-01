#!/bin/bash
# I don't know why I need this. W/o it, CUDA drivers don't get installed. I think it
# might have something to do w/ this error?
# Error reloading service: Failed to reload-or-restart sshd.service: Unit sshd.service not found..


set -uox

# sudo apt-get -y remove --purge openssh-server
# sudo apt-get update
# sudo apt-get -y install openssh-server

# Rest of script is from here:
# https://github.com/GoogleCloudPlatform/compute-gpu-installation/tree/main/linux

if test -f /opt/google/cuda-installer
then
    exit
fi

mkdir -p /opt/google/cuda-installer/
cd /opt/google/cuda-installer/ || exit

curl -fSsL -O https://github.com/GoogleCloudPlatform/compute-gpu-installation/releases/download/cuda-installer-v1.0.0/cuda_installer.pyz
sudo python3 cuda_installer.pyz install_cuda
