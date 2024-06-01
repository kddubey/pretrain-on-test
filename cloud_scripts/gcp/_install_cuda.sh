#!/bin/bash
# I don't know why I ended up needing this. W/o it, CUDA drivers don't get installed. I
# think it might have something to do w/ this error I see in the logs:
# Error reloading service: Failed to reload-or-restart sshd.service: Unit sshd.service not found..


set -uox
# https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425


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
