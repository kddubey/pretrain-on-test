#!/bin/bash


set -uox pipefail  # No e b/c want to shut down regardless of success or failure


sudo apt-get update
sudo apt-get -y install apache2
cat <<EOF > /var/www/html/index.html
<html><body><p>Linux startup script from a local file.</p></body></html>
EOF
# preamble taken from
# https://cloud.google.com/compute/docs/instances/startup-scripts/linux#passing-local


export PRETRAIN_ON_TEST_CLOUD_PROVIDER="gcp"


source ~/.bashrc
# W/o this, the GPU image says conda and venv don't exist. Probably has to do w/ not
# reloading:
# Error reloading service: Failed to reload-or-restart sshd.service: Unit sshd.service not found..
