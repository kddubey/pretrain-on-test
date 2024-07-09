#!/bin/bash


set -uox pipefail
# https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
# No e b/c want to shut down regardless of success or failure


sudo apt-get update
sudo apt-get -y install apache2
cat <<EOF > /var/www/html/index.html
<html><body><p>Linux startup script from a local file.</p></body></html>
EOF
# preamble taken from
# https://cloud.google.com/compute/docs/instances/startup-scripts/linux#passing-local


export PRETRAIN_ON_TEST_CLOUD_PROVIDER="gcp"
export HF_TOKEN=$(gcloud secrets versions access 1 --secret="HF_TOKEN")


set +x
source ~/.bashrc
set -x
# W/o this, the GPU image says conda and venv don't exist. Probably has to do w/ not
# reloading:
# Error reloading service: Failed to reload-or-restart sshd.service: Unit sshd.service not found..
