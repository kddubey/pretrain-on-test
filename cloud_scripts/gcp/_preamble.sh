#!/bin/bash
set -uo pipefail  # No e b/c want to shut down regardless of success or failure

sudo apt update
sudo apt -y install apache2
cat <<EOF > /var/www/html/index.html
<html><body><p>Linux startup script from a local file.</p></body></html>
EOF
# pre-amble taken from
# https://cloud.google.com/compute/docs/instances/startup-scripts/linux#passing-local

export PRETRAIN_ON_TEST_CLOUD_PROVIDER="gcp"
