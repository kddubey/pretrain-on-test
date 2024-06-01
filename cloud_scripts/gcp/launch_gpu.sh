#!/bin/bash
# This script launches a CPU instance which runs a mini experiment to check that your
# cloud setup works.


set -euo pipefail
# https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425


PROJECT_NAME=$(gcloud config get-value project)  # you may need to set this manually
INSTANCE_NAME="instance-pretrain-on-test-gpu"
ZONE="us-west4-a"


cat ./_install_cuda.sh ./_preamble.sh > _preamble_gpu.sh
cat _preamble_gpu.sh ../run.sh > run_gcp.sh
rm _preamble_gpu.sh


gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_NAME \
    --zone=$ZONE \
    --machine-type=n1-highmem-2 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=$SERVICE_ACCOUNT_EMAIL \
    --scopes=https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/devstorage.read_write \
    --accelerator=count=1,type=nvidia-tesla-t4 \
    --create-disk=auto-delete=yes,boot=yes,device-name=$INSTANCE_NAME,image=projects/ml-images/global/images/c2-deeplearning-pytorch-2-2-cu121-v20240514-debian-11,mode=rw,size=80,type=projects/$PROJECT_NAME/zones/$ZONE/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any \
    --metadata-from-file=startup-script=./run_gcp.sh


rm run_gcp.sh


PROJECT_NAME=$PROJECT_NAME INSTANCE_NAME=$INSTANCE_NAME ZONE=$ZONE ./_post_launch.sh
