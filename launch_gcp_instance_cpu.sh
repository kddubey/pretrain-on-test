#!/bin/bash
PROJECT_NAME="virtual-equator-423819-v6"  # you'll change this

gcloud compute instances create instance-pretrain-on-test-cpu-testing \
    --project=$PROJECT_NAME \
    --zone=us-central1-a \
    --machine-type=e2-standard-2 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --service-account=$SERVICE_ACCOUNT_EMAIL \
    --scopes=https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/devstorage.read_write \
    --create-disk=auto-delete=yes,boot=yes,device-name=instance-pretrain-on-test-cpu-testing,image=projects/debian-cloud/global/images/debian-12-bookworm-v20240515,mode=rw,size=80,type=projects/virtual-equator-423819-v6/zones/us-central1-a/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any

gcloud compute ssh \
    --zone "us-central1-a" \
    "instance-pretrain-on-test-cpu-testing" \
    --project $PROJECT_NAME
