#!/bin/bash
# TODO: add PRETRAIN_ON_TEST_BUCKET_NAME as an optional argument and export it
# TODO: maybe make these arguments
PROJECT_NAME=$(gcloud config get-value project)  # you may need to set this manually
INSTANCE_NAME="instance-pretrain-on-test-cpu-testing"
ZONE="us-central1-a"

gcloud beta compute instances create $INSTANCE_NAME \
    --project=$PROJECT_NAME \
    --zone=$ZONE \
    --machine-type=e2-standard-2 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --max-run-duration=7200s \
    --service-account=$SERVICE_ACCOUNT_EMAIL \
    --scopes=https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/devstorage.read_write \
    --create-disk=auto-delete=yes,boot=yes,device-name=$INSTANCE_NAME,image=projects/debian-cloud/global/images/debian-12-bookworm-v20240515,mode=rw,size=80,type=projects/$PROJECT_NAME/zones/$ZONE/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any \
    --metadata-from-file=startup-script=./gcp.sh

echo ""
echo "View raw logs here (cleaner logs are in the Logs Explorer page):"
echo "https://console.cloud.google.com/compute/instancesDetail/zones/$ZONE/instances/$INSTANCE_NAME?project=$PROJECT_NAME&tab=monitoring&pageState=(%22timeRange%22:(%22duration%22:%22PT1H%22),%22observabilityTab%22:(%22mainContent%22:%22logs%22))"

echo ""
echo "To SSH into the instance:"
echo "gcloud compute ssh --zone $ZONE $INSTANCE_NAME --project $PROJECT_NAME"
