#!/bin/bash
echo ""
echo "View raw logs here (cleaner logs are in the Logs Explorer page):"
echo "https://console.cloud.google.com/compute/instancesDetail/zones/$ZONE/instances/$INSTANCE_NAME?project=$PROJECT_NAME&tab=monitoring&pageState=(%22timeRange%22:(%22duration%22:%22PT1H%22),%22observabilityTab%22:(%22mainContent%22:%22logs%22))"
echo ""
echo "To SSH into the instance:"
echo gcloud compute ssh --zone \"$ZONE\" \"$INSTANCE_NAME\" --project \"$PROJECT_NAME\"
