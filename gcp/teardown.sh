#!/usr/bin/env bash
# Deletes the Cloud Run service. With min-instances 0 an idle service costs nothing,
# but deleting it is the only way to be sure.
set -euo pipefail
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to your Google Cloud project id}"
REGION="${GCP_REGION:-us-east4}"
SERVICE="${VLLM_SERVICE:-vllm-gemma}"
gcloud run services delete "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --quiet
