#!/usr/bin/env bash
# Copies the model weights from Hugging Face into a Cloud Storage bucket in the same region
# as the GPU, once. Cloud Build does the download inside Google's network, so your laptop
# never moves 16 GB. After this, every cold start reads the weights from the bucket.
#
#   GCP_PROJECT_ID=my-project bash gcp/stage_weights.sh
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to your Google Cloud project id}"
REGION="${GCP_REGION:-us-east4}"
MODEL="${VLLM_MODEL:-google/gemma-4-E4B-it}"
BUCKET="${WEIGHTS_BUCKET:-${PROJECT_ID}-vllm-weights}"
DEST="gs://${BUCKET}/${MODEL}"

gcloud storage buckets describe "gs://${BUCKET}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUCKET}" --project "${PROJECT_ID}" --location "${REGION}" \
    --uniform-bucket-level-access

cat > /tmp/stage_weights_cloudbuild.yaml <<YAML
steps:
- name: python:3.12-slim
  entrypoint: bash
  args: ["-c", "pip install -q huggingface_hub && hf download ${MODEL} --local-dir /workspace/model --exclude '*.gguf'"]
- name: gcr.io/google.com/cloudsdktool/cloud-sdk:slim
  entrypoint: bash
  args: ["-c", "gcloud storage cp -r /workspace/model/* ${DEST}/"]
timeout: 3600s
options:
  machineType: E2_HIGHCPU_8
  diskSizeGb: 100
YAML
gcloud builds submit --no-source --project "${PROJECT_ID}" --region "${REGION}" \
  --config /tmp/stage_weights_cloudbuild.yaml
gcloud storage ls "${DEST}/"
