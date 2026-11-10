#!/usr/bin/env bash
# Deploys the official vLLM image with Gemma on Cloud Run with one NVIDIA L4 GPU.
# On-demand: min-instances 0, so the GPU is billed only while a request keeps it awake.
#
#   GCP_PROJECT_ID=my-project bash gcp/deploy_vllm.sh
#
# Then open a local tunnel (no public URL, no API key in your code):
#   gcloud run services proxy vllm-gemma --region us-east4 --port 8000
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to your Google Cloud project id}"
REGION="${GCP_REGION:-us-east4}"            # a region with Cloud Run L4 quota
SERVICE="${VLLM_SERVICE:-vllm-gemma}"
IMAGE="${VLLM_IMAGE:-docker.io/vllm/vllm-openai:v0.33.0}"
MODEL="${VLLM_MODEL:-google/gemma-4-E4B-it}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
BUCKET="${WEIGHTS_BUCKET:-${PROJECT_ID}-vllm-weights}"

# Weights: from the bucket filled by gcp/stage_weights.sh when it exists (fast, predictable
# cold starts), otherwise straight from Hugging Face (simplest, but anonymous downloads are
# throttled and a cold start can take several minutes).
SOURCE_FLAGS=()
MODEL_ARG="${MODEL}"
if gcloud storage ls "gs://${BUCKET}/${MODEL}/config.json" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  SOURCE_FLAGS=(--add-volume "name=weights,type=cloud-storage,bucket=${BUCKET},readonly=true"
                --add-volume-mount "volume=weights,mount-path=/models")
  MODEL_ARG="/models/${MODEL}"
  echo "Weights from gs://${BUCKET}/${MODEL}"
else
  echo "Weights from Hugging Face (${MODEL}); run gcp/stage_weights.sh for faster cold starts"
fi

gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" --region "${REGION}" \
  --image "${IMAGE}" \
  --args="--model=${MODEL_ARG},--served-model-name=gemma,--max-model-len=${MAX_LEN},--port=8080,--host=0.0.0.0" \
  --port 8080 \
  ${SOURCE_FLAGS[@]+"${SOURCE_FLAGS[@]}"} \
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu 8 --memory 32Gi --no-cpu-throttling \
  --min-instances 0 --max-instances 1 --concurrency 64 --timeout 900 \
  --startup-probe "httpGet.path=/health,httpGet.port=8080,periodSeconds=10,failureThreshold=150,timeoutSeconds=5" \
  --no-allow-unauthenticated

gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" \
  --format 'value(status.url)'
