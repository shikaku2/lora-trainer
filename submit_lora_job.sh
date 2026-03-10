#!/usr/bin/env bash
# submit_lora_job.sh
# Encodes your JSONL + SSH key and fires the RunPod serverless job.
#
# Usage:
#   chmod +x submit_lora_job.sh
#   RUNPOD_API_KEY=rp_xxx RUNPOD_ENDPOINT_ID=abc123 ./submit_lora_job.sh
#
# Required env vars:
#   RUNPOD_API_KEY        your RunPod API key
#   RUNPOD_ENDPOINT_ID    the serverless endpoint ID
#
# Optional env vars (with defaults):
#   JSONL_FILE            path to training JSONL        [alastor_train.jsonl]
#   SSH_KEY_FILE          path to your private key      [~/.ssh/id_ed25519]
#   SSH_HOST              your home IP/hostname         [required]
#   SSH_PORT              SSH port                      [22]
#   SSH_USER              SSH username                  [$(whoami)]
#   SSH_DEST              remote path for output        [~/alastor-lora.tar.gz]
#   MODEL_PATH            HF repo or local path         [mistralai/Magistral-Small-2509]
#   EPOCHS                training epochs               [3]
#   RANK                  LoRA rank                     [16]

set -euo pipefail

: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
: "${RUNPOD_ENDPOINT_ID:?Set RUNPOD_ENDPOINT_ID}"
: "${SSH_HOST:?Set SSH_HOST to your home IP/hostname}"

JSONL_FILE="${JSONL_FILE:-alastor_train.jsonl}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/id_ed25519}"
SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-$(whoami)}"
SSH_DEST="${SSH_DEST:-$HOME/alastor-lora.tar.gz}"
MODEL_PATH="${MODEL_PATH:-mistralai/Magistral-Small-2509}"
EPOCHS="${EPOCHS:-3}"
RANK="${RANK:-16}"

echo "Encoding JSONL: $JSONL_FILE"
JSONL_B64=$(base64 -w 0 "$JSONL_FILE")

echo "Reading SSH key: $SSH_KEY_FILE"
SSH_KEY=$(cat "$SSH_KEY_FILE")

PAYLOAD=$(jq -n \
  --arg jsonl_b64  "$JSONL_B64" \
  --arg ssh_host   "$SSH_HOST" \
  --argjson ssh_port "$SSH_PORT" \
  --arg ssh_user   "$SSH_USER" \
  --arg ssh_key    "$SSH_KEY" \
  --arg ssh_dest   "$SSH_DEST" \
  --arg model_path "$MODEL_PATH" \
  --argjson epochs "$EPOCHS" \
  --argjson rank   "$RANK" \
  '{input: {
      jsonl_b64:  $jsonl_b64,
      ssh_host:   $ssh_host,
      ssh_port:   $ssh_port,
      ssh_user:   $ssh_user,
      ssh_key:    $ssh_key,
      ssh_dest:   $ssh_dest,
      model_path: $model_path,
      epochs:     $epochs,
      rank:       $rank
  }}')

echo "Submitting job to endpoint $RUNPOD_ENDPOINT_ID..."
RESPONSE=$(curl -s -X POST \
  "https://api.runpod.io/v2/$RUNPOD_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d "$PAYLOAD")

echo ""
echo "=== Response ==="
echo "$RESPONSE" | jq .
