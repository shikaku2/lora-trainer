#!/usr/bin/env python3
"""
submit_lora_job.py

Usage:
  RUNPOD_API_KEY=rp_xxx RUNPOD_ENDPOINT_ID=abc123 HF_WRITE_TOKEN=hf_xxx python3 submit_lora_job.py

Required env vars:
  RUNPOD_API_KEY        your RunPod API key
  RUNPOD_ENDPOINT_ID    the serverless endpoint ID
  HF_WRITE_TOKEN        HuggingFace write token

Optional env vars (with defaults):
  JSONL_FILE            path to training JSONL        [alastor_train.jsonl]
  HF_REPO               HuggingFace repo ID           [shikaku2/magistral-alastor-lora]
  MODEL_PATH            HF repo                       [unsloth/Magistral-Small-2509]
  EPOCHS                training epochs               [3]
  RANK                  LoRA rank                     [16]
"""

import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

def env(key, default=None, required=False):
    val = os.environ.get(key, default)
    if required and not val:
        print(f"ERROR: {key} is required.")
        sys.exit(1)
    return val

api_key     = env("RUNPOD_API_KEY",     required=True)
endpoint_id = env("RUNPOD_ENDPOINT_ID", required=True)
hf_token    = env("HF_WRITE_TOKEN",     required=True)

jsonl_file  = env("JSONL_FILE",  "alastor_train.jsonl")
hf_repo     = env("HF_REPO",    "shikaku2/magistral-alastor-lora")
model_path  = env("MODEL_PATH", "unsloth/Magistral-Small-2509")
epochs      = int(env("EPOCHS", "3"))
rank        = int(env("RANK",   "16"))

print(f"Encoding JSONL: {jsonl_file}")
jsonl_b64 = base64.b64encode(Path(jsonl_file).read_bytes()).decode()

payload = json.dumps({"input": {
    "jsonl_b64":  jsonl_b64,
    "hf_token":   hf_token,
    "hf_repo":    hf_repo,
    "model_path": model_path,
    "epochs":     epochs,
    "rank":       rank,
}}).encode()

HEADERS = {
    "Content-Type":  "application/json",
    "Authorization": f"Bearer {api_key}",
    "User-Agent":    "Mozilla/5.0",
}

def api(method, path, data=None):
    req = urllib.request.Request(
        f"https://api.runpod.ai/v2/{endpoint_id}/{path}",
        data=data,
        headers=HEADERS,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)

print(f"Payload size: {len(payload) / 1024:.1f} KB")
print(f"Submitting job to endpoint {endpoint_id}...")
result = api("POST", "run", data=payload)
job_id = result.get("id")
if not job_id:
    print(f"ERROR: no job id in response: {result}")
    sys.exit(1)
print(f"Job submitted: {job_id}")

start = time.time()
while True:
    time.sleep(15)
    elapsed = int(time.time() - start)
    status = api("GET", f"status/{job_id}")
    state = status.get("status", "UNKNOWN")
    print(f"[{elapsed:4d}s] {state}")

    if state == "COMPLETED":
        print("\n=== Response ===")
        print(json.dumps(status.get("output", status), indent=2))
        break
    elif state in ("FAILED", "CANCELLED", "TIMED_OUT"):
        print("\n=== FAILED ===")
        print(json.dumps(status, indent=2))
        sys.exit(1)
