#!/usr/bin/env python3
"""
submit_lora_job.py

Usage:
  RUNPOD_API_KEY=rp_xxx RUNPOD_ENDPOINT_ID=abc123 SSH_HOST=ssh.kung.pw python3 submit_lora_job.py

Required env vars:
  RUNPOD_API_KEY        your RunPod API key
  RUNPOD_ENDPOINT_ID    the serverless endpoint ID
  SSH_HOST              your server hostname/IP

Optional env vars (with defaults):
  JSONL_FILE            path to training JSONL        [alastor_train.jsonl]
  SSH_KEY_FILE          path to your private key      [~/.ssh/id_ed25519]
  SSH_PORT              SSH port                      [12369]
  SSH_USER              SSH username                  [current user]
  SSH_DEST              remote path for output        [/home/<user>/alastor-lora.tar.gz]
  MODEL_PATH            HF repo                       [unsloth/Magistral-Small-2509]
  EPOCHS                training epochs               [3]
  RANK                  LoRA rank                     [16]
"""

import base64
import getpass
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
ssh_host    = env("SSH_HOST",           required=True)

jsonl_file   = env("JSONL_FILE",   "alastor_train.jsonl")
ssh_key_file = env("SSH_KEY_FILE", str(Path.home() / ".ssh" / "id_ed25519"))
ssh_port     = int(env("SSH_PORT", "22"))
ssh_user     = env("SSH_USER",     getpass.getuser())
ssh_dest     = env("SSH_DEST",     f"/home/{ssh_user}/alastor-lora.tar.gz")
model_path   = env("MODEL_PATH",   "unsloth/Magistral-Small-2509")
epochs       = int(env("EPOCHS",   "3"))
rank         = int(env("RANK",     "16"))

print(f"Encoding JSONL: {jsonl_file}")
jsonl_b64 = base64.b64encode(Path(jsonl_file).read_bytes()).decode()

print(f"Reading SSH key: {ssh_key_file}")
ssh_key = Path(ssh_key_file).read_text()

payload = json.dumps({"input": {
    "jsonl_b64":  jsonl_b64,
    "ssh_host":   ssh_host,
    "ssh_port":   ssh_port,
    "ssh_user":   ssh_user,
    "ssh_key":    ssh_key,
    "ssh_dest":   ssh_dest,
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

# --- Submit ---
print(f"Payload size: {len(payload) / 1024:.1f} KB")
print(f"Submitting job to endpoint {endpoint_id}...")
result = api("POST", "run", data=payload)
job_id = result.get("id")
if not job_id:
    print(f"ERROR: no job id in response: {result}")
    sys.exit(1)
print(f"Job submitted: {job_id}")

# --- Poll ---
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
