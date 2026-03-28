#!/usr/bin/env python3
"""
submit_lora_job.py — submit a full CPT→QLoRA→DPO training run as a RunPod pod.

Uploads training files to a temporary private HF repo, creates a RunPod pod
that runs the full pipeline, polls until the pod terminates, then reports.

Usage:
  RUNPOD_API_KEY=rp_xxx HF_WRITE_TOKEN=hf_xxx python3 submit_lora_job.py

Required env vars:
  RUNPOD_API_KEY        RunPod API key
  HF_WRITE_TOKEN        HuggingFace write token

Optional env vars (with defaults):
  CPT_FILE        plain-text CPT corpus            [cpt.txt]
  LORA_FILE       dialogue examples (txt or jsonl) [lora.txt]
  DPO_FILE        DPO preference pairs (jsonl)     [dpo.jsonl]
  HF_REPO         HuggingFace repo for final adapter  [shikaku2/magistral-alastor-lora]
  MODEL_PATH      base model HF repo or local path [unsloth/Magistral-Small-2509]
  DOCKER_IMAGE    pod container image              [ghcr.io/shikaku2/lora-trainer:latest]
  GPU_TYPE        RunPod GPU type ID               [NVIDIA A40]
  EPOCHS_CPT      CPT epochs                       [1]
  EPOCHS_LORA     QLoRA epochs                     [3]
  EPOCHS_DPO      DPO epochs                       [1]
  RANK            LoRA rank                        [16]
  MAX_SEQ_LEN     token sequence length cap        [2048]
  FORCE_CPT       re-run CPT even if cached (0/1)  [0]
  FORCE_QLORA     re-run QLoRA even if cached      [0]
  FORCE_DPO       re-run DPO even if cached        [0]
"""

import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def env(key, default=None, required=False):
    val = os.environ.get(key, default)
    if required and not val:
        print(f"ERROR: {key} is required.")
        sys.exit(1)
    return val


def parse_lora_examples(text_path: str) -> bytes:
    """
    Parse lora.txt format into JSONL bytes.

    File format:
        SYSTEM:
        <system prompt>
        =====

        EXAMPLE N:
        USER: user message
        REPLY: possibly
               multi-line reply
        =====

    If a SYSTEM block is present it is injected into every [INST] block.
    Output: one {"text": "[INST] SYSTEM\\n\\nUSER [/INST] REPLY"} per line.
    """
    text = Path(text_path).read_text()
    blocks = re.split(r"\n=====\n?", text)

    system_prompt = ""
    lines = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        if block.startswith("SYSTEM:"):
            system_prompt = block[len("SYSTEM:"):].strip()
            continue

        if not re.match(r"EXAMPLES?\s+\d+", block) and not block.startswith("USER:"):
            continue

        user_m  = re.search(r"^USER:\s*(.+?)(?=\nREPLY:)", block,
                             re.DOTALL | re.MULTILINE)
        reply_m = re.search(r"^REPLY:\s*(.+)$",           block,
                             re.DOTALL | re.MULTILINE)
        if not user_m or not reply_m:
            continue

        user  = user_m.group(1).strip()
        reply = reply_m.group(1).strip()
        inst  = f"{system_prompt}\n\n{user}" if system_prompt else user
        lines.append(json.dumps({"text": f"[INST] {inst} [/INST] {reply}"}))

    if not lines:
        print(f"ERROR: No examples parsed from {text_path}")
        sys.exit(1)

    sys_info = f" (with system prompt, ~{len(system_prompt)//4} tokens)" if system_prompt else ""
    print(f"  Parsed {len(lines)} QLoRA examples from {Path(text_path).name}{sys_info}")
    return ("\n".join(lines) + "\n").encode()


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
api_key      = env("RUNPOD_API_KEY",  required=True)
hf_token     = env("HF_WRITE_TOKEN",  required=True)

cpt_file     = env("CPT_FILE",   "cpt.txt")
lora_file    = env("LORA_FILE",  "lora.txt")
dpo_file     = env("DPO_FILE",   "dpo.jsonl")
hf_repo      = env("HF_REPO",   "shikaku2/magistral-alastor-lora")
model_path   = env("MODEL_PATH", "unsloth/Magistral-Small-2509")
docker_image = env("DOCKER_IMAGE", "ghcr.io/shikaku2/lora-trainer:latest")
gpu_type     = env("GPU_TYPE",   "NVIDIA A40")
max_seq_len  = int(env("MAX_SEQ_LEN",  "2048"))
epochs_cpt   = int(env("EPOCHS_CPT",   "1"))
epochs_lora  = int(env("EPOCHS_LORA",  "3"))
epochs_dpo   = int(env("EPOCHS_DPO",   "1"))
rank         = int(env("RANK",         "16"))
force_cpt    = env("FORCE_CPT",   "0") == "1"
force_qlora  = env("FORCE_QLORA", "0") == "1"
force_dpo    = env("FORCE_DPO",   "0") == "1"

training_data_repo = f"{hf_repo}-training-data"

# ----------------------------------------------------------------
# Preflight: verify HF token
# ----------------------------------------------------------------
print(f"\nChecking HuggingFace access to {hf_repo}...")
try:
    hf_req = urllib.request.Request(
        f"https://huggingface.co/api/models/{urllib.parse.quote(hf_repo, safe='/')}",
        headers={"Authorization": f"Bearer {hf_token}"},
        method="GET",
    )
    with urllib.request.urlopen(hf_req, timeout=10) as r:
        info = json.loads(r.read())
    print(f"  HF repo accessible ({len(info.get('siblings', []))} files currently)")
except urllib.error.HTTPError as e:
    if e.code == 401:
        print("ERROR: HF token is invalid or expired.")
        sys.exit(1)
    elif e.code == 403:
        print("ERROR: HF token does not have write access to this repo.")
        sys.exit(1)
    elif e.code == 404:
        print(f"  Repo {hf_repo} not found — will be created by the training job.")
    else:
        print(f"ERROR: HF preflight HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: HF preflight failed: {e}")
    sys.exit(1)

# ----------------------------------------------------------------
# Prepare training files
# ----------------------------------------------------------------
print("\nPreparing training files...")

print(f"  CPT corpus:    {cpt_file}")
cpt_bytes = Path(cpt_file).read_bytes()

print(f"  LoRA examples: {lora_file}")
if lora_file.endswith(".jsonl"):
    lora_bytes = Path(lora_file).read_bytes()
    count = sum(1 for l in lora_bytes.decode().splitlines() if l.strip())
    print(f"  (JSONL — {count} records)")
else:
    lora_bytes = parse_lora_examples(lora_file)

print(f"  DPO pairs:     {dpo_file}")
dpo_bytes = Path(dpo_file).read_bytes()

# ----------------------------------------------------------------
# Upload training files to temporary HF repo
# ----------------------------------------------------------------
print(f"\nUploading training data to {training_data_repo}...")
try:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(training_data_repo, repo_type="model", exist_ok=True, private=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "cpt.txt").write_bytes(cpt_bytes)
        (tmp / "lora.jsonl").write_bytes(lora_bytes)
        (tmp / "dpo.jsonl").write_bytes(dpo_bytes)
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=training_data_repo,
            repo_type="model",
            commit_message="training data upload",
        )
    print(f"  Uploaded: cpt.txt ({len(cpt_bytes)//1024}KB)"
          f"  lora.jsonl ({len(lora_bytes)//1024}KB)"
          f"  dpo.jsonl ({len(dpo_bytes)//1024}KB)")
except Exception as e:
    print(f"ERROR: Failed to upload training data: {e}")
    sys.exit(1)

# ----------------------------------------------------------------
# Create RunPod pod
# ----------------------------------------------------------------
print(f"\nCreating RunPod pod ({gpu_type}, {docker_image})...")
try:
    import runpod
    runpod.api_key = api_key

    pod_env = {
        "HF_TOKEN":           hf_token,
        "HF_WRITE_TOKEN":     hf_token,
        "HF_REPO":            hf_repo,
        "TRAINING_DATA_REPO": training_data_repo,
        "MODEL_PATH":         model_path,
        "RUNPOD_API_KEY":     api_key,
        "EPOCHS_CPT":         str(epochs_cpt),
        "EPOCHS_LORA":        str(epochs_lora),
        "EPOCHS_DPO":         str(epochs_dpo),
        "RANK":               str(rank),
        "MAX_SEQ_LEN":        str(max_seq_len),
        "FORCE_CPT":          "1" if force_cpt  else "0",
        "FORCE_QLORA":        "1" if force_qlora else "0",
        "FORCE_DPO":          "1" if force_dpo   else "0",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }

    pod = runpod.create_pod(
        name=f"lora-training-{int(time.time())}",
        image_name=docker_image,
        gpu_type_id=gpu_type,
        cloud_type="SECURE",
        gpu_count=1,
        container_disk_in_gb=100,  # ephemeral: model (~47GB) + adapters + workspace
        env=pod_env,
        ports=None,
        support_public_ip=False,
    )
    pod_id = pod["id"]
    print(f"  Pod created: {pod_id}")
    print(f"  Dashboard:   https://www.runpod.io/console/pods/{pod_id}")
except Exception as e:
    print(f"ERROR: Failed to create pod: {e}")
    # Clean up training data repo
    try:
        HfApi(token=hf_token).delete_repo(training_data_repo, repo_type="model")
    except Exception:
        pass
    sys.exit(1)

# ----------------------------------------------------------------
# Poll until pod terminates
# ----------------------------------------------------------------
print("\nWaiting for pod to complete (pod self-terminates when done)...")
start = time.time()
last_status = None
while True:
    time.sleep(20)
    elapsed = int(time.time() - start)
    try:
        pod_info = runpod.get_pod(pod_id)
        if pod_info is None:
            print(f"\n[{elapsed:5d}s] Pod terminated (no longer found)")
            break
        status = pod_info.get("desiredStatus") or pod_info.get("lastStatusChange") or "RUNNING"
        if status != last_status:
            print(f"[{elapsed:5d}s] {status}")
            last_status = status
        else:
            print(f"[{elapsed:5d}s] {status}", end="\r")
        if status in ("EXITED", "TERMINATED", "DEAD", "STOPPED"):
            print()
            break
    except Exception as e:
        print(f"\n[{elapsed:5d}s] Status check error: {e} — pod may have terminated")
        break

# ----------------------------------------------------------------
# Check result via HF
# ----------------------------------------------------------------
print(f"\nChecking HuggingFace repo for results...")
try:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    try:
        files = list(api.list_repo_files(hf_repo, repo_type="model"))
    except Exception:
        files = []

    if "pod_error.log" in files:
        print("\n  !! Pod reported an error. Log:\n")
        log_path = api.hf_hub_download(
            repo_id=hf_repo,
            filename="pod_error.log",
            repo_type="model",
            token=hf_token,
        )
        with open(str(log_path)) as f:
            print(f.read())
        api.delete_file("pod_error.log", repo_id=hf_repo, repo_type="model",
                        commit_message="remove error log")
        sys.exit(1)

    has_adapter = any("adapter_model" in f for f in files)
    if has_adapter:
        print(f"  Adapter found at https://huggingface.co/{hf_repo}")
        print("  Training complete!")
    else:
        print(f"  No adapter found in {hf_repo} — training may have failed.")
        print(f"  Check pod logs at https://www.runpod.io/console/pods/{pod_id}")
        sys.exit(1)
except SystemExit:
    raise
except Exception as e:
    print(f"  Could not check HF repo: {e}")
