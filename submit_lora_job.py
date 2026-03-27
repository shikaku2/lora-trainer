#!/usr/bin/env python3
"""
submit_lora_job.py — submit a full CPT→QLoRA→DPO training run to RunPod.

Reads three source files from build1/, encodes them, and sends a single
job to the serverless endpoint.  The LoRA examples file (plain text) is
automatically parsed into JSONL format before submission.

Usage:
  RUNPOD_API_KEY=rp_xxx RUNPOD_ENDPOINT_ID=abc123 HF_WRITE_TOKEN=hf_xxx \
      python3 submit_lora_job.py

Required env vars:
  RUNPOD_API_KEY        your RunPod API key
  RUNPOD_ENDPOINT_ID    the serverless endpoint ID
  HF_WRITE_TOKEN        HuggingFace write token

Optional env vars (with defaults):
  CPT_FILE      plain-text CPT corpus            [cpt.txt]
  LORA_FILE     dialogue examples (txt or jsonl) [lora.txt]
  DPO_FILE      DPO preference pairs (jsonl)     [dpo.jsonl]
  HF_REPO       HuggingFace repo ID              [shikaku2/magistral-alastor-lora]
  MODEL_PATH    HF repo or local path            [unsloth/Magistral-Small-2509]
  EPOCHS_CPT    CPT epochs                       [1]
  EPOCHS_LORA   QLoRA epochs                     [3]
  EPOCHS_DPO    DPO epochs                       [1]
  RANK          LoRA rank                        [16]
  MAX_SEQ_LEN   token sequence length cap        [2048]
"""

import base64
import json
import os
import re
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

        # Extract optional system prompt from SYSTEM: block
        if block.startswith("SYSTEM:"):
            system_prompt = block[len("SYSTEM:"):].strip()
            continue

        # Accept "EXAMPLE N:", "EXAMPLES N:" (typo), or blocks starting with USER:
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
        text_val = f"[INST] {inst} [/INST] {reply}"
        lines.append(json.dumps({"text": text_val}))

    if not lines:
        print(f"ERROR: No examples parsed from {text_path}")
        sys.exit(1)

    sys_info = f" (with system prompt, ~{len(system_prompt)//4} tokens)" if system_prompt else ""
    print(f"  Parsed {len(lines)} QLoRA examples from {Path(text_path).name}{sys_info}")
    return ("\n".join(lines) + "\n").encode()


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
api_key     = env("RUNPOD_API_KEY",     required=True)
endpoint_id = env("RUNPOD_ENDPOINT_ID", required=True)
hf_token    = env("HF_WRITE_TOKEN",     required=True)

cpt_file    = env("CPT_FILE",      "cpt.txt")
lora_file   = env("LORA_FILE",     "lora.txt")
dpo_file    = env("DPO_FILE",      "dpo.jsonl")
max_seq_len = int(env("MAX_SEQ_LEN", "2048"))
hf_repo     = env("HF_REPO",    "shikaku2/magistral-alastor-lora")
model_path  = env("MODEL_PATH", "unsloth/Magistral-Small-2509")
epochs_cpt   = int(env("EPOCHS_CPT",  "1"))
epochs_lora  = int(env("EPOCHS_LORA", "3"))
epochs_dpo   = int(env("EPOCHS_DPO",  "1"))
rank         = int(env("RANK",        "16"))
force_cpt   = env("FORCE_CPT",   "0") == "1"
force_qlora = env("FORCE_QLORA", "0") == "1"
force_dpo   = env("FORCE_DPO",   "0") == "1"

# ----------------------------------------------------------------
# Encode files
# ----------------------------------------------------------------
print("Encoding training files...")

print(f"  CPT corpus:  {cpt_file}")
cpt_b64 = base64.b64encode(Path(cpt_file).read_bytes()).decode()

print(f"  LoRA examples: {lora_file}")
if lora_file.endswith(".jsonl"):
    lora_bytes = Path(lora_file).read_bytes()
    print(f"  (JSONL format — skipping parse, {sum(1 for l in lora_bytes.decode().splitlines() if l.strip())} records)")
else:
    lora_bytes = parse_lora_examples(lora_file)
lora_b64   = base64.b64encode(lora_bytes).decode()

print(f"  DPO pairs:   {dpo_file}")
dpo_b64 = base64.b64encode(Path(dpo_file).read_bytes()).decode()

payload = json.dumps({"input": {
    "cpt_b64":     cpt_b64,
    "lora_b64":    lora_b64,
    "dpo_b64":     dpo_b64,
    "hf_token":    hf_token,
    "hf_repo":     hf_repo,
    "model_path":  model_path,
    "epochs_cpt":   epochs_cpt,
    "epochs_lora":  epochs_lora,
    "epochs_dpo":   epochs_dpo,
    "rank":         rank,
    "max_seq_len":  max_seq_len,
    "force_cpt":    force_cpt,
    "force_qlora":  force_qlora,
    "force_dpo":    force_dpo,
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


# ----------------------------------------------------------------
# Preflight: verify HF token before burning GPU time
# ----------------------------------------------------------------
print(f"\nChecking HuggingFace access to {hf_repo}...")
try:
    import urllib.parse
    hf_req = urllib.request.Request(
        f"https://huggingface.co/api/models/{urllib.parse.quote(hf_repo, safe='/')}",
        headers={"Authorization": f"Bearer {hf_token}"},
        method="GET",
    )
    with urllib.request.urlopen(hf_req, timeout=10) as r:
        info = json.loads(r.read())
    siblings = info.get("siblings", [])
    print(f"HF repo accessible. ({len(siblings)} files currently)")
except urllib.error.HTTPError as e:
    if e.code == 401:
        print("ERROR: HF token is invalid or expired.")
        sys.exit(1)
    elif e.code == 403:
        print("ERROR: HF token does not have write access to this repo.")
        sys.exit(1)
    elif e.code == 404:
        print(f"Repo {hf_repo} not found — will be created by the training job.")
    else:
        print(f"ERROR: HF preflight HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: HF preflight failed: {e}")
    sys.exit(1)
print("HuggingFace preflight passed.")

# ----------------------------------------------------------------
# Submit job
# ----------------------------------------------------------------
print(f"\nPayload size: {len(payload) / 1024:.1f} KB")
print(f"Submitting CPT→QLoRA→DPO job to endpoint {endpoint_id}...")
result = api("POST", "run", data=payload)
job_id = result.get("id")
if not job_id:
    print(f"ERROR: no job id in response: {result}")
    sys.exit(1)
print(f"Job submitted: {job_id}")

# ----------------------------------------------------------------
# Poll until complete
# ----------------------------------------------------------------
start = time.time()
while True:
    time.sleep(15)
    elapsed = int(time.time() - start)
    status  = api("GET", f"status/{job_id}")
    state   = status.get("status", "UNKNOWN")
    print(f"[{elapsed:4d}s] {state}")

    if state == "COMPLETED":
        print("\n=== Response ===")
        print(json.dumps(status.get("output", status), indent=2))
        break
    elif state in ("FAILED", "CANCELLED", "TIMED_OUT"):
        print("\n=== FAILED ===")
        print(json.dumps(status, indent=2))
        sys.exit(1)
