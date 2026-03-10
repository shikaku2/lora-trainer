#!/usr/bin/env python3
"""
RunPod serverless handler for QLoRA fine-tuning.
Uploads the resulting LoRA adapter to HuggingFace Hub.

Job input fields:
  jsonl_b64   (str)  – base64-encoded JSONL training data         [required]
  hf_token    (str)  – HuggingFace write token                    [required]
  hf_repo     (str)  – HF repo ID e.g. "shikaku2/my-lora"        [required]
  model_path  (str)  – HF repo or local path (defaults to env MODEL_PATH)
  epochs      (int)  – default 3
  rank        (int)  – LoRA rank, default 16
  max_seq_len (int)  – default 2048
  lr          (float)– default 2e-4
  no_4bit     (bool) – disable 4-bit quant (default False)
"""

import os
import sys
import base64
import json
import logging
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import runpod

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[lora-trainer] %(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("lora-trainer")

DEFAULT_MODEL = os.getenv("MODEL_PATH", "unsloth/Magistral-Small-2509")
TRAIN_SCRIPT = Path(__file__).parent / "train_lora.py"


def _run(cmd, cwd=None, timeout=7200, log_prefix=""):
    log.info("%sRunning: %s", log_prefix, " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True,
        bufsize=1,
    )
    lines = []
    deadline = time.time() + timeout
    try:
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            log.info("%s%s", log_prefix, line)
            if time.time() > deadline:
                proc.kill()
                raise TimeoutError(f"Command exceeded {timeout}s timeout")
    finally:
        proc.wait()
    return proc.returncode, lines[-100:]


def run_training_job(event: dict) -> dict:
    jsonl_b64  = event.get("jsonl_b64")
    hf_token   = event.get("hf_token")
    hf_repo    = event.get("hf_repo")

    if not jsonl_b64:
        return {"status": "error", "message": "Missing required field: jsonl_b64"}
    if not hf_token:
        return {"status": "error", "message": "Missing required field: hf_token"}
    if not hf_repo:
        return {"status": "error", "message": "Missing required field: hf_repo"}

    model_path = event.get("model_path") or DEFAULT_MODEL
    epochs     = int(event.get("epochs",      3))
    rank       = int(event.get("rank",        16))
    max_seq    = int(event.get("max_seq_len", 2048))
    lr         = float(event.get("lr",        2e-4))
    no_4bit    = bool(event.get("no_4bit",    False))

    with tempfile.TemporaryDirectory(prefix="lora_job_") as workdir:
        workdir = Path(workdir)

        # Write JSONL
        try:
            jsonl_bytes = base64.b64decode(jsonl_b64)
        except Exception as e:
            return {"status": "error", "message": f"Failed to decode jsonl_b64: {e}"}

        jsonl_path = workdir / "train.jsonl"
        jsonl_path.write_bytes(jsonl_bytes)
        line_count = sum(1 for l in jsonl_bytes.decode().splitlines() if l.strip())
        log.info("Wrote %d training examples", line_count)

        # Train
        output_dir = workdir / "lora-output"
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--model",       model_path,
            "--data",        str(jsonl_path),
            "--output",      str(output_dir),
            "--epochs",      str(epochs),
            "--rank",        str(rank),
            "--max-seq-len", str(max_seq),
            "--lr",          str(lr),
        ]
        if no_4bit:
            cmd.append("--no-4bit")

        log.info("Starting training...")
        t0 = time.time()
        rc, train_lines = _run(cmd, cwd=str(workdir))
        elapsed = time.time() - t0

        if rc != 0:
            return {
                "status": "error",
                "message": f"Training script exited with code {rc}",
                "logs": train_lines,
            }

        log.info("Training complete in %.1fs", elapsed)

        # Upload to HuggingFace
        log.info("Uploading adapter to %s...", hf_repo)
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_folder(
                folder_path=str(output_dir),
                repo_id=hf_repo,
                repo_type="model",
                commit_message=f"LoRA adapter trained for {elapsed:.0f}s on {line_count} examples",
            )
        except Exception as e:
            return {
                "status": "error",
                "message": f"HuggingFace upload failed: {e}",
                "logs": train_lines,
            }

        return {
            "status": "ok",
            "message": (
                f"Training complete in {elapsed:.0f}s. "
                f"Adapter uploaded to https://huggingface.co/{hf_repo}"
            ),
            "hf_repo": hf_repo,
            "training_seconds": round(elapsed),
            "examples": line_count,
            "logs": train_lines,
        }


def handler(job: dict) -> dict:
    job_id = job.get("id") or job.get("requestId")
    log.info("Received training job id=%s", job_id)
    try:
        event = job.get("input", {}) or {}
        if not isinstance(event, dict):
            return {"status": "error", "message": "job['input'] must be a JSON object"}
        return run_training_job(event)
    except Exception as e:
        log.exception("Unhandled exception: %s", e)
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    log.info("Starting RunPod LoRA trainer worker.")
    runpod.serverless.start({"handler": handler})
