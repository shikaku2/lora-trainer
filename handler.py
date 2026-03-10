#!/usr/bin/env python3
"""
RunPod serverless handler for QLoRA fine-tuning.

Job input fields:
  jsonl_b64   (str)  – base64-encoded JSONL training data         [required]
  ssh_host    (str)  – hostname/IP to SCP the result to           [required]
  ssh_port    (int)  – SSH port, default 22
  ssh_user    (str)  – SSH username, default "root"
  ssh_key     (str)  – PEM private key content (the actual key text, not a path)
  ssh_dest    (str)  – remote path to write the tar.gz, e.g. "/home/aaron/lora.tar.gz"
  model_path  (str)  – HF repo or local path (defaults to env MODEL_PATH)
  epochs      (int)  – default 3
  rank        (int)  – LoRA rank, default 16
  max_seq_len (int)  – default 2048
  lr          (float)– default 2e-4
  no_4bit     (bool) – disable 4-bit quant (default False)

Job output (JSON):
  status      – "ok" or "error"
  message     – human-readable summary
  scp_dest    – where the file was sent
  logs        – last N lines of training stdout
"""

import os
import sys
import base64
import json
import logging
import stat
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

DEFAULT_MODEL = os.getenv(
    "MODEL_PATH",
    "unsloth/Magistral-Small-2509",
)

TRAIN_SCRIPT = Path(__file__).parent / "train_lora.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_ssh_key(key_text: str) -> Path:
    """Write the private key to a temp file with 0600 perms."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".pem", delete=False, prefix="runpod_sshkey_"
    )
    # Normalise line endings — people sometimes paste with \\n literals
    key_text = key_text.replace("\\n", "\n").strip() + "\n"
    tmp.write(key_text)
    tmp.close()
    os.chmod(tmp.name, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    return Path(tmp.name)


def _run(cmd: list, cwd=None, timeout=7200, log_prefix=""):
    """
    Run a subprocess, stream output to our logger, return (returncode, last_lines).
    timeout is in seconds (default 2h).
    """
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

    return proc.returncode, lines[-100:]  # return last 100 lines


def _scp_result(local_path: Path, host: str, port: int, user: str,
                key_path: Path, remote_path: str) -> None:
    """SCP a file to the user's machine."""
    cmd = [
        "scp",
        "-i", str(key_path),
        "-P", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30",
        str(local_path),
        f"{user}@{host}:{remote_path}",
    ]
    rc, lines = _run(cmd, log_prefix="[scp] ")
    if rc != 0:
        raise RuntimeError(f"SCP failed (exit {rc}). Last output:\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Core training job
# ---------------------------------------------------------------------------

def run_training_job(event: dict) -> dict:
    # ---- Validate required inputs ----
    jsonl_b64 = event.get("jsonl_b64")
    if not jsonl_b64:
        return {"status": "error", "message": "Missing required field: jsonl_b64"}

    ssh_host = event.get("ssh_host")
    ssh_key  = event.get("ssh_key")
    ssh_dest = event.get("ssh_dest", "/tmp/alastor-lora.tar.gz")
    if not ssh_host or not ssh_key:
        return {"status": "error", "message": "Missing required fields: ssh_host and ssh_key"}

    ssh_port = int(event.get("ssh_port", 22))
    ssh_user = event.get("ssh_user", "root")
    model_path = event.get("model_path") or DEFAULT_MODEL
    epochs     = int(event.get("epochs", 3))
    rank       = int(event.get("rank", 16))
    max_seq    = int(event.get("max_seq_len", 2048))
    lr         = float(event.get("lr", 2e-4))
    no_4bit    = bool(event.get("no_4bit", False))

    with tempfile.TemporaryDirectory(prefix="lora_job_") as workdir:
        workdir = Path(workdir)

        # ---- Write JSONL ----
        try:
            jsonl_bytes = base64.b64decode(jsonl_b64)
        except Exception as e:
            return {"status": "error", "message": f"Failed to decode jsonl_b64: {e}"}

        jsonl_path = workdir / "train.jsonl"
        jsonl_path.write_bytes(jsonl_bytes)
        line_count = sum(1 for _ in jsonl_bytes.decode().splitlines() if _.strip())
        log.info("Wrote %d training examples to %s", line_count, jsonl_path)

        # ---- Write SSH key ----
        key_path = None
        try:
            key_path = _write_ssh_key(ssh_key)

            # ---- Run training ----
            output_dir = workdir / "lora-output"
            cmd = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--model", model_path,
                "--data", str(jsonl_path),
                "--output", str(output_dir),
                "--epochs", str(epochs),
                "--rank", str(rank),
                "--max-seq-len", str(max_seq),
                "--lr", str(lr),
            ]
            if no_4bit:
                cmd.append("--no-4bit")

            log.info("Starting training run...")
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

            # ---- Package the adapter ----
            archive_path = workdir / "alastor-lora.tar.gz"
            rc2, _ = _run(
                ["tar", "-czf", str(archive_path), "-C", str(output_dir), "."],
            )
            if rc2 != 0:
                return {"status": "error", "message": "Failed to create tar archive"}

            size_mb = archive_path.stat().st_size / (1024 * 1024)
            log.info("Adapter archive: %.1f MB", size_mb)

            # ---- SCP home ----
            log.info("SCPing result to %s@%s:%s%s", ssh_user, ssh_host, ssh_port, ssh_dest)
            _scp_result(archive_path, ssh_host, ssh_port, ssh_user, key_path, ssh_dest)

            return {
                "status": "ok",
                "message": (
                    f"Training complete in {elapsed:.0f}s. "
                    f"Adapter ({size_mb:.1f} MB) sent to {ssh_user}@{ssh_host}:{ssh_dest}"
                ),
                "scp_dest": ssh_dest,
                "training_seconds": round(elapsed),
                "examples": line_count,
                "logs": train_lines,
            }

        finally:
            if key_path and key_path.exists():
                key_path.unlink()


# ---------------------------------------------------------------------------
# RunPod entrypoint
# ---------------------------------------------------------------------------

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
