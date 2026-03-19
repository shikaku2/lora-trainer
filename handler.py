#!/usr/bin/env python3
"""
RunPod serverless handler — full 3-stage training pipeline:
  Stage 1  CPT   (Continued Pre-Training)  on plain-text corpus
  Stage 2  QLoRA (instruction fine-tuning) on dialogue examples
  Stage 3  DPO   (preference alignment)    on chosen/rejected pairs

Each stage uploads its adapter to HuggingFace immediately after completing.
On the next run, completed stages are downloaded and skipped automatically,
so a failure mid-pipeline only reruns the remaining stages.

Intermediate repos:
  {hf_repo}-cpt   → CPT adapter
  {hf_repo}-qlora → QLoRA adapter
  {hf_repo}       → final DPO adapter (the one you actually use)

Job input fields:
  cpt_b64     (str)  – base64-encoded plain-text CPT corpus          [required]
  lora_b64    (str)  – base64-encoded JSONL dialogue examples        [required]
  dpo_b64     (str)  – base64-encoded JSONL DPO preference pairs     [required]
  hf_token    (str)  – HuggingFace write token                       [required]
  hf_repo     (str)  – HF repo ID  e.g. "shikaku2/my-model"         [required]
  model_path  (str)  – HF repo or local path (default: MODEL_PATH env)
  epochs_cpt  (int)  – CPT epochs  (default 1)
  epochs_lora (int)  – QLoRA epochs (default 3)
  epochs_dpo  (int)  – DPO epochs  (default 1)
  rank        (int)  – LoRA rank   (default 16)
  max_seq_len (int)  – sequence length cap (default 2048)
  lr_cpt      (float)– CPT learning rate   (default 1e-4)
  lr_lora     (float)– QLoRA learning rate (default 2e-4)
  lr_dpo      (float)– DPO learning rate   (default 5e-5)
  beta        (float)– DPO beta            (default 0.1)
  no_4bit     (bool) – disable 4-bit quant (default False)
  force_cpt   (bool) – re-run CPT even if checkpoint exists (default False)
  force_qlora (bool) – re-run QLoRA even if checkpoint exists (default False)
  force_dpo   (bool) – re-run DPO even if checkpoint exists (default False)
"""

import os
import sys
import base64
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

DEFAULT_MODEL  = os.getenv("MODEL_PATH", "unsloth/Magistral-Small-2509")
SCRIPT_DIR     = Path(__file__).parent
TRAIN_CPT      = SCRIPT_DIR / "train_cpt.py"
TRAIN_LORA     = SCRIPT_DIR / "train_lora.py"
TRAIN_DPO      = SCRIPT_DIR / "train_dpo.py"


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


def _decode(b64_field: str, label: str):
    try:
        return base64.b64decode(b64_field)
    except Exception as e:
        raise ValueError(f"Failed to decode {label}: {e}") from e


def _hf_repo_has_adapter(repo_id: str, token: str) -> bool:
    """Return True if the HF repo exists and contains adapter weights."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        files = [f.rfilename for f in api.list_repo_files(repo_id, repo_type="model")]
        return any("adapter_model" in f for f in files)
    except Exception:
        return False


def _hf_download(repo_id: str, token: str, local_dir: Path):
    """Download a HF repo to local_dir."""
    from huggingface_hub import snapshot_download
    log.info("Downloading existing adapter from %s...", repo_id)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        local_dir=str(local_dir),
    )
    log.info("Downloaded to %s", local_dir)


def _hf_upload(local_dir: Path, repo_id: str, token: str, commit_message: str):
    """Upload local_dir to a HF repo, creating it if needed."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    log.info("Uploaded %s to https://huggingface.co/%s", local_dir.name, repo_id)


def run_training_job(event: dict) -> dict:
    # ----------------------------------------------------------------
    # Validate required inputs
    # ----------------------------------------------------------------
    for field in ("cpt_b64", "lora_b64", "dpo_b64", "hf_token", "hf_repo"):
        if not event.get(field):
            return {"status": "error", "message": f"Missing required field: {field}"}

    model_path  = event.get("model_path") or DEFAULT_MODEL
    epochs_cpt  = int(event.get("epochs_cpt",  1))
    epochs_lora = int(event.get("epochs_lora", 3))
    epochs_dpo  = int(event.get("epochs_dpo",  1))
    rank        = int(event.get("rank",        16))
    max_seq     = int(event.get("max_seq_len", 2048))
    lr_cpt      = float(event.get("lr_cpt",   1e-4))
    lr_lora     = float(event.get("lr_lora",  2e-4))
    lr_dpo      = float(event.get("lr_dpo",   5e-5))
    beta        = float(event.get("beta",      0.1))
    no_4bit     = bool(event.get("no_4bit",     False))
    force_cpt   = bool(event.get("force_cpt",   False))
    force_qlora = bool(event.get("force_qlora", False))
    force_dpo   = bool(event.get("force_dpo",   False))
    hf_token    = event["hf_token"]
    hf_repo     = event["hf_repo"]

    hf_repo_cpt   = f"{hf_repo}-cpt"
    hf_repo_qlora = f"{hf_repo}-qlora"

    with tempfile.TemporaryDirectory(prefix="lora_job_") as workdir:
        workdir = Path(workdir)

        # Write input files
        cpt_path  = workdir / "cpt_corpus.txt"
        lora_path = workdir / "lora_train.jsonl"
        dpo_path  = workdir / "dpo_train.jsonl"

        cpt_path.write_bytes(_decode(event["cpt_b64"],  "cpt_b64"))
        lora_path.write_bytes(_decode(event["lora_b64"], "lora_b64"))
        dpo_path.write_bytes(_decode(event["dpo_b64"],  "dpo_b64"))

        lora_count = sum(1 for l in lora_path.read_text().splitlines() if l.strip())
        dpo_count  = sum(1 for l in dpo_path.read_text().splitlines()  if l.strip())
        log.info("Inputs: CPT %d bytes, %d QLoRA examples, %d DPO pairs",
                 cpt_path.stat().st_size, lora_count, dpo_count)

        cpt_out  = workdir / "cpt-output"
        lora_out = workdir / "lora-output"
        dpo_out  = workdir / "dpo-output"

        all_logs = []
        t_total  = time.time()
        skipped  = []

        no4bit_flag = ["--no-4bit"] if no_4bit else []

        # ----------------------------------------------------------------
        # Stage 1 — CPT
        # ----------------------------------------------------------------
        cpt_on_hf = _hf_repo_has_adapter(hf_repo_cpt, hf_token)
        if not force_cpt and (force_dpo or force_qlora or cpt_on_hf):
            if not cpt_on_hf:
                return {"status": "error",
                        "message": f"FORCE_DPO/FORCE_QLORA set but no CPT checkpoint found at "
                                   f"{hf_repo_cpt}. Run without force flags to train from scratch."}
            log.info("=== Stage 1/3: CPT — skipping (found %s) ===", hf_repo_cpt)
            _hf_download(hf_repo_cpt, hf_token, cpt_out)
            skipped.append("cpt")
        else:
            log.info("=== Stage 1/3: CPT ===")
            t0 = time.time()
            cmd = [
                sys.executable, str(TRAIN_CPT),
                "--model",       model_path,
                "--data",        str(cpt_path),
                "--output",      str(cpt_out),
                "--epochs",      str(epochs_cpt),
                "--rank",        str(rank),
                "--max-seq-len", str(max_seq),
                "--lr",          str(lr_cpt),
                *no4bit_flag,
            ]
            rc, lines = _run(cmd, cwd=str(workdir), log_prefix="[CPT] ")
            all_logs += lines
            if rc != 0:
                return {"status": "error", "message": "CPT stage failed",
                        "stage": "cpt", "logs": all_logs}
            log.info("CPT done in %.1fs — uploading checkpoint...", time.time() - t0)
            try:
                _hf_upload(cpt_out, hf_repo_cpt, hf_token,
                           f"CPT adapter after {epochs_cpt} epoch(s)")
            except Exception as e:
                return {"status": "error", "message": f"CPT checkpoint upload failed: {e}",
                        "stage": "cpt_upload", "logs": all_logs}

        # ----------------------------------------------------------------
        # Stage 2 — QLoRA  (continues from CPT adapter)
        # ----------------------------------------------------------------
        qlora_on_hf = _hf_repo_has_adapter(hf_repo_qlora, hf_token)
        if not force_qlora and (force_dpo or qlora_on_hf):
            if not qlora_on_hf:
                return {"status": "error",
                        "message": f"FORCE_DPO set but no QLoRA checkpoint found at "
                                   f"{hf_repo_qlora}. Run without force flags to train from scratch."}
            log.info("=== Stage 2/3: QLoRA — skipping (found %s) ===", hf_repo_qlora)
            _hf_download(hf_repo_qlora, hf_token, lora_out)
            skipped.append("qlora")
        else:
            log.info("=== Stage 2/3: QLoRA ===")
            t0 = time.time()
            cmd = [
                sys.executable, str(TRAIN_LORA),
                "--model",       model_path,
                "--adapter",     str(cpt_out),
                "--data",        str(lora_path),
                "--output",      str(lora_out),
                "--epochs",      str(epochs_lora),
                "--rank",        str(rank),
                "--max-seq-len", str(max_seq),
                "--lr",          str(lr_lora),
                *no4bit_flag,
            ]
            rc, lines = _run(cmd, cwd=str(workdir), log_prefix="[QLoRA] ")
            all_logs += lines
            if rc != 0:
                return {"status": "error", "message": "QLoRA stage failed",
                        "stage": "qlora", "logs": all_logs}
            log.info("QLoRA done in %.1fs — uploading checkpoint...", time.time() - t0)
            try:
                _hf_upload(lora_out, hf_repo_qlora, hf_token,
                           f"QLoRA adapter after {epochs_lora} epoch(s)")
            except Exception as e:
                return {"status": "error", "message": f"QLoRA checkpoint upload failed: {e}",
                        "stage": "qlora_upload", "logs": all_logs}

        # ----------------------------------------------------------------
        # Stage 3 — DPO  (continues from QLoRA adapter)
        # ----------------------------------------------------------------
        if not force_dpo and _hf_repo_has_adapter(hf_repo, hf_token):
            log.info("=== Stage 3/3: DPO — skipping (found %s) ===", hf_repo)
            skipped.append("dpo")
            total_elapsed = time.time() - t_total
            skipped_str = f" (skipped: {', '.join(skipped)})"
            return {
                "status": "ok",
                "message": f"Pipeline complete in {total_elapsed:.0f}s{skipped_str}. "
                           f"Adapter at https://huggingface.co/{hf_repo}",
                "hf_repo":          hf_repo,
                "training_seconds": round(total_elapsed),
                "qlora_examples":   lora_count,
                "dpo_pairs":        dpo_count,
                "skipped_stages":   skipped,
                "logs":             all_logs,
            }

        log.info("=== Stage 3/3: DPO ===")
        t0 = time.time()
        cmd = [
            sys.executable, str(TRAIN_DPO),
            "--model",       model_path,
            "--adapter",     str(lora_out),
            "--data",        str(dpo_path),
            "--output",      str(dpo_out),
            "--epochs",      str(epochs_dpo),
            "--max-seq-len", str(max_seq),
            "--lr",          str(lr_dpo),
            "--beta",        str(beta),
            *no4bit_flag,
        ]
        rc, lines = _run(cmd, cwd=str(workdir), log_prefix="[DPO] ")
        all_logs += lines
        if rc != 0:
            return {"status": "error", "message": "DPO stage failed",
                    "stage": "dpo", "logs": all_logs}
        log.info("DPO done in %.1fs", time.time() - t0)

        total_elapsed = time.time() - t_total
        log.info("All stages complete in %.1fs", total_elapsed)

        # ----------------------------------------------------------------
        # Upload final DPO adapter
        # ----------------------------------------------------------------
        log.info("Uploading final adapter to %s...", hf_repo)
        try:
            _hf_upload(dpo_out, hf_repo, hf_token,
                       f"CPT→QLoRA→DPO adapter ({lora_count} QLoRA, {dpo_count} DPO pairs)")
        except Exception as e:
            return {"status": "error",
                    "message": f"HuggingFace upload failed: {e}",
                    "logs": all_logs}

        skipped_str = f" (skipped: {', '.join(skipped)})" if skipped else ""
        return {
            "status": "ok",
            "message": (
                f"Pipeline complete in {total_elapsed:.0f}s{skipped_str}. "
                f"Adapter at https://huggingface.co/{hf_repo}"
            ),
            "hf_repo":          hf_repo,
            "training_seconds": round(total_elapsed),
            "qlora_examples":   lora_count,
            "dpo_pairs":        dpo_count,
            "skipped_stages":   skipped,
            "logs":             all_logs,
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
            "status":    "error",
            "message":   str(e),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    log.info("Starting RunPod CPT+QLoRA+DPO trainer worker.")
    runpod.serverless.start({"handler": handler})
