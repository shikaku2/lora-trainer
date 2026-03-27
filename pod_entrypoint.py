#!/usr/bin/env python3
"""
RunPod pod entrypoint — downloads training data from HF, runs the full
CPT→QLoRA→DPO pipeline via handler.run_training_job(), cleans up the
temporary training-data repo, then self-terminates the pod.

Environment variables (set by submit_lora_job.py at pod creation):
  HF_WRITE_TOKEN        HuggingFace write token
  HF_REPO               Target repo for the final adapter
  TRAINING_DATA_REPO    Temp HF repo with cpt.txt / lora.jsonl / dpo.jsonl
  MODEL_PATH            Base model (default: unsloth/Magistral-Small-2509)
  RUNPOD_API_KEY        RunPod API key for self-termination
  RUNPOD_POD_ID         Set automatically by RunPod
  EPOCHS_CPT/LORA/DPO  Training epochs
  RANK                  LoRA rank
  MAX_SEQ_LEN           Sequence length cap
  FORCE_CPT/QLORA/DPO   Re-run stage even if checkpoint exists (0/1)
"""

import base64
import logging
import os
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[pod] %(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("pod")


def download_training_data(repo: str, token: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download
    log.info("Downloading training data from %s ...", repo)
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        token=token,
        local_dir=str(dest),
        ignore_patterns=["*.gitattributes"],
    )
    log.info("Downloaded to %s", dest)


def delete_training_data_repo(repo: str, token: str) -> None:
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).delete_repo(repo_id=repo, repo_type="model")
        log.info("Deleted temporary training data repo: %s", repo)
    except Exception as e:
        log.warning("Could not delete training data repo %s: %s", repo, e)


def terminate_pod() -> None:
    pod_id  = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not pod_id or not api_key:
        log.warning("RUNPOD_POD_ID or RUNPOD_API_KEY not set — cannot self-terminate")
        return
    try:
        import runpod
        runpod.api_key = api_key
        runpod.terminate_pod(pod_id)
        log.info("Terminated pod %s", pod_id)
    except Exception as e:
        log.error("Failed to terminate pod: %s", e)


def main() -> None:
    hf_token    = os.environ["HF_WRITE_TOKEN"]
    hf_repo     = os.environ["HF_REPO"]
    data_repo   = os.environ["TRAINING_DATA_REPO"]
    model_path  = os.environ.get("MODEL_PATH",   "unsloth/Magistral-Small-2509")
    epochs_cpt  = int(os.environ.get("EPOCHS_CPT",  "1"))
    epochs_lora = int(os.environ.get("EPOCHS_LORA", "3"))
    epochs_dpo  = int(os.environ.get("EPOCHS_DPO",  "1"))
    rank        = int(os.environ.get("RANK",        "16"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "2048"))
    force_cpt   = os.environ.get("FORCE_CPT",   "0") == "1"
    force_qlora = os.environ.get("FORCE_QLORA", "0") == "1"
    force_dpo   = os.environ.get("FORCE_DPO",   "0") == "1"

    with tempfile.TemporaryDirectory(prefix="lora_pod_") as workdir:
        data_dir = Path(workdir) / "training-data"
        download_training_data(data_repo, hf_token, data_dir)

        # Re-encode as base64 so we can reuse handler.run_training_job() as-is
        event = {
            "cpt_b64":     base64.b64encode((data_dir / "cpt.txt").read_bytes()).decode(),
            "lora_b64":    base64.b64encode((data_dir / "lora.jsonl").read_bytes()).decode(),
            "dpo_b64":     base64.b64encode((data_dir / "dpo.jsonl").read_bytes()).decode(),
            "hf_token":    hf_token,
            "hf_repo":     hf_repo,
            "model_path":  model_path,
            "epochs_cpt":  epochs_cpt,
            "epochs_lora": epochs_lora,
            "epochs_dpo":  epochs_dpo,
            "rank":        rank,
            "max_seq_len": max_seq_len,
            "force_cpt":   force_cpt,
            "force_qlora": force_qlora,
            "force_dpo":   force_dpo,
        }

        from handler import run_training_job
        result = run_training_job(event)

    if result.get("status") == "ok":
        log.info("Pipeline complete: %s", result.get("message"))
    else:
        log.error("Pipeline failed [%s]: %s", result.get("stage"), result.get("message"))
        for line in (result.get("logs") or [])[-30:]:
            log.error("  %s", line)
        sys.exit(1)


if __name__ == "__main__":
    data_repo = os.environ.get("TRAINING_DATA_REPO", "")
    hf_token  = os.environ.get("HF_WRITE_TOKEN", "")
    try:
        main()
    finally:
        if data_repo and hf_token:
            delete_training_data_repo(data_repo, hf_token)
        terminate_pod()
