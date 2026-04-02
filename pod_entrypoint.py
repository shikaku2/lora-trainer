#!/usr/bin/env python3
"""
RunPod pod entrypoint — full CPT→QLoRA→DPO training pipeline.

Downloads training data from a temporary HF repo, runs the three stages in
sequence (each uploading its adapter to HF immediately on completion), then
deletes the temporary data repo and self-terminates the pod.

On failure the pod is stopped (paused) so the network volume is preserved.
Run submit_lora_job.py again to restart: it will patch the pod with the latest
image and resume.  Training automatically skips any stage whose adapter is
already on HF.

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
  LR_CPT/LORA/DPO       Learning rates
  BETA                  DPO beta
  NO_4BIT               Disable 4-bit quant (0/1)
  FORCE_CPT/QLORA/DPO   Re-run stage even if checkpoint exists (0/1)
"""

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

VERSION = 3

logging.basicConfig(
    level=logging.INFO,
    format="[pod] %(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("pod")

SCRIPT_DIR = Path(__file__).parent
TRAIN      = SCRIPT_DIR / "train.py"

RUNPOD_REST = "https://rest.runpod.io/v1"


# ----------------------------------------------------------------
# RunPod REST helpers
# ----------------------------------------------------------------

def _rest(method: str, path: str) -> None:
    """Make a bodyless RunPod REST call. No Content-Type header — bodyless POST/DELETE."""
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    pod_id  = os.environ.get("RUNPOD_POD_ID", "")
    log.info("RunPod REST %s %s (pod=%s key=%s...)", method, path, pod_id, api_key[:8] if api_key else "MISSING")
    req = urllib.request.Request(
        f"{RUNPOD_REST}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
            log.info("RunPod REST %s %s → %d %s", method, path, r.status, body[:200])
    except urllib.error.HTTPError as e:
        log.error("RunPod REST %s %s → HTTP %d: %s", method, path, e.code, e.read().decode())
        raise


def terminate_pod() -> None:
    pod_id  = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not pod_id or not api_key:
        log.warning("RUNPOD_POD_ID or RUNPOD_API_KEY not set — cannot self-terminate")
        return
    try:
        _rest("DELETE", f"/pods/{pod_id}")
        log.info("Terminated pod %s", pod_id)
    except Exception as e:
        log.error("Failed to terminate pod: %s", e)


def stop_pod() -> None:
    """Stop (pause) the pod on failure."""
    pod_id  = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not pod_id or not api_key:
        log.warning("RUNPOD_POD_ID or RUNPOD_API_KEY not set — cannot stop pod")
        return
    try:
        _rest("POST", f"/pods/{pod_id}/stop")
        log.info("Stopped pod %s", pod_id)
    except Exception as e:
        log.error("Failed to stop pod: %s", e)


# ----------------------------------------------------------------
# HuggingFace helpers
# ----------------------------------------------------------------

def hf_repo_has_adapter(repo_id: str, token: str) -> bool:
    try:
        from huggingface_hub import HfApi
        files = list(HfApi(token=token).list_repo_files(repo_id, repo_type="model"))
        return any("adapter_model" in f for f in files)
    except Exception as e:
        log.debug("hf_repo_has_adapter(%s): %s", repo_id, e)
        return False


def hf_download(repo_id: str, token: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    log.info("Downloading adapter from %s ...", repo_id)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        local_dir=str(local_dir),
    )
    log.info("Downloaded to %s", local_dir)


def hf_upload(local_dir: Path, repo_id: str, token: str, commit_message: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    log.info("Uploaded %s → https://huggingface.co/%s", local_dir.name, repo_id)


def upload_error_log(repo_id: str, token: str, text: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    except Exception as e:
        log.warning("Could not create error log repo %s: %s", repo_id, e)
        return
    for attempt in range(3):
        try:
            api.upload_file(
                path_or_fileobj=io.BytesIO(text.encode()),
                path_in_repo="pod_error.log",
                repo_id=repo_id,
                repo_type="model",
                commit_message="pod error log",
            )
            log.info("Error log uploaded to %s/pod_error.log", repo_id)
            return
        except Exception as e:
            if attempt < 2:
                log.warning("Upload attempt %d failed, retrying: %s", attempt + 1, e)
                time.sleep(5)
            else:
                log.warning("Could not upload error log after 3 attempts: %s", e)


def delete_training_data_repo(repo_id: str, token: str) -> None:
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).delete_repo(repo_id=repo_id, repo_type="model")
        log.info("Deleted temporary training data repo: %s", repo_id)
    except Exception as e:
        log.warning("Could not delete training data repo %s: %s", repo_id, e)


# ----------------------------------------------------------------
# Training subprocess
# ----------------------------------------------------------------

def _run(cmd, log_prefix="", timeout=21600):
    log.info("%sRunning: %s", log_prefix, " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    deadline = time.time() + timeout
    assert proc.stdout is not None
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
    return proc.returncode, lines[-200:]


# ----------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------

def run_pipeline(
    data_dir:    Path,
    workdir:     Path,
    model_path:  str,
    hf_repo:     str,
    hf_token:    str,
    epochs_cpt:  int,
    epochs_lora: int,
    epochs_dpo:  int,
    rank:        int,
    max_seq_len: int,
    lr_cpt:      float,
    lr_lora:     float,
    lr_dpo:      float,
    beta:        float,
    no_4bit:     bool,
    force_cpt:   bool,
    force_qlora: bool,
    force_dpo:   bool,
) -> None:
    """
    Run CPT→QLoRA→DPO.  Uploads adapter to HF after each stage.
    Skips stages whose adapter is already on HF (unless force_* is set).
    Raises on any failure.
    """
    hf_repo_cpt   = f"{hf_repo}-cpt"
    hf_repo_qlora = f"{hf_repo}-qlora"

    cpt_data  = data_dir / "cpt.txt"
    lora_data = data_dir / "lora.jsonl"
    dpo_data  = data_dir / "dpo.jsonl"

    cpt_out  = workdir / "cpt-output"
    lora_out = workdir / "lora-output"
    dpo_out  = workdir / "dpo-output"

    no4bit_flag = ["--no-4bit"] if no_4bit else []
    all_logs    = []
    t_total     = time.time()

    # ----------------------------------------------------------------
    # Stage 1 — CPT
    # ----------------------------------------------------------------
    cpt_on_hf = hf_repo_has_adapter(hf_repo_cpt, hf_token)
    if not force_cpt and (force_dpo or force_qlora or cpt_on_hf):
        if not cpt_on_hf:
            raise RuntimeError(
                f"FORCE_QLORA/FORCE_DPO set but no CPT adapter at {hf_repo_cpt}. "
                "Remove force flags to train from scratch."
            )
        log.info("=== Stage 1/3: CPT — skipping (found %s) ===", hf_repo_cpt)
        hf_download(hf_repo_cpt, hf_token, cpt_out)
    else:
        log.info("=== Stage 1/3: CPT ===")
        t0 = time.time()
        rc, lines = _run([
            sys.executable, str(TRAIN), "cpt",
            "--model",       model_path,
            "--data",        str(cpt_data),
            "--output",      str(cpt_out),
            "--epochs",      str(epochs_cpt),
            "--rank",        str(rank),
            "--max-seq-len", str(max_seq_len),
            "--lr",          str(lr_cpt),
            *no4bit_flag,
        ], log_prefix="[CPT] ")
        all_logs += lines
        if rc != 0:
            raise RuntimeError(f"CPT stage failed (exit {rc})\n" + "\n".join(lines[-50:]))
        log.info("CPT done in %.1fs — uploading checkpoint...", time.time() - t0)
        hf_upload(cpt_out, hf_repo_cpt, hf_token, f"CPT adapter after {epochs_cpt} epoch(s)")

    # ----------------------------------------------------------------
    # Stage 2 — QLoRA
    # ----------------------------------------------------------------
    qlora_on_hf = hf_repo_has_adapter(hf_repo_qlora, hf_token)
    if not force_qlora and (force_dpo or qlora_on_hf):
        if not qlora_on_hf:
            raise RuntimeError(
                f"FORCE_DPO set but no QLoRA adapter at {hf_repo_qlora}. "
                "Remove force flags to train from scratch."
            )
        log.info("=== Stage 2/3: QLoRA — skipping (found %s) ===", hf_repo_qlora)
        hf_download(hf_repo_qlora, hf_token, lora_out)
    else:
        log.info("=== Stage 2/3: QLoRA ===")
        t0 = time.time()
        rc, lines = _run([
            sys.executable, str(TRAIN), "qlora",
            "--model",       model_path,
            "--adapter",     str(cpt_out),
            "--data",        str(lora_data),
            "--output",      str(lora_out),
            "--epochs",      str(epochs_lora),
            "--rank",        str(rank),
            "--max-seq-len", str(max_seq_len),
            "--lr",          str(lr_lora),
            *no4bit_flag,
        ], log_prefix="[QLoRA] ")
        all_logs += lines
        if rc != 0:
            raise RuntimeError(f"QLoRA stage failed (exit {rc})\n" + "\n".join(lines[-50:]))
        log.info("QLoRA done in %.1fs — uploading checkpoint...", time.time() - t0)
        hf_upload(lora_out, hf_repo_qlora, hf_token, f"QLoRA adapter after {epochs_lora} epoch(s)")

    # ----------------------------------------------------------------
    # Stage 3 — DPO
    # ----------------------------------------------------------------
    if not force_dpo and hf_repo_has_adapter(hf_repo, hf_token):
        log.info("=== Stage 3/3: DPO — skipping (found %s) ===", hf_repo)
    else:
        log.info("=== Stage 3/3: DPO ===")
        t0 = time.time()
        rc, lines = _run([
            sys.executable, str(TRAIN), "dpo",
            "--model",       model_path,
            "--adapter",     str(lora_out),
            "--data",        str(dpo_data),
            "--output",      str(dpo_out),
            "--epochs",      str(epochs_dpo),
            "--rank",        str(rank),
            "--max-seq-len", str(max_seq_len),
            "--lr",          str(lr_dpo),
            "--beta",        str(beta),
            *no4bit_flag,
        ], log_prefix="[DPO] ")
        all_logs += lines
        if rc != 0:
            raise RuntimeError(f"DPO stage failed (exit {rc})\n" + "\n".join(lines[-50:]))
        log.info("DPO done in %.1fs — uploading final adapter...", time.time() - t0)
        lora_count = sum(1 for l in lora_data.read_text().splitlines() if l.strip())
        dpo_count  = sum(1 for l in dpo_data.read_text().splitlines()  if l.strip())
        hf_upload(
            dpo_out, hf_repo, hf_token,
            f"CPT→QLoRA→DPO adapter ({lora_count} QLoRA, {dpo_count} DPO pairs)",
        )

    log.info("Pipeline complete in %.1fs", time.time() - t_total)

    # Upload a success indicator file
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(hf_repo, repo_type="model", exist_ok=True, private=True)
        api.upload_file(
            path_or_fileobj=io.BytesIO(b"Training pipeline completed successfully."),
            path_in_repo="SUCCESS.txt",
            repo_id=hf_repo,
            repo_type="model",
            commit_message="Training pipeline completed successfully",
        )
        log.info("Uploaded SUCCESS.txt to %s", hf_repo)
    except Exception as e:
        log.warning("Failed to upload SUCCESS.txt to %s: %s", hf_repo, e)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def _debug_env() -> None:
    log.info("=== pod_entrypoint.py version %d ===", VERSION)
    for cmd in [
        ["which", "axolotl"],
        ["axolotl", "--version"],
        [sys.executable, "-c", "import axolotl; print(axolotl.__version__)"],
        [sys.executable, "-c", "import axolotl.cli.main; print('axolotl.cli OK')"],
        ["pip", "show", "axolotl"],
    ]:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            result = (out.stdout + out.stderr).strip()
            log.info("[env] %s → %s", " ".join(cmd), result or "(no output)")
        except Exception as e:
            log.info("[env] %s → ERROR: %s", " ".join(cmd), e)


def main() -> None:
    _debug_env()
    hf_token    = os.environ["HF_WRITE_TOKEN"]
    hf_repo     = os.environ["HF_REPO"]
    data_repo   = os.environ["TRAINING_DATA_REPO"]
    model_path  = os.environ.get("MODEL_PATH",   "unsloth/Magistral-Small-2509")
    epochs_cpt  = int(os.environ.get("EPOCHS_CPT",  "1"))
    epochs_lora = int(os.environ.get("EPOCHS_LORA", "3"))
    epochs_dpo  = int(os.environ.get("EPOCHS_DPO",  "1"))
    rank        = int(os.environ.get("RANK",        "16"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "2048"))
    lr_cpt      = float(os.environ.get("LR_CPT",   "1e-4"))
    lr_lora     = float(os.environ.get("LR_LORA",  "2e-4"))
    lr_dpo      = float(os.environ.get("LR_DPO",   "5e-5"))
    beta        = float(os.environ.get("BETA",      "0.1"))
    no_4bit     = os.environ.get("NO_4BIT",    "0") == "1"
    force_cpt   = os.environ.get("FORCE_CPT",   "0") == "1"
    force_qlora = os.environ.get("FORCE_QLORA", "0") == "1"
    force_dpo   = os.environ.get("FORCE_DPO",   "0") == "1"

    # Download training data from the temporary HF repo
    with tempfile.TemporaryDirectory(prefix="lora_pod_") as workdir:
        workdir  = Path(workdir)
        data_dir = workdir / "training-data"
        out_dir  = workdir / "outputs"
        out_dir.mkdir()

        log.info("Downloading training data from %s ...", data_repo)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=data_repo,
            repo_type="model",
            token=hf_token,
            local_dir=str(data_dir),
            ignore_patterns=["*.gitattributes"],
        )
        log.info("Downloaded training data to %s", data_dir)

        run_pipeline(
            data_dir=data_dir,
            workdir=out_dir,
            model_path=model_path,
            hf_repo=hf_repo,
            hf_token=hf_token,
            epochs_cpt=epochs_cpt,
            epochs_lora=epochs_lora,
            epochs_dpo=epochs_dpo,
            rank=rank,
            max_seq_len=max_seq_len,
            lr_cpt=lr_cpt,
            lr_lora=lr_lora,
            lr_dpo=lr_dpo,
            beta=beta,
            no_4bit=no_4bit,
            force_cpt=force_cpt,
            force_qlora=force_qlora,
            force_dpo=force_dpo,
        )


if __name__ == "__main__":
    data_repo = os.environ.get("TRAINING_DATA_REPO", "")
    hf_token  = os.environ.get("HF_WRITE_TOKEN", "")
    hf_repo   = os.environ.get("HF_REPO", "")
    success = False
    try:
        main()
        success = True
        log.info("Pipeline complete.")
    except Exception:
        import traceback
        err_text = traceback.format_exc()
        log.error("Pipeline failed:\n%s", err_text)
        if hf_repo and hf_token:
            upload_error_log(hf_repo, hf_token, err_text)
        else:
            log.error("Cannot upload error log: HF_REPO or HF_WRITE_TOKEN not set.")
    finally:
        if success:
            if data_repo and hf_token:
                delete_training_data_repo(data_repo, hf_token)
            terminate_pod()
        else:
            # On failure: preserve training data for retry, stop (not delete) pod
            stop_pod()
    # Always exit 0 — RunPod restarts containers that exit non-zero, causing a loop
    sys.exit(0)
