#!/usr/bin/env python3
"""
submit_lora_job.py — submit a full CPT→QLoRA→DPO training run as a RunPod pod.

Uploads training files to a temporary private HF repo, creates a RunPod pod, polls
until done, then cleans everything up.

On failure the pod is paused and state is written to .lora_trainer.state.
Run the script again to restart: the pod is patched with the latest Docker image digest
and updated env vars, then resumed on the same machine (Docker layers may be cached).
Training resumes from whatever HF checkpoints were already uploaded.

To abandon a failed run and start completely fresh, delete .lora_trainer.state.

Usage:
  RUNPOD_API_KEY=rp_xxx HF_WRITE_TOKEN=hf_xxx python3 submit_lora_job.py

Required env vars:
  RUNPOD_API_KEY        RunPod API key
  HF_WRITE_TOKEN        HuggingFace write token

Optional env vars (with defaults):
  CPT_FILE        plain-text CPT corpus            [cpt.txt]
  LORA_FILE       dialogue examples (txt or jsonl) [lora.txt]
  DPO_FILE        DPO preference pairs (txt or jsonl) [dpo.txt]
  HF_REPO         HuggingFace repo for final adapter  [shikaku2/magistral-alastor-lora]
  MODEL_PATH      base model HF repo or local path [unsloth/Magistral-Small-2509]
  GPU_TYPE        RunPod GPU type ID               [NVIDIA A40]
  GITHUB_REPO     URL of this repo for script injection [https://github.com/shikaku2/lora-trainer3]
  GH_TOKEN        GitHub token (required for private repos, used to clone at pod startup)
  EPOCHS_CPT      CPT epochs                       [1]
  EPOCHS_LORA     QLoRA epochs                     [3]
  EPOCHS_DPO      DPO epochs                       [1]
  RANK            LoRA rank                        [16]
  MAX_SEQ_LEN     token sequence length cap        [2048]
  FORCE_CPT       re-run CPT even if cached (0/1)  [0]
  FORCE_QLORA     re-run QLoRA even if cached      [0]
  FORCE_DPO       re-run DPO even if cached        [0]
  LR_CPT          CPT learning rate                [1e-4]
  LR_LORA         QLoRA learning rate              [2e-4]
  LR_DPO          DPO learning rate                [5e-5]
  BETA            DPO beta                         [0.1]

  GITHUB_REPO     scripts repo URL                 [https://github.com/shikaku2/lora-trainer]
  GH_TOKEN        GitHub token for private repo clone (optional)
"""

import base64
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

from huggingface_hub import HfApi

STATE_FILE = Path(".lora_trainer.state")


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return None


def save_state(data):
    STATE_FILE.write_text(json.dumps(data, indent=2))


def clear_state():
    STATE_FILE.unlink(missing_ok=True)


def estimate_disk_gb(model_path, hf_token, cpt_bytes, lora_bytes, dpo_bytes, rank):
    """
    Estimate pod disk usage.

    Returns (model_gb, adapter_gb_each, adapters_gb, data_gb, total_gb).

    Model size: actual weight file sizes queried from the HF Hub.
    Adapters:   ~0.4% of model bf16 weight size per stage, scaled by rank/16.
                (Derived from: 7 target modules × all layers × 2×rank×hidden at bf16.)
    Data caches: 2× the raw file sizes (Arrow tokenized datasets).
    Overhead:   3 GB fixed (OS, unsloth compiled cache, temp files).
    """
    model_gb = 0.0
    if not Path(model_path).exists():  # skip local paths
        try:
            info = HfApi(token=hf_token).model_info(model_path, files_metadata=True)
            model_gb = sum(
                f.size for f in info.siblings
                if f.size and f.rfilename.endswith((".safetensors", ".bin"))
            ) / 1e9
        except Exception:
            pass  # non-critical; disk allocation falls back to minimum

    adapter_gb_each = model_gb * (rank / 16) * 0.004
    adapters_gb = adapter_gb_each * 3                  # CPT + QLoRA + DPO

    data_gb = (len(cpt_bytes) + len(lora_bytes) + len(dpo_bytes)) * 2 / 1e9

    overhead_gb = 3.0
    total_gb = model_gb + adapters_gb + data_gb + overhead_gb
    return model_gb, adapter_gb_each, adapters_gb, data_gb, total_gb


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
        ---

    If a SYSTEM block is present it is injected into every [INST] block.
    Output: one {"text": "[INST] SYSTEM\\n\\nUSER [/INST] REPLY"} per line.
    """
    text = Path(text_path).read_text()
    blocks = re.split(r"\n(?:=====|-{2,})\n?", text)

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


def parse_dpo_examples(text_path: str) -> bytes:
    """
    Parse dpo.txt format into JSONL bytes.

    File format:
        ====...====
        EXAMPLE N
        ====...====
        PROMPT:
        SYSTEM: <system text>

        USER: <user text>

        CHOSEN:
        <chosen response>

        REJECTED:
        <rejected response>

    Output: one {"system": ..., "prompt": ..., "chosen": ..., "rejected": ...} per line.
    """
    text = Path(text_path).read_text()
    # Split on separator lines (=====...)
    blocks = re.split(r"={10,}\n?", text)

    records = []
    i = 0
    while i < len(blocks):
        block = blocks[i].strip()
        # Skip header blocks (EXAMPLE N lines and empty blocks)
        if not block or re.match(r"EXAMPLE\s+\d+", block):
            i += 1
            continue

        # This block should start with PROMPT:
        if not block.startswith("PROMPT:"):
            i += 1
            continue

        prompt_m  = re.search(r"^PROMPT:\s*\nSYSTEM:\s*(.+?)(?=\nUSER:)", block,
                               re.DOTALL | re.MULTILINE)
        user_m    = re.search(r"^USER:\s*(.+?)$", block, re.DOTALL | re.MULTILINE)
        chosen_m  = re.search(r"^CHOSEN:\s*\n(.+?)(?=\nREJECTED:|\Z)", block,
                               re.DOTALL | re.MULTILINE)
        rejected_m = re.search(r"^REJECTED:\s*\n(.+?)$", block, re.DOTALL | re.MULTILINE)

        if user_m and chosen_m and rejected_m:
            system = prompt_m.group(1).strip() if prompt_m else ""
            records.append(json.dumps({
                "system":   system,
                "prompt":   user_m.group(1).strip(),
                "chosen":   chosen_m.group(1).strip(),
                "rejected": rejected_m.group(1).strip(),
            }, ensure_ascii=False))
        i += 1

    if not records:
        print(f"ERROR: No DPO examples parsed from {text_path}")
        sys.exit(1)

    print(f"  Parsed {len(records)} DPO pairs from {Path(text_path).name}")
    return ("\n".join(records) + "\n").encode()


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
api_key       = env("RUNPOD_API_KEY",  required=True)
hf_token      = env("HF_WRITE_TOKEN",  required=True)

cpt_file      = env("CPT_FILE",   "cpt.txt")
lora_file     = env("LORA_FILE",  "lora.txt")
dpo_file      = env("DPO_FILE",   "dpo.txt")
hf_repo       = env("HF_REPO",   "shikaku2/magistral-alastor-lora")
model_path    = env("MODEL_PATH", "unsloth/Magistral-Small-2509")
github_repo   = env("GITHUB_REPO", "https://github.com/shikaku2/lora-trainer")
gpu_type      = env("GPU_TYPE",   "NVIDIA A40")
max_seq_len   = int(env("MAX_SEQ_LEN",  "2048"))
epochs_cpt    = int(env("EPOCHS_CPT",   "1"))
epochs_lora   = int(env("EPOCHS_LORA",  "3"))
epochs_dpo    = int(env("EPOCHS_DPO",   "1"))
rank          = int(env("RANK",         "16"))
force_cpt     = env("FORCE_CPT",   "0") == "1"
force_qlora   = env("FORCE_QLORA", "0") == "1"
force_dpo     = env("FORCE_DPO",   "0") == "1"
lr_cpt        = env("LR_CPT",   "1e-4")
lr_lora       = env("LR_LORA",  "2e-4")
lr_dpo        = env("LR_DPO",   "5e-5")
beta          = env("BETA",     "0.1")

training_data_repo = f"{hf_repo}-training-data"

RUNPOD_REST = "https://rest.runpod.io/v1"


def _rest(method: str, path: str, body=None):
    """Make a RunPod REST API call. Returns parsed JSON response."""
    req = urllib.request.Request(
        f"{RUNPOD_REST}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def resolve_image_digest(image: str, gh_token: str = "") -> str:
    """
    Resolve a Docker image :tag to an @sha256:digest reference.
    This guarantees RunPod re-pulls the image on PATCH even when the tag is :latest.
    Returns the original image string unchanged if resolution fails.
    """
    if "@sha256:" in image:
        return image  # already base_image to a digest

    # Split base:tag
    last_segment = image.split("/")[-1]
    if ":" in last_segment:
        base, tag = image.rsplit(":", 1)
    else:
        base, tag = image, "latest"

    # Derive registry and repository path
    first, *rest = base.split("/")
    if "." in first or ":" in first:   # has a hostname (e.g. ghcr.io)
        registry = first
        repo = "/".join(rest)
    else:
        registry = "registry-1.docker.io"
        repo = base if "/" in base else f"library/{base}"

    # Fetch bearer token — GHCR supports anonymous pull for public images;
    # pass GH_TOKEN as Basic auth password for private packages.
    try:
        auth_headers = {}
        if gh_token:
            creds = base64.b64encode(f"token:{gh_token}".encode()).decode()
            auth_headers["Authorization"] = f"Basic {creds}"
        token_req = urllib.request.Request(
            f"https://{registry}/token?scope=repository:{repo}:pull&service={registry}",
            headers=auth_headers,
        )
        with urllib.request.urlopen(token_req, timeout=10) as r:
            bearer = json.loads(r.read()).get("token", "")
    except Exception as e:
        print(f"  (image digest: token fetch failed — {e})")
        return image

    # HEAD the manifest — the registry returns Docker-Content-Digest in the headers
    try:
        manifest_req = urllib.request.Request(
            f"https://{registry}/v2/{repo}/manifests/{tag}",
            headers={
                "Authorization": f"Bearer {bearer}",
                "Accept": ", ".join([
                    "application/vnd.oci.image.index.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.v2+json",
                ]),
            },
            method="HEAD",
        )
        with urllib.request.urlopen(manifest_req, timeout=10) as r:
            digest = r.headers.get("Docker-Content-Digest", "")
        if digest:
            return f"{base}@{digest}"
        print(f"  (image digest: no Docker-Content-Digest header returned)")
    except Exception as e:
        print(f"  (image digest: manifest fetch failed — {e})")

    return image

# ----------------------------------------------------------------
# Check for existing state (restart mode)
# ----------------------------------------------------------------
state = load_state()
if state:
    print(f"\nFound {STATE_FILE} — resuming from previous run")
    print(f"  Existing pod: {state['pod_id']}")
    print(f"  Will patch pod with latest image and resume on same machine")

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
if dpo_file.endswith(".jsonl"):
    dpo_bytes = Path(dpo_file).read_bytes()
    count = sum(1 for l in dpo_bytes.decode().splitlines() if l.strip())
    print(f"  (JSONL — {count} records)")
else:
    dpo_bytes = parse_dpo_examples(dpo_file)

# ----------------------------------------------------------------
# Disk estimate
# ----------------------------------------------------------------
print("\nEstimating disk requirements...")
model_gb, adapter_gb_each, adapters_gb, data_gb, total_gb = estimate_disk_gb(
    model_path, hf_token, cpt_bytes, lora_bytes, dpo_bytes, rank,
)
container_disk_gb = max(80, int(total_gb * 1.25))

if model_gb:
    print(f"  Model weights:  {model_gb:.1f} GB")
else:
    print(f"  Model weights:  unknown (local path or HF query failed)")
print(f"  LoRA adapters:  {adapters_gb:.2f} GB  (3 stages × {adapter_gb_each:.2f} GB, rank {rank})")
print(f"  Dataset caches: {data_gb * 1000:.1f} MB")
print(f"  ──────────────────────────────────────────")
print(f"  Container disk: {container_disk_gb} GB")

# ----------------------------------------------------------------
# Upload training files to temporary HF repo
# ----------------------------------------------------------------
print(f"\nUploading training data to {training_data_repo}...")
try:
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
# Build pod environment (used for both fresh and restart paths)
# ----------------------------------------------------------------
pod_env = {
    "HF_TOKEN":           hf_token,
    "HF_WRITE_TOKEN":     hf_token,
    "HF_REPO":            hf_repo,
    "TRAINING_DATA_REPO": training_data_repo,
    "MODEL_PATH":         model_path,
    "RUNPOD_ACCT_KEY":    api_key,
    "EPOCHS_CPT":         str(epochs_cpt),
    "EPOCHS_LORA":        str(epochs_lora),
    "EPOCHS_DPO":         str(epochs_dpo),
    "RANK":               str(rank),
    "MAX_SEQ_LEN":        str(max_seq_len),
    "FORCE_CPT":          "1" if force_cpt  else "0",
    "FORCE_QLORA":        "1" if force_qlora else "0",
    "FORCE_DPO":          "1" if force_dpo   else "0",
    "LR_CPT":             str(lr_cpt),
    "LR_LORA":            str(lr_lora),
    "LR_DPO":             str(lr_dpo),
    "BETA":               str(beta),

    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    **({ "CUDA_LAUNCH_BLOCKING": os.environ["CUDA_LAUNCH_BLOCKING"] }
       if "CUDA_LAUNCH_BLOCKING" in os.environ else {}),
}

# ----------------------------------------------------------------
# Pod helpers
# ----------------------------------------------------------------
_RETRY_DELAYS = [2, 4, 8]  # seconds between attempts before giving up


def _create_fresh_pod(base_image: str) -> str:
    """Create a brand-new pod. Retries with _RETRY_DELAYS on failure. Returns pod_id."""
    for attempt, delay in enumerate([0] + _RETRY_DELAYS, 1):
        if delay:
            print(f"  Retrying pod creation in {delay}s (attempt {attempt}/{1 + len(_RETRY_DELAYS)})...")
            time.sleep(delay)
        try:
            pod = _rest("POST", "/pods", {
                "name":               f"lora-training-{int(time.time())}",
                "imageName":          base_image,
                "gpuTypeIds":         [gpu_type],
                "cloudType":          "SECURE",
                "gpuCount":           1,
                "containerDiskInGb":  container_disk_gb,
                "volumeInGb":         0,
                "env":                pod_env,
                "dockerEntrypoint":   startup_entrypoint,
            })
            pid = pod["id"]
            print(f"  Pod created: {pid}")
            return pid
        except Exception as e:
            print(f"  Pod creation failed: {e}")
    print("ERROR: Could not create pod after all retries.")
    try:
        HfApi(token=hf_token).delete_repo(training_data_repo, repo_type="model")
    except Exception:
        pass
    sys.exit(1)


# ----------------------------------------------------------------
# Create pod (fresh) or patch + resume existing pod (restart)
# ----------------------------------------------------------------
gh_token   = os.environ.get("GH_TOKEN", "")
base_image = resolve_image_digest("axolotlai/axolotl:main-latest", gh_token)

# Inject token into clone URL for private repos, e.g. https://TOKEN@github.com/...
_clone_url = (github_repo or "").replace("https://", f"https://{gh_token}@") if gh_token else (github_repo or "")
startup_entrypoint = [
    "bash", "-c",
    "rm -rf /workspace/lora-trainer && "
    f"git clone {_clone_url} /workspace/lora-trainer && "
    "python3 -u /workspace/lora-trainer/pod_entrypoint.py"
]

if state:
    # ── Restart: just start the stopped pod — entrypoint pulls fresh code from GitHub ──
    pod_id = state["pod_id"]
    print(f"\nResuming pod {pod_id} (entrypoint will pull latest code from {github_repo})...")
    started = False
    for attempt, delay in enumerate([0] + _RETRY_DELAYS, 1):
        if delay:
            print(f"  GPU unavailable — retrying in {delay}s (attempt {attempt}/{1 + len(_RETRY_DELAYS)})...")
            time.sleep(delay)
        try:
            _rest("POST", f"/pods/{pod_id}/start")
            print(f"  Pod {pod_id} started.")
            started = True
            break
        except Exception as e:
            print(f"  Start failed: {e}")

    if not started:
        print(f"\n  Could not start pod {pod_id} — GPU may no longer be available.")
        print(f"  Terminating old pod and creating a fresh one...")
        try:
            _rest("DELETE", f"/pods/{pod_id}")
            print(f"  Old pod {pod_id} terminated.")
        except Exception as e:
            print(f"  Could not terminate old pod ({e}) — continuing anyway.")
        clear_state()
        pod_id = _create_fresh_pod(base_image)
else:
    # ── Fresh run: create pod ──
    print(f"\nCreating RunPod pod ({gpu_type}, {base_image})...")
    pod_id = _create_fresh_pod(base_image)

# Save state immediately so we can recover on interrupt or failure
save_state({"pod_id": pod_id})
print(f"  State saved to {STATE_FILE}  (rerun this script to retry on failure)")
print(f"  Dashboard:   https://www.runpod.io/console/pods/{pod_id}")

# ----------------------------------------------------------------
# Poll until pod terminates or pauses
# ----------------------------------------------------------------
print("\nWaiting for pod to complete...")
start = time.time()
last_status = ""
POLLING_TIMEOUT_SECONDS = 24 * 3600 # 24 hours
pod_final_status = None # To store the status when the loop breaks

while True:
    time.sleep(20)
    elapsed = int(time.time() - start)

    if elapsed > POLLING_TIMEOUT_SECONDS:
        print(f"\n[{elapsed:5d}s] Polling timed out after {POLLING_TIMEOUT_SECONDS} seconds.")
        pod_final_status = "TIMED_OUT"
        break

    try:
        pod_info = _rest("GET", f"/pods/{pod_id}")
        if pod_info is None:
            print(f"\n[{elapsed:5d}s] Pod terminated (no longer found)")
            pod_final_status = "TERMINATED" # Assume terminated if not found
            break
        status = pod_info.get("desiredStatus") or pod_info.get("runtime", {}).get("status") or "UNKNOWN"
        if status != last_status:
            print(f"[{elapsed:5d}s] {status}")
            last_status = status
        else:
            print(f"[{elapsed:5d}s] {status}", end="\r")

        if status in ("EXITED", "TERMINATED", "DEAD", "STOPPED", "PAUSED"):
            print()
            pod_final_status = status
            break
    except Exception as e:
        print(f"\n[{elapsed:5d}s] Status check error: {e} — pod may have terminated")
        pod_final_status = "UNKNOWN_ERROR"
        break
# ----------------------------------------------------------------
# Check result via HF
# ----------------------------------------------------------------
print(f"\nChecking HuggingFace repo for results...")
success = False
try:
    explicit_error_logged = False
    api = HfApi(token=hf_token)

    try:
        files = list(api.list_repo_files(hf_repo, repo_type="model"))
    except Exception:
        files = []

    if "SUCCESS.txt" in files:
        print(f"  SUCCESS.txt found in {hf_repo}. Training complete!")
        success = True
    elif "pod_error.log" in files:
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
        explicit_error_logged = True
    elif any("adapter_model" in f for f in files): # Fallback if SUCCESS.txt isn't explicitly uploaded
        print(f"  Adapter found at https://huggingface.co/{hf_repo}")
        print("  Training complete (no explicit SUCCESS.txt found, assuming success).")
        success = True
    else:
        print(f"  No SUCCESS.txt, pod_error.log, or adapter found in {hf_repo}.")
        if pod_final_status in ("STOPPED", "PAUSED"):
            print(f"  Pod was paused. This usually indicates a failure within the pod that wasn't explicitly logged.")
        elif pod_final_status in ("TERMINATED", "EXITED"):
            print(f"  Pod terminated without clear success or error log. This is unexpected.")
        else:
            print(f"  Pod status: {pod_final_status}. Training may have failed silently or encountered an unknown issue.")
except Exception as e:
    print(f"  Could not check HF repo: {e}")
# ----------------------------------------------------------------
# Cleanup on success / instructions on failure
# ----------------------------------------------------------------
if success:
    print("\nCleaning up...")
    try:
        _rest("DELETE", f"/pods/{pod_id}")
        print(f"  Pod {pod_id} terminated.")
    except Exception as e:
        print(f"  Pod {pod_id} already gone ({e}).")
    clear_state()
    print(f"  {STATE_FILE} cleared.")
elif not explicit_error_logged: # Only suggest rerun if no explicit error log was found
    print(f"\n  Pod is paused. Rerun this script to retry with the latest Docker image.")
    print(f"  To start completely fresh: rm {STATE_FILE}")
    sys.exit(1)
