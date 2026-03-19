#!/usr/bin/env python3
"""
Alastor LoRA trainer via RunPod Flash.

Usage:
  pip install runpod-flash
  flash login  (or export RUNPOD_API_KEY=...)

  JSONL_FILE=alastor_train.jsonl \
  HF_WRITE_TOKEN=hf_xxx \
  HF_REPO=shikaku2/magistral-alastor-lora \
  python3 lora_trainer.py
"""

import asyncio
import base64
import os
import sys
from pathlib import Path

from runpod_flash import Endpoint, GpuGroup, NetworkVolume

@Endpoint(
    name="lora-trainer",
    gpu=GpuGroup.AMPERE_48,          # A40/A6000 48GB — comfortable for 22B 4-bit
    workers=(0, 1),
    idle_timeout=120,
    execution_timeout_ms=0,           # unlimited — training takes ~17min
    dependencies=[
        "mistral-common",
        "peft",
        "bitsandbytes",
        "transformers",
        "datasets",
        "accelerate",
        "huggingface_hub",
        "hf_transfer",
    ],
    volume=NetworkVolume(name="lora-trainer-cache", size=100),
    env={
        "HF_HOME":                  "/runpod-volume/huggingface-cache",
        "HUGGINGFACE_HUB_CACHE":    "/runpod-volume/huggingface-cache/hub",
        "TRANSFORMERS_CACHE":       "/runpod-volume/huggingface-cache/hub",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    },
)
async def train_lora(
    jsonl_b64:  str,
    hf_token:   str,
    hf_repo:    str,
    model_path: str   = "mistralai/Magistral-Small-2509",
    epochs:     int   = 3,
    rank:       int   = 16,
    max_seq_len:int   = 2048,
    lr:         float = 2e-4,
    no_4bit:    bool  = False,
):
    # ---- ALL imports must be inside the function (cloudpickle rule) ----
    import base64
    import json
    import os
    import subprocess
    import sys
    import tempfile
    import time
    import torch
    from pathlib import Path

    os.environ["HF_TOKEN"] = hf_token

    # Write JSONL
    tmp = Path(tempfile.mkdtemp(prefix="lora_"))
    jsonl_path = tmp / "train.jsonl"
    jsonl_path.write_bytes(base64.b64decode(jsonl_b64))
    line_count = sum(1 for l in jsonl_path.read_text().splitlines() if l.strip())

    # ---- Tokenize ----
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from huggingface_hub import hf_hub_download
    from datasets import Dataset

    try:
        tekken = Path(hf_hub_download(repo_id=model_path, filename="tekken.json",
                                       token=hf_token))
    except Exception:
        tekken = Path(hf_hub_download(repo_id=model_path, filename="tokenizer.model",
                                       token=hf_token))

    mc_tok = MistralTokenizer.from_file(str(tekken))
    encode = mc_tok.instruct_tokenizer.tokenizer.encode

    records, skipped = [], 0
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ids = encode(json.loads(line)["text"], True, True)
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
            skipped += 1
        records.append({"input_ids": ids, "labels": ids.copy()})

    dataset = Dataset.from_list(records)

    # ---- Load model ----
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer

    use_4bit = not no_4bit and torch.cuda.is_available()
    output_dir = tmp / "lora-output"

    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb,
            device_map="auto", token=hf_token, trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto", token=hf_token, trust_remote_code=True,
        )

    model.config.use_cache = False

    # Freeze vision encoder
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in ("vision", "patch", "pixel")):
            param.requires_grad = False

    # ---- LoRA ----
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    model = get_peft_model(model, LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ))

    # ---- Collator ----
    class SimpleCollator:
        def __call__(self, features):
            max_len = ((max(len(f["input_ids"]) for f in features) + 7) // 8) * 8
            input_ids, labels, attention_mask = [], [], []
            for f in features:
                pad = max_len - len(f["input_ids"])
                input_ids.append(f["input_ids"] + [0] * pad)
                labels.append(f["labels"] + [-100] * pad)
                attention_mask.append([1] * len(f["input_ids"]) + [0] * pad)
            return {
                "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
                "labels":         torch.tensor(labels,         dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    # ---- Train ----
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            warmup_steps=10,
            learning_rate=lr,
            bf16=bf16,
            fp16=torch.cuda.is_available() and not bf16,
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        ),
        train_dataset=dataset,
        data_collator=SimpleCollator(),
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    model.save_pretrained(str(output_dir))

    # ---- Upload to HF ----
    from huggingface_hub import HfApi
    HfApi(token=hf_token).upload_folder(
        folder_path=str(output_dir),
        repo_id=hf_repo,
        repo_type="model",
        commit_message=f"LoRA adapter trained for {elapsed:.0f}s on {line_count} examples",
    )

    return {
        "status": "ok",
        "message": f"Training complete in {elapsed:.0f}s. Uploaded to https://huggingface.co/{hf_repo}",
        "training_seconds": round(elapsed),
        "examples": line_count,
        "skipped": skipped,
    }


async def main():
    # ---- Preflight ----
    import json
    import urllib.request
    import urllib.error
    import urllib.parse

    def env(key, default=None, required=False):
        val = os.environ.get(key, default)
        if required and not val:
            print(f"ERROR: {key} is required.")
            sys.exit(1)
        return val

    jsonl_file  = env("JSONL_FILE",     "alastor_train.jsonl")
    hf_token    = env("HF_WRITE_TOKEN", required=True)
    hf_repo     = env("HF_REPO",        "shikaku2/magistral-alastor-lora")
    model_path  = env("MODEL_PATH",     "mistralai/Magistral-Small-2509")
    epochs      = int(env("EPOCHS",     "3"))
    rank        = int(env("RANK",       "16"))

    # Check HF access before burning GPU time
    print(f"Checking HuggingFace access to {hf_repo}...")
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/api/models/{urllib.parse.quote(hf_repo, safe='/')}",
            headers={"Authorization": f"Bearer {hf_token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            info = json.loads(r.read())
        print(f"HF repo OK ({len(info.get('siblings', []))} files currently)")
    except urllib.error.HTTPError as e:
        msgs = {401: "token invalid", 403: "no write access", 404: "repo not found"}
        print(f"ERROR: {msgs.get(e.code, f'HTTP {e.code}')}: {e.read().decode()}")
        sys.exit(1)

    # Encode JSONL
    print(f"Encoding {jsonl_file}...")
    jsonl_b64 = base64.b64encode(Path(jsonl_file).read_bytes()).decode()
    print(f"Payload: {len(jsonl_b64) / 1024:.1f} KB (limit 10MB)")
    if len(jsonl_b64) > 9 * 1024 * 1024:
        print("ERROR: JSONL exceeds 10MB Flash payload limit.")
        sys.exit(1)

    # Submit
    print("Submitting training job...")
    result = await train_lora(
        jsonl_b64=jsonl_b64,
        hf_token=hf_token,
        hf_repo=hf_repo,
        model_path=model_path,
        epochs=epochs,
        rank=rank,
    )
    print("\n=== Result ===")
    print(json.dumps(result, indent=2))

    if result.get("status") == "ok":
        print(f"\nDownloading adapter to ./alastor-lora/...")
        import subprocess
        subprocess.run([
            sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
            "download", hf_repo,
            "--local-dir", "./alastor-lora",
            "--token", hf_token,
        ], check=True)
        print("Done! Adapter saved to ./alastor-lora/")


if __name__ == "__main__":
    asyncio.run(main())
