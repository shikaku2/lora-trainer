#!/usr/bin/env python3
"""
Axolotl-based training wrapper — CPT, QLoRA, and DPO stages.

Generates an axolotl YAML config and calls accelerate launch.
CPT and QLoRA use the unsloth plugin; DPO uses plain axolotl+TRL
(unsloth does not support DPO).

Usage:
    python train.py cpt   --model <id> --data corpus.txt   --output ./cpt-out
    python train.py qlora --model <id> --data train.jsonl  --output ./lora-out [--adapter ./cpt-out]
    python train.py dpo   --model <id> --data dpo.jsonl    --output ./dpo-out  --adapter ./lora-out
"""

VERSION = 16

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def gpu_check():
    print(f"train.py version {VERSION}")
    import torch
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected — aborting.")
        sys.exit(1)
    print(f"PyTorch: {torch.__version__}  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def run_axolotl(config_path: str):
    """Launch axolotl training with the given YAML config.

    The image installs axolotl as an editable install from /workspace/axolotl/src
    but the .pth file is broken, so axolotl.cli is unreachable without explicitly
    adding the source to PYTHONPATH.
    """
    env = os.environ.copy()
    src = "/workspace/axolotl/src"
    env["PYTHONPATH"] = src + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    num_gpus = int(os.environ.get("NUM_GPUS", "1"))
    cmd = ["axolotl", "train", config_path]
    if num_gpus > 1:
        # axolotl captures unknown CLI args in ctx.args and forwards them as
        # launcher_args to `accelerate launch`. This is the only reliable way
        # to set --num_processes before accelerate auto-detects GPUs and spawns DDP.
        cmd += ["--num_processes", "1", "--num_machines", "1",
                "--mixed_precision", "no", "--dynamo_backend", "no"]
    print(f"Running: {' '.join(cmd)}  (PYTHONPATH includes {src})")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"ERROR: axolotl exited with code {result.returncode}")
        sys.exit(result.returncode)


def base_config(args) -> dict:
    """Common axolotl config fields for LoRA-based training."""
    num_gpus = int(os.environ.get("NUM_GPUS", "1"))
    cfg = {
        "base_model":        args.model,
        "model_type":        "AutoModelForCausalLM",
        "tokenizer_type":    "AutoTokenizer",
        "trust_remote_code":           True,
        # is_multimodal is auto-detected from transformers MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
        # and ORed with user setting — cannot be forced False. axolotl 0.16.1 also requires
        # is_multimodal=True for gemma4 to inject mm_token_type_ids (needed even for text-only training).
        "load_in_4bit":                True,
        "low_cpu_mem_usage":           True,
        "adapter":           "lora",
        "lora_r":            args.rank,
        "lora_alpha":        args.rank * 2,
        "lora_dropout":      0.0,
        # Regex scoped to language_model only — prevents LoRA from touching vision layers.
        # Verified against model.safetensors.index.json:
        #   model.language_model.layers.X.self_attn.{q,k,v,o}_proj
        #   model.language_model.layers.X.mlp.{gate,up,down}_proj       (dense/shared layers)
        #   model.language_model.layers.X.experts.{gate_up,down}_proj   (MoE expert blocks, fused)
        "lora_target_modules": r"model\.language_model\.layers\.[\d]+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj|experts\.(gate_up|down)_proj)",
        "sequence_len":                  args.max_seq_len,
        "num_epochs":                    args.epochs,
        "micro_batch_size":              1,
        "gradient_accumulation_steps":   8,
        "optimizer":                     "adamw_bnb_8bit",
        "lr_scheduler":                  "cosine",
        "warmup_steps":                  10,
        "weight_decay":                  0.01,
        "max_grad_norm":                 1.0,
        "logging_steps":                 5,
        "save_strategy":                 "no",
        "bf16":                          "auto",
        "fp16":                          False,
        "flash_attention":               True,
        "gradient_checkpointing":        True,
        "pad_to_sequence_len":           True,
        "seed":                          42,
        "report_to":                     "none",
        "output_dir":                    str(args.output),
    }
    if num_gpus > 1:
        # device_map=auto distributes model layers across all visible GPUs.
        # --num_processes 1 is passed as a CLI launcher arg (see run_axolotl) to
        # prevent accelerate from spawning DDP before device_map can take effect.
        cfg["device_map"] = "auto"
    return cfg


def cpt_text_to_jsonl(text_path: str, max_seq_len: int, output_path: str) -> int:
    """
    Split CPT plain text into ~max_seq_len-token chunks and write as JSONL.
    Uses 3 chars/token as a conservative estimate so chunks don't exceed max_seq_len.
    """
    text = Path(text_path).read_text(encoding="utf-8")
    chars_per_chunk = max_seq_len * 3
    records = []
    for i in range(0, len(text), chars_per_chunk):
        chunk = text[i:i + chars_per_chunk].strip()
        if len(chunk) < 100:
            continue
        records.append({"text": chunk})
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  CPT: {len(text):,} chars → {len(records)} chunks → {output_path}")
    return len(records)


def cmd_cpt(args):
    gpu_check()

    # Chunk the plain-text corpus into JSONL records for axolotl
    cpt_jsonl = str(Path(args.data).with_suffix("")) + "_axolotl.jsonl"
    cpt_text_to_jsonl(args.data, args.max_seq_len, cpt_jsonl)

    cfg = base_config(args)
    cfg["learning_rate"] = args.lr
    cfg["sample_packing"] = True   # efficient packing for CPT
    cfg["datasets"] = [{
        "path":    cpt_jsonl,
        "ds_type": "json",
        "type":    "completion",
        "field":   "text",
    }]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="axolotl_cpt_"
    ) as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        config_path = f.name

    print(f"CPT config: {config_path}")
    try:
        run_axolotl(config_path)
    finally:
        os.unlink(config_path)


def cmd_qlora(args):
    gpu_check()

    cfg = base_config(args)
    cfg["learning_rate"]  = args.lr
    cfg["chat_template"]  = "gemma4"
    cfg["train_on_inputs"] = True  # diagnostic: bypass roles_to_train masking
    cfg["datasets"] = [{
        "path":           str(args.data),
        "ds_type":        "json",
        "type":           "chat_template",
        "field_messages": "messages",
    }]
    if args.adapter:
        cfg["lora_model_dir"] = str(args.adapter)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="axolotl_qlora_"
    ) as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        config_path = f.name

    print(f"QLoRA config: {config_path}")
    try:
        run_axolotl(config_path)
    finally:
        os.unlink(config_path)


def cmd_dpo(args):
    gpu_check()

    # chat_template.default expects {messages: [{role, content}...], chosen, rejected}
    # where messages is the prompt context and chosen/rejected are plain response strings.
    dpo_jsonl = str(Path(args.data).with_suffix("")) + "_axolotl.jsonl"
    records = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            system = r.get("system", "")
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": r["prompt"]})
            records.append({
                "messages": messages,
                "chosen":   r["chosen"],
                "rejected": r["rejected"],
            })
    with open(dpo_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  DPO: {len(records)} pairs → {dpo_jsonl}")

    cfg = base_config(args)
    cfg["learning_rate"]              = args.lr
    cfg["chat_template"]              = "gemma4"
    cfg["rl"]                         = "dpo"
    cfg["rl_beta"]                    = args.beta
    cfg["lora_model_dir"]             = str(args.adapter)
    cfg["merge_adapters_by_default"]  = False   # don't merge+save the 24GB base model
    cfg["warmup_steps"]               = 5
    cfg["max_length"]                 = args.max_seq_len
    cfg["max_prompt_length"]          = args.max_seq_len // 2
    cfg["datasets"] = [{
        "path":           dpo_jsonl,
        "ds_type":        "json",
        "type":           "chat_template.default",
        "field_messages": "messages",
        "field_chosen":   "chosen",
        "field_rejected": "rejected",
    }]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="axolotl_dpo_"
    ) as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        config_path = f.name

    print(f"DPO config: {config_path}")
    try:
        run_axolotl(config_path)
    finally:
        os.unlink(config_path)

    # Belt-and-suspenders: remove any merged base model weights axolotl may have
    # written to output_dir — we only want the adapter files for HF upload.
    output_path = Path(args.output)
    for pattern in ["model.safetensors", "model-*.safetensors",
                    "pytorch_model.bin", "pytorch_model-*.bin"]:
        for f in output_path.glob(pattern):
            print(f"  Removing merged model file {f.name} ({f.stat().st_size // 1024 // 1024}MB)")
            f.unlink()


def main():
    parser = argparse.ArgumentParser(description="Axolotl-based CPT / QLoRA / DPO trainer")
    sub = parser.add_subparsers(dest="stage", required=True)

    def add_common(p):
        p.add_argument("--model",       default="unsloth/gemma-4-26B-A4B-it")
        p.add_argument("--data",        required=True)
        p.add_argument("--output",      required=True)
        p.add_argument("--epochs",      type=int,   default=1)
        p.add_argument("--max-seq-len", type=int,   default=2048, dest="max_seq_len")
        p.add_argument("--rank",        type=int,   default=16)

    p_cpt = sub.add_parser("cpt",   help="Continued pre-training on plain text")
    add_common(p_cpt)
    p_cpt.add_argument("--lr", type=float, default=1e-4)

    p_qlora = sub.add_parser("qlora", help="QLoRA instruction fine-tuning")
    add_common(p_qlora)
    p_qlora.add_argument("--adapter", default=None, help="CPT adapter dir to resume from")
    p_qlora.add_argument("--lr",      type=float,   default=2e-4)

    p_dpo = sub.add_parser("dpo",   help="DPO preference alignment")
    add_common(p_dpo)
    p_dpo.add_argument("--adapter", required=True, help="QLoRA adapter dir to build on")
    p_dpo.add_argument("--lr",      type=float,   default=5e-5)
    p_dpo.add_argument("--beta",    type=float,   default=0.1)

    args = parser.parse_args()
    {"cpt": cmd_cpt, "qlora": cmd_qlora, "dpo": cmd_dpo}[args.stage](args)


if __name__ == "__main__":
    main()
