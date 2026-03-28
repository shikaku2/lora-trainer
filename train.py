#!/usr/bin/env python3
"""
Unified training script — CPT, QLoRA, and DPO stages.

Usage:
    python train.py cpt   --model <id> --data corpus.txt   --output ./cpt-out
    python train.py qlora --model <id> --data train.jsonl  --output ./lora-out [--adapter ./cpt-out]
    python train.py dpo   --model <id> --data dpo.jsonl    --output ./dpo-out   --adapter ./lora-out
"""

import argparse
import json
import os
os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # unsloth device_map conflicts with this env var
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Import unsloth before anything that might pull in transformers/peft
from unsloth import FastLanguageModel, PatchDPOTrainer
import sys
import torch
from pathlib import Path


# ----------------------------------------------------------------
# Shared utilities
# ----------------------------------------------------------------

def check_model_cached(model_path: str) -> None:
    """Exit early if model_path is not present in the HF cache (or as a local path)."""
    local = Path(model_path)
    if local.exists():
        return
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_name = "models--" + model_path.replace("/", "--")
    snapshots_dir = hf_home / "hub" / cache_name / "snapshots"
    if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
        print(f"ERROR: Model '{model_path}' not found in HF cache.")
        print(f"  Expected: {snapshots_dir}")
        sys.exit(1)
    print(f"  Model cache verified: {snapshots_dir}")


def load_tokenizer(model_path: str, token: str = None):
    """
    Returns encode(text, bos, eos) -> list[int] for Mistral-family models.
    Tries mistral_common (tekken.json / tokenizer.model) first — official Mistral models
    need this because AutoTokenizer has a broken regex.
    Falls back to AutoTokenizer for merged/derived models that only ship tokenizer.json.
    """
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from huggingface_hub import hf_hub_download

    local = Path(model_path)
    if local.exists():
        for filename in ("tekken.json", "tokenizer.model"):
            tekken = local / filename
            if tekken.exists():
                tok = MistralTokenizer.from_file(str(tekken))
                return tok.instruct_tokenizer.tokenizer.encode
    else:
        for filename in ("tekken.json", "tokenizer.model"):
            try:
                tekken = Path(hf_hub_download(repo_id=model_path, filename=filename,
                                              local_files_only=True))
                tok = MistralTokenizer.from_file(str(tekken))
                return tok.instruct_tokenizer.tokenizer.encode
            except Exception:
                pass

    print("  No tekken.json/tokenizer.model found — falling back to AutoTokenizer")
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    _tok_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")
    _tok_kwargs = dict(local_files_only=True, token=_tok_token)
    hf_tok = None
    for _attempt, _kwargs in [
        ("AutoTokenizer slow",           dict(**_tok_kwargs, trust_remote_code=True, use_fast=False)),
        ("AutoTokenizer fast",           dict(**_tok_kwargs, trust_remote_code=True, use_fast=True)),
        ("PreTrainedTokenizerFast",      _tok_kwargs),
    ]:
        try:
            cls = PreTrainedTokenizerFast if "PreTrainedTokenizerFast" in _attempt else AutoTokenizer
            hf_tok = cls.from_pretrained(model_path, **_kwargs)
            print(f"  Loaded tokenizer via {_attempt}")
            break
        except Exception as e:
            print(f"  {_attempt} failed: {e}")
    if hf_tok is None:
        raise RuntimeError(f"Could not load any tokenizer for {model_path}")

    def encode(text: str, bos: bool = True, eos: bool = True):
        ids = hf_tok.encode(text, add_special_tokens=False)
        if bos and hf_tok.bos_token_id is not None:
            ids = [hf_tok.bos_token_id] + ids
        if eos and hf_tok.eos_token_id is not None:
            ids = ids + [hf_tok.eos_token_id]
        return ids

    return encode


def load_dpo_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    attempts = [
        {"use_fast": False, "trust_remote_code": True},
        {"use_fast": True,  "trust_remote_code": True},
        {"use_fast": False},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, **kwargs)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "right"  # unsloth assumes right-padding in its embedding mask
            print(f"  Loaded tokenizer with {kwargs}")
            return tok
        except Exception as e:
            last_error = e
            print(f"  Tokenizer attempt failed {kwargs}: {e}")
    raise RuntimeError(f"Failed to load tokenizer for {model_path}: {last_error}")


class SimpleCollator:
    """Pads input_ids/labels to longest sequence; masks padding with -100."""
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = ((max_len + 7) // 8) * 8
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            ids = f["input_ids"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def load_model(model_path: str, use_4bit: bool, max_seq_len: int):
    import transformers.modeling_utils as _mu
    if hasattr(_mu, "caching_allocator_warmup"):
        _mu.caching_allocator_warmup = lambda *a, **kw: None
    print(f"\nLoading model (4-bit={use_4bit})...")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(f"  CUDA device 0: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA device 0 memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=use_4bit,
        device_map={"": 0},  # force all layers onto cuda:0
    )
    return model


def gpu_check():
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected — aborting.")
        sys.exit(1)
    print(f"PyTorch: {torch.__version__}  Device: {torch.cuda.get_device_name(0)}")


def apply_lora(model, rank: int):
    return FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )


def make_training_args(output, epochs, batch_size, grad_accum, lr, bf16, fp16,
                       warmup_steps=10, cls=None, **extra):
    from transformers import TrainingArguments
    klass = cls or TrainingArguments
    return klass(
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=False,
        warmup_steps=warmup_steps,
        learning_rate=lr,
        bf16=bf16,
        fp16=fp16,
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
        **extra,
    )


# ----------------------------------------------------------------
# Stage: CPT
# ----------------------------------------------------------------

def cmd_cpt(args):
    gpu_check()
    check_model_cached(args.model)

    use_4bit   = not args.no_4bit
    data_p     = Path(args.data)
    cache_path = str(data_p.parent / (data_p.stem + "_cpt_cache"))

    from datasets import Dataset

    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached CPT dataset from {cache_path}")
        dataset = Dataset.load_from_disk(cache_path)
    else:
        print("Tokenizing CPT corpus...")
        encode  = load_tokenizer(args.model)
        all_ids = encode(Path(args.data).read_text(), True, False)
        print(f"  Total tokens: {len(all_ids)}")
        records = []
        for i in range(0, len(all_ids), args.max_seq_len):
            chunk = all_ids[i : i + args.max_seq_len]
            if len(chunk) < 32:
                continue
            records.append({"input_ids": chunk, "labels": chunk})
        del all_ids
        dataset = Dataset.from_list(records)
        dataset.save_to_disk(cache_path)
        print(f"  Chunked into {len(dataset)} sequences")

    print(f"CPT dataset: {len(dataset)} chunks")

    model = load_model(args.model, use_4bit, args.max_seq_len)
    model = apply_lora(model, args.rank)
    model.print_trainable_parameters()

    bf16 = torch.cuda.is_bf16_supported()
    training_args = make_training_args(
        args.output, args.epochs, args.batch_size, args.grad_accum,
        args.lr, bf16, not bf16,
    )

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SimpleCollator(pad_id=0),
    )
    print("\nStarting CPT training...")
    trainer.train()

    print(f"\nSaving CPT adapter to {args.output}")
    model.save_pretrained(args.output)
    print("Done!")


# ----------------------------------------------------------------
# Stage: QLoRA
# ----------------------------------------------------------------

def cmd_qlora(args):
    gpu_check()
    check_model_cached(args.model)

    use_4bit   = not args.no_4bit
    data_p     = Path(args.data)
    cache_path = str(data_p.parent / (data_p.stem + "_tokenized_cache"))

    from datasets import Dataset

    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached dataset from {cache_path}")
        dataset = Dataset.load_from_disk(cache_path)
    else:
        print("Pre-tokenizing dataset...")
        encode = load_tokenizer(args.model)
        records, skipped = [], 0
        with open(args.data) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line)["text"]
                ids  = encode(text, True, True)
                if len(ids) > args.max_seq_len:
                    ids = ids[:args.max_seq_len]
                    skipped += 1
                records.append({"input_ids": ids, "labels": ids})
        if skipped:
            print(f"  Truncated {skipped} examples to {args.max_seq_len} tokens")
        dataset = Dataset.from_list(records)
        dataset.save_to_disk(cache_path)
        print(f"  Tokenized {len(dataset)} examples, saved to {cache_path}")

    print(f"Dataset size: {len(dataset)} examples")

    from peft import PeftModel
    model = load_model(args.model, use_4bit, args.max_seq_len)

    if args.adapter:
        print(f"Loading existing LoRA adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        model = apply_lora(model, args.rank)

    model.print_trainable_parameters()

    bf16 = torch.cuda.is_bf16_supported()
    training_args = make_training_args(
        args.output, args.epochs, args.batch_size, args.grad_accum,
        args.lr, bf16, not bf16,
    )

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SimpleCollator(pad_id=0),
    )
    print("\nStarting QLoRA training...")
    trainer.train()

    print(f"\nSaving QLoRA adapter to {args.output}")
    model.save_pretrained(args.output)
    print("Done!")


# ----------------------------------------------------------------
# Stage: DPO
# ----------------------------------------------------------------

def cmd_dpo(args):
    gpu_check()
    check_model_cached(args.model)

    use_4bit = not args.no_4bit
    data_p   = Path(args.data)
    cache_path = str(data_p.parent / (data_p.stem + "_dpo_cache"))

    from datasets import Dataset

    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached DPO dataset from {cache_path}")
        dataset = Dataset.load_from_disk(cache_path)
    else:
        records = []
        with open(args.data) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj["prompt"]
                if not prompt.endswith("\n"):
                    prompt += "\n"
                records.append({"prompt": prompt, "chosen": obj["chosen"], "rejected": obj["rejected"]})
        dataset = Dataset.from_list(records)
        dataset.save_to_disk(cache_path)
        print(f"Loaded {len(dataset)} DPO preference pairs")

    PatchDPOTrainer()

    # Load base model + existing adapter via unsloth so its patches stay intact
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=use_4bit,
        device_map={"": 0},
    )
    FastLanguageModel.for_training(model)
    # DPO passes a 4D causal mask; unsloth's _has_no_labels path assumes 2D and
    # crashes when it tries to broadcast (batch,1,seq,seq) against (batch,seq,hidden).
    # Disabling the flag skips that embedding-masking shortcut entirely.
    for _m in model.modules():
        if hasattr(_m, "_has_no_labels"):
            _m._has_no_labels = False
    model.print_trainable_parameters()

    tokenizer = load_dpo_tokenizer(args.model)

    from trl import DPOTrainer, DPOConfig
    bf16 = torch.cuda.is_bf16_supported()
    dpo_config = make_training_args(
        args.output, args.epochs, args.batch_size, args.grad_accum,
        args.lr, bf16, not bf16,
        warmup_steps=5,
        cls=DPOConfig,
        beta=args.beta,
        max_length=args.max_seq_len,
        max_prompt_length=args.max_seq_len // 2,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    print("\nStarting DPO training...")
    trainer.train()

    print(f"\nSaving DPO adapter to {args.output}")
    model.save_pretrained(args.output)
    print("Done!")


# ----------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CPT / QLoRA / DPO trainer")
    sub = parser.add_subparsers(dest="stage", required=True)

    # Shared args
    def add_common(p):
        p.add_argument("--model",      default="unsloth/Magistral-Small-2509")
        p.add_argument("--data",       required=True)
        p.add_argument("--output",     required=True)
        p.add_argument("--epochs",     type=int,   default=1)
        p.add_argument("--batch-size", type=int,   default=1,   dest="batch_size")
        p.add_argument("--grad-accum", type=int,   default=8,   dest="grad_accum")
        p.add_argument("--max-seq-len",type=int,   default=2048,dest="max_seq_len")
        p.add_argument("--no-4bit",    action="store_true",     dest="no_4bit")

    # CPT
    p_cpt = sub.add_parser("cpt", help="Continued pre-training on plain text")
    add_common(p_cpt)
    p_cpt.add_argument("--rank", type=int,   default=16)
    p_cpt.add_argument("--lr",   type=float, default=1e-4)

    # QLoRA
    p_qlora = sub.add_parser("qlora", help="QLoRA instruction fine-tuning")
    add_common(p_qlora)
    p_qlora.add_argument("--adapter", default=None, help="Existing adapter to continue from")
    p_qlora.add_argument("--rank",    type=int,   default=16)
    p_qlora.add_argument("--lr",      type=float, default=2e-4)

    # DPO
    p_dpo = sub.add_parser("dpo", help="DPO preference alignment")
    add_common(p_dpo)
    p_dpo.add_argument("--adapter", required=True, help="QLoRA adapter to continue from")
    p_dpo.add_argument("--lr",      type=float, default=5e-5)
    p_dpo.add_argument("--beta",    type=float, default=0.1)

    args = parser.parse_args()
    {"cpt": cmd_cpt, "qlora": cmd_qlora, "dpo": cmd_dpo}[args.stage](args)


if __name__ == "__main__":
    main()
