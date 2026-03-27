#!/usr/bin/env python3
"""
QLoRA fine-tune of Magistral-Small-2509 on Alastor character examples.
Uses mistral_common for correct tokenization + HuggingFace + PEFT for training.
Uses Unsloth for faster training and reduced VRAM usage.

Usage:
    python train_lora.py --model ~/models/magistral-small-2509-bf16 --data alastor_train.jsonl
"""

import argparse
import sys
import torch
import json
from pathlib import Path

class _AutoTokenizerWrapper:
    """
    Wraps AutoTokenizer to expose the same .instruct_tokenizer.tokenizer.encode(text, bos, eos)
    interface as MistralTokenizer, so callers don't need to branch.
    """
    class _Inner:
        def __init__(self, hf_tok):
            self._tok = hf_tok

        def encode(self, text: str, bos: bool = True, eos: bool = True):
            ids = self._tok.encode(text, add_special_tokens=False)
            if bos and self._tok.bos_token_id is not None:
                ids = [self._tok.bos_token_id] + ids
            if eos and self._tok.eos_token_id is not None:
                ids = ids + [self._tok.eos_token_id]
            return ids

    def __init__(self, hf_tok):
        inner = self._Inner(hf_tok)
        self.instruct_tokenizer = type("_IT", (), {"tokenizer": inner})()


def check_model_cached(model_path: str) -> None:
    """Exit early if model_path is not present in the HF cache (or as a local path)."""
    import os
    local = Path(model_path)
    if local.exists():
        return
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_name = "models--" + model_path.replace("/", "--")
    snapshots_dir = hf_home / "hub" / cache_name / "snapshots"
    if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
        print(f"ERROR: Model '{model_path}' not found in HF cache.")
        print(f"  Expected: {snapshots_dir}")
        print("  Pre-download the model to the RunPod volume before submitting a job.")
        sys.exit(1)
    print(f"  Model cache verified: {snapshots_dir}")


def load_tokenizer(model_path: str):
    """
    Load a tokenizer for Mistral-family models.
    Tries mistral_common (tekken.json / tokenizer.model) first — official Mistral models
    need this because AutoTokenizer has a broken regex.
    Falls back to AutoTokenizer for merged/derived models that only ship tokenizer.json.
    Handles both local paths and HF repo IDs.
    """
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from huggingface_hub import hf_hub_download

    local = Path(model_path)
    if local.exists():
        tekken = local / "tekken.json"
        if not tekken.exists():
            tekken = local / "tokenizer.model"
        if tekken.exists():
            return MistralTokenizer.from_file(str(tekken))
        # fall through to AutoTokenizer
    else:
        for filename in ("tekken.json", "tokenizer.model"):
            try:
                tekken = Path(hf_hub_download(repo_id=model_path, filename=filename,
                                              local_files_only=True))
                return MistralTokenizer.from_file(str(tekken))
            except Exception:
                pass
        # fall through to AutoTokenizer

    print(f"  No tekken.json/tokenizer.model found — falling back to AutoTokenizer")
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return _AutoTokenizerWrapper(hf_tok)

def pretokenize_dataset(data_path: str, model_path: str, max_seq_len: int, cache_path: str):
    """
    Pre-tokenize all examples using mistral_common and cache as a HF dataset.
    Returns a HuggingFace Dataset of input_ids + labels.
    """
    from datasets import Dataset

    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached tokenized dataset from {cache_path}")
        return Dataset.load_from_disk(cache_path)

    print("Pre-tokenizing dataset with mistral_common...")
    mc_tok = load_tokenizer(model_path)
    # Get the underlying fast tokenizer encode function
    encode = mc_tok.instruct_tokenizer.tokenizer.encode

    records = []
    skipped = 0
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            # encode(text, bos=True, eos=True)
            ids = encode(text, True, True)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
                skipped += 1
            records.append({"input_ids": ids, "labels": ids.copy()})

    if skipped:
        print(f"  Truncated {skipped} examples to {max_seq_len} tokens")

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(cache_path)
    print(f"  Tokenized {len(dataset)} examples, saved to {cache_path}")
    return dataset


class SimpleCollator:
    """
    Pads input_ids and labels to the longest sequence in the batch.
    Uses pad_token_id=0 (Mistral/tekken convention).
    Labels are masked at pad positions with -100 so loss ignores them.
    """
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        # Round up to multiple of 8 for efficiency
        max_len = ((max_len + 7) // 8) * 8

        input_ids = []
        labels = []
        attention_mask = []

        for f in features:
            ids = f["input_ids"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def main():
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected — aborting to avoid wasting compute.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Magistral-Small-2509")
    parser.add_argument("--data", default="alastor_train.jsonl")
    parser.add_argument("--output", default="./alastor-lora")
    parser.add_argument("--adapter", default=None,
                        help="Path to existing LoRA adapter to continue training from "
                             "(e.g. output of train_cpt.py). If omitted a fresh adapter is created.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (uses more RAM)")
    args = parser.parse_args()

    use_4bit = not args.no_4bit
    cache_path = args.data.replace(".jsonl", "_tokenized_cache")

    if use_4bit and not torch.cuda.is_available():
        print("WARNING: No GPU detected — disabling 4-bit quantization (CPU-only mode)")
        use_4bit = False

    # ----------------------------------------------------------------
    # 1. Sanity check GPU
    # ----------------------------------------------------------------
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    check_model_cached(args.model)

    # Import unsloth after CUDA is confirmed ready, before any transformers usage
    from unsloth import FastLanguageModel
    from peft import PeftModel

    # ----------------------------------------------------------------
    # 2. Pre-tokenize dataset using mistral_common
    # ----------------------------------------------------------------
    tokenized_dataset = pretokenize_dataset(
        args.data, args.model, args.max_seq_len, cache_path
    )
    print(f"Dataset size: {len(tokenized_dataset)} examples")

    # ----------------------------------------------------------------
    # 3. Load model + attach LoRA adapters (via Unsloth)
    # ----------------------------------------------------------------

    print(f"\nLoading model (4-bit={use_4bit})...")

    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,        # auto-detect bf16/fp16
        load_in_4bit=use_4bit,
        local_files_only=True,
    )

    if args.adapter:
        print(f"Loading existing LoRA adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.rank,
            lora_alpha=args.rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,    # must be 0 for Unsloth kernel optimisations
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    model.print_trainable_parameters()

    # ----------------------------------------------------------------
    # 6. Training arguments
    # ----------------------------------------------------------------
    from transformers import TrainingArguments, Trainer

    has_gpu = torch.cuda.is_available()
    bf16_supported = has_gpu and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=False,  # handled by use_gradient_checkpointing="unsloth" above
        warmup_steps=10,
        learning_rate=args.lr,
        bf16=bf16_supported,
        fp16=has_gpu and not bf16_supported,
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
    )

    # ----------------------------------------------------------------
    # 7. Train
    # ----------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=SimpleCollator(pad_id=0),
    )

    print("\nStarting training...")
    trainer.train()

    # ----------------------------------------------------------------
    # 8. Save adapter
    # ----------------------------------------------------------------
    print(f"\nSaving LoRA adapter to {args.output}")
    model.save_pretrained(args.output)

    print("\nDone!")
    print(f"\nNext step — convert adapter to GGUF for llama.cpp:")
    print(f"  python llama.cpp/convert_lora_to_gguf.py {args.output}")
    print(f"  llama-server --model ~/models/magistral-2509-vision/Magistral-Small-2509-Q3_K_S.gguf \\")
    print(f"    --lora {args.output}-gguf \\")
    print(f"    --n-gpu-layers 99 --port 8080")


if __name__ == "__main__":
    main()
