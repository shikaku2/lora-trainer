#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) fine-tuning using TRL's DPOTrainer.

Loads the LoRA adapter produced by the QLoRA stage and continues training
on preference pairs to align the model's outputs.

JSONL format (one object per line):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
    python train_dpo.py \
        --model  mistralai/Magistral-Small-2509 \
        --adapter ./lora-output \
        --data   build1/dpo_all_2026-03-19.jsonl \
        --output ./dpo-lora
"""

import argparse
import json
import sys
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def load_dpo_dataset(data_path: str):
    from datasets import Dataset

    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Ensure prompt ends with a newline so the tokenizer sees a clean
            # boundary between prompt and chosen/rejected and avoids merge weirdness.
            prompt = obj["prompt"]
            if not prompt.endswith("\n"):
                prompt += "\n"
            records.append({
                "prompt": prompt,
                "chosen": obj["chosen"],
                "rejected": obj["rejected"],
            })

    print(f"Loaded {len(records)} DPO preference pairs")
    return Dataset.from_list(records)


def load_dpo_tokenizer(model_path: str):
    """
    Magistral tokenizer behavior is flaky across transformers releases.
    Force the slow tokenizer path first, because that's the combo that stayed
    stable in the local sweep and it avoids the fast-tokenizer Mistral regex bug.
    """
    from transformers import AutoTokenizer

    attempts = [
        {"use_fast": False, "trust_remote_code": True},
        {"use_fast": True,  "trust_remote_code": True},
        {"use_fast": False},
    ]

    last_error = None
    for kwargs in attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            print(f"Loaded tokenizer with args: {kwargs}")
            return tokenizer
        except Exception as e:
            last_error = e
            print(f"Tokenizer load failed with args {kwargs}: {e}")

    raise RuntimeError(
        f"Failed to load tokenizer for {model_path}. Last error: {last_error}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="unsloth/Magistral-Small-2509")
    parser.add_argument("--adapter",    required=True,
                        help="Path to LoRA adapter from the QLoRA stage")
    parser.add_argument("--data",       required=True,
                        help="JSONL with prompt / chosen / rejected fields")
    parser.add_argument("--output",     default="./dpo-lora")
    parser.add_argument("--epochs",     type=int,   default=1)
    parser.add_argument("--batch-size", type=int,   default=1)
    parser.add_argument("--grad-accum", type=int,   default=8)
    parser.add_argument("--max-seq-len",type=int,   default=2048)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--beta",       type=float, default=0.1,
                        help="DPO beta - KL divergence penalty weight")
    parser.add_argument("--no-4bit",    action="store_true")
    args = parser.parse_args()

    use_4bit = not args.no_4bit
    if use_4bit and not torch.cuda.is_available():
        print("WARNING: No GPU detected - disabling 4-bit quantization")
        use_4bit = False

    if not torch.cuda.is_available():
        print("ERROR: No GPU detected - aborting to avoid wasting compute.")
        sys.exit(1)
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    from train_lora import check_model_cached
    check_model_cached(args.model)

    # ----------------------------------------------------------------
    # 1. Load dataset
    # ----------------------------------------------------------------
    dataset = load_dpo_dataset(args.data)

    # ----------------------------------------------------------------
    # 2. Load base model + QLoRA adapter
    # ----------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation="sdpa",
    )

    # ----------------------------------------------------------------
    # 3. Load the QLoRA adapter and make it trainable
    # ----------------------------------------------------------------
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ----------------------------------------------------------------
    # 4. Tokenizer
    # ----------------------------------------------------------------
    tokenizer = load_dpo_tokenizer(args.model)

    # ----------------------------------------------------------------
    # 5. DPO training
    # ----------------------------------------------------------------
    from trl import DPOTrainer, DPOConfig

    has_gpu = torch.cuda.is_available()
    bf16    = has_gpu and torch.cuda.is_bf16_supported()

    dpo_config = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        warmup_steps=5,
        learning_rate=args.lr,
        bf16=bf16,
        fp16=has_gpu and not bf16,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        remove_unused_columns=False,
        beta=args.beta,
        max_length=args.max_seq_len,
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

    # ----------------------------------------------------------------
    # 6. Save updated adapter
    # ----------------------------------------------------------------
    print(f"\nSaving DPO adapter to {args.output}")
    model.save_pretrained(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
