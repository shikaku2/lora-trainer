#!/usr/bin/env python3
"""
Continued Pre-Training (CPT) on plain text using LoRA.

Reads a raw text file, chunks it into token sequences, and trains
as a causal LM so the model internalises the corpus vocabulary and style.

Usage:
    python train_cpt.py --model unsloth/Magistral-Small-2509 \
        --data build1/Alastor_CPT_Master.txt --output ./cpt-lora
"""

import argparse
import torch
from pathlib import Path


def chunk_text(text_path: str, model_path: str, max_seq_len: int, cache_path: str):
    """Tokenize a plain-text file and split into fixed-length chunks."""
    from datasets import Dataset
    from train_lora import load_tokenizer

    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached CPT dataset from {cache_path}")
        return Dataset.load_from_disk(cache_path)

    print("Tokenizing CPT corpus...")
    mc_tok = load_tokenizer(model_path)
    encode = mc_tok.instruct_tokenizer.tokenizer.encode

    text = Path(text_path).read_text()
    all_ids = encode(text, True, False)   # BOS, no EOS — we chunk the stream
    print(f"  Total tokens in corpus: {len(all_ids)}")

    records = []
    for i in range(0, len(all_ids), max_seq_len):
        chunk = all_ids[i : i + max_seq_len]
        if len(chunk) < 32:          # skip tiny tail chunks
            continue
        records.append({"input_ids": chunk, "labels": chunk.copy()})

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(cache_path)
    print(f"  Chunked into {len(dataset)} sequences of up to {max_seq_len} tokens")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="unsloth/Magistral-Small-2509")
    parser.add_argument("--data",       required=True, help="Plain-text CPT corpus")
    parser.add_argument("--output",     default="./cpt-lora")
    parser.add_argument("--epochs",     type=int,   default=1)
    parser.add_argument("--rank",       type=int,   default=16)
    parser.add_argument("--batch-size", type=int,   default=1)
    parser.add_argument("--grad-accum", type=int,   default=8)
    parser.add_argument("--max-seq-len",type=int,   default=2048)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--no-4bit",    action="store_true")
    args = parser.parse_args()

    use_4bit   = not args.no_4bit
    cache_path = args.data.replace(".txt", "_cpt_cache")

    if use_4bit and not torch.cuda.is_available():
        print("WARNING: No GPU detected — disabling 4-bit quantization")
        use_4bit = False

    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    # ----------------------------------------------------------------
    # 1. Tokenize corpus into chunks
    # ----------------------------------------------------------------
    dataset = chunk_text(args.data, args.model, args.max_seq_len, cache_path)
    print(f"CPT dataset: {len(dataset)} chunks")

    # ----------------------------------------------------------------
    # 2. Load model
    # ----------------------------------------------------------------
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, quantization_config=bnb,
            device_map=device_map, trust_remote_code=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            device_map=device_map, trust_remote_code=True,
        )

    model.config.use_cache = False

    # ----------------------------------------------------------------
    # 3. Freeze vision encoder (if any)
    # ----------------------------------------------------------------
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in ("vision", "patch", "pixel")):
            param.requires_grad = False

    # ----------------------------------------------------------------
    # 4. Attach fresh LoRA adapters
    # ----------------------------------------------------------------
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ----------------------------------------------------------------
    # 5. Train
    # ----------------------------------------------------------------
    from transformers import TrainingArguments, Trainer
    from train_lora import SimpleCollator

    has_gpu = torch.cuda.is_available()
    bf16    = has_gpu and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        warmup_steps=10,
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
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SimpleCollator(pad_id=0),
    )

    print("\nStarting CPT training...")
    trainer.train()

    # ----------------------------------------------------------------
    # 6. Save adapter
    # ----------------------------------------------------------------
    print(f"\nSaving CPT LoRA adapter to {args.output}")
    model.save_pretrained(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
