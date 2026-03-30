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

os.environ.pop("CUDA_VISIBLE_DEVICES", None)


from unsloth import FastLanguageModel, PatchDPOTrainer
import sys
import torch
import transformers.modeling_utils as _mu
if hasattr(_mu, "caching_allocator_warmup"):
    _mu.caching_allocator_warmup = lambda *a, **kw: None
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


def _try_autotokenizer(model_path: str, token: str = None):
    """Try multiple AutoTokenizer loading strategies, return the first that works."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    _token = token or os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")
    base = dict(local_files_only=True, token=_token)
    for _cls, _label, _kwargs in [
        (AutoTokenizer,           "AutoTokenizer slow",      dict(**base, trust_remote_code=True, use_fast=False)),
        (AutoTokenizer,           "AutoTokenizer fast",      dict(**base, trust_remote_code=True, use_fast=True)),
        (PreTrainedTokenizerFast, "PreTrainedTokenizerFast", base),
    ]:
        try:
            tok = _cls.from_pretrained(model_path, **_kwargs)
            print(f"  Loaded tokenizer via {_label}")
            return tok
        except Exception as e:
            print(f"  {_label} failed: {e}")
    raise RuntimeError(f"Could not load any tokenizer for {model_path}")


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
    _tok_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")
    hf_tok = _try_autotokenizer(model_path, token=_tok_token)

    def encode(text: str, bos: bool = True, eos: bool = True):
        ids = hf_tok.encode(text, add_special_tokens=False)
        if bos and hf_tok.bos_token_id is not None:
            ids = [hf_tok.bos_token_id] + ids
        if eos and hf_tok.eos_token_id is not None:
            ids = ids + [hf_tok.eos_token_id]
        return ids

    return encode


def load_dpo_tokenizer(model_path: str):
    tok = _try_autotokenizer(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # unsloth assumes right-padding in its embedding mask
    return tok


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


def _patch_tokenizer_config(model_dir: str):
    """
    Fix tokenizer_config.json for models with broken tokenizer_class values.

    - 'TokenizersBackend': not a real transformers class (artifact of some merge tools)
    - 'MistralCommonTokenizer': rejects _from_auto/_commit_hash kwargs injected by AutoTokenizer
    Both are replaced with LlamaTokenizerFast, which reads tokenizer.json and works correctly
    for Mistral-family models.
    """
    import json
    cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(cfg_path):
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    tok_class = cfg.get("tokenizer_class", "")
    broken = {"TokenizersBackend", "MistralCommonTokenizer"}
    if tok_class not in broken:
        return
    print(f"  Replacing tokenizer_class={tok_class!r} with LlamaTokenizerFast in tokenizer_config.json")
    cfg["tokenizer_class"] = "LlamaTokenizerFast"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


def load_model(model_path: str, use_4bit: bool, max_seq_len: int):
    print(f"\nLoading model (4-bit={use_4bit})...")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(f"  CUDA device 0: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA device 0 memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Resolve to local cache path only to patch tokenizer_config.json — pass the original
    # model_path (HF repo ID) to FastLanguageModel so adapters record the correct base_model.
    local_path = model_path if os.path.isdir(model_path) else None
    if local_path is None:
        from huggingface_hub import snapshot_download
        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")
        local_path = snapshot_download(model_path, token=token)
    _patch_tokenizer_config(local_path)
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


def _is_vlm(model) -> bool:
    """Return True if model (raw or PEFT-wrapped) is a vision-language model."""
    # Check the unwrapped base class
    base = getattr(getattr(model, "base_model", model), "model", model)
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    return ("ConditionalGeneration" in type(base).__name__
            or model_type in ("mistral3", "llava", "pixtral", "idefics"))


def _patch_vlm(model) -> None:
    """
    For PEFT-wrapped VLMs, patch the outer forward to route text-only training through
    language_model + lm_head directly, bypassing the unsloth-compiled VLM wrapper.

    Model hierarchy (after get_peft_model / from_pretrained):
      model (PeftModel)
        └── .base_model (LoraModel)
              └── .model  =: vlm   (Mistral3ForConditionalGeneration)
                    └── .model  =: inner  (Mistral3Model)
                          ├── .language_model  (MistralModel — base transformer, no lm_head)
                          └── .vision_tower

    PEFT calls vlm.forward() directly. We must patch at that level so the
    unsloth-compiled Mistral3ForConditionalGeneration_forward never runs.
    """
    import types
    from transformers.modeling_outputs import CausalLMOutputWithPast

    vlm   = getattr(getattr(model, "base_model", model), "model", model)
    inner = getattr(vlm, "model", vlm)

    vt = getattr(inner, "vision_tower", None)
    if vt is not None:
        vt.requires_grad_(False)
        print(f"  Frozen vision tower ({sum(p.numel() for p in vt.parameters())/1e6:.1f}M params)")

    lm      = getattr(inner, "language_model", None)
    lm_head = getattr(vlm,   "lm_head",        None)
    if lm is None or lm_head is None:
        print(f"  WARNING: VLM detected but language_model/lm_head not found — skipping forward patch")
        return

    def _text_forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # requires_grad_(True) on the embedding output ensures PyTorch's gradient
        # checkpointing builds a backward graph even though input_ids are integers.
        embeds = lm.embed_tokens(input_ids).requires_grad_(True)
        hidden = lm(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
        logits = lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    vlm.forward = types.MethodType(_text_forward, vlm)
    print(f"  Patched {type(vlm).__name__}.forward → language_model + lm_head for text-only training")


def apply_lora(model, rank: int):
    # Unsloth's custom GC assumes a flat CausalLM structure. For VLMs the decoder
    # lives inside a wrapper (.language_model), so use standard PyTorch GC instead.
    vlm_model = _is_vlm(model)
    gc = True if vlm_model else "unsloth"
    if vlm_model:
        print(f"  VLM detected ({type(model).__name__}) — using standard gradient checkpointing")

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=gc,
    )

    if vlm_model:
        _patch_vlm(model)

    return model


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
        if _is_vlm(model):
            _patch_vlm(model)
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

    # Load base model + existing adapter via unsloth so its patches stay intact
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=use_4bit,
        device_map={"": 0},
    )
    FastLanguageModel.for_training(model)

    _vlm_entry = None  # may be set in the VLM block below; used to restore after DPOTrainer init

    if _is_vlm(model):
        # Pop mistral3 from TRL/Unsloth's VLM mapping so both treat this as a text-only
        # model for data preparation (no images column required in the dataset).
        # _patch_vlm handles the actual forward pass correctly for text-only input.
        _patch_vlm(model)
        # TRL v0.23 sets is_vision_model = model.config.model_type in
        # MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES and then routes dataset
        # preparation through process_row() which calls processing_class.tokenizer.
        # Temporarily remove mistral3 so DPOTrainer uses the text-only tokenize_row
        # path.  Restored after __init__ returns so nothing else is affected.
        try:
            from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES as _VLM_MAP
            _vlm_entry = _VLM_MAP.pop("mistral3", None)
            print("  Removed mistral3 from MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES — DPOTrainer will use text-only path")
        except Exception as e:
            _vlm_entry = None
            print(f"  Warning: could not patch VLM mapping: {e}")

    PatchDPOTrainer()

    # For Mistral-family models unsloth sets _has_no_labels=True when DPO data has no
    # labels field, causing the inner model forward to skip loss computation.  Our
    # _patch_vlm overrides vlm.forward entirely so this is only needed for non-VLM.
    if not _is_vlm(model):
        _inner = model.base_model.model.model
        _orig_fwd = _inner.forward
        def _safe_fwd(*args, **kwargs):
            _inner._has_no_labels = False
            return _orig_fwd(*args, **kwargs)
        _inner.forward = _safe_fwd
        print(f"  Patched {type(_inner).__name__} instance forward (disable _has_no_labels)")

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
    # Restore the VLM mapping now that DPOTrainer.__init__ (which reads it) has returned
    try:
        if _vlm_entry is not None:
            from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES as _VLM_MAP
            _VLM_MAP["mistral3"] = _vlm_entry
    except Exception:
        pass
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
