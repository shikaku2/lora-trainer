#!/usr/bin/env python3
"""
patch_lora_gguf.py — fix general.architecture mismatch in a LoRA GGUF.

When a LoRA is trained on a VLM (e.g. Mistral3ForConditionalGeneration) and
converted with convert_lora_to_gguf.py, the GGUF gets architecture=mistral3.
If the base model GGUF was patched for llama.cpp compatibility (e.g. the
nicoboss llamacppfixed models), it reports architecture=llama. llama.cpp
refuses to load a LoRA whose arch metadata doesn't match the base.

This script reads the architecture from the base GGUF and patches it into the
LoRA GGUF in-place (or writes a new file).

Usage:
  python3 patch_lora_gguf.py <base.gguf> <lora.gguf> [output.gguf]

  If output.gguf is omitted, overwrites lora.gguf in-place.
"""

import struct
import shutil
import sys
from pathlib import Path


def read_gguf_string_kv(data: bytes, key: str) -> tuple[str, int, int]:
    """
    Find a string KV entry in GGUF metadata. Returns (value, val_offset, val_len).
    Raises KeyError if not found.
    """
    key_b = key.encode()
    offset = 24  # skip magic(4) + version(4) + n_tensors(8) + n_kv(8)

    n_kv = struct.unpack_from('<Q', data, 16)[0]
    pos = 24
    for _ in range(n_kv):
        k_len = struct.unpack_from('<Q', data, pos)[0]
        pos += 8
        k = data[pos:pos + k_len]
        pos += k_len
        vtype = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        if vtype == 8:  # string
            v_len = struct.unpack_from('<Q', data, pos)[0]
            pos += 8
            if k == key_b:
                return data[pos:pos + v_len].decode(), pos, v_len
            pos += v_len
        elif vtype == 9:  # array — skip
            arr_type = struct.unpack_from('<I', data, pos)[0]
            pos += 4
            arr_len = struct.unpack_from('<Q', data, pos)[0]
            pos += 8
            elem_sizes = {0: 1, 1: 1, 2: 2, 3: 4, 4: 4, 5: 2, 6: 4,
                          7: 8, 10: 4, 11: 8, 12: 16}
            if arr_type == 8:  # array of strings
                for _ in range(arr_len):
                    sl = struct.unpack_from('<Q', data, pos)[0]
                    pos += 8 + sl
            elif arr_type in elem_sizes:
                pos += arr_len * elem_sizes[arr_type]
            else:
                raise ValueError(f"Unknown array element type {arr_type} at offset {pos}")
        else:
            elem_sizes = {0: 1, 1: 1, 2: 2, 3: 4, 4: 4, 5: 2, 6: 4,
                          7: 8, 10: 4, 11: 8, 12: 16}
            if vtype not in elem_sizes:
                raise ValueError(f"Unknown KV type {vtype} at offset {pos}")
            pos += elem_sizes[vtype]

    raise KeyError(f"Key '{key}' not found in GGUF metadata")


def patch_gguf_arch(base_path: Path, lora_path: Path, out_path: Path):
    base_data = base_path.read_bytes()
    lora_data = lora_path.read_bytes()

    base_arch, _, _ = read_gguf_string_kv(base_data, 'general.architecture')
    lora_arch, val_off, val_len = read_gguf_string_kv(lora_data, 'general.architecture')

    print(f"Base arch: {base_arch}")
    print(f"LoRA arch: {lora_arch}")

    if base_arch == lora_arch:
        print("Architectures already match — no patch needed.")
        if out_path != lora_path:
            shutil.copy2(lora_path, out_path)
            print(f"Copied to {out_path}")
        return

    new_arch = base_arch.encode()
    if len(new_arch) > val_len:
        raise ValueError(
            f"New arch string '{base_arch}' ({len(new_arch)} bytes) is longer than "
            f"existing '{lora_arch}' ({val_len} bytes) — cannot patch in-place. "
            f"Reconvert the LoRA with the correct base model."
        )

    # Pad with nulls to keep file size identical
    padded = new_arch + b'\x00' * (val_len - len(new_arch))
    patched = lora_data[:val_off] + padded + lora_data[val_off + val_len:]

    out_path.write_bytes(patched)
    print(f"Patched '{lora_arch}' → '{base_arch}'")
    print(f"Output:  {out_path}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    base_path = Path(sys.argv[1])
    lora_path = Path(sys.argv[2])
    out_path  = Path(sys.argv[3]) if len(sys.argv) > 3 else lora_path

    for p in (base_path, lora_path):
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    patch_gguf_arch(base_path, lora_path, out_path)


if __name__ == '__main__':
    main()
