"""Quantize the finetuned MLX ESM2 model to int8 and save as safetensors.

Reduces the ESM2 checkpoint from ~11 GB to ~5.5 GB on disk.
The quantized weights can be loaded directly — no runtime quantization needed.

Usage
-----
    python scripts/quantize_esm2_int8.py \
        --input  path/to/ESM2_MiniFold \
        --output path/to/ESM2_MiniFold_int8

Or download weights automatically and quantize from cache:
    python scripts/quantize_esm2_int8.py --from_hf
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "1")

HF_REPO = "z-ardern/MiniFoldX_weights"


def main():
    parser = argparse.ArgumentParser(description="Quantize ESM2 to int8 and save")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str,
                       help="Path to ESM2_MiniFold checkpoint directory")
    group.add_argument("--from_hf", action="store_true",
                       help="Download weights from HuggingFace and quantize")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: <input>_int8)")
    args = parser.parse_args()

    # ── Resolve input path ────────────────────────────────────────────────────
    if args.from_hf:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("ERROR: pip install huggingface_hub")
            sys.exit(1)
        print(f"Downloading weights from {HF_REPO} …")
        weights_dir = Path(snapshot_download(HF_REPO))
        input_path  = weights_dir / "ESM2_MiniFold"
    else:
        input_path = Path(args.input).expanduser()

    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    # ── Resolve output path ───────────────────────────────────────────────────
    output_path = Path(args.output).expanduser() if args.output else \
                  input_path.parent / (input_path.name + "_int8")

    if output_path.exists():
        print(f"Output already exists: {output_path}")
        print("Delete it first if you want to re-quantize.")
        sys.exit(0)

    # ── Load ESM2 ─────────────────────────────────────────────────────────────
    pkg = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(pkg))

    import mlx.core as mx
    import mlx.nn as nn_mlx
    from minifoldx.esm2 import ESM2

    print(f"\nLoading ESM2 from {input_path} …")
    t0 = time.perf_counter()
    tokenizer, esm_model = ESM2.from_pretrained(str(input_path))
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    # ── Quantize ──────────────────────────────────────────────────────────────
    print("Quantizing to int8 (group_size=32) …")
    t0 = time.perf_counter()
    nn_mlx.quantize(esm_model, bits=8, group_size=32)
    mx.eval(esm_model.parameters())
    print(f"  quantized in {time.perf_counter()-t0:.1f}s")

    # ── Save weights ──────────────────────────────────────────────────────────
    output_path.mkdir(parents=True)

    weights_file = str(output_path / "model.safetensors")
    print(f"Saving to {weights_file} …")
    t0 = time.perf_counter()
    esm_model.save_weights(weights_file)
    elapsed = time.perf_counter() - t0
    size_gb = Path(weights_file).stat().st_size / 1e9
    print(f"  saved {size_gb:.2f} GB in {elapsed:.1f}s")

    # Copy tokenizer config files
    for fname in ("config.json", "special_tokens_map.json", "vocab.txt"):
        src = input_path / fname
        if src.exists():
            shutil.copy(src, output_path / fname)

    print(f"\nDone. Quantized ESM2 saved to: {output_path}")


if __name__ == "__main__":
    main()
