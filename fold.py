"""fold.py — MiniFold-MLX command-line structure prediction.

Usage
-----
    python fold.py sequences.fasta --out_dir ./structures
    python fold.py sequences.fasta --out_dir ./structures --model_size 12L --no_compile
    python fold.py sequences.fasta --out_dir ./structures --recycling 1

Weights are downloaded automatically from HuggingFace on first run
and cached in ~/.cache/huggingface/hub/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


HF_REPO = "z-ardern/MiniFold_MLX_weights"


def _get_weights(model_size: str) -> tuple[str, str]:
    """Download weights from HuggingFace if not cached; return (esm_path, minifold_path)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)

    print(f"Checking weights ({HF_REPO}) …")
    weights_dir = Path(snapshot_download(HF_REPO))
    esm_path      = weights_dir / "ESM2_MiniFold"
    minifold_path = weights_dir / f"minifold_{model_size}"

    if not esm_path.exists():
        print(f"ERROR: ESM2_MiniFold/ not found in {weights_dir}")
        sys.exit(1)
    if not minifold_path.exists():
        print(f"ERROR: minifold_{model_size}/ not found in {weights_dir}")
        sys.exit(1)

    return str(esm_path), str(minifold_path)


def _read_fasta(path: str) -> list[tuple[str, str]]:
    sequences = []
    current_id, current_seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
    if current_id is not None:
        sequences.append((current_id, "".join(current_seq)))
    return sequences


def main():
    parser = argparse.ArgumentParser(description="MiniFold-MLX structure prediction")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("--out_dir", default="./structures",
                        help="Output directory for PDB files (default: ./structures)")
    parser.add_argument("--model_size", choices=["48L", "12L"], default="48L",
                        help="MiniFold model size (default: 48L)")
    parser.add_argument("--recycling", type=int, default=0,
                        help="Number of recycling iterations (default: 0)")
    parser.add_argument("--max_len", type=int, default=800,
                        help="Skip sequences longer than this (default: 800, 0=no limit)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable mx.compile on MiniFormer")
    parser.add_argument("--no_int8", action="store_true",
                        help="Disable int8 ESM2 quantization (uses more memory)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = _read_fasta(args.fasta)
    if not sequences:
        print(f"ERROR: no sequences found in {args.fasta}")
        sys.exit(1)
    print(f"Found {len(sequences)} sequence(s) in {args.fasta}")

    esm_path, minifold_path = _get_weights(args.model_size)

    import mlx.core as mx
    import mlx.nn as nn_mlx
    from minifold_mlx import load_model, predict_sequence

    print(f"\nLoading model ({args.model_size}) …")
    tokenizer, model = load_model(esm_path, minifold_path, bf16=False, fp16=False)
    model.convert_to_bf16()
    model.enable_sgmm_gate()
    model.fold._timing = False

    if not args.no_int8:
        esm_model = object.__getattribute__(model, "_esm_model")
        nn_mlx.quantize(esm_model, bits=8, group_size=32)
        mx.eval(esm_model.parameters())
        print("  ESM2 quantized to int8")

    if not args.no_compile:
        model.enable_compile_miniformer()

    print(f"\nPredicting {len(sequences)} sequence(s) → {out_dir}/\n")
    t_start = time.perf_counter()

    for seq_id, seq in sequences:
        t0 = time.perf_counter()
        pdb_str = predict_sequence(
            seq_id, seq, model, tokenizer,
            num_recycling=args.recycling,
            max_seq_len=args.max_len,
        )
        elapsed = time.perf_counter() - t0

        if pdb_str is None:
            print(f"  SKIP  {seq_id}  (len={len(seq)})")
            continue

        out_path = out_dir / f"{seq_id}.pdb"
        out_path.write_text(pdb_str)
        print(f"  {seq_id}  len={len(seq)}  {elapsed:.2f}s  → {out_path}")

    total = time.perf_counter() - t_start
    print(f"\nDone. {len(sequences)} sequence(s) in {total:.1f}s")


if __name__ == "__main__":
    main()
