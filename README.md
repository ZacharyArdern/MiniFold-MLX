# MiniFoldX

An extension of [MiniFold](https://github.com/jwohlwend/minifold) with multiple compute backends for protein structure prediction: Apple Silicon GPU via [MLX](https://github.com/ml-explore/mlx) (no PyTorch required), Google Colab GPU launched directly from the terminal, standard CUDA GPU, and CPU.

## Features

- Full ESM2 (3B) + MiniFold folding trunk in MLX
- int8 ESM2 quantization via `mlx.nn.quantize` (group_size=32)
- Custom Apple Metal SGMM gate kernel fusing LayerNorm + gating for TriangularUpdate blocks
- `mx.compile` support on MiniFormer for reduced Metal dispatch overhead
- 48-layer (full) and 12-layer (fast) MiniFold variants
- Simple three-function public API: `load_model`, `predict_sequence`, `predict_batch`
- Pre-converted weights hosted on HuggingFace for fast download-on-first-use
- `--colab` backend: run on a Colab GPU with results downloaded locally
- CUDA and CPU backends (same PyTorch path as standard MiniFold) with added `--int8-esm2`, `--num_recycling`, and safetensors fast-loading
- `--cpu` backend: PyTorch CPU inference using all available cores

## Installation

```bash
pip install minifold-mlx
```

**Requires macOS with Apple Silicon (M1 or later).** For Linux/Windows, use the original [MiniFold](https://github.com/jwohlwend/minifold). MLX >= 0.16.0 is installed automatically as a dependency.

## Quick Start

```bash
minifold example.fasta --out_dir ./structures
```

OR

```bash
python fold.py example.fasta --out_dir ./structures
```

Weights (~3.8 GB) are downloaded automatically on first run. PDB files are saved to `./structures/`.

## Backends

### Apple Silicon (default)

`minifoldx seqs.fasta` runs the native MLX path on the local Mac GPU/Neural Engine. No additional setup required.

### Google Colab GPU

Run predictions on a Colab GPU with results downloaded to your Mac:

```bash
pip install colab-cli

# Default (48L model, A100, 3 recyclings)
minifoldx --colab run seqs.fasta --gpu A100

# Lighter/faster
minifoldx --colab run seqs.fasta --gpu A100 --model 12L --rec 0

# Save GPU memory with int8 ESM2
minifoldx --colab run seqs.fasta --gpu A100 --int8-esm2
```

Available GPUs: T4, L4, A100. Results are saved to `./minifold_results/<job_id>/` by default.

**Drive caching (optional):** configure an rclone `gdrive` remote and model weights are cached to Google Drive, making repeat runs significantly faster. For long jobs (>200 sequences), results are checkpointed to Drive every 10 minutes to survive connection loss.

```bash
# List past jobs
minifoldx --colab list

# Re-download results from Drive
minifoldx --colab fetch JOB_ID
```

### CUDA / HPC

MiniFoldX can also be run on any CUDA GPU, the same as standard [MiniFold](https://github.com/jwohlwend/minifold), using `minifoldx/pytorch/predict.py`. Several options are added beyond the standard MiniFold CLI:

```bash
python minifoldx/pytorch/predict.py seqs.fasta \
    --out_dir ./structures \
    --model_size 48L \
    --num_recycling 3 \
    --int8-esm2
```

Added options:
- `--int8-esm2`: quantize ESM2 to int8 via bitsandbytes (~3× smaller GPU memory footprint; requires `pip install bitsandbytes`)
- `--num_recycling N`: set recycling iterations (default 3; 0 for fastest inference)
- Safetensors weights (hosted on HuggingFace as `z-ardern/MiniFoldX_weights`) load faster than the original `.ckpt` format and are used automatically when `--checkpoint` points to a `.safetensors` file

### CPU

```bash
minifoldx --cpu seqs.fasta --out_dir ./structures
```

Runs the PyTorch backend on CPU using all available cores. No GPU required. Slower than CUDA but useful for small jobs or environments without a GPU.

## Weights

Pre-converted MLX weights (finetuned ESM2 + MiniFold 48L/12L) are available on HuggingFace:

```
z-ardern/MiniFold_MLX_weights
├── ESM2_MiniFold_int8/  # finetuned MLX ESM2 3B, int8 quantized (~3.3 GB) ← downloaded by default
├── ESM2_MiniFold/       # finetuned MLX ESM2 3B, full precision (~11 GB)
├── minifold_48L/        # 48-layer MiniFold MLX weights (~285 MB)
└── minifold_12L/        # 12-layer MiniFold MLX weights (~259 MB)
```

`fold.py` downloads only `ESM2_MiniFold_int8` by default. Pass `--non-quantized-ESM2` to use the full-precision weights instead.

```python
from huggingface_hub import snapshot_download

weights = snapshot_download("z-ardern/MiniFold_MLX_weights")
esm_path      = f"{weights}/ESM2_MiniFold_int8"
minifold_path = f"{weights}/minifold_48L"
```

## Use in Python

```python
from minifoldx import load_model, predict_sequence, predict_batch

tokenizer, model = load_model(
    mlx_esm_path      = "path/to/esm2",
    mlx_minifold_path = "path/to/minifold_48L",
)

# Single sequence
pdb_str = predict_sequence("my_protein", "MKVLILSAVLFAASSA...", model, tokenizer)

# Batch
results = predict_batch(
    [("prot1", "MKVL..."), ("prot2", "MSYL...")],
    model, tokenizer,
)
# results = {"prot1": "<PDB string>", "prot2": "<PDB string>"}
```

## Acknowledgements

This package is a port of [MiniFold](https://github.com/jwohlwend/minifold) by Jeremy Wohlwend et al., adapted for Apple Silicon using MLX. The original MiniFold code and weights are the foundation of this work.

The use of MLX for ESM-2 is based on MLX-ESM-2 by Vincent Amato [https://github.com/vincentamato/mlx-esm-2] (https://github.com/vincentamato/mlx-esm-2), but re-implemented for the fine-tuned ESM-2 model used by MiniFold, and with int8 quantization.

Development assistance provided by [Claude Code](https://claude.ai/code) (Anthropic). The MLX port, SGMM Metal kernel, use of RMSNorm, and int8 quantization pipeline were developed, through iterations and with careful benchmarking, with the aid of Claude Sonnet 4.6.

## License

MIT License, following MiniFold. See [LICENSE](LICENSE) for details.
