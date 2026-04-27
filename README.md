# MiniFold-MLX

An Apple Silicon port of [MiniFold](https://github.com/jwohlwend/minifold) using the [MLX](https://github.com/ml-explore/mlx) framework. Runs protein structure prediction entirely on Apple Silicon GPU/Neural Engine with no PyTorch dependency at inference time.

## Features

- Full ESM2 (3B) + MiniFold folding trunk in MLX
- int8 ESM2 quantization via `mlx.nn.quantize` (group_size=32)
- Custom Apple Metal SGMM gate kernel fusing LayerNorm + gating for TriangularUpdate blocks
- `mx.compile` support on MiniFormer for reduced Metal dispatch overhead
- 48-layer (full) and 12-layer (fast) MiniFold variants
- Simple three-function public API: `load_model`, `predict_sequence`, `predict_batch`
- Pre-converted weights hosted on HuggingFace for fast download-on-first-use

## Installation

```bash
pip install git+https://github.com/ZacharyArdern/MiniFold-MLX
```

Requires macOS with Apple Silicon (M1 or later) and MLX >= 0.16.0.

## Weights

Pre-converted MLX weights (finetuned ESM2 + MiniFold 48L/12L) are available on HuggingFace:

```
z-ardern/MiniFold_MLX_weights
├── ESM2_MiniFold/  # finetuned MLX ESM2 3B (safetensors, ~11 GB)
├── minifold_48L/   # 48-layer MiniFold MLX weights (~285 MB)
└── minifold_12L/   # 12-layer MiniFold MLX weights (~259 MB)
```

```python
from huggingface_hub import snapshot_download

weights = snapshot_download("z-ardern/MiniFold_MLX_weights")
esm_path      = f"{weights}/ESM2_MiniFold"
minifold_path = f"{weights}/minifold_48L"
```

## Quick Start

```python
from minifold_mlx import load_model, predict_sequence, predict_batch

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

This package is a port of [MiniFold](https://github.com/jwohlwend/minifold) by Jonas Wohlwend et al., adapted for Apple Silicon using MLX. The original MiniFold code and weights are the foundation of this work.

Development assistance provided by [Claude Code](https://claude.ai/code) (Anthropic). The MLX port, SGMM Metal kernel, int8 quantization pipeline, and benchmarking infrastructure were developed in collaboration with Claude Sonnet 4.6.

## License

MIT License, following MiniFold. See [LICENSE](LICENSE) for details.
