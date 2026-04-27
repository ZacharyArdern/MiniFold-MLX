"""MLX implementation of auxiliary prediction heads (pLDDT)."""

import mlx.core as mx
import mlx.nn as nn


def _compute_plddt(logits: mx.array) -> mx.array:
    """Per-residue pLDDT from logits (matches minifold/train/loss.py::compute_plddt)."""
    num_bins  = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds    = mx.arange(0.5 * bin_width, 1.0, bin_width)   # (num_bins,)
    probs     = mx.softmax(logits, axis=-1)
    return (probs * bounds).sum(axis=-1) * 100.0


class PerResidueLDDTCaPredictorMLX(nn.Module):
    """Three-layer MLP pLDDT predictor.  Weight names match HuggingFace model exactly."""

    def __init__(self, no_bins: int, c_in: int, c_hidden: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_in)
        self.linear_1   = nn.Linear(c_in,     c_hidden)
        self.linear_2   = nn.Linear(c_hidden,  c_hidden)
        self.linear_3   = nn.Linear(c_hidden,  no_bins)

    def __call__(self, s: mx.array) -> mx.array:
        s = self.layer_norm(s)
        s = mx.maximum(self.linear_1(s), 0)   # ReLU
        s = mx.maximum(self.linear_2(s), 0)   # ReLU
        return self.linear_3(s)


class AuxiliaryHeadsMLX(nn.Module):
    """Wrapper that computes pLDDT from the single representation."""

    def __init__(self, no_bins: int, c_in: int, c_hidden: int):
        super().__init__()
        self.plddt = PerResidueLDDTCaPredictorMLX(no_bins, c_in, c_hidden)

    def __call__(self, s: mx.array) -> mx.array:
        """Return per-residue pLDDT scores, shape (B, N)."""
        logits = self.plddt(s)
        return _compute_plddt(logits)
