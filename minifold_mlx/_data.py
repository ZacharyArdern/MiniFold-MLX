"""Data preparation utilities for MLX MiniFold inference.

Pure numpy/MLX — no PyTorch required.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from minifold_mlx.utils.residue_constants import (
    restype_order_with_x,
    restype_order_with_x_inverse,
)
from minifold_mlx.utils.protein import Protein, to_pdb


# ── Input preparation ─────────────────────────────────────────────────────────

def prepare_input(seq: str, tokenizer):
    """Prepare inputs for a single sequence.

    Parameters
    ----------
    seq       : amino-acid string
    tokenizer : MLX ESM2 tokenizer

    Returns
    -------
    tokens   : mx.array  (L,)          — ESM2 token IDs (no BOS/EOS)
    mask_np  : np.ndarray (L,) float32 — all-ones valid-residue mask
    aatype   : np.ndarray (L,) int32   — residue-type indices (0–20, X=20)
    """
    # Map any non-standard residues to X (index 20)
    aatype = np.array(
        [restype_order_with_x.get(aa, restype_order_with_x["X"]) for aa in seq],
        dtype=np.int32,
    )

    # Reconstruct canonical sequence so the tokeniser sees the same residues
    canonical_seq = "".join(restype_order_with_x_inverse[i] for i in aatype)

    # Tokenise — strip BOS (pos 0) and EOS (last) to match fair-ESM2 behaviour
    raw    = tokenizer.encode(canonical_seq)
    ids    = np.asarray(raw, dtype=np.int32)[1:-1]
    tokens = mx.array(ids)                           # (L,)

    mask_np = np.ones(len(seq), dtype=np.float32)    # (L,)

    return tokens, mask_np, aatype


# ── Batching ─────────────────────────────────────────────────────────────────

def create_batches(sequences: list[tuple[str, str]], batch_tokens: int):
    """Group (seq_id, seq) pairs into batches.

    Batch size is bounded by max_len² (MiniFormer memory scales quadratically).

    Parameters
    ----------
    sequences    : list of (seq_id, seq_str)
    batch_tokens : quadratic budget — batches split when max_len² exceeds this

    Returns
    -------
    list of lists of (seq_id, seq_str)
    """
    sorted_seqs = sorted(sequences, key=lambda x: len(x[1]))
    batches: list[list] = []
    cur_batch: list = []
    cur_max_len = 0

    for seq_id, seq in sorted_seqs:
        L = len(seq)
        new_max = max(cur_max_len, L)
        if cur_batch and new_max * new_max > batch_tokens:
            batches.append(cur_batch)
            cur_batch, cur_max_len = [], 0
            new_max = L
        cur_batch.append((seq_id, seq))
        cur_max_len = new_max

    if cur_batch:
        batches.append(cur_batch)
    return batches


# ── Padding utilities ─────────────────────────────────────────────────────────

def pad_tokens(token_list: list, max_len: int, pad_id: int) -> mx.array:
    """Pad a list of 1-D mx.arrays to (B, max_len)."""
    rows = []
    for t in token_list:
        gap = max_len - t.shape[0]
        if gap > 0:
            t = mx.concatenate([t, mx.full((gap,), pad_id, dtype=mx.int32)])
        rows.append(t)
    return mx.stack(rows)


def pad_mask(mask_list: list, max_len: int) -> mx.array:
    """Pad a list of 1-D float32 arrays to (B, max_len)."""
    rows = []
    for m in mask_list:
        padded = np.zeros(max_len, dtype=np.float32)
        padded[: len(m)] = m
        rows.append(padded)
    return mx.array(np.stack(rows))


def pad_aatype(aatype_list: list, max_len: int) -> mx.array:
    """Pad a list of 1-D int32 aatype arrays to (B, max_len)."""
    rows = []
    for a in aatype_list:
        padded = np.zeros(max_len, dtype=np.int32)
        padded[: len(a)] = a
        rows.append(padded)
    return mx.array(np.stack(rows))


# ── Output ────────────────────────────────────────────────────────────────────

def output_to_pdb(seq: str, coords: np.ndarray, mask: np.ndarray,
                  plddt: np.ndarray) -> str:
    """Convert model output arrays to a PDB string."""
    af_seq = np.array([restype_order_with_x[r] for r in seq])
    plddt  = plddt[:, None].repeat(coords.shape[1], axis=1)
    pred   = Protein(
        aatype         = af_seq,
        atom_positions = coords,
        atom_mask      = mask,
        residue_index  = 1 + np.arange(len(seq)),
        b_factors      = plddt,
    )
    return to_pdb(pred)
