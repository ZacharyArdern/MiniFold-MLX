"""MLX implementation of the StructureModule neural-network layers.

StructureModuleMLX returns the intermediate MLX arrays (single repr ``s``,
normalised torsion angles, backbone-update vector ``n_ca_c``).  The caller
(MiniFoldMLX.__call__) passes these to mlx_rigid.build_all_atom_positions
for the pure-MLX rigid-body geometry step.

Weight key names are chosen to match the HuggingFace safetensors exactly.
The only exception is the dict-keyed Sequential layers where ReLU (no parameters)
is omitted, e.g.
  mlp = {"0": LayerNorm, "1": Linear, "3": Linear}   (skips idx-2 ReLU)
  mlp = {"1": Linear,    "3": Linear}                 (skips idx-0,2 ReLUs)
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn



# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_plddt(logits: mx.array) -> mx.array:
    """Compute per-residue pLDDT from logits."""
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = mx.arange(0.5 * bin_width, 1.0, bin_width)
    probs = mx.softmax(logits, axis=-1)
    return (probs * bounds).sum(axis=-1) * 100.0


# ── Attention ─────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Gated multi-head self-attention used inside StructureModule."""

    def __init__(self, dim: int, num_heads: int, head_width: int):
        super().__init__()
        assert dim == num_heads * head_width
        self.num_heads  = num_heads
        self.head_width = head_width
        self.rescale_factor = head_width ** -0.5

        self.layer_norm = nn.LayerNorm(dim)
        self.proj   = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim)
        self.g_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, bias: mx.array, mask: mx.array) -> mx.array:
        """
        x    : (B, N, D)
        bias : (B, H, N, N)   per-head attention bias
        mask : (B, N)          1 = valid residue, 0 = padding
        """
        B, N, D = x.shape
        x = self.layer_norm(x)

        # QKV  (B, N, 3*D) -> (B, H, N, C) for each of q, k, v
        t = self.proj(x)                              # (B, N, 3D)
        t = t.reshape(B, N, self.num_heads, -1)       # (B, N, H, 3C)
        t = t.transpose(0, 2, 1, 3)                   # (B, H, N, 3C)
        qkv = t.shape[-1] // 3
        q = t[..., :qkv]
        k = t[..., qkv : 2 * qkv]
        v = t[..., 2 * qkv :]

        q = self.rescale_factor * q
        a = mx.einsum("...qc,...kc->...qk", q, k)    # (B, H, N, N)
        a = a + bias

        # Mask out padding (mask=0 -> -1e9)
        neg_inf = mx.full(a.shape, -1e9, dtype=a.dtype)
        a = mx.where(mask[:, None, None, :] != 0, a, neg_inf)
        a = mx.softmax(a, axis=-1)

        # Weighted sum -> (B, N, H, C)
        y = mx.einsum("...hqk,...hkc->...qhc", a, v)
        y = y.reshape(B, N, -1)                       # (B, N, D)

        # Gating (uses the same layer-normed x)
        g = mx.sigmoid(self.g_proj(x))
        y = g * y
        return self.o_proj(y)


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Two-layer MLP: LayerNorm -> Linear -> ReLU -> Linear.
    Flat attribute naming (idx 2 is ReLU, no params): mlp_0=LN, mlp_1=L, mlp_3=L.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp_0 = nn.LayerNorm(in_dim)
        self.mlp_1 = nn.Linear(in_dim, in_dim)
        self.mlp_3 = nn.Linear(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.mlp_0(x)
        x = self.mlp_1(x)
        x = mx.maximum(x, 0)   # ReLU
        x = self.mlp_3(x)
        return x


# ── AngleResnet ───────────────────────────────────────────────────────────────

class AngleResnetBlock(nn.Module):
    """Residual block for torsion-angle prediction.
    Flat attribute naming (idx 0,2 are ReLUs, no params): mlp_1=L, mlp_3=L.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.mlp_1 = nn.Linear(dim, dim)
        self.mlp_3 = nn.Linear(dim, dim)

    def __call__(self, a: mx.array) -> mx.array:
        r = a
        a = mx.maximum(a, 0)   # ReLU
        a = self.mlp_1(a)
        a = mx.maximum(a, 0)   # ReLU
        a = self.mlp_3(a)
        return r + a


class AngleResnet(nn.Module):
    """Predicts torsion angles from single representation."""

    def __init__(self, c_in: int, c_hidden: int, no_blocks: int,
                 no_angles: int, epsilon: float):
        super().__init__()
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in      = nn.Linear(c_in, c_hidden)
        self.linear_initial = nn.Linear(c_in, c_hidden)
        self.layers         = [AngleResnetBlock(c_hidden) for _ in range(no_blocks)]
        self.linear_out     = nn.Linear(c_hidden, no_angles * 2)

    def __call__(self, s: mx.array, s_initial: mx.array):
        s_initial = mx.maximum(s_initial, 0)
        s_initial = self.linear_initial(s_initial)
        s = mx.maximum(s, 0)
        s = self.linear_in(s) + s_initial

        for layer in self.layers:
            s = layer(s)

        s = mx.maximum(s, 0)
        s = self.linear_out(s)                         # (..., no_angles*2)
        s = s.reshape(*s.shape[:-1], -1, 2)            # (..., no_angles, 2)

        unnormalized = s
        norm = mx.sqrt(mx.clip((s * s).sum(axis=-1, keepdims=True), self.eps, None))
        s = s / norm
        return unnormalized, s


# ── StructureModuleMLX ────────────────────────────────────────────────────────

class StructureModuleMLX(nn.Module):
    """Neural-network layers of the StructureModule in MLX.

    Produces:
      s       : (B, N, c_s)   — single representation after all transformer blocks
      angles  : (B, N, 7, 2)  — normalised torsion angles (sin/cos pairs)
      n_ca_c  : (B, N, 9)     — backbone-update vector from bb_update linear layer
                                (caller splits into n, ca, c each (B,N,3) for Rigid)

    Rigid-body geometry (Rigid.make_transform_from_reference, torsion_angles_to_frames,
    frames_and_literature_positions_to_atom14_pos) must be executed by the caller
    using the PyTorch utilities in minifold.utils.rigid_utils / feats.
    The relevant residue-constant tensors are exposed as ``_default_frames`` etc.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_resnet: int,
        head_dim: int,
        no_heads: int,
        no_blocks: int,
        no_resnet_blocks: int,
        no_angles: int,
        epsilon: float,
    ):
        super().__init__()
        self.no_blocks = no_blocks
        self.no_heads  = no_heads

        self.layer_norm_s = nn.LayerNorm(c_s)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_in    = nn.Linear(c_s, c_s)
        self.linear_b     = nn.Linear(c_z, no_blocks * no_heads)

        self.attn        = [Attention(c_s, no_heads, head_dim) for _ in range(no_blocks)]
        self.transitions = [MLP(c_s, c_s)                      for _ in range(no_blocks)]

        self.bb_update   = nn.Linear(c_s, 9)
        self.angle_resnet = AngleResnet(c_s, c_resnet, no_resnet_blocks, no_angles, epsilon)

    def __call__(self, s: mx.array, z: mx.array, mask: mx.array):
        """
        s    : (B, N, c_s)    single representation from PairToSequence
        z    : (B, N, N, c_z) pair representation from FoldingTrunk
        mask : (B, N)          float/bool residue mask

        Returns dict with keys: s, angles, unnormalized_angles, n_ca_c
        """
        B, N, _ = s.shape

        # Input norms
        s = self.layer_norm_s(s)
        s_initial = s
        s = self.linear_in(s)

        # Pair -> attention bias  (B, no_blocks, no_heads, N, N)
        z = self.layer_norm_z(z)
        b = self.linear_b(z)                                  # (B, N, N, no_blocks*no_heads)
        b = b.transpose(0, 3, 1, 2)                           # (B, no_blocks*no_heads, N, N)
        b = b.reshape(B, self.no_blocks, self.no_heads, N, N)

        # Transformer blocks
        for i in range(self.no_blocks):
            s = s + self.attn[i](s, b[:, i], mask)
            s = s + self.transitions[i](s)

        # Torsion angles
        unnormalized_angles, angles = self.angle_resnet(s, s_initial)

        # Backbone update (used by caller for Rigid)
        n_ca_c = self.bb_update(s.astype(mx.float32))         # (B, N, 9)

        return {
            "s":                   s,
            "angles":              angles,
            "unnormalized_angles": unnormalized_angles,
            "n_ca_c":              n_ca_c,
        }
