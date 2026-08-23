"""Pure MLX rigid-body geometry for structure prediction.

Ports the OpenFold rigid-body pipeline to MLX, removing the PyTorch bridge
in mlx_model.py.  All operations run on the Apple Metal back-end via MLX.

Conventions (matching OpenFold / minifold weights):
  - Rotation matrices are stored as (..., 3, 3) arrays (row-major).
  - A rigid body is a (rot (...,3,3), trans (...,3)) pair.
  - Angles are (sin, cos) pairs: alpha[..., 0] = sin, alpha[..., 1] = cos.
  - make_transform_from_reference(n, ca, c) places CA at the origin with C
    along the x-axis — exactly matching OpenFold Rigid.make_transform_from_reference.

Entry point
-----------
    atom37_pos, atom37_mask = build_all_atom_positions(n_ca_c, angles, aatype)
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from minifoldx.utils.residue_constants import (
    restype_rigid_group_default_frame,   # (21, 8, 4, 4)
    restype_atom14_to_rigid_group,        # (21, 14)
    restype_atom14_mask,                  # (21, 14)
    restype_atom14_rigid_group_positions, # (21, 14, 3)
    restype_atom37_mask,                  # (21, 37)
    restypes,
    restype_1to3,
    restype_name_to_atom14_names,
    atom_order,
    atom_types,
)


# ── Pre-computed residue-constant tables (MLX arrays) ────────────────────────

def _build_restype_atom37_to_atom14() -> np.ndarray:
    """Build (21, 37) int array mapping atom37 index → atom14 index per residue."""
    table = []
    for rt in restypes:
        atom_names = restype_name_to_atom14_names[restype_1to3[rt]]
        name_to_idx14 = {name: i for i, name in enumerate(atom_names) if name}
        table.append(
            [name_to_idx14.get(name, 0) for name in atom_types]
        )
    table.append([0] * 37)  # UNK residue
    return np.array(table, dtype=np.int32)


_DEFAULT_FRAMES     = mx.array(restype_rigid_group_default_frame,    dtype=mx.float32)  # (21,8,4,4)
_ATOM14_GROUP_IDX   = mx.array(restype_atom14_to_rigid_group,         dtype=mx.int32)    # (21,14)
_ATOM14_MASK        = mx.array(restype_atom14_mask,                   dtype=mx.float32)  # (21,14)
_ATOM14_LIT_POS     = mx.array(restype_atom14_rigid_group_positions,  dtype=mx.float32)  # (21,14,3)
_ATOM37_MASK        = mx.array(restype_atom37_mask,                   dtype=mx.float32)  # (21,37)
_ATOM37_TO_ATOM14   = mx.array(_build_restype_atom37_to_atom14(),     dtype=mx.int32)    # (21,37)


# ── Low-level rigid-body helpers ─────────────────────────────────────────────

def _apply_rot(rot: mx.array, v: mx.array) -> mx.array:
    """Apply rotation matrix to vectors: (...,3,3) x (...,3) → (...,3)."""
    return mx.einsum('...ij,...j->...i', rot, v)


def _rigid_compose(
    rot_a: mx.array, trans_a: mx.array,
    rot_b: mx.array, trans_b: mx.array | None,
):
    """Compose two rigid transforms: apply b then a (world <- b <- a convention).

    If trans_b is None it is treated as zero.
    Returns (rot_out, trans_out).
    """
    rot_out = mx.matmul(rot_a, rot_b)
    if trans_b is None:
        trans_out = trans_a
    else:
        trans_out = _apply_rot(rot_a, trans_b) + trans_a
    return rot_out, trans_out


# ── Backbone-frame construction ───────────────────────────────────────────────

def make_transform_from_reference(
    n_xyz:  mx.array,   # (..., 3)
    ca_xyz: mx.array,   # (..., 3)
    c_xyz:  mx.array,   # (..., 3)
    eps: float = 1e-20,
):
    """Build backbone rigid frame from N / CA / C coordinates.

    Exact MLX port of OpenFold ``Rigid.make_transform_from_reference``.
    Places CA at the origin with C along the x-axis and N in the xy-plane.

    Returns
    -------
    rot   : (..., 3, 3)  rotation (rows = world-space axes of local frame)
    trans : (..., 3)     translation = ca_xyz
    """
    translation = -ca_xyz
    n_xyz = n_xyz + translation
    c_xyz = c_xyz + translation

    c_x = c_xyz[..., 0]
    c_y = c_xyz[..., 1]
    c_z = c_xyz[..., 2]

    # Rotation 1: around z to put C in xz-plane
    norm_xy  = mx.sqrt(eps + c_x * c_x + c_y * c_y)
    sin_c1   = -c_y / norm_xy
    cos_c1   =  c_x / norm_xy
    zeros    = mx.zeros_like(sin_c1)
    ones     = mx.ones_like(sin_c1)

    c1_row0 = mx.stack([cos_c1, -sin_c1, zeros], axis=-1)
    c1_row1 = mx.stack([sin_c1,  cos_c1, zeros], axis=-1)
    c1_row2 = mx.stack([zeros,   zeros,   ones], axis=-1)
    c1_rots = mx.stack([c1_row0, c1_row1, c1_row2], axis=-2)  # (...,3,3)

    # Rotation 2: around y to put C on positive x-axis
    norm_xyz = mx.sqrt(eps + c_x * c_x + c_y * c_y + c_z * c_z)
    sin_c2   =  c_z / norm_xyz
    cos_c2   = mx.sqrt(mx.maximum(c_x * c_x + c_y * c_y, mx.array(0.0))) / norm_xyz

    c2_row0 = mx.stack([ cos_c2, zeros, sin_c2], axis=-1)
    c2_row1 = mx.stack([  zeros,  ones,  zeros], axis=-1)
    c2_row2 = mx.stack([-sin_c2, zeros, cos_c2], axis=-1)
    c2_rots = mx.stack([c2_row0, c2_row1, c2_row2], axis=-2)  # (...,3,3)

    # Compose c2 after c1
    c_rots = mx.matmul(c2_rots, c1_rots)  # (...,3,3)

    # Apply combined rotation to N
    n_xyz = _apply_rot(c_rots, n_xyz)

    # Rotation 3: around x to put N in xy-plane
    n_y = n_xyz[..., 1]
    n_z = n_xyz[..., 2]
    norm_yz = mx.sqrt(eps + n_y * n_y + n_z * n_z)
    sin_n   = -n_z / norm_yz
    cos_n   =  n_y / norm_yz

    n_row0 = mx.stack([ones,   zeros,   zeros], axis=-1)
    n_row1 = mx.stack([zeros,  cos_n,  -sin_n], axis=-1)
    n_row2 = mx.stack([zeros,  sin_n,   cos_n], axis=-1)
    n_rots = mx.stack([n_row0, n_row1, n_row2], axis=-2)  # (...,3,3)

    # Final rotation and transpose (OpenFold convention)
    rots  = mx.swapaxes(mx.matmul(n_rots, c_rots), -1, -2)  # (...,3,3)
    trans = -translation  # = ca_xyz

    return rots, trans


# ── Torsion-angle frames ──────────────────────────────────────────────────────

def torsion_angles_to_frames(
    aatype:   mx.array,  # (B, N)    int
    bb_rot:   mx.array,  # (B, N, 3, 3)
    bb_trans: mx.array,  # (B, N, 3)
    angles:   mx.array,  # (B, N, 7, 2)  (sin, cos)
):
    """Compute all 8 rigid frames per residue from backbone frame + torsions.

    MLX port of OpenFold ``torsion_angles_to_frames`` (feats.py).

    Returns
    -------
    frames_rot   : (B, N, 8, 3, 3)
    frames_trans : (B, N, 8, 3)
    """
    # Default frames for each residue type (B, N, 8, 4, 4)
    default_4x4   = _DEFAULT_FRAMES[aatype]           # (B, N, 8, 4, 4)
    default_rot   = default_4x4[..., :3, :3]          # (B, N, 8, 3, 3)
    default_trans = default_4x4[..., :3,  3]          # (B, N, 8, 3)

    # Prepend backbone frame (sin=0, cos=1) to the 7 torsion angles → 8 angles
    B, N = aatype.shape
    bb_sin = mx.zeros((B, N, 1, 1), dtype=angles.dtype)
    bb_cos = mx.ones( (B, N, 1, 1), dtype=angles.dtype)
    bb_angle = mx.concatenate([bb_sin, bb_cos], axis=-1)   # (B, N, 1, 2)
    alpha = mx.concatenate([bb_angle, angles], axis=-2)    # (B, N, 8, 2)

    sin_a = alpha[..., 0]   # (B, N, 8)
    cos_a = alpha[..., 1]   # (B, N, 8)
    z = mx.zeros_like(sin_a)
    o = mx.ones_like(sin_a)

    # Rotation around x-axis: [[1,0,0],[0,cos,-sin],[0,sin,cos]]
    t_row0 = mx.stack([o,     z,     z    ], axis=-1)
    t_row1 = mx.stack([z,  cos_a, -sin_a  ], axis=-1)
    t_row2 = mx.stack([z,  sin_a,  cos_a  ], axis=-1)
    torsion_rots = mx.stack([t_row0, t_row1, t_row2], axis=-2)  # (B, N, 8, 3, 3)

    # Compose: all_frames = default composed with torsion rotation
    # all_frames_rot   = default_rot @ torsion_rots
    # all_frames_trans = default_trans (torsion has no translation)
    all_rot   = mx.matmul(default_rot, torsion_rots)   # (B, N, 8, 3, 3)
    all_trans = default_trans                           # (B, N, 8, 3)

    # Extract chi frames (indices 4-7)
    chi1_rot,  chi1_trans  = all_rot[:, :, 4], all_trans[:, :, 4]
    chi2f_rot, chi2f_trans = all_rot[:, :, 5], all_trans[:, :, 5]
    chi3f_rot, chi3f_trans = all_rot[:, :, 6], all_trans[:, :, 6]
    chi4f_rot, chi4f_trans = all_rot[:, :, 7], all_trans[:, :, 7]

    # Chain chi2-4 through previous frame
    chi2_rot, chi2_trans = _rigid_compose(chi1_rot,  chi1_trans,  chi2f_rot, chi2f_trans)
    chi3_rot, chi3_trans = _rigid_compose(chi2_rot,  chi2_trans,  chi3f_rot, chi3f_trans)
    chi4_rot, chi4_trans = _rigid_compose(chi3_rot,  chi3_trans,  chi4f_rot, chi4f_trans)

    # Replace frames 5-7 with chained versions
    frames_to_bb_rot = mx.concatenate([
        all_rot[:, :, :5],
        chi2_rot[:, :, None],
        chi3_rot[:, :, None],
        chi4_rot[:, :, None],
    ], axis=2)   # (B, N, 8, 3, 3)

    frames_to_bb_trans = mx.concatenate([
        all_trans[:, :, :5],
        chi2_trans[:, :, None],
        chi3_trans[:, :, None],
        chi4_trans[:, :, None],
    ], axis=2)   # (B, N, 8, 3)

    # Compose with backbone frame to get global frames
    # bb_rot: (B, N, 3, 3), frames_to_bb_rot: (B, N, 8, 3, 3)
    # frames_to_bb_trans: (B, N, 8, 3), bb_trans: (B, N, 3)
    frames_rot = mx.matmul(
        bb_rot[:, :, None],       # (B, N, 1, 3, 3)
        frames_to_bb_rot,         # (B, N, 8, 3, 3)
    )                             # (B, N, 8, 3, 3)

    # Apply bb_rot to each of the 8 frame translations, then add bb_trans
    frames_trans = (
        mx.einsum('bnij,bnkj->bnki', bb_rot, frames_to_bb_trans)  # (B, N, 8, 3)
        + bb_trans[:, :, None]                                     # (B, N, 1, 3)
    )

    return frames_rot, frames_trans


# ── Atom-position computation ─────────────────────────────────────────────────

def frames_and_literature_positions_to_atom14_pos(
    aatype:      mx.array,  # (B, N)
    frames_rot:  mx.array,  # (B, N, 8, 3, 3)
    frames_trans: mx.array, # (B, N, 8, 3)
) -> mx.array:              # (B, N, 14, 3)
    """Place literature atom positions into global frames.

    MLX port of OpenFold ``frames_and_literature_positions_to_atom14_pos``.
    """
    # For each residue, which frame does each atom14 belong to?
    group_idx = _ATOM14_GROUP_IDX[aatype]            # (B, N, 14)  int
    # one-hot via identity-matrix indexing  (B, N, 14, 8)
    group_mask = mx.eye(8)[group_idx].astype(mx.float32)

    # Select the frame transform for each atom via weighted sum over frames
    # t_atoms_rot  : (B, N, 14, 3, 3)
    # t_atoms_trans: (B, N, 14, 3)
    t_atoms_rot   = mx.einsum('...af,...fij->...aij', group_mask, frames_rot)
    t_atoms_trans = mx.einsum('...af,...fi->...ai',   group_mask, frames_trans)

    # Literature positions in local frame
    lit_pos = _ATOM14_LIT_POS[aatype]                # (B, N, 14, 3)

    # Apply frame: rot @ lit_pos + trans
    pred_pos = _apply_rot(t_atoms_rot, lit_pos) + t_atoms_trans  # (B, N, 14, 3)

    # Mask non-existent atoms
    atom14_mask = _ATOM14_MASK[aatype]               # (B, N, 14)
    pred_pos = pred_pos * atom14_mask[..., None]

    return pred_pos


def atom14_to_atom37(
    atom14_pos: mx.array,  # (B, N, 14, 3)
    aatype:     mx.array,  # (B, N)
):
    """Convert atom14 positions to atom37 representation.

    Returns
    -------
    atom37_pos  : (B, N, 37, 3)
    atom37_mask : (B, N, 37)
    """
    # Index mapping and atom-existence mask
    residx_atom37_to_atom14 = _ATOM37_TO_ATOM14[aatype]  # (B, N, 37)  int
    atom37_mask = _ATOM37_MASK[aatype]                    # (B, N, 37)

    # Gather atom14 positions at the atom37 indices
    B, N, _, _ = atom14_pos.shape
    atom14_flat = atom14_pos.reshape(B * N, 14, 3)
    idx_flat    = residx_atom37_to_atom14.reshape(B * N, 37)

    # take_along_axis: indices (B*N, 37, 3) gathering from axis=1 of (B*N, 14, 3)
    idx_exp     = mx.broadcast_to(idx_flat[:, :, None], (B * N, 37, 3))
    atom37_flat = mx.take_along_axis(atom14_flat, idx_exp, axis=1)  # (B*N, 37, 3)
    atom37_pos  = atom37_flat.reshape(B, N, 37, 3)

    # Mask
    atom37_pos = atom37_pos * atom37_mask[..., None]

    return atom37_pos, atom37_mask


# ── Top-level entry point ────────────────────────────────────────────────────

def build_all_atom_positions(
    n_ca_c: mx.array,  # (B, N, 9)   backbone-update linear output
    angles: mx.array,  # (B, N, 7, 2) normalised torsion (sin, cos) pairs
    aatype: mx.array,  # (B, N)       residue-type indices (int)
    trans_scale_factor: float = 10.0,
):
    """Compute all-atom positions in global frame.

    Replaces the PyTorch bridge in mlx_model.py.

    Parameters
    ----------
    n_ca_c             : (B, N, 9)   — bb_update linear layer output, split as [N|CA|C]×3
    angles             : (B, N, 7, 2)— AngleResnet output (sin/cos per torsion)
    aatype             : (B, N)      — integer residue types
    trans_scale_factor : scalar multiplied onto backbone translations (default 10.0)

    Returns
    -------
    atom37_pos  : (B, N, 37, 3)  mx.array  — Cartesian coordinates
    atom37_mask : (B, N, 37)     mx.array  — 1 for existing atoms
    """
    n_ca_c  = n_ca_c.astype(mx.float32)
    angles  = angles.astype(mx.float32)

    n_xyz   = n_ca_c[..., :3]    # (B, N, 3)
    ca_xyz  = n_ca_c[..., 3:6]   # (B, N, 3)
    c_xyz   = n_ca_c[..., 6:9]   # (B, N, 3)

    bb_rot, bb_trans = make_transform_from_reference(n_xyz, ca_xyz, c_xyz)
    bb_trans = bb_trans * trans_scale_factor

    frames_rot, frames_trans = torsion_angles_to_frames(aatype, bb_rot, bb_trans, angles)

    atom14_pos = frames_and_literature_positions_to_atom14_pos(aatype, frames_rot, frames_trans)

    atom37_pos, atom37_mask = atom14_to_atom37(atom14_pos, aatype)

    return atom37_pos, atom37_mask
