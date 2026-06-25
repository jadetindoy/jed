"""
model/rope.py
-------------
Rotary Position Embeddings (RoPE).

Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864

RoPE encodes position by rotating query/key vectors in the complex plane.
This gives strong relative-position awareness without extra parameters.
"""

import torch
from typing import Tuple


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10_000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute cosine and sine rotation matrices.

    Returns:
        cos: (seq_len, head_dim/2)
        sin: (seq_len, head_dim/2)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    half_dim = head_dim // 2
    # Frequency bands: θ_i = 1 / (theta^(2i/head_dim))
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    # Outer product: (seq_len, half_dim)
    angles = torch.outer(positions, freqs)

    cos = angles.cos()
    sin = angles.sin()
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to query or key tensors.

    Args:
        x:   (batch, seq_len, n_heads, head_dim)
        cos: (seq_len, head_dim/2)
        sin: (seq_len, head_dim/2)

    Returns:
        Rotated tensor of same shape as x.
    """
    B, T, H, D = x.shape
    half = D // 2

    # Split into two halves
    x1 = x[..., :half]    # (B, T, H, half)
    x2 = x[..., half:]    # (B, T, H, half)

    # Broadcast: (seq_len, half) → (1, T, 1, half)
    c = cos[:T].unsqueeze(0).unsqueeze(2)
    s = sin[:T].unsqueeze(0).unsqueeze(2)

    # Rotation in 2D: [x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return rotated.type_as(x)
