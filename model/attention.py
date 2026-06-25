"""
model/attention.py
------------------
Multi-Head Causal Self-Attention with RoPE and optional Grouped-Query Attention (GQA).

GQA (Ainslie et al., 2023) uses fewer key/value heads than query heads,
reducing the KV-cache memory footprint during inference without significant
quality loss. When n_kv_heads == n_heads, this is standard MHA.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.rope import apply_rope, build_rope_cache


class CausalSelfAttention(nn.Module):
    """
    Grouped-Query Multi-Head Causal Self-Attention.

    Shapes throughout (B=batch, T=seq_len, H=n_heads, Hkv=n_kv_heads, D=head_dim):
        q: (B, T, H,   D)
        k: (B, T, Hkv, D)
        v: (B, T, Hkv, D)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.scale = math.sqrt(self.head_dim)
        self.n_rep = self.n_heads // self.n_kv_heads  # GQA repetition factor

        d = config.d_model
        # Projections
        self.q_proj = nn.Linear(d, self.n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # RoPE cache (will be built/extended lazily)
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None
        self._rope_theta = config.rope_theta

    # ------------------------------------------------------------------
    # RoPE helpers
    # ------------------------------------------------------------------

    def _get_rope(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._rope_cos is None
            or self._rope_cos.shape[0] < seq_len
            or self._rope_cos.device != device
        ):
            self._rope_cos, self._rope_sin = build_rope_cache(
                seq_len=seq_len,
                head_dim=self.head_dim,
                theta=self._rope_theta,
                device=device,
                dtype=torch.float32,
            )
        return self._rope_cos[:seq_len], self._rope_sin[:seq_len]

    # ------------------------------------------------------------------
    # GQA helper: repeat k/v heads to match q heads
    # ------------------------------------------------------------------

    @staticmethod
    def _repeat_kv(
        x: torch.Tensor, n_rep: int
    ) -> torch.Tensor:
        """(B, T, Hkv, D) → (B, T, H, D)"""
        if n_rep == 1:
            return x
        B, T, Hkv, D = x.shape
        return (
            x[:, :, :, None, :]
            .expand(B, T, Hkv, n_rep, D)
            .reshape(B, T, Hkv * n_rep, D)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, d_model)
            mask: (T, T) causal mask — True where attention should be blocked
        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        # Project
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        cos, sin = self._get_rope(T, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: expand k/v to match q
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        # (B, T, H, D) → (B, H, T, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # (B, H, T, D) → (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out)
