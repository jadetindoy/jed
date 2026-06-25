"""
model/config.py
---------------
ModelConfig dataclass — single source of truth for all architectural hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_size: int = 32_000
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    # ── Architecture ─────────────────────────────────────────────────────────
    d_model: int = 512          # hidden / embedding dimension
    n_heads: int = 8            # number of attention heads
    n_kv_heads: Optional[int] = None  # grouped-query attention heads (None = MHA)
    n_layers: int = 6           # number of transformer blocks
    max_seq_len: int = 2048     # maximum sequence length

    # ── Feed-forward ─────────────────────────────────────────────────────────
    # SwiGLU FFN hidden dim = floor(d_model * ffn_mult / 64) * 64
    ffn_mult: float = 2.6875    # approx 4 * (2/3) for SwiGLU
    ffn_hidden_dim: Optional[int] = None  # override if set explicitly

    # ── Regularization ───────────────────────────────────────────────────────
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # ── Normalization ────────────────────────────────────────────────────────
    rms_norm_eps: float = 1e-5

    # ── Positional encoding ───────────────────────────────────────────────────
    rope_theta: float = 10_000.0
    rope_scaling: Optional[float] = None  # scale factor for extended context

    # ── Weight tying ──────────────────────────────────────────────────────────
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        # Default n_kv_heads to n_heads (standard MHA)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        # Compute FFN hidden dim if not explicitly set
        if self.ffn_hidden_dim is None:
            raw = int(self.d_model * self.ffn_mult)
            # Round up to nearest multiple of 64 for hardware efficiency
            self.ffn_hidden_dim = (raw + 63) // 64 * 64

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)
