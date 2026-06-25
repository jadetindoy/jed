"""
model/feedforward.py
--------------------
SwiGLU Feed-Forward Network.

Paper: "GLU Variants Improve Transformer" (Noam Shazeer, 2020)
https://arxiv.org/abs/2002.05202

SwiGLU replaces the standard FFN (Linear → ReLU → Linear) with:
    FFN(x) = (xW₁ ⊙ Swish(xW_gate)) W₂

This gating mechanism empirically outperforms vanilla FFN and GELU FFN
at the same parameter count. Used in LLaMA, PaLM, Mistral, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU FFN: gate(x) × silu(up(x)) → down projection.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.ffn_hidden_dim

        self.gate_proj = nn.Linear(config.d_model, hidden, bias=False)
        self.up_proj   = nn.Linear(config.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # SwiGLU: Swish(gate) ⊙ up
        gate   = F.silu(self.gate_proj(x))
        up     = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(self.dropout(hidden))
