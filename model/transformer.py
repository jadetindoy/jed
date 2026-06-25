"""
model/transformer.py
---------------------
Transformer block and full Transformer stack.

Each block follows the Pre-Norm architecture used in LLaMA / modern LLMs:

    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))

Pre-norm training is more stable than post-norm at large scale.
"""

import torch
import torch.nn as nn

from model.attention import CausalSelfAttention
from model.config import ModelConfig
from model.feedforward import SwiGLUFeedForward
from model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Single decoder-only Transformer block (Pre-LN style).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn  = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn   = SwiGLUFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Attention sub-layer with residual
        x = x + self.attn(self.norm1(x))
        # Feed-forward sub-layer with residual
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    """
    Stack of N TransformerBlocks.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return x
