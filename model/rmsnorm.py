"""
model/rmsnorm.py
----------------
Root Mean Square Layer Normalization.

Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
https://arxiv.org/abs/1910.07467

RMSNorm is preferred over LayerNorm in modern LLMs (LLaMA, Mistral, etc.)
because it omits the mean-centering step, reducing compute while preserving
training stability.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMSNorm(x) = x / RMS(x) * weight
    where RMS(x) = sqrt( mean(x^2) + eps )
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability, then back to original dtype
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
