"""
model/model.py
--------------
Top-level JedAI language model.

Architecture:
    Input IDs → TokenEmbedding → N × TransformerBlock → RMSNorm → Linear (LM head)

The LM head may share weights with the embedding table (weight tying) to
reduce parameter count and often improve perplexity.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.embeddings import TokenEmbedding
from model.rmsnorm import RMSNorm
from model.transformer import Transformer


class JedAI(nn.Module):
    """
    Decoder-only causal language model.

    Usage:
        config = ModelConfig(d_model=512, n_heads=8, n_layers=6)
        model  = JedAI(config)
        logits = model(input_ids)           # (B, T, vocab_size)
        tokens = model.generate(input_ids)  # greedy / sampled output
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding   = TokenEmbedding(config)
        self.transformer = Transformer(config)
        self.norm        = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head     = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch, seq_len) — token IDs
            labels:    (batch, seq_len) — shifted targets for LM loss (optional)

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss:   scalar cross-entropy loss (only if labels provided)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        x = self.embedding(input_ids)    # (B, T, d_model)
        x = self.transformer(x)           # (B, T, d_model)
        x = self.norm(x)                  # (B, T, d_model)
        logits = self.lm_head(x)          # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Standard causal LM loss: shift left by 1
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Auto-regressive token generation.

        Args:
            input_ids:      (1, prompt_len) — tokenized prompt
            max_new_tokens: maximum tokens to generate
            temperature:    softmax temperature (1.0 = no change)
            top_k:          keep only top-k logits (0 = disabled)
            top_p:          nucleus sampling threshold (1.0 = disabled)
            eos_token_id:   stop generation when this token is produced

        Returns:
            (1, prompt_len + n_generated) token IDs
        """
        from inference.sampling import sample_logits  # local import to avoid circular

        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            ctx = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self.forward(ctx)
            next_logits = logits[:, -1, :]  # (1, vocab_size)

            next_token = sample_logits(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == eos:
                break

        return input_ids

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def num_parameters(self, non_embedding: bool = True) -> int:
        """Count trainable parameters (optionally excluding embeddings)."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            total -= self.embedding.embedding.weight.numel()
        return total

    def __repr__(self) -> str:
        params_m = self.num_parameters() / 1e6
        return (
            f"JedAI(\n"
            f"  d_model={self.config.d_model}, "
            f"n_heads={self.config.n_heads}, "
            f"n_layers={self.config.n_layers}\n"
            f"  params={params_m:.1f}M (non-embedding)\n"
            f")"
        )
