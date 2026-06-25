"""
model/embeddings.py
-------------------
Token embedding table with optional output-projection weight tying.

Weight tying (Press & Wolf, 2016) shares the parameters between the input
embedding and the final linear projection to vocabulary logits, which:
  - Reduces parameter count significantly (vocab_size × d_model params saved)
  - Often improves perplexity at the same model size
"""

import torch
import torch.nn as nn

from model.config import ModelConfig


class TokenEmbedding(nn.Module):
    """
    Learnable token embedding table.
    Scaling by sqrt(d_model) follows the original "Attention Is All You Need".
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        # Zero-out the padding embedding
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].fill_(0.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)
        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        return self.embedding(token_ids)

    @property
    def weight(self) -> torch.Tensor:
        """Expose weight for output-projection tying."""
        return self.embedding.weight
