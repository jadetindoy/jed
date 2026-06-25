"""
inference/sampling.py
---------------------
Sampling strategies for auto-regressive generation.

Supported modes (can be combined):
  - Temperature scaling
  - Top-k filtering
  - Top-p (nucleus) filtering
  - Greedy decoding (temperature=0)
"""

import torch
import torch.nn.functional as F


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Zero out all logits except the top-k.

    Args:
        logits: (batch, vocab_size) raw logits
        k:      number of top logits to keep (0 = keep all)
    Returns:
        Filtered logits (same shape).
    """
    if k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k=k, dim=-1)
    threshold = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability mass ≥ p.

    Args:
        logits: (batch, vocab_size)
        p:      nucleus probability threshold (1.0 = no filtering)
    Returns:
        Filtered logits.
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative prob above threshold
    sorted_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[sorted_to_remove] = float("-inf")

    # Scatter back to original ordering
    return logits.scatter(-1, sorted_indices, sorted_logits)


@torch.inference_mode()
def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Sample the next token from a logit distribution.

    Args:
        logits:      (batch, vocab_size) — raw (un-normalized) logits
        temperature: controls sharpness (< 1 = sharper, > 1 = flatter)
                     temperature=0 → greedy (argmax)
        top_k:       top-k filtering (0 = disabled)
        top_p:       nucleus filtering (1.0 = disabled)

    Returns:
        (batch, 1) tensor of sampled token IDs.
    """
    if temperature == 0.0:
        # Greedy decoding
        return logits.argmax(dim=-1, keepdim=True)

    # Apply temperature
    logits = logits / temperature

    # Apply filters
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    # Sample from the resulting distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
