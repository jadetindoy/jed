"""
training/optimizer.py
---------------------
AdamW optimizer factory with correct weight-decay filtering.

Best practice: do NOT apply weight decay to:
  - Bias terms
  - LayerNorm / RMSNorm gains (1-D parameters)
  - Embedding weights (optional, but common)

Applying decay only to weight matrices helps regularize without
harming norm/bias terms that should not be penalized.
"""

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool = False,
) -> torch.optim.AdamW:
    """
    Create an AdamW optimizer with separate parameter groups:
      - Group 1 (decay):    weight matrices (≥2D)
      - Group 2 (no-decay): biases, norms, embeddings

    Args:
        model:        the JedAI model (or any nn.Module)
        lr:           peak learning rate
        weight_decay: L2 regularization strength for decayed params
        betas:        AdamW beta1, beta2
        eps:          AdamW epsilon
        fused:        use torch.optim.AdamW(fused=True) if available (CUDA only)

    Returns:
        Configured AdamW optimizer.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            # 1-D: biases, RMSNorm weights, etc.
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # fused AdamW is faster on CUDA but not available on CPU
    use_fused = fused and torch.cuda.is_available()

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        fused=use_fused,
    )

    n_decay    = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(
        f"[optimizer] decay={n_decay/1e6:.2f}M params | "
        f"no_decay={n_no_decay/1e6:.2f}M params | "
        f"fused={use_fused}"
    )
    return optimizer
