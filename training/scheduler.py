"""
training/scheduler.py
---------------------
Cosine learning-rate scheduler with linear warmup.

Schedule:
  Phase 1 (0 → warmup_steps):        linear ramp from lr_min → lr_max
  Phase 2 (warmup_steps → max_steps): cosine decay from lr_max → lr_min
"""

import math
import torch


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    lr_max: float,
    lr_min: float = 1e-5,
) -> float:
    """
    Compute the learning rate for a given training step.

    Args:
        step:         current global step (0-indexed)
        warmup_steps: number of linear warm-up steps
        max_steps:    total training steps
        lr_max:       peak learning rate
        lr_min:       minimum (floor) learning rate

    Returns:
        Scalar learning rate value.
    """
    if step < warmup_steps:
        # Linear warm-up
        return lr_max * step / max(warmup_steps, 1)

    if step >= max_steps:
        return lr_min

    # Cosine decay
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + cosine * (lr_max - lr_min)


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Update all param-group learning rates in-place."""
    for group in optimizer.param_groups:
        group["lr"] = lr
