"""
training/checkpoint.py
-----------------------
Utilities for saving and loading training checkpoints.

Each checkpoint stores:
  - model state_dict
  - optimizer state_dict
  - current global step
  - training configuration dict
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    output_dir: str | Path,
    tag: str | None = None,
    max_keep: int = 3,
) -> Path:
    """
    Save a training checkpoint to disk.

    Args:
        model:      the JedAI model
        optimizer:  the AdamW optimizer
        step:       current global step
        config:     serializable dict of ModelConfig/training config
        output_dir: directory to save checkpoint to
        tag:        custom filename tag (default: step number)
        max_keep:   keep only the N most recent checkpoints (0 = keep all)

    Returns:
        Path to the saved checkpoint file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"ckpt_{tag or step:07d}.pt"
    path = output_dir / filename

    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    torch.save(state, path)
    log.info(f"Checkpoint saved → {path}")

    # Prune old checkpoints
    if max_keep > 0:
        _prune_checkpoints(output_dir, max_keep)

    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Load a checkpoint from disk.

    Args:
        path:      path to the .pt checkpoint file
        model:     model to load weights into
        optimizer: optimizer to restore state (optional)
        device:    target device for tensors

    Returns:
        The full checkpoint dict (contains 'step', 'config', etc.)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device or "cpu"
    state = torch.load(path, map_location=map_location, weights_only=True)

    model.load_state_dict(state["model_state_dict"])
    log.info(f"Loaded model weights from {path} (step={state['step']})")

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        log.info("Loaded optimizer state.")

    return state


def latest_checkpoint(directory: str | Path) -> Path | None:
    """Return the path to the most recently saved checkpoint, or None."""
    directory = Path(directory)
    checkpoints = sorted(directory.glob("ckpt_*.pt"))
    return checkpoints[-1] if checkpoints else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prune_checkpoints(directory: Path, max_keep: int) -> None:
    """Delete old checkpoints, keeping only the N most recent."""
    checkpoints = sorted(directory.glob("ckpt_*.pt"))
    to_delete = checkpoints[:-max_keep]
    for ckpt in to_delete:
        try:
            ckpt.unlink()
            log.debug(f"Deleted old checkpoint: {ckpt}")
        except OSError as exc:
            log.warning(f"Could not delete {ckpt}: {exc}")
