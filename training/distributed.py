"""
training/distributed.py
------------------------
Helpers for initializing and tearing down PyTorch Distributed Data Parallel (DDP)
training across multiple GPUs or nodes.

Usage (launch via torchrun):
    torchrun --nproc_per_node=8 training/trainer.py --config configs/1b.yaml
"""

import logging
import os

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


def is_distributed() -> bool:
    """Return True if a distributed process group is active."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the global rank of this process (0 if not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    """Return the local GPU rank (from LOCAL_RANK env var)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Return the total number of processes (1 if not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """Return True only for the rank-0 process (or non-distributed runs)."""
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> None:
    """
    Initialize the process group for DDP.

    Automatically detects RANK / WORLD_SIZE from environment variables set
    by torchrun. No-op if WORLD_SIZE <= 1.

    Args:
        backend: 'nccl' for GPU, 'gloo' for CPU-only
    """
    if dist.is_initialized():
        return

    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size <= 1:
        log.info("Single-process run; skipping distributed init.")
        return

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    log.info(
        f"[DDP] Initialized | rank={rank}/{world_size} | "
        f"local_rank={local_rank} | backend={backend}"
    )


def cleanup() -> None:
    """Destroy the process group. Call at the end of training."""
    if is_distributed():
        dist.destroy_process_group()
        log.info("[DDP] Process group destroyed.")


def barrier() -> None:
    """Synchronize all processes (no-op if not distributed)."""
    if is_distributed():
        dist.barrier()
