"""
training/trainer.py
--------------------
Main Trainer class: orchestrates the full pre-training loop.

Features:
  - Gradient accumulation
  - Mixed-precision training (torch.amp)
  - Cosine LR schedule with warmup
  - Periodic checkpoint saving
  - DDP-aware logging (only rank-0 prints)
  - Gradient clipping

Usage:
    python training/trainer.py --config configs/50m.yaml
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

from model.config import ModelConfig
from model.model import JedAI
from training.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from training.distributed import (
    barrier, cleanup, get_rank, init_distributed, is_main_process
)
from training.optimizer import build_optimizer
from training.scheduler import apply_lr, get_lr

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: JedAI,
        train_loader,
        config: dict,
        output_dir: str = "checkpoints",
        resume: bool = True,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)

        if torch.distributed.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=False,
            )

        self.train_loader = train_loader

        self.lr_max       = config.get("lr", 3e-4)
        self.lr_min       = config.get("lr_min", 1e-5)
        self.warmup_steps = config.get("warmup_steps", 200)
        self.max_steps    = config.get("max_steps", 100_000)
        self.grad_clip    = config.get("grad_clip", 1.0)
        self.grad_accum   = config.get("grad_accum_steps", 1)
        self.save_every   = config.get("save_every", 1000)
        self.log_every    = config.get("log_every", 10)

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = build_optimizer(
            raw_model,
            lr=self.lr_max,
            weight_decay=config.get("weight_decay", 0.1),
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        self.step = 0
        if resume:
            ckpt = latest_checkpoint(self.output_dir)
            if ckpt is not None:
                state = load_checkpoint(ckpt, raw_model, self.optimizer, self.device)
                self.step = state.get("step", 0)
                log.info(f"Resumed from step {self.step}")

    def train(self) -> None:
        self.model.train()
        loader_iter = iter(self.train_loader)

        t0 = time.perf_counter()
        total_loss = 0.0
        tokens_seen = 0

        while self.step < self.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0

            for micro_step in range(self.grad_accum):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels    = batch.get("labels", input_ids).to(self.device)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    _, loss = self.model(input_ids, labels=labels)
                    loss = loss / self.grad_accum

                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                tokens_seen += input_ids.numel()

            lr = get_lr(self.step, self.warmup_steps, self.max_steps, self.lr_max, self.lr_min)
            apply_lr(self.optimizer, lr)

            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.step += 1
            total_loss += accumulated_loss

            if self.step % self.log_every == 0 and is_main_process():
                avg_loss = total_loss / self.log_every
                dt = time.perf_counter() - t0
                tokens_per_sec = tokens_seen / dt
                log.info(
                    f"step={self.step:>7d} | loss={avg_loss:.4f} | "
                    f"lr={lr:.2e} | tok/s={tokens_per_sec:,.0f}"
                )
                total_loss = 0.0
                tokens_seen = 0
                t0 = time.perf_counter()

            if self.step % self.save_every == 0 and is_main_process():
                raw_model = self.model.module if hasattr(self.model, "module") else self.model
                save_checkpoint(
                    model=raw_model,
                    optimizer=self.optimizer,
                    step=self.step,
                    config=self.config,
                    output_dir=self.output_dir,
                )

        if is_main_process():
            log.info("Training complete.")
        cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JedAI language model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    init_distributed()

    model_cfg = ModelConfig(**{
        k: v for k, v in cfg.items()
        if k in ModelConfig.__dataclass_fields__
    })
    model = JedAI(model_cfg)

    if is_main_process():
        print(model)
        print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Dummy DataLoader — replace with real tokenized dataset
    dummy_ids = torch.randint(0, model_cfg.vocab_size, (64, min(128, model_cfg.max_seq_len)))
    loader = DataLoader(TensorDataset(dummy_ids), batch_size=cfg.get("batch_size", 4), shuffle=True)

    class _DictLoader:
        def __init__(self, ds): self._ds = ds
        def __iter__(self):
            for (ids,) in self._ds:
                yield {"input_ids": ids}
        def __len__(self): return len(self._ds)

    output_dir = args.output_dir or cfg.get("output_dir", "checkpoints")
    trainer = Trainer(
        model=model,
        train_loader=_DictLoader(loader),
        config=cfg,
        output_dir=output_dir,
        resume=not args.no_resume,
    )
    trainer.train()
