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
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path to allow running training/trainer.py directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

        use_cuda = torch.cuda.is_available()
        self.device_type = "cuda" if use_cuda else "cpu"
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=use_cuda)

        self.step = 0
        if resume:
            ckpt = latest_checkpoint(self.output_dir)
            if ckpt is not None:
                state = load_checkpoint(ckpt, raw_model, self.optimizer, self.device)
                self.step = state.get("step", 0)
                log.info(f"Resumed from step {self.step}")

    def train(self) -> None:
        from tqdm import tqdm

        use_amp = torch.cuda.is_available()
        self.model.train()
        loader_iter = iter(self.train_loader)

        t0 = time.perf_counter()
        total_loss = 0.0
        tokens_seen = 0

        pbar = tqdm(
            total=self.max_steps,
            initial=self.step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

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

                with torch.amp.autocast(self.device_type, enabled=use_amp):
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
            pbar.update(1)

            if self.step % self.log_every == 0 and is_main_process():
                avg_loss = total_loss / self.log_every
                dt = time.perf_counter() - t0
                tokens_per_sec = tokens_seen / dt
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", tok_s=f"{tokens_per_sec:,.0f}")
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

        pbar.close()
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

    # Auto-sync vocab_size to the actual tokenized dataset so the model's
    # embedding / LM head always matches the tokenizer it was built with.
    _meta_path = Path("data/tokenized/metadata.json")
    if _meta_path.exists():
        import json
        with open(_meta_path) as _mf:
            _meta = json.load(_mf)
        _data_vocab = _meta.get("vocab_size")
        if _data_vocab and cfg.get("vocab_size") != _data_vocab:
            log.info(
                f"Overriding config vocab_size={cfg.get('vocab_size')} "
                f"with dataset vocab_size={_data_vocab} (from metadata.json)"
            )
            cfg["vocab_size"] = _data_vocab

    model_cfg = ModelConfig(**{
        k: v for k, v in cfg.items()
        if k in ModelConfig.__dataclass_fields__
    })
    model = JedAI(model_cfg)

    if is_main_process():
        print(model)
        print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Load dataset
    tokenized_dir = Path("data/tokenized")
    train_bin = tokenized_dir / "train.bin"
    
    if train_bin.exists():
        if is_main_process():
            log.info(f"Loading real tokenized dataset from {train_bin}")
        from data.dataset import PretokenizedDataset, get_dataset_metadata
        
        meta = get_dataset_metadata(tokenized_dir)
        dtype = meta.get("dtype", "uint16")
        
        train_dataset = PretokenizedDataset(
            train_bin,
            max_seq_len=model_cfg.max_seq_len,
            dtype=dtype
        )
        
        sampler = None
        shuffle = True
        if torch.distributed.is_initialized():
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(train_dataset, shuffle=True)
            shuffle = False
            
        loader = DataLoader(
            train_dataset,
            batch_size=cfg.get("batch_size", 4),
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            num_workers=cfg.get("num_workers", 0),
        )
    else:
        if is_main_process():
            log.warning(
                "Pre-tokenized dataset not found at data/tokenized/train.bin! "
                "Falling back to dummy DataLoader. "
                "To use real data, please place raw text in data/raw/, then run:\n"
                "  python -m data.download_dataset   # or add your own .txt files\n"
                "  python -m data.clean\n"
                "  python -m tokenizer.train\n"
                "  python -m data.tokenize_dataset"
            )
        dummy_ids = torch.randint(0, model_cfg.vocab_size, (64, min(128, model_cfg.max_seq_len)))
        ds = TensorDataset(dummy_ids)
        
        sampler = None
        shuffle = True
        if torch.distributed.is_initialized():
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(ds, shuffle=True)
            shuffle = False
            
        dummy_loader = DataLoader(ds, batch_size=cfg.get("batch_size", 4), shuffle=shuffle, sampler=sampler)
        
        class _DictLoader:
            def __init__(self, ds_loader):
                self.ds_loader = ds_loader
            def __iter__(self):
                for batch in self.ds_loader:
                    ids = batch[0] if isinstance(batch, (list, tuple)) else batch
                    yield {"input_ids": ids, "labels": ids}
            def __len__(self):
                return len(self.ds_loader)
                
        loader = _DictLoader(dummy_loader)

    output_dir = args.output_dir or cfg.get("output_dir", "checkpoints")
    trainer = Trainer(
        model=model,
        train_loader=loader,
        config=cfg,
        output_dir=output_dir,
        resume=not args.no_resume,
    )
    trainer.train()
