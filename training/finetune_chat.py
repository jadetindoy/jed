"""
training/finetune_chat.py
-------------------------
Fine-tune a pretrained JedAI BASE model on conversational data so it learns
to *answer* (chat) instead of just continuing text.

This is the step that turns a "text continuer" into a "chatbot". It takes your
pretrained checkpoint, continues training it on user/assistant conversations
(from data/download_dialogue.py) at a low learning rate, and saves a new
chat-tuned checkpoint.

Usage:
    python -m training.finetune_chat \
        --base_checkpoint checkpoints/free/ckpt_0003000.pt \
        --data data/dialogue/conversations.txt \
        --output_dir checkpoints/chat \
        --epochs 3
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer as HFTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.model import JedAI
from training.checkpoint import load_checkpoint, save_checkpoint
from training.optimizer import build_optimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class TextChunkDataset(Dataset):
    """Tokenize a text file once, then serve non-overlapping max_seq_len chunks."""

    def __init__(self, token_ids: np.ndarray, max_seq_len: int) -> None:
        self.ids = token_ids
        self.max_seq_len = max_seq_len
        self.n = max(len(token_ids) // max_seq_len, 0)
        if self.n == 0:
            raise ValueError(
                f"Dialogue data has only {len(token_ids)} tokens, "
                f"less than max_seq_len {max_seq_len}. Use more data or a smaller seq len."
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        s = idx * self.max_seq_len
        chunk = np.asarray(self.ids[s : s + self.max_seq_len], dtype=np.int64)
        x = torch.from_numpy(chunk)
        return {"input_ids": x, "labels": x}


def load_base_model(checkpoint: str, device: torch.device) -> tuple[JedAI, ModelConfig]:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg_dict = state.get("config", {})
    cfg = ModelConfig.from_dict(
        {k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__}
    )
    model = JedAI(cfg).to(device)
    load_checkpoint(checkpoint, model, device=device)
    return model, cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune JedAI on conversations (chat tuning)")
    p.add_argument("--base_checkpoint", required=True, help="Pretrained base .pt checkpoint")
    p.add_argument("--data", default="data/dialogue/conversations.txt")
    p.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    p.add_argument("--output_dir", default="checkpoints/chat")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)  # low LR: don't wreck the base
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=300,
                   help="Save a checkpoint every N steps (crash insurance)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # 1. Load base model
    model, cfg = load_base_model(args.base_checkpoint, device)
    log.info(f"Loaded base model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # 2. Tokenize the dialogue text
    tok = HFTokenizer.from_file(args.tokenizer)
    text = Path(args.data).read_text(encoding="utf-8", errors="replace")
    ids = np.array(tok.encode(text).ids, dtype=np.int64)
    log.info(f"Dialogue data: {len(ids):,} tokens")

    ds = TextChunkDataset(ids, max_seq_len=cfg.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        pin_memory=torch.cuda.is_available())

    # 3. Optimizer (low LR continued training)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=0.0)
    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda" if use_cuda else "cpu", enabled=use_cuda)

    # 4. Fine-tune
    model.train()
    step = 0
    total_steps = args.epochs * len(loader)
    log.info(f"Fine-tuning for {args.epochs} epoch(s) = {total_steps} steps")

    pbar = tqdm(total=total_steps, desc="Chat fine-tune", unit="step", dynamic_ncols=True)
    for epoch in range(args.epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda" if use_cuda else "cpu", enabled=use_cuda):
                _, loss = model(input_ids, labels=labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            loss_val = loss.item()
            ppl = math.exp(min(loss_val, 20))
            # Live progress bar: ticks every step so you can see it working
            pbar.update(1)
            pbar.set_postfix(epoch=f"{epoch+1}/{args.epochs}",
                             loss=f"{loss_val:.4f}", ppl=f"{ppl:.1f}")
            if step % args.save_every == 0:
                save_checkpoint(model=model, optimizer=optimizer, step=step,
                                config=cfg.to_dict(), output_dir=args.output_dir, tag="chat")
    pbar.close()

    # 5. Save the chat-tuned model
    save_checkpoint(
        model=model, optimizer=optimizer, step=step,
        config=cfg.to_dict(), output_dir=args.output_dir, tag="chat",
    )
    log.info(f"Done. Chat model saved in {args.output_dir}/")
    log.info(f"Test it: python chat.py --checkpoint {args.output_dir}/ckpt_chat.pt --chat")


if __name__ == "__main__":
    main()
