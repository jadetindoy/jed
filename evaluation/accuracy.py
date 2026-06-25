"""
evaluation/accuracy.py
-----------------------
Token-level accuracy evaluation.

Measures what fraction of next-token predictions match the ground truth.

Usage:
    python -m evaluation.accuracy \
        --checkpoint checkpoints/ckpt_0010000.pt \
        --data_dir datasets/cleaned \
        --tokenizer tokenizer/tokenizer.json
"""

import argparse
import logging
from pathlib import Path

import torch
from tokenizers import Tokenizer as HFTokenizer
from tqdm import tqdm

log = logging.getLogger(__name__)


@torch.inference_mode()
def compute_accuracy(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    device: torch.device,
    max_seq_len: int = 2048,
    ignore_index: int = 0,
) -> tuple[float, int]:
    model.eval()
    chunk = token_ids[:, :max_seq_len].to(device)

    logits, _ = model(chunk)
    preds  = logits[:, :-1].argmax(dim=-1)
    labels = chunk[:, 1:]

    mask    = labels != ignore_index
    correct = (preds == labels) & mask
    n_tokens  = mask.sum().item()
    n_correct = correct.sum().item()

    return n_correct / max(n_tokens, 1), int(n_tokens)


def evaluate_directory(
    model: torch.nn.Module,
    tokenizer: HFTokenizer,
    data_dir: Path,
    device: torch.device,
    max_docs: int = 100,
) -> dict:
    files = list(data_dir.rglob("*.txt"))[:max_docs]
    if not files:
        log.warning("No .txt files found.")
        return {}

    total_correct_tokens = 0
    total_tokens = 0

    for path in tqdm(files, desc="Accuracy"):
        text = path.read_text(encoding="utf-8", errors="replace")
        enc  = tokenizer.encode(text)
        if len(enc.ids) < 2:
            continue
        ids = torch.tensor([enc.ids], dtype=torch.long)
        acc, n = compute_accuracy(model, ids, device)
        total_correct_tokens += int(acc * n)
        total_tokens += n

    if total_tokens == 0:
        return {}

    return {
        "n_docs":         len(files),
        "total_tokens":   total_tokens,
        "token_accuracy": total_correct_tokens / total_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate JedAI token accuracy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("datasets/cleaned"))
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--max_docs", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    from inference.generate import Generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator(args.checkpoint, args.tokenizer, device=str(device))
    tokenizer = HFTokenizer.from_file(args.tokenizer)
    results = evaluate_directory(gen.model, tokenizer, args.data_dir, device, max_docs=args.max_docs)
    print("\n=== Accuracy Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
