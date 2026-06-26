"""
evaluation/perplexity.py
------------------------
Evaluate language model perplexity on a held-out dataset.

Perplexity = exp(average negative log-likelihood per token)

Lower is better. A perplexity of K means the model is, on average,
as uncertain as if choosing uniformly from K options at each step.

Usage:
    python -m evaluation.perplexity \
        --checkpoint checkpoints/ckpt_0010000.pt \
        --data_dir datasets/cleaned \
        --tokenizer tokenizer/tokenizer.json
"""

import argparse
import logging
import math
import sys
from pathlib import Path

# Add project root to sys.path to allow running evaluation/perplexity.py directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer as HFTokenizer
from tqdm import tqdm

log = logging.getLogger(__name__)


def compute_perplexity(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    device: torch.device,
    stride: int = 512,
    max_seq_len: int = None,
) -> float:
    """
    Sliding-window perplexity evaluation to avoid truncation bias.
    """
    if max_seq_len is None:
        max_seq_len = getattr(getattr(model, "config", None), "max_seq_len", 2048)
    stride = min(stride, max_seq_len)

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    seq_len = token_ids.size(1)

    prev_end = 0
    with torch.inference_mode():
        for begin in range(0, seq_len, stride):
            end = min(begin + max_seq_len, seq_len)
            target_len = end - max(begin, prev_end)
            if target_len <= 0:
                continue

            chunk = token_ids[:, begin:end].to(device)
            logits, _ = model(chunk)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()

            shift_logits = shift_logits[:, -target_len:, :]
            shift_labels = shift_labels[:, -target_len:]

            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_tokens += shift_labels.numel()
            prev_end = end

    avg_nll = total_nll / max(total_tokens, 1)
    return math.exp(avg_nll)


def evaluate_directory(
    model: torch.nn.Module,
    tokenizer: HFTokenizer,
    data_dir: Path,
    device: torch.device,
    max_docs: int = 100,
) -> dict:
    files = list(data_dir.rglob("*.txt"))[:max_docs]
    if not files:
        log.warning("No .txt files found for evaluation.")
        return {}

    ppls = []
    for path in tqdm(files, desc="Perplexity"):
        text = path.read_text(encoding="utf-8", errors="replace")
        enc  = tokenizer.encode(text)
        if len(enc.ids) < 2:
            continue
        ids = torch.tensor([enc.ids], dtype=torch.long)
        ppl = compute_perplexity(model, ids, device)
        ppls.append(ppl)

    if not ppls:
        return {}

    return {
        "n_docs":   len(ppls),
        "mean_ppl": sum(ppls) / len(ppls),
        "min_ppl":  min(ppls),
        "max_ppl":  max(ppls),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate JedAI perplexity")
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
    print("\n=== Perplexity Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
