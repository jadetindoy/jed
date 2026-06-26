"""
evaluation/benchmark.py
------------------------
Orchestrates all evaluation tasks and produces a consolidated report.

Usage:
    python -m evaluation.benchmark \
        --checkpoint checkpoints/ckpt_0010000.pt \
        --data_dir datasets/cleaned \
        --output eval_results.json
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path

# Add project root to sys.path to allow running evaluation/benchmark.py directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tokenizers import Tokenizer as HFTokenizer
from rich.console import Console
from rich.table import Table

from evaluation.accuracy import evaluate_directory as eval_accuracy
from evaluation.perplexity import evaluate_directory as eval_perplexity

log = logging.getLogger(__name__)
console = Console()


def run_benchmark(
    model: torch.nn.Module,
    tokenizer: HFTokenizer,
    data_dir: Path,
    device: torch.device,
    max_docs: int = 100,
) -> dict:
    results: dict = {"data_dir": str(data_dir), "max_docs": max_docs}

    console.rule("[bold cyan]Perplexity")
    t0 = time.perf_counter()
    ppl_results = eval_perplexity(model, tokenizer, data_dir, device, max_docs)
    results["perplexity"] = ppl_results
    results["perplexity"]["elapsed_s"] = round(time.perf_counter() - t0, 2)

    console.rule("[bold cyan]Token Accuracy")
    t0 = time.perf_counter()
    acc_results = eval_accuracy(model, tokenizer, data_dir, device, max_docs)
    results["accuracy"] = acc_results
    results["accuracy"]["elapsed_s"] = round(time.perf_counter() - t0, 2)

    return results


def print_report(results: dict) -> None:
    table = Table(title="JedAI Evaluation Report", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    ppl = results.get("perplexity", {})
    acc = results.get("accuracy", {})

    if ppl:
        table.add_row("Perplexity (mean)", f"{ppl.get('mean_ppl', 0):.2f}")
        table.add_row("Perplexity (min)",  f"{ppl.get('min_ppl',  0):.2f}")
        table.add_row("Perplexity (max)",  f"{ppl.get('max_ppl',  0):.2f}")
        table.add_row("Docs evaluated",    str(ppl.get("n_docs", 0)))
    if acc:
        table.add_row("Token Accuracy",    f"{acc.get('token_accuracy', 0):.4f}")
        table.add_row("Total Tokens",      f"{acc.get('total_tokens', 0):,}")

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JedAI evaluation benchmark")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("datasets/cleaned"))
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--max_docs", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    from inference.generate import Generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator(args.checkpoint, args.tokenizer, device=str(device))
    tokenizer = HFTokenizer.from_file(args.tokenizer)
    results = run_benchmark(gen.model, tokenizer, args.data_dir, device, max_docs=args.max_docs)
    print_report(results)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        console.print(f"\n[green]Report saved → {args.output}")
