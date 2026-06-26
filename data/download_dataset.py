"""
data/download_dataset.py
------------------------
Download a real public text corpus from HuggingFace and dump it into
data/raw/ as plain .txt, ready for the existing clean -> tokenize pipeline.

This replaces the tiny 0.5 MB starter corpus with a real dataset so the
model has enough text to actually learn from.

Recommended datasets (small -> large):
    wikitext-2     ~2M tokens   (~12 MB)   first end-to-end smoke test, trains in minutes
    wikitext-103   ~100M tokens (~520 MB)  realistic small-model target
    openwebtext    ~8B tokens   (~38 GB)   GPT-2-scale reproduction (needs a real GPU + time)

Usage:
    python -m data.download_dataset --dataset wikitext-2
    python -m data.download_dataset --dataset wikitext-103
    python -m data.download_dataset --dataset openwebtext --max_docs 200000
"""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# name -> (hf_repo, hf_config, text_column)
DATASETS = {
    "wikitext-2":   ("Salesforce/wikitext", "wikitext-2-raw-v1", "text"),
    "wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
    "openwebtext":  ("Skylion007/openwebtext", None, "text"),
}


def download(
    dataset: str,
    output_dir: Path,
    max_docs: int | None = None,
    shard_size: int = 50_000,
) -> None:
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from: {', '.join(DATASETS)}"
        )

    repo, config, text_col = DATASETS[dataset]
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading '{dataset}' from HuggingFace ({repo}) ...")
    # streaming=True avoids downloading the whole thing into RAM/disk at once
    ds = load_dataset(repo, config, split="train", streaming=True)

    # We concatenate many short rows into bigger shard files so we don't
    # create millions of tiny files (which would slow the pipeline down).
    buffer: list[str] = []
    buffer_chars = 0
    shard_idx = 0
    n_docs = 0
    n_written_chars = 0

    def flush() -> None:
        nonlocal shard_idx, buffer, buffer_chars
        if not buffer:
            return
        shard_path = output_dir / f"{dataset}_{shard_idx:05d}.txt"
        shard_path.write_text("\n\n".join(buffer), encoding="utf-8")
        shard_idx += 1
        buffer = []
        buffer_chars = 0

    target_shard_chars = shard_size * 200  # ~ shard_size lines of typical text

    pbar = tqdm(ds, desc=f"Downloading {dataset}", unit="doc")
    for row in pbar:
        text = (row.get(text_col) or "").strip()
        if not text:
            continue
        buffer.append(text)
        buffer_chars += len(text)
        n_docs += 1
        n_written_chars += len(text)

        if buffer_chars >= target_shard_chars:
            flush()

        if max_docs is not None and n_docs >= max_docs:
            break

    flush()
    pbar.close()

    log.info(
        f"Done. Wrote {shard_idx} shard file(s), {n_docs:,} documents, "
        f"~{n_written_chars/1e6:.1f} MB of text into {output_dir}"
    )
    log.info("Next steps:")
    log.info("  python -m data.clean")
    log.info("  python -m tokenizer.train --vocab_size 8000")
    log.info("  python -m data.tokenize_dataset")
    log.info("  python training/trainer.py --config configs/free.yaml")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a public text corpus for JedAI")
    p.add_argument(
        "--dataset", default="wikitext-2", choices=list(DATASETS),
        help="Which corpus to download (default: wikitext-2, the fast smoke test)",
    )
    p.add_argument(
        "--output_dir", type=Path, default=Path("data/raw"),
        help="Where to write the .txt shards",
    )
    p.add_argument(
        "--max_docs", type=int, default=None,
        help="Cap the number of documents (useful to keep openwebtext small)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download(args.dataset, args.output_dir, max_docs=args.max_docs)
