"""
tokenizer/train.py
------------------
Train a Byte-Pair Encoding (BPE) tokenizer on the cleaned corpus.

Outputs:
  - tokenizer/tokenizer.json  (HuggingFace tokenizers format)
  - tokenizer/vocab.json      (token → id mapping)

Usage:
    python -m tokenizer.train \
        --input_dir datasets/cleaned \
        --vocab_size 32000 \
        --output_dir tokenizer
"""

import argparse
import json
import logging
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# Special tokens — order matters: index 0 = <pad>, 1 = <unk>, etc.
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<mask>"]


def collect_files(input_dir: Path) -> list[str]:
    """Recursively collect all .txt files under input_dir."""
    files = [str(p) for p in input_dir.rglob("*.txt")]
    log.info(f"Found {len(files)} text file(s) for tokenizer training.")
    return files


def train(
    input_dir: Path,
    output_dir: Path,
    vocab_size: int = 32_000,
) -> None:
    files = collect_files(input_dir)
    if not files:
        log.warning("No files found. Add .txt files to datasets/cleaned/ first.")
        return

    # Build tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )

    log.info(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(files)} file(s) ...")
    tokenizer.train(files=files, trainer=trainer)

    # Verify special tokens were added correctly
    vocab = tokenizer.get_vocab()
    log.info(f"Trained vocab size: {len(vocab)}")
    for tok in SPECIAL_TOKENS:
        if tok in vocab:
            log.info(f"  {tok} → id={vocab[tok]}")
        else:
            log.warning(f"  {tok} NOT found in vocab!")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    log.info(f"Saved tokenizer → {tokenizer_path}")

    # Save plain vocab mapping for convenience
    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"Saved vocab ({len(vocab)} tokens) → {vocab_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for jed-ai")
    parser.add_argument(
        "--input_dir", type=Path, default=Path("datasets/cleaned"),
        help="Directory of cleaned .txt files"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("tokenizer"),
        help="Where to save tokenizer.json and vocab.json"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=32_000,
        help="Target vocabulary size"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
    )
