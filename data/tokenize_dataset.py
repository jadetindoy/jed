"""
data/tokenize_dataset.py
-------------------------
Pre-tokenizes cleaned text files and splits them into binary train/val datasets.

Usage:
    python -m data.tokenize_dataset \
        --input_dir data/cleaned \
        --output_dir data/tokenized \
        --tokenizer tokenizer/tokenizer.json \
        --val_ratio 0.1
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def tokenize_files(
    input_dir: Path,
    tokenizer_path: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
) -> None:
    # 1. Load tokenizer
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    log.info(f"Loaded tokenizer with vocab size: {vocab_size}")

    # Detect proper dtype
    dtype = np.uint16 if vocab_size <= 65535 else np.int32
    dtype_str = "uint16" if dtype == np.uint16 else "int32"

    eos_token_id = tokenizer.token_to_id("<eos>")
    if eos_token_id is None:
        log.warning("Could not find '<eos>' token in tokenizer. Defaulting ID to 3.")
        eos_token_id = 3

    # 2. Collect files
    files = sorted(list(input_dir.rglob("*.txt")))
    # Filter out placeholder files
    files = [f for f in files if f.name != ".gitkeep" and f.stat().st_size > 0]
    
    if not files:
        log.warning(f"No non-empty text files found in {input_dir}. Cannot proceed.")
        return

    log.info(f"Tokenizing {len(files)} files...")

    all_doc_tokens = []
    total_tokens = 0

    for file_path in tqdm(files, desc="Encoding documents"):
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.error(f"Error reading {file_path}: {e}")
            continue

        encoded = tokenizer.encode(text)
        tokens = encoded.ids
        
        # Append <eos> if not already present at the end
        if not tokens or tokens[-1] != eos_token_id:
            tokens.append(eos_token_id)
            
        all_doc_tokens.append(tokens)
        total_tokens += len(tokens)

    if total_tokens == 0:
        log.warning("No tokens collected. Make sure you have non-empty cleaned text files.")
        return

    log.info(f"Total tokens collected: {total_tokens:,}")

    # 3. Split into Train & Val
    # If we have multiple documents, we can split by document boundary.
    # Otherwise, split the single document's token list.
    train_tokens = []
    val_tokens = []

    if len(all_doc_tokens) > 1:
        # Split documents
        val_target = int(total_tokens * val_ratio)
        current_val_tokens = 0
        
        # We can put the last few documents into val split
        for tokens in reversed(all_doc_tokens):
            if current_val_tokens < val_target:
                val_tokens.extend(tokens)
                current_val_tokens += len(tokens)
            else:
                train_tokens.extend(tokens)
        
        # Reverse train back to keep original order if possible, though order doesn't strictly matter
        # Since we popped from the end, the remaining goes to train (which is currently the front docs)
        # Let's rebuild train/val properly:
        train_tokens = []
        val_tokens = []
        accumulated = 0
        train_target = total_tokens - val_target
        
        for tokens in all_doc_tokens:
            if accumulated < train_target:
                train_tokens.extend(tokens)
                accumulated += len(tokens)
            else:
                val_tokens.extend(tokens)
    else:
        # Single document case
        single_doc = all_doc_tokens[0]
        split_idx = int(len(single_doc) * (1 - val_ratio))
        train_tokens = single_doc[:split_idx]
        val_tokens = single_doc[split_idx:]

    log.info(f"Train split: {len(train_tokens):,} tokens ({len(train_tokens)/total_tokens*100:.1f}%)")
    log.info(f"Val split: {len(val_tokens):,} tokens ({len(val_tokens)/total_tokens*100:.1f}%)")

    # 4. Save to binary files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_arr = np.array(train_tokens, dtype=dtype)
    val_arr = np.array(val_tokens, dtype=dtype)

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    log.info(f"Saved train binary to {train_path}")
    log.info(f"Saved val binary to {val_path}")

    # 5. Save metadata
    metadata = {
        "vocab_size": vocab_size,
        "dtype": dtype_str,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "total_tokens": total_tokens,
    }
    
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info(f"Saved metadata to {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenize cleaned text files")
    parser.add_argument(
        "--input_dir", type=Path, default=Path("data/cleaned"),
        help="Directory of cleaned .txt files"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/tokenized"),
        help="Directory to save tokenized binary files"
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"),
        help="Path to trained tokenizer.json"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Ratio of tokens to assign to validation split"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenize_files(
        input_dir=args.input_dir,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
    )
