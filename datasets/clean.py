"""
datasets/clean.py
-----------------
Data cleaning pipeline for jed-ai.

Reads raw text files from datasets/raw/, applies cleaning transformations,
deduplicates, and writes processed output to datasets/cleaned/.

Usage:
    python -m datasets.clean --input_dir datasets/raw --output_dir datasets/cleaned
"""

import argparse
import hashlib
import logging
import re
import unicodedata
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form."""
    return unicodedata.normalize("NFC", text)


def remove_control_characters(text: str) -> str:
    """Strip non-printable / control characters (except newline & tab)."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t")
    )


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into one; strip trailing whitespace per line."""
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in lines]
    # Collapse runs of more than 2 blank lines into 2
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def remove_urls(text: str, placeholder: str = "") -> str:
    """Optionally replace URLs with a placeholder."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(placeholder, text)


def clean_document(text: str, remove_url: bool = False) -> str:
    """Apply the full cleaning pipeline to a single document string."""
    text = normalize_unicode(text)
    text = remove_control_characters(text)
    if remove_url:
        text = remove_urls(text)
    text = normalize_whitespace(text)
    return text.strip()


def document_hash(text: str) -> str:
    """SHA-256 hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_directory(
    input_dir: Path,
    output_dir: Path,
    remove_url: bool = False,
    min_chars: int = 50,
) -> None:
    """Process all .txt files in input_dir and write to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    total = skipped_short = skipped_dup = written = 0

    files = list(input_dir.rglob("*.txt"))
    log.info(f"Found {len(files)} raw text file(s) in {input_dir}")

    for path in tqdm(files, desc="Cleaning"):
        total += 1
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            log.warning(f"Could not read {path}: {exc}")
            continue

        cleaned = clean_document(raw, remove_url=remove_url)

        if len(cleaned) < min_chars:
            skipped_short += 1
            continue

        doc_hash = document_hash(cleaned)
        if doc_hash in seen_hashes:
            skipped_dup += 1
            continue
        seen_hashes.add(doc_hash)

        # Preserve relative path structure
        rel = path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(cleaned, encoding="utf-8")
        written += 1

    log.info(
        f"Done. Total={total}, written={written}, "
        f"skipped_short={skipped_short}, skipped_dup={skipped_dup}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="jed-ai data cleaning pipeline")
    parser.add_argument("--input_dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/cleaned"))
    parser.add_argument(
        "--remove_urls", action="store_true", help="Strip URLs from text"
    )
    parser.add_argument(
        "--min_chars", type=int, default=50,
        help="Minimum characters for a document to be kept"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        remove_url=args.remove_urls,
        min_chars=args.min_chars,
    )
