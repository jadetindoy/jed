"""
data/clean.py
--------------
Data cleaning pipeline for jed-ai.

Reads raw text from data/raw/ — both plain ``.txt`` files and ``.jsonl``
files (one JSON record per line, text under a ``"text"`` field) — applies
cleaning transformations, deduplicates at the document level across ALL
files, and writes processed output to data/cleaned/.

Usage:
    python -m data.clean --input_dir data/raw --output_dir data/cleaned
"""

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path

from tqdm import tqdm

# Candidate field names to pull document text from, in priority order.
TEXT_FIELDS = ("text", "content", "body", "raw")

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

def _extract_text(record: dict) -> str | None:
    """Pull the document text from a JSONL record using known field names."""
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _iter_documents(path: Path):
    """
    Yield raw (uncleaned) document strings from a single source file.

    - .txt  -> the whole file is one document.
    - .jsonl/.json -> one document per line, read from the "text" field
      (falls back to other common field names).
    """
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson", ".json"):
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    log.warning(f"{path.name}:{line_no} is not valid JSON, skipping")
                    continue
                text = _extract_text(record) if isinstance(record, dict) else None
                if text is None:
                    log.warning(
                        f"{path.name}:{line_no} has no usable text field "
                        f"(looked for {TEXT_FIELDS}), skipping"
                    )
                    continue
                yield text
    else:
        yield path.read_text(encoding="utf-8", errors="replace")


def process_directory(
    input_dir: Path,
    output_dir: Path,
    remove_url: bool = False,
    min_chars: int = 50,
) -> None:
    """Clean every .txt and .jsonl file in input_dir and write to output_dir.

    Documents are deduplicated by content hash across ALL files, so the same
    article appearing in both an aggregate file (e.g. all_books.jsonl) and an
    individual file is only kept once.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    files_seen = docs_total = skipped_short = skipped_dup = docs_written = 0

    patterns = ("*.txt", "*.jsonl", "*.ndjson", "*.json")
    files = sorted(
        p for pat in patterns for p in input_dir.rglob(pat)
        if p.name != ".gitkeep"
    )
    log.info(f"Found {len(files)} raw file(s) in {input_dir}")

    for path in tqdm(files, desc="Cleaning"):
        files_seen += 1
        try:
            raw_docs = list(_iter_documents(path))
        except Exception as exc:
            log.warning(f"Could not read {path}: {exc}")
            continue

        kept_docs: list[str] = []
        for raw in raw_docs:
            docs_total += 1
            cleaned = clean_document(raw, remove_url=remove_url)

            if len(cleaned) < min_chars:
                skipped_short += 1
                continue

            doc_hash = document_hash(cleaned)
            if doc_hash in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(doc_hash)
            kept_docs.append(cleaned)
            docs_written += 1

        if not kept_docs:
            continue

        # One cleaned .txt per source file. Multiple docs (from a .jsonl) are
        # joined with a blank-line separator; the tokenizer appends <eos> at
        # the file boundary.
        rel = path.relative_to(input_dir).with_suffix(".txt")
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n\n".join(kept_docs), encoding="utf-8")

    log.info(
        f"Done. files={files_seen}, docs={docs_total}, written={docs_written}, "
        f"skipped_short={skipped_short}, skipped_dup={skipped_dup}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="jed-ai data cleaning pipeline")
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/cleaned"))
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
