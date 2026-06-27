"""
data/download_dialogue.py
-------------------------
Download conversational OR instruction (Q&A) data and format it for chat
fine-tuning. Both are written with the same user/assistant markers so
training/finetune_chat.py can use them directly.

    <|user|>
    What is the capital of France?
    <|assistant|>
    The capital of France is Paris.<eos>

Dataset types:
  - conversational ("soda") -> teaches casual, human-like CHATTING
  - instruction   ("dolly", "alpaca") -> teaches ANSWERING questions / following instructions

Tip: for a bot that BOTH chats and answers, stack them with --append:
    python -m data.download_dialogue --dataset soda  --max_items 5000
    python -m data.download_dialogue --dataset dolly --append

Usage:
    python -m data.download_dialogue --dataset dolly
    python -m data.download_dialogue --dataset alpaca --max_items 20000
"""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

USER = "<|user|>"
ASSISTANT = "<|assistant|>"
EOS = "<eos>"

# name -> dict(repo, config, split, type)
#   type "dialogue"    -> multi-turn list of utterances
#   type "instruction" -> single instruction + response (Q&A)
DATASETS = {
    "soda":   {"repo": "allenai/soda",                    "config": None, "split": "train", "type": "dialogue"},
    "dolly":  {"repo": "databricks/databricks-dolly-15k", "config": None, "split": "train", "type": "instruction"},
    "alpaca": {"repo": "tatsu-lab/alpaca",                "config": None, "split": "train", "type": "instruction"},
}


def format_dialogue(turns: list[str]) -> str:
    lines = []
    for i, utt in enumerate(turns):
        utt = (utt or "").strip()
        if not utt:
            continue
        role = USER if i % 2 == 0 else ASSISTANT
        lines.append(f"{role}\n{utt}")
    return "\n".join(lines) + EOS + "\n\n" if lines else ""


def format_instruction(instruction: str, context: str, response: str) -> str:
    instruction = (instruction or "").strip()
    response = (response or "").strip()
    if not instruction or not response:
        return ""
    user = instruction
    if context and context.strip():
        user += "\n\n" + context.strip()
    return f"{USER}\n{user}\n{ASSISTANT}\n{response}{EOS}\n\n"


def download(dataset: str, output_dir: Path, max_items: int | None, append: bool) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose: {', '.join(DATASETS)}")
    info = DATASETS[dataset]
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading '{dataset}' ({info['repo']}) as type '{info['type']}' ...")
    ds = load_dataset(info["repo"], info["config"], split=info["split"], streaming=True)

    out_path = output_dir / "conversations.txt"
    mode = "a" if append else "w"
    n_items = 0
    n_chars = 0

    with open(out_path, mode, encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"Formatting {dataset}", unit="item"):
            if info["type"] == "dialogue":
                turns = row.get("dialogue") or row.get("dialog") or row.get("turns")
                block = format_dialogue(turns) if turns else ""
            else:  # instruction
                block = format_instruction(
                    row.get("instruction"),
                    row.get("context") or row.get("input"),
                    row.get("response") or row.get("output"),
                )
            if not block:
                continue
            f.write(block)
            n_items += 1
            n_chars += len(block)
            if max_items is not None and n_items >= max_items:
                break

    verb = "Appended" if append else "Wrote"
    log.info(f"{verb} {n_items:,} examples (~{n_chars/1e6:.1f} MB) -> {out_path}")
    log.info("Next: python -m training.finetune_chat --base_checkpoint <your base ckpt>")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download chat / instruction data for fine-tuning")
    p.add_argument("--dataset", default="dolly", choices=list(DATASETS))
    p.add_argument("--output_dir", type=Path, default=Path("data/dialogue"))
    p.add_argument("--max_items", type=int, default=None, help="Cap number of examples")
    p.add_argument("--append", action="store_true",
                   help="Append to conversations.txt instead of overwriting (to mix datasets)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download(args.dataset, args.output_dir, args.max_items, args.append)
