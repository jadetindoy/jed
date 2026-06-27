"""
chat.py
-------
Interactive tester for a trained JedAI model. Type a prompt, see what the
model generates. Type 'quit' (or Ctrl-C) to exit.

Usage:
    python chat.py
    python chat.py --checkpoint checkpoints/50m_cpu/ckpt_0050000.pt
    python chat.py --temperature 0.9 --max_new_tokens 80
"""

import argparse
import sys
from pathlib import Path


def find_latest_checkpoint() -> str | None:
    """Grab the newest ckpt_*.pt under checkpoints/ if none is given."""
    ckpts = sorted(Path("checkpoints").rglob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1]) if ckpts else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive JedAI tester")
    parser.add_argument("--checkpoint", default=None, help="Path to a .pt checkpoint")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--chat", action="store_true",
                        help="Use chat format (for models fine-tuned with finetune_chat.py)")
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_latest_checkpoint()
    if checkpoint is None:
        print("No checkpoint found under checkpoints/. Train a model first.")
        sys.exit(1)

    print(f"Loading model: {checkpoint}")
    from inference.generate import Generator

    gen = Generator(model_path=checkpoint, tokenizer_path=args.tokenizer)
    print("Model ready. Type a prompt and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            print("bye!")
            break

        # In --chat mode, wrap the prompt in the same markers used for fine-tuning
        # so a chat-tuned model knows to produce an assistant reply.
        full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n" if args.chat else prompt

        output = gen.generate(
            full_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        # Show only the newly generated continuation
        continuation = output[len(full_prompt):] if output.startswith(full_prompt) else output
        # Stop at the next turn marker if the model produced one
        continuation = continuation.split("<|user|>")[0].split("<eos>")[0]
        print(f"ai  > {continuation.strip()}\n")


if __name__ == "__main__":
    main()
