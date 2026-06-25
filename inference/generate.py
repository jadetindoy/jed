"""
inference/generate.py
---------------------
Generator class: wraps the JedAI model + tokenizer for easy text generation.

Usage:
    from inference.generate import Generator
    gen = Generator(model_path="checkpoints/ckpt_0010000.pt",
                    tokenizer_path="tokenizer/tokenizer.json")
    print(gen.generate("Once upon a time"))
"""

import torch
from pathlib import Path
from typing import Optional

from tokenizers import Tokenizer as HFTokenizer

from model.config import ModelConfig
from model.model import JedAI
from training.checkpoint import load_checkpoint


class Generator:
    """
    High-level text generation interface.

    Handles:
      - Loading a trained checkpoint
      - Encoding a text prompt to token IDs
      - Running auto-regressive generation
      - Decoding token IDs back to text
    """

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path = "tokenizer/tokenizer.json",
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load tokenizer
        self.tokenizer = HFTokenizer.from_file(str(tokenizer_path))

        # Reconstruct model from checkpoint
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        cfg_dict = state.get("config", {})
        model_cfg = ModelConfig.from_dict(
            {k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__}
        )
        self.model = JedAI(model_cfg).to(self.device)
        load_checkpoint(model_path, self.model, device=self.device)
        self.model.eval()

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate text continuation from a prompt.

        Args:
            prompt:         Input text to condition on.
            max_new_tokens: Maximum number of tokens to generate.
            temperature:    Sampling temperature (0 = greedy).
            top_k:          Top-k filtering (0 = disabled).
            top_p:          Nucleus filtering (1.0 = disabled).

        Returns:
            Full generated string (prompt + continuation).
        """
        enc = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([enc.ids], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        generated = output_ids[0].tolist()
        return self.tokenizer.decode(generated)

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)
