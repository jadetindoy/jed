"""
datasets/dataset.py
--------------------
PyTorch Dataset using memory-mapped files (np.memmap) for high-performance data loading.
"""

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class PretokenizedDataset(Dataset):
    """
    Memory-mapped pre-tokenized dataset for training and validation.

    Loads a binary file of token IDs and serves chunks of size `max_seq_len`.
    """

    def __init__(
        self,
        bin_path: Union[str, Path],
        max_seq_len: int,
        dtype: str = "uint16",
    ) -> None:
        super().__init__()
        self.bin_path = Path(bin_path)
        self.max_seq_len = max_seq_len
        
        # Resolve dtype
        if dtype == "uint16":
            self.np_dtype = np.uint16
        elif dtype == "int32":
            self.np_dtype = np.int32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary file not found at {self.bin_path}")

        # Memory map the file
        self.data = np.memmap(self.bin_path, dtype=self.np_dtype, mode="r")
        self.total_tokens = len(self.data)

        # Number of non-overlapping chunks of size max_seq_len
        # We need max_seq_len tokens per sequence.
        self.num_samples = self.total_tokens // self.max_seq_len

        if self.num_samples == 0:
            raise ValueError(
                f"Binary file {self.bin_path} only has {self.total_tokens} tokens, "
                f"which is less than max_seq_len {self.max_seq_len}."
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Calculate start and end indices
        start_idx = idx * self.max_seq_len
        end_idx = start_idx + self.max_seq_len

        # Fetch the chunk and convert to torch.Tensor
        # Note: np.memmap slices need to be converted to numpy array first before torch.from_numpy,
        # to ensure it owns its memory and is writable in PyTorch context.
        chunk = np.array(self.data[start_idx:end_idx], dtype=np.int64)
        x = torch.from_numpy(chunk)

        # In trainer.py, model(input_ids, labels=labels) shifts input/labels.
        # So we can pass the exact same tensor for both input_ids and labels.
        return {
            "input_ids": x,
            "labels": x,
        }


def get_dataset_metadata(tokenized_dir: Union[str, Path]) -> Dict:
    """Helper to load tokenizer dataset metadata if it exists."""
    meta_path = Path(tokenized_dir) / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
