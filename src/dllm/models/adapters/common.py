from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]


def dtype(name: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[name]
    except KeyError as error:
        raise ValueError(f"unsupported precision: {name}") from error


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def round_generation_length(length: int, block_size: int) -> int:
    return ((length + block_size - 1) // block_size) * block_size

