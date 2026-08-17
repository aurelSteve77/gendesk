"""Determinism helpers.

Backtest results are only meaningful if they can be regenerated bit-for-bit, so
every entry point seeds Python, NumPy and Torch from a single integer.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs used by the project.

    Args:
        seed: Master seed.
        deterministic: If True, ask cuDNN for deterministic kernels. This costs
            throughput and is reserved for reproduction runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def new_generator(seed: int) -> np.random.Generator:
    """Return an independent NumPy generator (preferred over global state)."""
    return np.random.default_rng(seed)
