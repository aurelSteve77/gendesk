"""Optimiser and learning-rate schedule helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


def build_optimizer(
    model: nn.Module, lr: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.95)
) -> torch.optim.Optimizer:
    """AdamW with decay applied only to matrix parameters.

    Norm gains, biases and embeddings are excluded, which is the standard recipe:
    decaying them mostly just shrinks the model's ability to represent rare tokens.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith("bias") or "embed" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up into a cosine decay floored at ``min_lr_ratio``."""
    warmup = max(1, int(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def clip_gradients(params: Iterable[nn.Parameter], max_norm: float) -> float:
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


def resolve_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
