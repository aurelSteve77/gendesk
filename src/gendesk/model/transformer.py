"""Decoder-only transformer backbone.

A conventional modern stack -- pre-norm, RMSNorm, rotary positions, grouped-query
attention, SwiGLU -- deliberately kept small. The interesting part of this project
is what the tokens *mean*, not the block design, and a ~6M parameter backbone is
what the ablation in :mod:`gendesk.evaluation.ablations` needs in order to scale
capacity up and down cheaply.

The KV cache exists because page generation is autoregressive over ~40 tokens and
the hybrid decoder ( :mod:`gendesk.decoding` ) reuses the prompt prefill across
every row.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gendesk.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device) -> Tensor:
    """Return a ``(seq_len, head_dim/2, 2)`` cache of cos/sin rotations."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inv_freq)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def apply_rope(x: Tensor, cache: Tensor) -> Tensor:
    """Apply rotary embeddings to ``(batch, heads, seq, head_dim)``."""
    batch, heads, seq, head_dim = x.shape
    x = x.float().reshape(batch, heads, seq, head_dim // 2, 2)
    cos = cache[:seq, :, 0].view(1, 1, seq, head_dim // 2)
    sin = cache[:seq, :, 1].view(1, 1, seq, head_dim // 2)
    x0, x1 = x[..., 0], x[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.reshape(batch, heads, seq, head_dim)


@dataclass
class KVCache:
    """Per-layer key/value cache for incremental decoding."""

    keys: list[Tensor]
    values: list[Tensor]
    length: int = 0

    @classmethod
    def empty(cls, n_layers: int) -> KVCache:
        """Layers are appended lazily on the first prefill; ``n_layers`` documents intent."""
        del n_layers
        return cls(keys=[], values=[], length=0)

    def clone(self) -> KVCache:
        return KVCache(
            keys=[k.clone() for k in self.keys],
            values=[v.clone() for v in self.values],
            length=self.length,
        )

    def expand(self, repeats: int) -> KVCache:
        """Repeat the cache along the batch dimension (one prompt, many samples)."""
        return KVCache(
            keys=[k.repeat_interleave(repeats, dim=0) for k in self.keys],
            values=[v.repeat_interleave(repeats, dim=0) for v in self.values],
            length=self.length,
        )


class Attention(nn.Module):
    """Causal grouped-query attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.repeats = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
        offset: int = 0,
    ) -> Tensor:
        batch, seq, _ = x.shape

        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        window = rope[offset : offset + seq]
        q = apply_rope(q, window).to(x.dtype)
        k = apply_rope(k, window).to(x.dtype)

        if cache is not None:
            if layer_idx < len(cache.keys):
                k = torch.cat([cache.keys[layer_idx], k], dim=2)
                v = torch.cat([cache.values[layer_idx], v], dim=2)
                cache.keys[layer_idx], cache.values[layer_idx] = k, v
            else:
                cache.keys.append(k)
                cache.values.append(v)

        if self.repeats > 1:
            k = k.repeat_interleave(self.repeats, dim=1)
            v = v.repeat_interleave(self.repeats, dim=1)

        # A single query step attends to the whole cache, so causality is implicit;
        # a multi-token step needs the triangular mask.
        causal = seq > 1
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
        offset: int = 0,
    ) -> Tensor:
        x = x + self.drop(self.attn(self.attn_norm(x), rope, cache, layer_idx, offset))
        return x + self.drop(self.ffn(self.ffn_norm(x)))


class TransformerBackbone(nn.Module):
    """Stack of decoder blocks operating on pre-embedded inputs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.register_buffer(
            "rope",
            build_rope_cache(
                config.max_seq_len,
                config.d_model // config.n_heads,
                config.rope_theta,
                torch.device("cpu"),
            ),
            persistent=False,
        )

    def forward(self, x: Tensor, cache: KVCache | None = None, offset: int = 0) -> Tensor:
        rope = self.rope
        for i, block in enumerate(self.blocks):
            x = block(x, rope, cache, i, offset)
        if cache is not None:
            cache.length = offset + x.shape[1]
        return self.norm(x)
