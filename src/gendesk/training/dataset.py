"""Datasets and batching.

The whole standardised feature tensor lives on the accelerator (about 125 MB at the
default catalog size), so a batch only carries *date positions* and the model gathers
the market state it needs. That keeps collation cheap and, more importantly, keeps
one copy of the truth: every stage reads the same tensor by the same index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from gendesk.config import Config, PersonaConfig
from gendesk.corpus.build import CorpusExample
from gendesk.features.store import FeatureStore
from gendesk.portfolio.reward import slot_rewards
from gendesk.tokenization.page import ContextSpec, PageSequence
from gendesk.tokenization.vocab import Vocab


@dataclass
class PageBatch:
    """A padded batch ready for the model."""

    tokens: Tensor  # (B, L)
    loss_weights: Tensor  # (B, L)
    positions: Tensor  # (B,) date index into the feature tensor
    #: (B, L) True where the token is an instrument inside the page block.
    slot_mask: Tensor
    rewards: Tensor  # (B,) page-level reward
    persona_ids: Tensor  # (B,)

    def to(self, device: torch.device) -> PageBatch:
        return PageBatch(
            tokens=self.tokens.to(device, non_blocking=True),
            loss_weights=self.loss_weights.to(device, non_blocking=True),
            positions=self.positions.to(device, non_blocking=True),
            slot_mask=self.slot_mask.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            persona_ids=self.persona_ids.to(device, non_blocking=True),
        )

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


class PageDataset(Dataset):
    """Encodes corpus examples into token sequences on the fly.

    Encoding lazily (rather than materialising a token matrix) is what makes the
    context-enrichment ablation cheap: swapping :class:`ContextSpec` re-encodes the
    same corpus with a different prompt, with no rebuild.
    """

    def __init__(
        self,
        examples: list[CorpusExample],
        vocab: Vocab,
        config: Config,
        spec: ContextSpec | None = None,
    ) -> None:
        self.examples = examples
        self.vocab = vocab
        self.config = config
        self.spec = spec or ContextSpec(history_pages=config.corpus.history_pages)
        self.sequence = PageSequence(vocab, self.spec)
        self.personas: dict[str, PersonaConfig] = {p.name: p for p in config.personas}
        self.persona_ids = {name: i for i, name in enumerate(self.personas)}
        self.max_len = config.model.max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        persona = self.personas[example.persona]
        encoded = self.sequence.encode(example.to_page(), example.to_context(persona))

        length = min(len(encoded), self.max_len)
        tokens = encoded.tokens[:length]

        weights = np.full(
            length, self.config.training.pretrain.context_loss_weight, dtype=np.float32
        )
        weights[encoded.page_mask[:length]] = 1.0

        slot_mask = np.zeros(length, dtype=bool)
        in_range = encoded.entity_positions[encoded.entity_positions < length]
        slot_mask[in_range] = True

        return {
            "tokens": tokens,
            "weights": weights,
            "slot_mask": slot_mask,
            "position": example.position,
            "reward": example.reward,
            "persona_id": self.persona_ids[example.persona],
        }


def collate_pages(items: list[dict], pad_id: int) -> PageBatch:
    """Right-pad a list of dataset items."""
    length = max(len(item["tokens"]) for item in items)
    batch = len(items)

    tokens = np.full((batch, length), pad_id, dtype=np.int64)
    weights = np.zeros((batch, length), dtype=np.float32)
    slots = np.zeros((batch, length), dtype=bool)

    for i, item in enumerate(items):
        take = len(item["tokens"])
        tokens[i, :take] = item["tokens"]
        weights[i, :take] = item["weights"]
        slots[i, :take] = item["slot_mask"]

    return PageBatch(
        tokens=torch.from_numpy(tokens),
        loss_weights=torch.from_numpy(weights),
        positions=torch.tensor([item["position"] for item in items], dtype=torch.long),
        slot_mask=torch.from_numpy(slots),
        rewards=torch.tensor([item["reward"] for item in items], dtype=torch.float32),
        persona_ids=torch.tensor([item["persona_id"] for item in items], dtype=torch.long),
    )


class FeatureBank:
    """The standardised feature tensor, resident on the training device."""

    def __init__(self, store: FeatureStore, device: torch.device) -> None:
        self.device = device
        self.values = torch.as_tensor(store.values, dtype=torch.float32, device=device)
        self.available = torch.as_tensor(store.available.to_numpy().astype(bool), device=device)

    def gather(self, positions: Tensor) -> Tensor:
        """``(B, n_instruments, n_features)`` market state for each example's date."""
        return self.values.index_select(0, positions)

    def eligibility(self, positions: Tensor) -> Tensor:
        return self.available.index_select(0, positions)


class RewardBank:
    """Per-instrument forward rewards, precomputed for every corpus date.

    These are labels. They are materialised once here so the WBC stage never touches
    the return frame directly, which makes the "no forward data reaches the model as
    an input" invariant checkable in one place.
    """

    def __init__(
        self,
        store: FeatureStore,
        positions: list[int] | np.ndarray,
        horizon: int,
        clip: float,
        device: torch.device,
    ) -> None:
        unique = sorted({int(p) for p in positions})
        self.index = {pos: i for i, pos in enumerate(unique)}
        table = np.stack([slot_rewards(store, pos, horizon, clip) for pos in unique])
        self.table = torch.as_tensor(table, dtype=torch.float32, device=device)
        self.device = device

    def gather(self, positions: Tensor) -> Tensor:
        rows = torch.tensor(
            [self.index[int(p)] for p in positions.tolist()],
            dtype=torch.long,
            device=self.table.device,
        )
        return self.table.index_select(0, rows)
