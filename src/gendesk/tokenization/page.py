"""Page objects and their sequence encoding.

A *page* is the unit the model generates: an ordered list of rows, each row an
ordered list of instruments. The encoder lays a page out in reading order -- context
first, then the desk's recent holdings, then the page itself -- so that generation
is a plain left-to-right continuation of a prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gendesk.features.regimes import REGIME_AXES
from gendesk.tokenization.vocab import Vocab


@dataclass(frozen=True)
class Row:
    """One themed row of a page."""

    archetype: str
    symbols: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.symbols)


@dataclass(frozen=True)
class Page:
    """A generated desk page for one mandate on one date."""

    date: pd.Timestamp
    persona: str
    rows: tuple[Row, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sym for row in self.rows for sym in row.symbols)

    @property
    def archetypes(self) -> tuple[str, ...]:
        return tuple(row.archetype for row in self.rows)

    def __len__(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class PageContext:
    """Everything the model conditions on before it writes a page."""

    persona: str
    risk_budget: str
    horizon_days: int
    #: Regime axis -> tercile bucket, as produced by :func:`build_regimes`.
    regimes: dict[str, int]
    #: Previous pages' holdings, oldest first. The desk's own recent behaviour is
    #: the direct analogue of a member's viewing history.
    history: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class ContextSpec:
    """Which context blocks to include -- the axis of the enrichment ablation."""

    persona: bool = True
    risk: bool = True
    horizon: bool = True
    regimes: bool = True
    history: bool = True
    row_tokens: bool = True
    history_pages: int = 3

    @property
    def name(self) -> str:
        parts = [
            key
            for key, on in (
                ("persona", self.persona),
                ("risk", self.risk),
                ("horizon", self.horizon),
                ("regime", self.regimes),
                ("history", self.history),
                ("rows", self.row_tokens),
            )
            if on
        ]
        return "+".join(parts) if parts else "bare"


@dataclass(frozen=True)
class EncodedPage:
    """Token sequence plus the index structure the training stages need."""

    tokens: np.ndarray
    #: True on positions belonging to the generated page block.
    page_mask: np.ndarray
    #: Positions holding an instrument token inside the page block.
    entity_positions: np.ndarray
    #: Row index (0-based) of each entity slot.
    slot_row: np.ndarray
    #: Catalog index of each entity slot.
    slot_entity: np.ndarray
    #: Positions holding a ``<row:...>`` token.
    row_positions: np.ndarray
    #: Index of the first page token; everything before it is prompt.
    prompt_len: int

    def __len__(self) -> int:
        return len(self.tokens)


class PageSequence:
    """Encoder / decoder between :class:`Page` objects and token sequences."""

    def __init__(self, vocab: Vocab, spec: ContextSpec | None = None) -> None:
        self.vocab = vocab
        self.spec = spec or ContextSpec()

    # -- prompt --------------------------------------------------------------

    def encode_prompt(self, context: PageContext) -> list[int]:
        """Tokens preceding the page block."""
        v = self.vocab
        tokens: list[int] = [v.bos, v.ctx]

        if self.spec.persona:
            tokens.append(v.persona(context.persona))
        if self.spec.risk:
            tokens.append(v.risk(context.risk_budget))
        if self.spec.horizon:
            tokens.append(v.horizon(context.horizon_days))
        if self.spec.regimes:
            tokens.extend(v.regime(axis, context.regimes[axis]) for axis in REGIME_AXES)

        if self.spec.history and self.spec.history_pages > 0:
            tokens.append(v.hist)
            recent = context.history[-self.spec.history_pages :]
            for holdings in recent:
                tokens.extend(v.entity(sym) for sym in holdings)
                tokens.append(v.eor)

        tokens.append(v.page)
        return tokens

    # -- full page -----------------------------------------------------------

    def encode(self, page: Page, context: PageContext) -> EncodedPage:
        """Encode a complete page, returning tokens and their index structure."""
        v = self.vocab
        prompt = self.encode_prompt(context)
        tokens = list(prompt)

        entity_positions: list[int] = []
        slot_row: list[int] = []
        slot_entity: list[int] = []
        row_positions: list[int] = []

        for row_idx, row in enumerate(page.rows):
            if self.spec.row_tokens:
                row_positions.append(len(tokens))
                tokens.append(v.row(row.archetype))
            for symbol in row.symbols:
                entity_positions.append(len(tokens))
                token_id = v.entity(symbol)
                slot_entity.append(v.entity_index(token_id))
                slot_row.append(row_idx)
                tokens.append(token_id)
            tokens.append(v.eor)
        tokens.append(v.eop)

        page_mask = np.zeros(len(tokens), dtype=bool)
        page_mask[len(prompt) :] = True

        return EncodedPage(
            tokens=np.asarray(tokens, dtype=np.int64),
            page_mask=page_mask,
            entity_positions=np.asarray(entity_positions, dtype=np.int64),
            slot_row=np.asarray(slot_row, dtype=np.int64),
            slot_entity=np.asarray(slot_entity, dtype=np.int64),
            row_positions=np.asarray(row_positions, dtype=np.int64),
            prompt_len=len(prompt),
        )

    def decode(self, tokens: Sequence[int], date: pd.Timestamp, persona: str) -> Page:
        """Reconstruct a :class:`Page` from a generated token sequence.

        Tolerant by design: generation under constraints can still emit a truncated
        tail, and a partially written page is better surfaced than dropped.
        """
        v = self.vocab
        rows: list[Row] = []
        current_archetype: str | None = None
        current: list[str] = []

        started = False
        for token_id in tokens:
            token_id = int(token_id)
            if token_id == v.page:
                started = True
                continue
            if not started:
                continue
            if token_id == v.eop:
                break
            if token_id == v.eor:
                if current:
                    rows.append(Row(current_archetype or "UNSPECIFIED", tuple(current)))
                current, current_archetype = [], None
                continue
            if v.is_row(token_id):
                if current:
                    rows.append(Row(current_archetype or "UNSPECIFIED", tuple(current)))
                    current = []
                current_archetype = v.row_archetype(token_id)
                continue
            if v.is_entity(token_id):
                current.append(v.tokens[token_id])

        if current:
            rows.append(Row(current_archetype or "UNSPECIFIED", tuple(current)))
        return Page(date=date, persona=persona, rows=tuple(rows))


@dataclass
class PaddedBatch:
    """A right-padded batch of encoded pages."""

    tokens: np.ndarray
    attention_mask: np.ndarray
    page_mask: np.ndarray
    lengths: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)


def pad_batch(encoded: Sequence[EncodedPage], pad_id: int, max_len: int) -> PaddedBatch:
    """Right-pad a list of encoded pages into rectangular arrays."""
    if not encoded:
        raise ValueError("cannot pad an empty batch")
    length = min(max(len(e) for e in encoded), max_len)

    tokens = np.full((len(encoded), length), pad_id, dtype=np.int64)
    attention = np.zeros((len(encoded), length), dtype=bool)
    page_mask = np.zeros((len(encoded), length), dtype=bool)
    lengths = np.zeros(len(encoded), dtype=np.int64)

    for i, item in enumerate(encoded):
        take = min(len(item), length)
        tokens[i, :take] = item.tokens[:take]
        attention[i, :take] = True
        page_mask[i, :take] = item.page_mask[:take]
        lengths[i] = take

    return PaddedBatch(
        tokens=tokens, attention_mask=attention, page_mask=page_mask, lengths=lengths
    )
