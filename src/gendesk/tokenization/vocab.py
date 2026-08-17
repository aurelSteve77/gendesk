"""The GenDesk vocabulary.

Layout (contiguous, order-critical -- a checkpoint is only valid for the vocabulary
it was trained with, which is why :meth:`Vocab.fingerprint` is stamped into every
checkpoint):

``[specials] [personas] [risk budgets] [horizons] [regime buckets] [row archetypes] [entities]``

Entities occupy the final, contiguous block. That is deliberate: generation only
ever has to sample inside ``[entity_offset, vocab_size)``, so the constrained
decoder can work on a compact ``n_instruments`` mask instead of the full vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gendesk.data.universe import Catalog
from gendesk.features.regimes import BUCKET_LABELS, REGIME_AXES
from gendesk.utils.hashing import hash_obj

#: Structural markers.
SPECIALS: tuple[str, ...] = (
    "<pad>",
    "<bos>",
    "<ctx>",
    "<hist>",
    "<page>",
    "<eor>",
    "<eop>",
    "<unk>",
)

#: Row archetypes: the "row types" of GenPage. Each is a thesis about *why* a group
#: of instruments belongs together, and each has a teacher scoring rule in
#: :mod:`gendesk.corpus.rows`.
ROW_ARCHETYPES: tuple[str, ...] = (
    "MOMENTUM_LEADERS",
    "TREND_BREAKOUT",
    "QUALITY_BALLAST",
    "MEAN_REVERSION",
    "DISPERSION_HARVEST",
    "HIGH_BETA_RISK_ON",
    "CROWDING_UNWIND",
    "MACRO_HEDGE",
)

RISK_BUDGETS: tuple[str, ...] = ("low", "medium", "high")

#: Horizon buckets in trading days. A mandate's horizon is snapped to the nearest.
HORIZON_BUCKETS: tuple[int, ...] = (10, 21, 42, 63)


@dataclass(frozen=True)
class Vocab:
    """Bidirectional token <-> id mapping with typed accessors."""

    tokens: tuple[str, ...]
    entity_offset: int
    n_instruments: int
    symbols: tuple[str, ...]
    persona_names: tuple[str, ...]
    _ids: dict[str, int] = field(repr=False, default_factory=dict)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def size(self) -> int:
        return len(self.tokens)

    # -- generic ------------------------------------------------------------

    def id(self, token: str) -> int:
        try:
            return self._ids[token]
        except KeyError:
            return self._ids["<unk>"]

    def token(self, token_id: int) -> str:
        return self.tokens[token_id]

    # -- specials -----------------------------------------------------------

    @property
    def pad(self) -> int:
        return self._ids["<pad>"]

    @property
    def bos(self) -> int:
        return self._ids["<bos>"]

    @property
    def ctx(self) -> int:
        return self._ids["<ctx>"]

    @property
    def hist(self) -> int:
        return self._ids["<hist>"]

    @property
    def page(self) -> int:
        return self._ids["<page>"]

    @property
    def eor(self) -> int:
        return self._ids["<eor>"]

    @property
    def eop(self) -> int:
        return self._ids["<eop>"]

    # -- typed lookups ------------------------------------------------------

    def persona(self, name: str) -> int:
        return self.id(f"<persona:{name}>")

    def risk(self, budget: str) -> int:
        return self.id(f"<risk:{budget}>")

    def horizon(self, days: int) -> int:
        bucket = min(HORIZON_BUCKETS, key=lambda b: abs(b - days))
        return self.id(f"<horizon:{bucket}d>")

    def regime(self, axis: str, bucket: int) -> int:
        label = BUCKET_LABELS[axis][int(min(max(bucket, 0), 2))]
        return self.id(f"<regime:{axis}={label}>")

    def row(self, archetype: str) -> int:
        return self.id(f"<row:{archetype}>")

    def entity(self, symbol: str) -> int:
        return self.id(symbol)

    def entity_index(self, token_id: int) -> int:
        """Catalog position of an entity token id."""
        idx = token_id - self.entity_offset
        if not 0 <= idx < self.n_instruments:
            raise ValueError(f"token id {token_id} is not an entity")
        return idx

    def index_to_token(self, index: int) -> int:
        """Token id of the instrument at catalog position ``index``."""
        return self.entity_offset + index

    def is_entity(self, token_id: int) -> bool:
        return self.entity_offset <= token_id < self.entity_offset + self.n_instruments

    def is_row(self, token_id: int) -> bool:
        return self.tokens[token_id].startswith("<row:")

    def row_archetype(self, token_id: int) -> str:
        return self.tokens[token_id][len("<row:") : -1]

    @property
    def row_token_ids(self) -> tuple[int, ...]:
        return tuple(self.id(f"<row:{arch}>") for arch in ROW_ARCHETYPES)

    def fingerprint(self) -> str:
        """Stable hash of the vocabulary, stamped into checkpoints."""
        return hash_obj({"tokens": self.tokens})


def build_vocab(catalog: Catalog, persona_names: tuple[str, ...]) -> Vocab:
    """Assemble the vocabulary from the catalog and the configured mandates."""
    tokens: list[str] = list(SPECIALS)
    tokens += [f"<persona:{name}>" for name in persona_names]
    tokens += [f"<risk:{budget}>" for budget in RISK_BUDGETS]
    tokens += [f"<horizon:{days}d>" for days in HORIZON_BUCKETS]
    tokens += [f"<regime:{axis}={label}>" for axis in REGIME_AXES for label in BUCKET_LABELS[axis]]
    tokens += [f"<row:{arch}>" for arch in ROW_ARCHETYPES]

    entity_offset = len(tokens)
    tokens += list(catalog.symbols)

    if len(set(tokens)) != len(tokens):
        raise ValueError("vocabulary contains duplicate tokens")

    return Vocab(
        tokens=tuple(tokens),
        entity_offset=entity_offset,
        n_instruments=len(catalog),
        symbols=catalog.symbols,
        persona_names=persona_names,
        _ids={tok: i for i, tok in enumerate(tokens)},
    )
