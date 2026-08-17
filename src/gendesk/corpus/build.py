"""Corpus construction.

Netflix pretrains GenPage on *positive production impressions* -- pages that were
actually served and that members responded well to. There is no production desk
here, so the corpus is manufactured in the same spirit:

1. a stochastic teacher proposes several candidate pages per (date, mandate) cell,
2. each candidate is scored by what it went on to earn over the forward window,
3. candidates below a reward quantile are discarded.

What survives is a set of pages that were *good given the regime they were written
in*. Next-token pretraining on that set teaches the model the language of pages that
worked, without ever showing it a forward return.

Split discipline: an example belongs to the training split only if its entire
forward reward window closes before ``backtest.train_end``. The same purge is
applied at the validation/test boundary. Without it, a page dated one day before the
cutoff would carry a label computed from three weeks of out-of-sample data.
"""

from __future__ import annotations

import gzip
import json
from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from gendesk.config import Config, PersonaConfig
from gendesk.corpus.teacher import TeacherPolicy
from gendesk.features.regimes import REGIME_AXES
from gendesk.features.store import FeatureStore, load_features
from gendesk.portfolio.reward import evaluate_page
from gendesk.portfolio.weights import page_weights
from gendesk.tokenization.page import Page, PageContext, Row
from gendesk.utils.hashing import hash_obj
from gendesk.utils.logging import get_logger
from gendesk.utils.paths import MANIFEST_DIR, PROCESSED_DIR, ensure_dirs

log = get_logger(__name__)

Split = Literal["train", "valid", "test", "purged"]


@dataclass
class CorpusExample:
    """One (context, page, outcome) triple."""

    position: int
    date: str
    persona: str
    rows: list[list]  # [[archetype, [symbols...]], ...]
    history: list[list[str]]
    regimes: dict[str, int]
    reward: float
    active_return: float
    max_drawdown: float
    turnover: float
    effective_bets: float
    split: Split
    #: True when the example survived the positive-outcome filter.
    keep_pretrain: bool = False
    #: True for the teacher's deterministic book at this cell.
    is_book: bool = False

    def to_page(self) -> Page:
        return Page(
            date=pd.Timestamp(self.date),
            persona=self.persona,
            rows=tuple(Row(arch, tuple(syms)) for arch, syms in self.rows),
        )

    def to_context(self, persona: PersonaConfig) -> PageContext:
        return PageContext(
            persona=self.persona,
            risk_budget=persona.risk_budget,
            horizon_days=persona.horizon_days,
            regimes=self.regimes,
            history=tuple(tuple(h) for h in self.history),
        )


@dataclass
class PageCorpus:
    """A collection of examples plus the metadata needed to reproduce it."""

    examples: list[CorpusExample]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[CorpusExample]:
        return iter(self.examples)

    def split(self, name: Split, pretrain_only: bool = False) -> list[CorpusExample]:
        out = [ex for ex in self.examples if ex.split == name]
        if pretrain_only:
            out = [ex for ex in out if ex.keep_pretrain]
        return out

    def save(self, path: Path | None = None) -> Path:
        path = path or PROCESSED_DIR / "corpus.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"__meta__": self.meta}) + "\n")
            for example in self.examples:
                handle.write(json.dumps(asdict(example)) + "\n")
        return path


def load_corpus(path: Path | None = None) -> PageCorpus:
    """Read a corpus written by :meth:`PageCorpus.save`."""
    path = path or PROCESSED_DIR / "corpus.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `gendesk corpus build` first.")

    meta: dict = {}
    examples: list[CorpusExample] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if "__meta__" in payload:
                meta = payload["__meta__"]
                continue
            examples.append(CorpusExample(**payload))
    return PageCorpus(examples=examples, meta=meta)


def _assign_split(date: pd.Timestamp, horizon_end: pd.Timestamp, config: Config) -> Split:
    """Purged split assignment: the whole label window must lie inside the split."""
    train_end = pd.Timestamp(config.backtest.train_end)
    valid_end = pd.Timestamp(config.backtest.valid_end)

    if horizon_end <= train_end:
        return "train"
    if date <= train_end:
        return "purged"  # label leaks across the training boundary
    if horizon_end <= valid_end:
        return "valid"
    if date <= valid_end:
        return "purged"
    return "test"


def build_corpus(
    config: Config, store: FeatureStore | None = None, force: bool = False
) -> PageCorpus:
    """Generate, score and filter the page corpus."""
    ensure_dirs()
    store = store or load_features()

    manifest_path = MANIFEST_DIR / "corpus.json"
    corpus_path = PROCESSED_DIR / "corpus.jsonl.gz"
    fingerprint = hash_obj(
        {
            "corpus": config.corpus.model_dump(mode="json"),
            "personas": [p.model_dump(mode="json") for p in config.personas],
            "backtest": {
                "train_end": str(config.backtest.train_end),
                "valid_end": str(config.backtest.valid_end),
            },
            "symbols": store.symbols,
            "decode_sector_cap": config.decode.max_names_per_sector,
        }
    )

    if manifest_path.exists() and corpus_path.exists() and not force:
        cached = json.loads(manifest_path.read_text())
        if cached.get("fingerprint") == fingerprint:
            log.info("corpus_cache_hit", fingerprint=fingerprint, n=cached.get("n_examples"))
            return load_corpus(corpus_path)

    cfg = config.corpus
    teacher = TeacherPolicy(store.catalog, cfg, config.decode.max_names_per_sector)
    rng = np.random.default_rng(cfg.seed)

    positions = store.eligible_dates()
    # Leave room for the forward reward window at the end of the sample.
    positions = positions[positions < len(store.dates) - cfg.reward_horizon - 1]
    positions = positions[:: cfg.stride_days]
    log.info("corpus_build_start", n_dates=len(positions), n_personas=len(config.personas))

    examples: list[CorpusExample] = []

    for persona in config.personas:
        history: deque[tuple[str, ...]] = deque(maxlen=max(cfg.history_pages, 1))
        previous_weights = None

        for position in positions:
            date = store.dates[int(position)]
            regimes = {axis: int(store.regimes[axis].iloc[int(position)]) for axis in REGIME_AXES}
            horizon_end = store.dates[min(int(position) + cfg.reward_horizon, len(store.dates) - 1)]
            split = _assign_split(date, horizon_end, config)

            book = teacher.greedy_page(store, int(position), persona, regimes, rng)
            candidates: list[tuple[Page, bool]] = [(book, True)]
            for _ in range(max(cfg.candidates_per_cell - 1, 0)):
                candidates.append(
                    (teacher.sample_page(store, int(position), persona, regimes, rng), False)
                )

            snapshot_history = [list(h) for h in history]
            for page, is_book in candidates:
                if not page.rows:
                    continue
                reward = evaluate_page(
                    page,
                    store,
                    int(position),
                    persona,
                    cfg,
                    previous_weights=previous_weights,
                )
                examples.append(
                    CorpusExample(
                        position=int(position),
                        date=str(date.date()),
                        persona=persona.name,
                        rows=[[row.archetype, list(row.symbols)] for row in page.rows],
                        history=snapshot_history,
                        regimes=regimes,
                        reward=reward.total,
                        active_return=reward.active_return,
                        max_drawdown=reward.max_drawdown,
                        turnover=reward.turnover,
                        effective_bets=reward.effective_bets,
                        split=split,
                        is_book=is_book,
                    )
                )

            history.append(book.symbols)
            previous_weights = page_weights(book, store, int(position), persona)

        log.info("corpus_persona_done", persona=persona.name, n_examples=len(examples))

    # --- positive-outcome filter -------------------------------------------
    train_rewards = np.array([ex.reward for ex in examples if ex.split == "train"])
    threshold = (
        float(np.quantile(train_rewards, cfg.positive_quantile)) if train_rewards.size else 0.0
    )
    for example in examples:
        example.keep_pretrain = example.reward >= threshold

    counts = {
        name: sum(ex.split == name for ex in examples)
        for name in ("train", "valid", "test", "purged")
    }
    kept = sum(ex.keep_pretrain and ex.split == "train" for ex in examples)

    corpus = PageCorpus(
        examples=examples,
        meta={
            "fingerprint": fingerprint,
            "built_at": pd.Timestamp.utcnow().isoformat(),
            "n_examples": len(examples),
            "threshold": threshold,
            "counts": counts,
            "kept_train": kept,
            "config": config.corpus.model_dump(mode="json"),
            "personas": [p.name for p in config.personas],
        },
    )
    corpus.save(corpus_path)
    manifest_path.write_text(json.dumps(corpus.meta, indent=2, default=str))

    log.info(
        "corpus_built",
        n_examples=len(examples),
        threshold=round(threshold, 4),
        kept_train=kept,
        **counts,
    )
    return corpus
