"""GenDesk as a tradable strategy.

The wrapper is what makes the backtest an honest test of the *system* rather than of
a single page: the model's context at each rebalance contains the pages it itself
generated at the previous rebalances, exactly as it would in production. There is no
oracle history and no teacher assistance at inference.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from gendesk.config import Config, PersonaConfig
from gendesk.decoding.generate import PageGenerator
from gendesk.evaluation.diversity import page_diversity
from gendesk.features.regimes import REGIME_AXES
from gendesk.features.store import FeatureStore
from gendesk.model.gendesk import GenDeskModel
from gendesk.portfolio.weights import page_weights
from gendesk.tokenization.page import ContextSpec, Page, PageContext, PageSequence
from gendesk.tokenization.vocab import Vocab


@dataclass
class GenDeskStrategy:
    """Weight function backed by the generative model."""

    model: GenDeskModel
    vocab: Vocab
    store: FeatureStore
    config: Config
    persona: PersonaConfig
    head: str = "lm"
    temperature: float = 0.0
    hybrid: bool | None = None
    spec: ContextSpec | None = None
    #: Pages the strategy produced, keyed by date; used by the report and the UI.
    pages: dict[pd.Timestamp, Page] = field(default_factory=dict)
    diagnostics: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model.eval()
        self._generator = PageGenerator(self.model, self.vocab, self.store, self.config)
        self._spec = self.spec or ContextSpec(history_pages=self.config.corpus.history_pages)
        self._sequence = PageSequence(self.vocab, self._spec)
        self._history: deque[tuple[str, ...]] = deque(
            maxlen=max(self.config.corpus.history_pages, 1)
        )

    def reset(self) -> None:
        self._history.clear()
        self.pages.clear()
        self.diagnostics.clear()

    def context(self, position: int) -> PageContext:
        regimes = {axis: int(self.store.regimes[axis].iloc[position]) for axis in REGIME_AXES}
        return PageContext(
            persona=self.persona.name,
            risk_budget=self.persona.risk_budget,
            horizon_days=self.persona.horizon_days,
            regimes=regimes,
            history=tuple(self._history),
        )

    @torch.no_grad()
    def generate(self, position: int) -> Page:
        result = self._generator.generate(
            self.context(position),
            self.persona,
            position,
            n_samples=1,
            temperature=self.temperature,
            hybrid=self.hybrid,
            head=self.head,
            spec_sequence=self._sequence,
        )
        page = result.pages[0]
        self.diagnostics.append(
            {
                "date": self.store.dates[position],
                "model_calls": result.model_calls,
                "latency_ms": result.latency_ms,
                **result.report.as_dict(),
            }
        )
        return page

    def __call__(self, position: int) -> pd.Series:
        page = self.generate(position)
        date = pd.Timestamp(self.store.dates[position])
        self.pages[date] = page
        self._history.append(page.symbols)
        return page_weights(page, self.store, position, self.persona)

    # -- reporting -----------------------------------------------------------

    def diversity_frame(self) -> pd.DataFrame:
        """Diversity of every generated page, computed with trailing data only."""
        rows = []
        for date, page in self.pages.items():
            position = self.store.date_position(date)
            metrics = page_diversity(page, self.store, position, self.persona)
            rows.append({"date": date, **metrics.as_dict()})
        return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()

    def archetype_mix(self) -> pd.Series:
        """How often each row archetype was chosen."""
        counts: dict[str, int] = {}
        for page in self.pages.values():
            for row in page.rows:
                counts[row.archetype] = counts.get(row.archetype, 0) + 1
        total = sum(counts.values()) or 1
        return pd.Series({k: v / total for k, v in sorted(counts.items())})

    def latency_summary(self) -> dict:
        if not self.diagnostics:
            return {}
        frame = pd.DataFrame(self.diagnostics)
        return {
            "median_latency_ms": float(frame["latency_ms"].median()),
            "p95_latency_ms": float(np.percentile(frame["latency_ms"], 95)),
            "mean_model_calls": float(frame["model_calls"].mean()),
        }
