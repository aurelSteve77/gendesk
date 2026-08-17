"""The teacher policy.

The teacher writes the pretraining corpus. It is a stochastic, regime-aware factor
screen: it picks a set of row archetypes appropriate to the macro state, then samples
instruments inside each row from a softmax over that archetype's score.

The teacher is intentionally *mediocre and diverse* rather than optimal. Pretraining
on a single greedy policy would teach the model to imitate a fixed screen; sampling
at temperature produces a spread of pages whose realised outcomes differ, which is
exactly what the outcome filter and the later RL stage need in order to have
something to select on.

Crucially, the teacher never sees forward returns. Selection pressure enters only
later, when :mod:`gendesk.corpus.build` discards the candidates that did badly.
"""

from __future__ import annotations

import numpy as np

from gendesk.config import CorpusConfig, PersonaConfig
from gendesk.corpus.rows import ARCHETYPES, archetype_order, eligibility_matrix
from gendesk.data.universe import FUND_SECTOR, Catalog
from gendesk.features.store import FeatureStore
from gendesk.tokenization.page import Page, Row


class TeacherPolicy:
    """Regime-aware stochastic factor screen over row archetypes."""

    def __init__(self, catalog: Catalog, config: CorpusConfig, max_names_per_sector: int = 4):
        self.catalog = catalog
        self.config = config
        self.max_names_per_sector = max_names_per_sector
        self.archetypes = archetype_order()
        self._eligible = eligibility_matrix(catalog)
        self._sector_of = np.array([inst.sector for inst in catalog])
        self._coefs = np.stack([ARCHETYPES[name].coefficient_vector() for name in self.archetypes])

    # -- row selection -------------------------------------------------------

    def choose_rows(
        self, persona: PersonaConfig, regimes: dict[str, int], rng: np.random.Generator
    ) -> list[str]:
        """Pick the page's row archetypes: pinned rows first, then regime-weighted."""
        n_rows = self.config.n_rows
        chosen = [row for row in persona.pinned_rows if row in ARCHETYPES][:n_rows]

        pool = [row for row in persona.allowed_rows if row in ARCHETYPES and row not in chosen]
        while len(chosen) < n_rows and pool:
            affinities = np.array([ARCHETYPES[row].affinity(regimes) for row in pool])
            probs = affinities / affinities.sum()
            pick = int(rng.choice(len(pool), p=probs))
            chosen.append(pool.pop(pick))

        # A mandate with fewer allowed rows than n_rows repeats its favourite.
        while len(chosen) < n_rows and chosen:
            chosen.append(chosen[len(chosen) % max(len(chosen), 1)])
        return chosen[:n_rows]

    # -- instrument selection ------------------------------------------------

    def _base_mask(self, store: FeatureStore, position: int, persona: PersonaConfig) -> np.ndarray:
        available = store.available.iloc[position].to_numpy().astype(bool)
        if persona.excluded_assets:
            excluded = np.isin(np.asarray(store.symbols), np.asarray(persona.excluded_assets))
            available = available & ~excluded
        return available

    def sample_page(
        self,
        store: FeatureStore,
        position: int,
        persona: PersonaConfig,
        regimes: dict[str, int],
        rng: np.random.Generator,
        temperature: float | None = None,
        greedy: bool = False,
    ) -> Page:
        """Sample one page for ``persona`` at ``position``."""
        temperature = self.config.teacher_temperature if temperature is None else temperature
        features = store.values[position]
        scores = self._coefs @ features.T  # (n_archetypes, n_instruments)

        base = self._base_mask(store, position, persona)
        used = np.zeros(len(self.catalog), dtype=bool)
        sector_counts: dict[str, int] = {}

        rows: list[Row] = []
        for archetype in self.choose_rows(persona, regimes, rng):
            a = self.archetypes.index(archetype)
            picks = self._select(
                scores[a],
                base & self._eligible[a],
                used,
                sector_counts,
                rng,
                temperature,
                greedy,
            )
            if not picks:
                continue
            rows.append(Row(archetype, tuple(self.catalog[int(i)].symbol for i in picks)))

        return Page(date=store.dates[position], persona=persona.name, rows=tuple(rows))

    def _capped_sectors(self, sector_counts: dict[str, int]) -> list[str]:
        if self.max_names_per_sector <= 0:
            return []
        return [
            sector
            for sector, count in sector_counts.items()
            if sector != FUND_SECTOR and count >= self.max_names_per_sector
        ]

    def _select(
        self,
        scores: np.ndarray,
        row_mask: np.ndarray,
        used: np.ndarray,
        sector_counts: dict[str, int],
        rng: np.random.Generator,
        temperature: float,
        greedy: bool,
    ) -> list[int]:
        """Choose ``row_size`` instruments, one at a time.

        Picking the whole row in one draw would let a single row breach the sector cap
        internally -- the cap would only ever be checked *between* rows. Selecting
        sequentially and updating the counters after each pick applies exactly the
        same rule the constrained decoder applies at generation time, which is what
        keeps the corpus and the model's own output governed by one set of rules.
        """
        picks: list[int] = []
        for _ in range(self.config.row_size):
            mask = row_mask & ~used
            capped = self._capped_sectors(sector_counts)
            if capped:
                mask = mask & ~np.isin(self._sector_of, capped)

            candidates = np.flatnonzero(mask)
            if candidates.size == 0:
                break

            values = scores[candidates].astype(np.float64)
            if greedy or temperature <= 0:
                choice = int(candidates[int(np.argmax(values))])
            else:
                logits = values / max(temperature, 1e-6)
                logits -= logits.max()
                probs = np.exp(logits)
                total = probs.sum()
                if not np.isfinite(total) or total <= 0:
                    choice = int(candidates[int(np.argmax(values))])
                else:
                    choice = int(rng.choice(candidates, p=probs / total))

            picks.append(choice)
            used[choice] = True
            sector = self._sector_of[choice]
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Present the row best-first: the row's own ranking is information the model
        # should learn to reproduce in its generation order.
        return [int(i) for i in np.array(picks)[np.argsort(-scores[picks])]] if picks else []

    def greedy_page(
        self,
        store: FeatureStore,
        position: int,
        persona: PersonaConfig,
        regimes: dict[str, int],
        rng: np.random.Generator,
    ) -> Page:
        """Deterministic page: the desk's actual book under the teacher.

        This is what defines the interaction history and the turnover baseline. It
        is chosen ex ante -- selecting the book by realised reward would leak the
        forward window of one page into the context of the next.
        """
        return self.sample_page(store, position, persona, regimes, rng, greedy=True)
