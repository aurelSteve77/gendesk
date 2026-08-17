"""Baseline strategies.

The list is chosen so that each one isolates a different explanation for whatever
the model does:

* ``benchmark`` -- did it beat simply owning the market?
* ``equal_weight`` -- did it beat owning everything in the catalog?
* ``momentum_12_1`` / ``low_volatility`` -- did it beat the individual risk premia
  its own row archetypes are built from?
* ``risk_parity`` -- did it beat a pure sizing rule with no selection at all?
* ``pipeline`` -- **the important one.** A faithful stand-in for the classical
  multi-stage recommender GenPage replaces: candidate generation, then a ranking
  model, then a separate diversification pass. It uses the same features, the same
  sizing and the same constraints as GenDesk. The gap between them is therefore
  attributable to end-to-end page generation rather than to better inputs.
* ``teacher_book`` -- the deterministic screen the corpus was written by. The gap
  between this and GenDesk is what the generative layer added on top of its teacher.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from gendesk.config import Config, PersonaConfig
from gendesk.corpus.rows import ARCHETYPES, archetype_order
from gendesk.corpus.teacher import TeacherPolicy
from gendesk.data.universe import FUND_SECTOR
from gendesk.features.regimes import REGIME_AXES
from gendesk.features.store import FeatureStore
from gendesk.portfolio.weights import MIN_VOL, page_weights

WeightFn = Callable[[int], pd.Series]


def _inverse_vol(store: FeatureStore, position: int, symbols: list[str]) -> pd.Series:
    if not symbols:
        return pd.Series(dtype=float)
    vol = store.vol.iloc[position][symbols].to_numpy(dtype=np.float64)
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
    fallback = float(np.nanmedian(vol)) if np.isfinite(vol).any() else 0.2
    vol = np.maximum(np.nan_to_num(vol, nan=fallback), MIN_VOL)
    w = 1.0 / vol
    return pd.Series(w / w.sum(), index=symbols)


def _available(store: FeatureStore, position: int, equities_only: bool = True) -> list[str]:
    mask = store.available.iloc[position]
    symbols = [s for s in store.symbols if bool(mask.get(s, False))]
    if equities_only:
        by_symbol = store.catalog.by_symbol
        symbols = [s for s in symbols if not by_symbol[s].is_fund]
    return symbols


def benchmark_weights(store: FeatureStore) -> WeightFn:
    """Buy and hold the benchmark."""
    symbol = store.catalog.benchmark

    def fn(position: int) -> pd.Series:
        del position  # buy and hold is date-independent by definition
        return pd.Series({symbol: 1.0})

    return fn


def equal_weight(store: FeatureStore) -> WeightFn:
    """Equal weight across every available single name."""

    def fn(position: int) -> pd.Series:
        symbols = _available(store, position)
        if not symbols:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(symbols), index=symbols)

    return fn


def factor_screen(
    store: FeatureStore, feature: str, n_names: int = 30, sign: float = 1.0
) -> WeightFn:
    """Top-``n`` names by a single standardised feature, inverse-volatility weighted."""
    idx = store.feature_names.index(feature)

    def fn(position: int) -> pd.Series:
        symbols = _available(store, position)
        if not symbols:
            return pd.Series(dtype=float)
        order = store.symbol_index
        scores = np.array([sign * store.values[position, order[s], idx] for s in symbols])
        picks = [symbols[i] for i in np.argsort(-scores)[:n_names]]
        return _inverse_vol(store, position, picks)

    return fn


def risk_parity(store: FeatureStore, n_names: int = 30) -> WeightFn:
    """Sizing with no selection: the ``n`` lowest-volatility names, inverse-vol weighted."""

    def fn(position: int) -> pd.Series:
        symbols = _available(store, position)
        if not symbols:
            return pd.Series(dtype=float)
        vol = store.vol.iloc[position][symbols]
        picks = list(vol.nsmallest(n_names).index)
        return _inverse_vol(store, position, picks)

    return fn


def multistage_pipeline(
    store: FeatureStore,
    config: Config,
    persona: PersonaConfig,
    n_candidates: int = 100,
    n_names: int = 30,
) -> WeightFn:
    """The classical three-stage recommender, transposed to a desk.

    Stage 1 (retrieval): score the catalog with a blended factor model and keep the
    top ``n_candidates``. Stage 2 (ranking): re-score the survivors with the mandate's
    own row-archetype blend. Stage 3 (diversification): a greedy pass that enforces the
    sector cap and the exclusion list.

    This is the architecture GenPage argues against: three independently tuned stages,
    where the diversification pass can only remove what the ranker already chose and
    the ranker never sees the page it is contributing to.
    """
    archetypes = archetype_order()
    blend = np.mean(
        [
            ARCHETYPES[name].coefficient_vector()
            for name in persona.allowed_rows
            if name in ARCHETYPES
        ],
        axis=0,
    )
    retrieval = np.mean([ARCHETYPES[name].coefficient_vector() for name in archetypes], axis=0)
    by_symbol = store.catalog.by_symbol
    excluded = set(persona.excluded_assets)

    def fn(position: int) -> pd.Series:
        features = store.values[position]
        available = store.available.iloc[position].to_numpy().astype(bool)
        order = store.symbol_index

        # Stage 1: retrieval on a generic score.
        generic = features @ retrieval
        generic = np.where(available, generic, -np.inf)
        candidates = np.argsort(-generic)[:n_candidates]

        # Stage 2: ranking with the mandate's blend.
        ranked = sorted(candidates, key=lambda i: -float(features[i] @ blend))

        # Stage 3: post-hoc diversification.
        picks: list[str] = []
        sector_counts: dict[str, int] = {}
        for i in ranked:
            symbol = store.symbols[i]
            if symbol in excluded:
                continue
            sector = by_symbol[symbol].sector
            if sector != FUND_SECTOR:
                if sector_counts.get(sector, 0) >= config.decode.max_names_per_sector:
                    continue
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            picks.append(symbol)
            if len(picks) >= n_names:
                break

        _ = order  # symbol_index kept for readability of the mapping above
        return _inverse_vol(store, position, picks)

    return fn


def teacher_book(
    store: FeatureStore, config: Config, persona: PersonaConfig, seed: int = 0
) -> WeightFn:
    """The teacher's deterministic page, sized exactly like a generated page."""
    teacher = TeacherPolicy(store.catalog, config.corpus, config.decode.max_names_per_sector)
    rng = np.random.default_rng(seed)

    def fn(position: int) -> pd.Series:
        regimes = {axis: int(store.regimes[axis].iloc[position]) for axis in REGIME_AXES}
        page = teacher.greedy_page(store, position, persona, regimes, rng)
        return page_weights(page, store, position, persona)

    return fn


def build_baselines(
    store: FeatureStore, config: Config, persona: PersonaConfig
) -> dict[str, WeightFn]:
    """The full baseline suite for one mandate."""
    return {
        "benchmark_spy": benchmark_weights(store),
        "equal_weight": equal_weight(store),
        "momentum_12_1": factor_screen(store, "mom_12_1"),
        "low_volatility": factor_screen(store, "vol_63", sign=-1.0),
        "risk_parity": risk_parity(store),
        "pipeline_multistage": multistage_pipeline(store, config, persona),
        "teacher_book": teacher_book(store, config, persona, seed=config.backtest.seed),
    }
