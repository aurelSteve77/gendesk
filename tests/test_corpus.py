"""Teacher policy and corpus construction."""

from __future__ import annotations

import numpy as np

from gendesk.config import Config
from gendesk.corpus.rows import ARCHETYPES, archetype_scores, eligibility_matrix
from gendesk.corpus.teacher import TeacherPolicy
from gendesk.data.universe import FUND_SECTOR
from gendesk.features.regimes import REGIME_AXES
from gendesk.features.store import FeatureStore


def _regimes(store: FeatureStore, position: int) -> dict[str, int]:
    return {axis: int(store.regimes[axis].iloc[position]) for axis in REGIME_AXES}


def test_archetype_scores_match_their_coefficients(store: FeatureStore) -> None:
    position = 600
    scores = archetype_scores(store.values[position])
    assert scores.shape == (len(ARCHETYPES), store.n_symbols)

    name = "MOMENTUM_LEADERS"
    index = list(ARCHETYPES).index(name)
    manual = store.values[position] @ ARCHETYPES[name].coefficient_vector()
    np.testing.assert_allclose(scores[index], manual, rtol=1e-5)


def test_eligibility_reserves_macro_funds_for_the_hedge_row(store: FeatureStore) -> None:
    matrix = eligibility_matrix(store.catalog)
    names = list(ARCHETYPES)
    hedge = names.index("MACRO_HEDGE")
    momentum = names.index("MOMENTUM_LEADERS")

    for i, instrument in enumerate(store.catalog):
        if instrument.is_hedge_candidate:
            assert matrix[hedge, i]
            assert not matrix[momentum, i]
        elif instrument.asset_class == "equity":
            assert matrix[momentum, i]
            assert not matrix[hedge, i]


def test_regime_affinity_shifts_row_selection(store: FeatureStore, config: Config) -> None:
    """A stressed regime must make the teacher reach for defensive rows more often."""
    teacher = TeacherPolicy(store.catalog, config.corpus, config.decode.max_names_per_sector)
    persona = config.personas[0]
    rng = np.random.default_rng(0)

    calm = dict.fromkeys(REGIME_AXES, 1) | {"vol_level": 0, "market_trend": 2}
    stressed = dict.fromkeys(REGIME_AXES, 1) | {"vol_level": 2, "market_trend": 0}

    def frequency(regimes: dict[str, int]) -> float:
        picks = [teacher.choose_rows(persona, regimes, rng) for _ in range(300)]
        return float(np.mean(["QUALITY_BALLAST" in rows for rows in picks]))

    assert frequency(stressed) > frequency(calm)


def test_pinned_rows_always_appear(store: FeatureStore, config: Config) -> None:
    teacher = TeacherPolicy(store.catalog, config.corpus, config.decode.max_names_per_sector)
    persona = config.personas[0]
    rng = np.random.default_rng(1)
    for _ in range(50):
        rows = teacher.choose_rows(persona, _regimes(store, 600), rng)
        assert "MACRO_HEDGE" in rows
        assert len(rows) == config.corpus.n_rows


def test_greedy_page_is_deterministic(store: FeatureStore, config: Config) -> None:
    teacher = TeacherPolicy(store.catalog, config.corpus, config.decode.max_names_per_sector)
    persona = config.personas[1]
    regimes = _regimes(store, 600)

    first = teacher.greedy_page(store, 600, persona, regimes, np.random.default_rng(0))
    second = teacher.greedy_page(store, 600, persona, regimes, np.random.default_rng(0))
    assert first.symbols == second.symbols


def test_sampled_pages_vary_but_stay_legal(store: FeatureStore, config: Config) -> None:
    teacher = TeacherPolicy(store.catalog, config.corpus, config.decode.max_names_per_sector)
    persona = config.personas[1]
    regimes = _regimes(store, 600)
    rng = np.random.default_rng(2)

    pages = [teacher.sample_page(store, 600, persona, regimes, rng) for _ in range(12)]
    assert len({p.symbols for p in pages}) > 1

    by_symbol = store.catalog.by_symbol
    for page in pages:
        symbols = list(page.symbols)
        assert len(symbols) == len(set(symbols))
        counts: dict[str, int] = {}
        for symbol in symbols:
            sector = by_symbol[symbol].sector
            if sector != FUND_SECTOR:
                counts[sector] = counts.get(sector, 0) + 1
        assert max(counts.values(), default=0) <= config.decode.max_names_per_sector


def test_teacher_respects_mandate_exclusions(store: FeatureStore, config: Config) -> None:
    teacher = TeacherPolicy(store.catalog, config.corpus, config.decode.max_names_per_sector)
    persona = config.personas[0]  # excludes GLD
    rng = np.random.default_rng(3)
    for _ in range(20):
        page = teacher.sample_page(store, 600, persona, _regimes(store, 600), rng)
        assert "GLD" not in page.symbols


def test_row_is_ordered_best_first(store: FeatureStore, config: Config) -> None:
    """The row's own ranking is signal the model should learn to reproduce."""
    teacher = TeacherPolicy(store.catalog, config.corpus, config.decode.max_names_per_sector)
    persona = config.personas[1]
    page = teacher.greedy_page(store, 600, persona, _regimes(store, 600), np.random.default_rng(0))

    scores = archetype_scores(store.values[600])
    names = list(ARCHETYPES)
    index = store.symbol_index
    for row in page.rows:
        row_scores = [scores[names.index(row.archetype), index[s]] for s in row.symbols]
        assert row_scores == sorted(row_scores, reverse=True)
