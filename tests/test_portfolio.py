"""Sizing and the reward model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gendesk.config import Config
from gendesk.features.store import FeatureStore
from gendesk.portfolio.reward import evaluate_page, slot_rewards, store_benchmark
from gendesk.portfolio.weights import MAX_NAME_WEIGHT, page_weights, row_budgets, turnover
from gendesk.tokenization.page import Page, Row


def _page(store: FeatureStore) -> Page:
    equities = [s for s in store.symbols if not store.catalog.by_symbol[s].is_fund]
    hedges = [s for s in store.symbols if store.catalog.by_symbol[s].is_hedge_candidate]
    return Page(
        date=store.dates[600],
        persona="core",
        rows=(
            Row("MOMENTUM_LEADERS", tuple(equities[:3])),
            Row("QUALITY_BALLAST", tuple(equities[3:6])),
            Row("MACRO_HEDGE", tuple(hedges[:2])),
        ),
    )


def test_weights_are_long_only_and_fully_invested(config: Config, store: FeatureStore) -> None:
    persona = config.personas[0]
    weights = page_weights(_page(store), store, 600, persona)

    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    # The cap is raised to 1/n when it cannot be met by a fully invested book.
    assert weights.max() <= max(MAX_NAME_WEIGHT, 1.0 / len(weights)) + 1e-9


def test_single_name_cap_survives_renormalisation() -> None:
    """Clip-then-renormalise silently breaches the cap; water-filling does not."""
    from gendesk.portfolio.weights import cap_and_renormalise

    raw = pd.Series({"A": 0.70, "B": 0.10, "C": 0.10, "D": 0.05, "E": 0.05})
    capped = cap_and_renormalise(raw, 0.25)

    assert capped.sum() == pytest.approx(1.0)
    assert capped.max() <= 0.25 + 1e-9
    # Order among the uncapped names is preserved.
    assert capped["B"] == pytest.approx(capped["C"])
    assert capped["B"] > capped["D"]


def test_defensive_rows_are_overweighted_for_a_low_risk_mandate(
    config: Config, store: FeatureStore
) -> None:
    page = _page(store)
    low = config.personas[0]
    high = low.model_copy(update={"risk_budget": "high"})

    low_budgets = row_budgets(page, low)
    high_budgets = row_budgets(page, high)

    # Rows 1 (quality) and 2 (hedge) are defensive.
    assert low_budgets[1] > high_budgets[1]
    assert low_budgets[2] > high_budgets[2]
    assert low_budgets.sum() == pytest.approx(1.0)


def test_inverse_volatility_prefers_the_calmer_name(config: Config, store: FeatureStore) -> None:
    persona = config.personas[0]
    page = _page(store)
    # The cap is measured separately; here it is lifted so the sizing rule itself is
    # what is under test.
    weights = page_weights(page, store, 600, persona, max_name_weight=1.0)

    row = page.rows[0].symbols
    vols = store.vol.iloc[600][list(row)]
    calmest, wildest = vols.idxmin(), vols.idxmax()
    assert weights[calmest] > weights[wildest]


def test_turnover_bounds(config: Config, store: FeatureStore) -> None:
    persona = config.personas[0]
    weights = page_weights(_page(store), store, 600, persona)

    assert turnover(None, weights) == pytest.approx(1.0, abs=1e-9)
    assert turnover(weights, weights) == pytest.approx(0.0, abs=1e-9)

    other = pd.Series(1.0, index=["SPY"])
    assert 0.0 < turnover(other, weights) <= 1.0


def test_benchmark_only_page_earns_no_active_reward(config: Config, store: FeatureStore) -> None:
    """A page that is just the benchmark must score zero active return, by construction."""
    benchmark = store_benchmark(store)
    page = Page(store.dates[600], "core", (Row("MOMENTUM_LEADERS", (benchmark,)),))
    reward = evaluate_page(page, store, 600, config.personas[0], config.corpus)

    assert reward.active_return == pytest.approx(0.0, abs=1e-9)
    # The only cost left is turnover from an empty starting book.
    assert reward.total < 0.0


def test_reward_penalises_turnover(config: Config, store: FeatureStore) -> None:
    persona = config.personas[0]
    page = _page(store)
    weights = page_weights(page, store, 600, persona)

    fresh = evaluate_page(page, store, 600, persona, config.corpus, previous_weights=None)
    held = evaluate_page(page, store, 600, persona, config.corpus, previous_weights=weights)

    assert held.turnover == pytest.approx(0.0, abs=1e-9)
    assert held.total > fresh.total


def test_reward_is_bounded(config: Config, store: FeatureStore) -> None:
    persona = config.personas[0]
    for position in (400, 600, 800):
        reward = evaluate_page(_page(store), store, position, persona, config.corpus)
        assert -8.0 <= reward.total <= 8.0
        assert np.isfinite(reward.active_return)


def test_slot_rewards_are_volatility_scaled(store: FeatureStore) -> None:
    """Scaling by volatility is what stops the objective becoming 'prefer high beta'."""
    position, horizon = 600, 10
    rewards = slot_rewards(store, position, horizon)
    raw = store.forward_return(position, horizon)
    bench = store.symbols.index(store_benchmark(store))
    active = raw - raw[bench]

    vol = store.vol.iloc[position].to_numpy()
    high_vol = np.nanargmax(vol)
    # The scaled reward of the most volatile name must be smaller in magnitude than its
    # unscaled active return would suggest, unless that return is already tiny.
    if abs(active[high_vol]) > 0.01:
        assert abs(rewards[high_vol]) < abs(active[high_vol]) / 0.05


def test_empty_page_scores_the_floor(config: Config, store: FeatureStore) -> None:
    reward = evaluate_page(
        Page(store.dates[600], "core", ()), store, 600, config.personas[0], config.corpus
    )
    assert reward.total == pytest.approx(-1.0)
