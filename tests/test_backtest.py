"""The backtest engine, on inputs whose answer is known in advance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gendesk.config import Config
from gendesk.evaluation.backtest import compare, rebalance_positions, run_backtest
from gendesk.evaluation.baselines import build_baselines
from gendesk.features.store import FeatureStore


def _window(store: FeatureStore) -> tuple[pd.Timestamp, pd.Timestamp]:
    return store.dates[400], store.dates[-1]


def test_buy_and_hold_reproduces_the_instrument(config: Config, store: FeatureStore) -> None:
    """A single-name strategy must earn exactly that name's return, net of one trade."""
    start, end = _window(store)
    result = run_backtest(
        "spy_only",
        lambda position: pd.Series({"SPY": 1.0}),
        store,
        config,
        start,
        end,
    )
    realised = (1.0 + result.gross_returns).prod()
    expected = (1.0 + store.returns.loc[result.returns.index, "SPY"]).prod()
    assert realised == pytest.approx(expected, rel=1e-9)

    # Turnover is reported one-way: half the traded notional. Entering from an empty
    # book trades 100% of notional once, so the first rebalance reads 0.5 and every
    # later one reads 0.
    assert result.turnover.iloc[0] == pytest.approx(0.5)
    assert result.turnover.iloc[1:].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_costs_reduce_returns_by_the_expected_amount(config: Config, store: FeatureStore) -> None:
    start, end = _window(store)
    result = run_backtest(
        "spy_only", lambda position: pd.Series({"SPY": 1.0}), store, config, start, end
    )
    drag = result.gross_returns.sum() - result.returns.sum()
    # Exactly one round of trading at the configured rate.
    assert drag == pytest.approx(config.backtest.cost_bps / 1e4, rel=1e-6)


def test_turnover_is_charged_on_every_rebalance(config: Config, store: FeatureStore) -> None:
    """A strategy that flips between two names must pay every time it flips."""
    start, end = _window(store)
    names = [s for s in store.symbols if not store.catalog.by_symbol[s].is_fund][:2]
    flip = {"n": 0}

    def weights(position: int) -> pd.Series:
        flip["n"] += 1
        return pd.Series({names[flip["n"] % 2]: 1.0})

    result = run_backtest("flip", weights, store, config, start, end)
    drag = result.gross_returns.sum() - result.returns.sum()
    assert drag > config.backtest.cost_bps / 1e4 * 5


def test_weights_drift_between_rebalances(config: Config, store: FeatureStore) -> None:
    """Between rebalances the book is held, not re-weighted daily."""
    start, end = _window(store)
    names = [s for s in store.symbols if not store.catalog.by_symbol[s].is_fund][:2]

    def weights(position: int) -> pd.Series:
        return pd.Series({names[0]: 0.5, names[1]: 0.5})

    held = run_backtest("held", weights, store, config, start, end)
    daily = (
        store.returns.loc[held.returns.index, names[0]] * 0.5
        + store.returns.loc[held.returns.index, names[1]] * 0.5
    )
    # A daily-rebalanced book would match `daily` exactly; a drifting one will not.
    assert not np.allclose(held.gross_returns.to_numpy(), daily.to_numpy(), atol=1e-9)


def test_rebalance_positions_respect_the_window(config: Config, store: FeatureStore) -> None:
    start, end = store.dates[400], store.dates[600]
    positions = rebalance_positions(store, start, end, config.backtest.rebalance_days)
    assert positions
    assert all(start <= store.dates[p] <= end for p in positions)
    gaps = np.diff(positions)
    assert set(gaps.tolist()) <= {config.backtest.rebalance_days}


def test_baselines_all_run_and_produce_valid_books(config: Config, store: FeatureStore) -> None:
    start, end = _window(store)
    persona = config.personas[0]
    results = []
    for name, fn in build_baselines(store, config, persona).items():
        result = run_backtest(name, fn, store, config, start, end)
        assert result.returns.notna().all()
        assert np.isfinite(result.stats["sharpe"])
        weights = result.weights
        assert (weights.sum(axis=1) - 1.0).abs().max() < 1e-6
        results.append(result)

    table = compare(results)
    assert len(table) == len(results)
    assert table["sharpe"].is_monotonic_decreasing


def test_pipeline_baseline_respects_the_sector_cap(config: Config, store: FeatureStore) -> None:
    """The multi-stage baseline must be held to the same rules as the model."""
    from gendesk.data.universe import FUND_SECTOR

    persona = config.personas[0]
    weights = build_baselines(store, config, persona)["pipeline_multistage"](600)
    by_symbol = store.catalog.by_symbol

    counts: dict[str, int] = {}
    for symbol in weights.index:
        sector = by_symbol[str(symbol)].sector
        if sector != FUND_SECTOR:
            counts[sector] = counts.get(sector, 0) + 1
    assert max(counts.values(), default=0) <= config.decode.max_names_per_sector
    assert not set(weights.index) & set(persona.excluded_assets)
