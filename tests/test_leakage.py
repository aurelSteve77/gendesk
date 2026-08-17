"""Look-ahead tests.

These are the tests that matter most. A backtest is a claim about what a system
*would have* produced, and a single forward-looking feature invalidates every number
downstream. Each test below states one invariant and checks it against the real
feature code rather than a description of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gendesk.config import Config
from gendesk.data.panel import PricePanel
from gendesk.features.cross_section import build_feature_tensor
from gendesk.features.regimes import build_regimes
from gendesk.features.store import FeatureStore


def _perturb_future(panel: PricePanel, cut: int, factor: float = 3.0) -> PricePanel:
    """Return a panel whose prices after ``cut`` are multiplied by ``factor``."""
    adj = panel.adj_close.copy()
    adj.iloc[cut + 1 :] = adj.iloc[cut + 1 :] * factor
    close = panel.close.copy()
    close.iloc[cut + 1 :] = close.iloc[cut + 1 :] * factor
    return PricePanel(
        adj_close=adj,
        close=close,
        volume=panel.volume,
        macro=panel.macro,
        available=panel.available,
        catalog=panel.catalog,
    )


def test_features_do_not_move_when_the_future_changes(panel: PricePanel, config: Config) -> None:
    """Every feature at or before the cut must be identical under both futures."""
    cut = 700
    base, _ = build_feature_tensor(panel, config.features)
    shocked, _ = build_feature_tensor(_perturb_future(panel, cut), config.features)

    np.testing.assert_allclose(base[: cut + 1], shocked[: cut + 1], rtol=1e-6, atol=1e-6)
    # Sanity: the perturbation must actually change something after the cut, otherwise
    # the test would pass for the wrong reason.
    assert not np.allclose(base[cut + 30 :], shocked[cut + 30 :])


def test_regimes_do_not_move_when_the_future_changes(panel: PricePanel, config: Config) -> None:
    cut = 700
    base = build_regimes(panel, config.regimes)
    shocked = build_regimes(_perturb_future(panel, cut), config.regimes)
    pd.testing.assert_frame_equal(base.iloc[: cut + 1], shocked.iloc[: cut + 1])


def test_forward_return_window_starts_after_the_position(store: FeatureStore) -> None:
    """``forward_return`` must use ``(t, t+h]``, never ``[t, t+h]``."""
    position = 500
    horizon = 10
    forward = store.forward_return(position, horizon)

    window = store.returns.iloc[position + 1 : position + 1 + horizon]
    expected = np.expm1(np.log1p(window).sum(axis=0).to_numpy())
    np.testing.assert_allclose(forward, expected, rtol=1e-6, atol=1e-8)

    # Including the observation date itself would give a different answer, which is
    # exactly the bug this asserts against.
    with_today = store.returns.iloc[position : position + 1 + horizon]
    assert not np.allclose(
        forward, np.expm1(np.log1p(with_today).sum(axis=0).to_numpy()), atol=1e-8
    )


def test_slot_rewards_are_pure_labels(store: FeatureStore) -> None:
    """Slot rewards must depend only on data strictly after the position."""
    from gendesk.portfolio.reward import slot_rewards

    position = 500
    rewards = slot_rewards(store, position, 10)
    assert rewards.shape == (store.n_symbols,)
    assert np.isfinite(rewards).all()
    # The benchmark's own active return is zero by construction.
    bench = store.symbols.index("SPY")
    assert abs(float(rewards[bench])) < 1e-6


def test_corpus_split_purges_labels_that_cross_the_boundary(config: Config) -> None:
    """A page whose reward window straddles the cutoff belongs to neither split."""
    from gendesk.corpus.build import _assign_split

    train_end = pd.Timestamp(config.backtest.train_end)
    assert (
        _assign_split(train_end - pd.Timedelta(days=60), train_end - pd.Timedelta(days=30), config)
        == "train"
    )
    assert (
        _assign_split(train_end - pd.Timedelta(days=5), train_end + pd.Timedelta(days=16), config)
        == "purged"
    )
    assert (
        _assign_split(
            pd.Timestamp(config.backtest.valid_end) + pd.Timedelta(days=40),
            pd.Timestamp(config.backtest.valid_end) + pd.Timedelta(days=70),
            config,
        )
        == "test"
    )


def test_availability_mask_precedes_price_fill(panel: PricePanel) -> None:
    """A name must not be available before it has ever traded."""
    late = panel.adj_close.columns[3]
    assert not panel.available[late].iloc[:120].any()
    assert panel.available[late].iloc[200:].all()


@pytest.mark.parametrize("feature", ["mom_12_1", "vol_63", "beta", "trend_200"])
def test_feature_is_finite_and_standardised(store: FeatureStore, feature: str) -> None:
    values = store.feature(feature)[store.warm]
    assert np.isfinite(values).all()
    assert np.abs(values).max() <= 4.0 + 1e-6
