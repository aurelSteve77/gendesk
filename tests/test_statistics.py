"""Inference helpers, checked against cases with a known answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gendesk.evaluation.statistics import (
    block_bootstrap_difference,
    block_bootstrap_sharpe,
    deflated_sharpe_ratio,
    newey_west_tstat,
    performance_summary,
    probability_of_backtest_overfitting,
)


def _series(mean: float, vol: float, n: int = 1500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rng.normal(mean, vol, n), index=index)


def test_performance_summary_recovers_known_moments() -> None:
    daily = 0.0004
    vol = 0.01
    stats = performance_summary(_series(daily, vol, n=5000))

    assert stats["vol"] == pytest.approx(vol * np.sqrt(252), rel=0.05)
    assert stats["sharpe"] == pytest.approx(daily / vol * np.sqrt(252), rel=0.25)
    assert stats["max_drawdown"] < 0.0
    assert 0.4 < stats["hit_rate"] < 0.6


def test_summary_of_a_constant_series() -> None:
    flat = pd.Series(0.0, index=pd.bdate_range("2020-01-01", periods=300))
    stats = performance_summary(flat)
    assert stats["sharpe"] == 0.0
    assert stats["max_drawdown"] == pytest.approx(0.0)


def test_bootstrap_interval_covers_the_estimate() -> None:
    test = block_bootstrap_sharpe(_series(0.0006, 0.01, seed=3), n_samples=400, block=21)
    assert test.ci_low < test.estimate < test.ci_high
    assert 0.0 <= test.p_value <= 1.0


def test_bootstrap_rejects_a_real_edge_and_spares_pure_noise() -> None:
    strong = block_bootstrap_sharpe(_series(0.0012, 0.008, n=2500, seed=5), n_samples=500)
    noise = block_bootstrap_sharpe(_series(0.0, 0.01, n=2500, seed=6), n_samples=500)

    assert strong.p_value < 0.05
    assert noise.p_value > 0.10


def test_paired_difference_is_tighter_than_independent_intervals() -> None:
    """Common-block resampling must exploit the correlation between the two legs."""
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2015-01-01", periods=2000)
    common = rng.normal(0.0, 0.01, 2000)

    a = pd.Series(common + rng.normal(0.0004, 0.002, 2000), index=index)
    b = pd.Series(common + rng.normal(0.0000, 0.002, 2000), index=index)

    paired = block_bootstrap_difference(a, b, n_samples=500, block=21)
    solo_a = block_bootstrap_sharpe(a, n_samples=500, block=21)
    solo_b = block_bootstrap_sharpe(b, n_samples=500, block=21)

    paired_width = paired.ci_high - paired.ci_low
    naive_width = (solo_a.ci_high - solo_a.ci_low) + (solo_b.ci_high - solo_b.ci_low)
    assert paired_width < naive_width
    assert paired.estimate > 0


def test_deflated_sharpe_falls_as_trials_rise() -> None:
    series = _series(0.0006, 0.01, n=2000, seed=7)
    few = deflated_sharpe_ratio(series, n_trials=2)
    many = deflated_sharpe_ratio(series, n_trials=500)

    assert few["dsr"] >= many["dsr"]
    assert many["sharpe0"] > few["sharpe0"]


def test_deflated_sharpe_is_scale_invariant_in_the_null() -> None:
    noise = _series(0.0, 0.01, n=2000, seed=8)
    result = deflated_sharpe_ratio(noise, n_trials=50)
    assert result["dsr"] < 0.9


def test_newey_west_handles_autocorrelation() -> None:
    rng = np.random.default_rng(13)
    n = 1500
    noise = rng.normal(0.0005, 0.01, n)
    autocorrelated = pd.Series(noise).rolling(5, min_periods=1).mean()

    plain = newey_west_tstat(pd.Series(noise), lags=0)
    corrected = newey_west_tstat(autocorrelated)
    # Smoothing inflates the naive t-stat; the HAC correction must pull it back.
    naive = float(autocorrelated.mean() / autocorrelated.std() * np.sqrt(n))
    assert abs(corrected) < abs(naive)
    assert np.isfinite(plain)


def test_pbo_flags_pure_noise_selection() -> None:
    """With no real edge, the in-sample winner should land below the median half the time."""
    rng = np.random.default_rng(17)
    index = pd.bdate_range("2015-01-01", periods=1200)
    frame = pd.DataFrame({f"cfg{i}": rng.normal(0.0, 0.01, 1200) for i in range(12)}, index=index)
    result = probability_of_backtest_overfitting(frame, n_splits=8)
    assert 0.25 <= result["pbo"] <= 0.75


def test_pbo_is_low_when_one_configuration_genuinely_dominates() -> None:
    rng = np.random.default_rng(19)
    index = pd.bdate_range("2015-01-01", periods=1200)
    frame = pd.DataFrame({f"cfg{i}": rng.normal(0.0, 0.01, 1200) for i in range(6)}, index=index)
    frame["winner"] = rng.normal(0.0015, 0.008, 1200)
    result = probability_of_backtest_overfitting(frame, n_splits=8)
    assert result["pbo"] < 0.2
