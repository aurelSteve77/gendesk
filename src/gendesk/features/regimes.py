"""Macro regime discretisation.

GenPage compresses a member's context into a handful of tokens rather than a wide
float vector. The financial analogue is a small set of *regime axes* -- volatility,
the shape of the curve, breadth, dispersion, correlation -- each bucketed into
terciles against its own trailing distribution.

Ranking each axis against a trailing window (rather than the full sample) is what
keeps the discretisation point-in-time: the tercile boundaries at date ``t`` are a
function of data up to ``t`` only.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from gendesk.config import RegimeConfig
from gendesk.data.panel import PricePanel

#: Axis order fixes the token layout; changing it invalidates checkpoints.
REGIME_AXES: tuple[str, ...] = (
    "vol_level",
    "vol_shock",
    "curve_slope",
    "rate_impulse",
    "market_trend",
    "breadth",
    "dispersion",
    "implied_corr",
)

#: Human-readable labels for the terciles, used by the UI and the verbaliser.
BUCKET_LABELS: dict[str, tuple[str, str, str]] = {
    "vol_level": ("calm", "normal", "stressed"),
    "vol_shock": ("vol_falling", "vol_flat", "vol_spiking"),
    "curve_slope": ("inverted", "flat", "steep"),
    "rate_impulse": ("rates_falling", "rates_stable", "rates_rising"),
    "market_trend": ("downtrend", "range", "uptrend"),
    "breadth": ("narrow", "mixed", "broad"),
    "dispersion": ("low_disp", "mid_disp", "high_disp"),
    "implied_corr": ("low_corr", "mid_corr", "high_corr"),
}


def _trailing_bucket(series: pd.Series, window: int, n_buckets: int) -> pd.Series:
    """Bucket a series by its percentile rank inside a trailing window."""
    pct = series.rolling(window, min_periods=max(60, window // 8)).rank(pct=True)
    bucket = np.floor(pct * n_buckets).clip(0, n_buckets - 1)
    return bucket.fillna((n_buckets - 1) // 2).astype(np.int64)


def compute_regime_signals(panel: PricePanel) -> pd.DataFrame:
    """Compute the continuous value behind each regime axis."""
    macro = panel.macro
    prices = panel.adj_close
    rets = cast(pd.DataFrame, np.log(prices / prices.shift(1))).where(panel.available)

    signals = pd.DataFrame(index=panel.calendar)

    vix = macro["^VIX"] if "^VIX" in macro else pd.Series(np.nan, index=panel.calendar)
    signals["vol_level"] = vix
    signals["vol_shock"] = vix / vix.shift(21) - 1.0

    tnx = macro.get("^TNX", pd.Series(np.nan, index=panel.calendar))
    irx = macro.get("^IRX", pd.Series(np.nan, index=panel.calendar))
    signals["curve_slope"] = tnx - irx
    signals["rate_impulse"] = tnx - tnx.shift(63)

    bench = prices[panel.catalog.benchmark]
    signals["market_trend"] = bench / bench.shift(126) - 1.0

    ma200 = prices.rolling(200, min_periods=150).mean()
    above = (prices > ma200).where(panel.available)
    signals["breadth"] = above.sum(axis=1) / panel.available.sum(axis=1).replace(0, np.nan)

    monthly = prices / prices.shift(21) - 1.0
    signals["dispersion"] = monthly.where(panel.available).std(axis=1)

    # Average-correlation proxy: an equally weighted index is only as volatile as
    # its constituents when they all move together, so the ratio of index variance
    # to mean constituent variance rises with average pairwise correlation.
    window = 63
    index_ret = rets.mean(axis=1)
    index_var = index_ret.rolling(window, min_periods=window // 2).var()
    mean_var = rets.rolling(window, min_periods=window // 2).var().mean(axis=1)
    n_names = panel.available.sum(axis=1).clip(lower=2)
    ratio = (index_var / mean_var.replace(0.0, np.nan)).clip(0.0, 1.0)
    signals["implied_corr"] = ((ratio * n_names - 1.0) / (n_names - 1.0)).clip(0.0, 1.0)

    return signals[list(REGIME_AXES)]


def build_regimes(panel: PricePanel, config: RegimeConfig) -> pd.DataFrame:
    """Return a date x axis frame of integer regime buckets."""
    signals = compute_regime_signals(panel)
    buckets = {
        axis: _trailing_bucket(signals[axis], config.rank_window, config.n_buckets)
        for axis in REGIME_AXES
    }
    return pd.DataFrame(buckets, index=signals.index)


def describe_regime(row: pd.Series) -> dict[str, str]:
    """Map a row of integer buckets to human-readable labels."""
    return {
        axis: BUCKET_LABELS[axis][int(min(max(row[axis], 0), 2))]
        for axis in REGIME_AXES
        if axis in row.index
    }
