"""Cross-sectional instrument features.

The feature set is intentionally small and interpretable: it is the *evidence* the
teacher policy and the row archetypes are defined on, not a kitchen sink. Each
feature is a well-known risk-premium or microstructure signal, and every one is
computed strictly from information available at the close of the observation date.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from gendesk.config import FeatureConfig
from gendesk.data.panel import PricePanel

#: Feature order is part of the model contract: it fixes the layout of the tensor
#: consumed by the semantic-fusion embedding and by the teacher policy.
FEATURE_NAMES: tuple[str, ...] = (
    "mom_12_1",
    "mom_6_1",
    "mom_3m",
    "mom_1m",
    "rev_5d",
    "vol_63",
    "vol_ratio",
    "beta",
    "idio_vol",
    "trend_200",
    "dist_52w_high",
    "drawdown_126",
    "sharpe_126",
    "corr_bench",
    "turnover",
    "downside_ratio",
)

TRADING_DAYS = 252


def _log_return(prices: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Log return between ``t-start`` and ``t-end`` (``end`` closer to today)."""
    # NumPy ufuncs preserve the frame at runtime; pandas-stubs types them as ndarray.
    return cast(pd.DataFrame, np.log(prices.shift(end) / prices.shift(start)))


def compute_raw_features(panel: PricePanel, config: FeatureConfig) -> dict[str, pd.DataFrame]:
    """Compute every feature in its natural units.

    Returns a mapping from feature name to a date x symbol frame. Values are NaN
    wherever the trailing window is incomplete or the instrument was unavailable.
    """
    prices = panel.adj_close
    rets = cast(pd.DataFrame, np.log(prices / prices.shift(1)))
    bench = rets[panel.catalog.benchmark]

    vol_w = config.vol_window
    beta_w = config.beta_window
    corr_w = config.corr_window

    out: dict[str, pd.DataFrame] = {}

    # --- trend / momentum family -------------------------------------------
    out["mom_12_1"] = _log_return(prices, TRADING_DAYS, config.momentum_skip)
    out["mom_6_1"] = _log_return(prices, 126, config.momentum_skip)
    out["mom_3m"] = _log_return(prices, 63, 0)
    out["mom_1m"] = _log_return(prices, 21, 0)
    out["rev_5d"] = -_log_return(prices, config.reversal_window, 0)

    ma200 = prices.rolling(200, min_periods=150).mean()
    out["trend_200"] = prices / ma200 - 1.0

    high_252 = prices.rolling(TRADING_DAYS, min_periods=180).max()
    out["dist_52w_high"] = prices / high_252 - 1.0

    # --- risk family --------------------------------------------------------
    vol_63 = rets.rolling(vol_w, min_periods=vol_w // 2).std() * np.sqrt(TRADING_DAYS)
    vol_252 = rets.rolling(beta_w, min_periods=beta_w // 2).std() * np.sqrt(TRADING_DAYS)
    out["vol_63"] = vol_63
    out["vol_ratio"] = vol_63 / vol_252.replace(0.0, np.nan)

    bench_var = bench.rolling(beta_w, min_periods=beta_w // 2).var()
    cov = rets.rolling(beta_w, min_periods=beta_w // 2).cov(bench)
    beta = cov.div(bench_var.replace(0.0, np.nan), axis=0)
    out["beta"] = beta

    # One-factor residual volatility: total variance net of the systematic part.
    total_var = rets.rolling(beta_w, min_periods=beta_w // 2).var()
    idio_var = (total_var - beta.pow(2).mul(bench_var, axis=0)).clip(lower=0.0)
    out["idio_vol"] = cast(pd.DataFrame, np.sqrt(idio_var * TRADING_DAYS))

    roll_max = prices.rolling(126, min_periods=90).max()
    out["drawdown_126"] = prices / roll_max - 1.0

    mean_126 = rets.rolling(126, min_periods=90).mean()
    std_126 = rets.rolling(126, min_periods=90).std()
    out["sharpe_126"] = (mean_126 / std_126.replace(0.0, np.nan)) * np.sqrt(TRADING_DAYS)

    bench_std = bench.rolling(corr_w, min_periods=corr_w // 2).std()
    inst_std = rets.rolling(corr_w, min_periods=corr_w // 2).std()
    corr_cov = rets.rolling(corr_w, min_periods=corr_w // 2).cov(bench)
    out["corr_bench"] = corr_cov.div(bench_std, axis=0) / inst_std.replace(0.0, np.nan)

    downside = rets.where(rets < 0.0).rolling(vol_w, min_periods=vol_w // 3).std()
    out["downside_ratio"] = (downside * np.sqrt(TRADING_DAYS)) / vol_63.replace(0.0, np.nan)

    # --- liquidity ----------------------------------------------------------
    dollar_volume = panel.dollar_volume
    adv = dollar_volume.rolling(config.dollar_volume_window, min_periods=10).median()
    out["turnover"] = cast(pd.DataFrame, np.log1p(adv))

    for name, frame in out.items():
        out[name] = frame.where(panel.available)

    return out


def cross_sectional_zscore(frame: pd.DataFrame, winsor: tuple[float, float]) -> pd.DataFrame:
    """Winsorise then standardise each row across the available cross-section.

    Standardising per date is what makes the features comparable through time: a
    momentum z-score of +2 means the same thing in 2008 and in 2024, which is a
    prerequisite for the discretisation used by the tokenizer.
    """
    lo = frame.quantile(winsor[0], axis=1)
    hi = frame.quantile(winsor[1], axis=1)
    clipped = frame.clip(lower=lo, upper=hi, axis=0)

    mean = clipped.mean(axis=1)
    std = clipped.std(axis=1).replace(0.0, np.nan)
    z = clipped.sub(mean, axis=0).div(std, axis=0)
    return z.clip(-4.0, 4.0)


def build_feature_tensor(
    panel: PricePanel, config: FeatureConfig
) -> tuple[np.ndarray, dict[str, pd.DataFrame]]:
    """Return the standardised ``(dates, symbols, features)`` tensor and raw frames."""
    raw = compute_raw_features(panel, config)
    layers = [
        cross_sectional_zscore(raw[name], config.winsor).to_numpy(dtype=np.float32)
        for name in FEATURE_NAMES
    ]
    tensor = np.stack(layers, axis=-1)
    return np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0), raw
