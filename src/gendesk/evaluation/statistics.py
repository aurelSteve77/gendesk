"""Performance statistics and inference.

A backtest that reports only a Sharpe ratio is not evidence. Three tools are used
here to say something defensible about whether a result is real:

* **Stationary block bootstrap** for confidence intervals on the Sharpe ratio and on
  the *difference* between two strategies' Sharpe ratios. Daily returns are
  autocorrelated and fat-tailed, so the textbook standard error is optimistic.
* **Deflated Sharpe ratio** (Bailey and Lopez de Prado), which discounts the observed
  Sharpe by how many configurations were tried, and by the non-normality of the
  return series. The number of trials is taken from the ablation grid actually run,
  not chosen after the fact.
* **Probability of backtest overfitting** via combinatorially symmetric
  cross-validation: how often the in-sample best configuration lands below the median
  out of sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------- descriptive


def performance_summary(
    returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS
) -> dict:
    """Standard performance statistics for a daily return series."""
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return dict.fromkeys(("cagr", "vol", "sharpe", "sortino", "max_drawdown", "calmar"), 0.0)

    equity = (1.0 + r).cumprod()
    years = len(r) / periods
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(r.std() * np.sqrt(periods))

    excess = r - risk_free / periods
    sharpe = float(excess.mean() / excess.std() * np.sqrt(periods)) if excess.std() > 0 else 0.0

    downside = excess[excess < 0]
    sortino = (
        float(excess.mean() / downside.std() * np.sqrt(periods))
        if len(downside) > 1 and downside.std() > 0
        else 0.0
    )

    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())

    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else 0.0,
        "hit_rate": float((r > 0).mean()),
        "skew": float(stats.skew(r)),
        "kurtosis": float(stats.kurtosis(r, fisher=False)),
        "n_days": len(r),
        "total_return": float(equity.iloc[-1] - 1.0),
        "worst_day": float(r.min()),
        "best_day": float(r.max()),
    }


def rolling_drawdown(returns: pd.Series) -> pd.Series:
    equity = (1.0 + pd.Series(returns).fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


# ----------------------------------------------------------------- bootstrap


@dataclass
class SharpeTest:
    """Bootstrap inference on a Sharpe ratio or a Sharpe difference."""

    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    n_samples: int

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n_samples": self.n_samples,
        }


def _sharpe(x: np.ndarray, periods: int = TRADING_DAYS) -> float:
    sd = x.std()
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one stationary block-bootstrap resample of length ``n``."""
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, s + block) % n for s in starts])
    return idx[:n]


def block_bootstrap_sharpe(
    returns: pd.Series,
    n_samples: int = 2000,
    block: int = 21,
    seed: int = 0,
    periods: int = TRADING_DAYS,
) -> SharpeTest:
    """Bootstrap CI and a two-sided p-value against ``Sharpe = 0``."""
    x = pd.Series(returns).dropna().to_numpy(dtype=float)
    if x.size < block * 2:
        return SharpeTest(_sharpe(x, periods), np.nan, np.nan, np.nan, 0)

    rng = np.random.default_rng(seed)
    draws = np.array(
        [_sharpe(x[_block_indices(x.size, block, rng)], periods) for _ in range(n_samples)]
    )
    estimate = _sharpe(x, periods)
    # p-value from the bootstrap distribution recentred on the null.
    centred = draws - draws.mean()
    p = float((np.abs(centred) >= abs(estimate)).mean())
    return SharpeTest(
        estimate=estimate,
        ci_low=float(np.quantile(draws, 0.025)),
        ci_high=float(np.quantile(draws, 0.975)),
        p_value=p,
        n_samples=n_samples,
    )


def block_bootstrap_difference(
    a: pd.Series,
    b: pd.Series,
    n_samples: int = 2000,
    block: int = 21,
    seed: int = 0,
    periods: int = TRADING_DAYS,
) -> SharpeTest:
    """Bootstrap the Sharpe difference ``a - b`` on the common date index.

    Resampling both series with the *same* block indices preserves their
    contemporaneous correlation, which is what makes the difference test far tighter
    than comparing two independent intervals.
    """
    joined = pd.concat([pd.Series(a), pd.Series(b)], axis=1, join="inner").dropna()
    if len(joined) < block * 2:
        return SharpeTest(np.nan, np.nan, np.nan, np.nan, 0)

    xa = joined.iloc[:, 0].to_numpy(dtype=float)
    xb = joined.iloc[:, 1].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    draws = np.empty(n_samples)
    for i in range(n_samples):
        idx = _block_indices(len(xa), block, rng)
        draws[i] = _sharpe(xa[idx], periods) - _sharpe(xb[idx], periods)

    estimate = _sharpe(xa, periods) - _sharpe(xb, periods)
    centred = draws - draws.mean()
    return SharpeTest(
        estimate=float(estimate),
        ci_low=float(np.quantile(draws, 0.025)),
        ci_high=float(np.quantile(draws, 0.975)),
        p_value=float((np.abs(centred) >= abs(estimate)).mean()),
        n_samples=n_samples,
    )


def newey_west_tstat(returns: pd.Series, lags: int | None = None) -> float:
    """t-statistic of the mean return with a Newey-West HAC standard error."""
    x = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 10:
        return 0.0
    lags = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    demeaned = x - x.mean()

    variance = float(demeaned @ demeaned / n)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        cov = float(demeaned[lag:] @ demeaned[:-lag] / n)
        variance += 2.0 * weight * cov
    if variance <= 0:
        return 0.0
    return float(x.mean() / np.sqrt(variance / n))


# ------------------------------------------------------- multiple-testing


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpes: np.ndarray | None = None,
    periods: int = TRADING_DAYS,
) -> dict:
    """Deflated Sharpe ratio.

    The observed Sharpe is compared against the Sharpe one would expect from the
    *best* of ``n_trials`` independent attempts on a strategy with no edge, and the
    comparison is corrected for the skew and kurtosis of the realised returns.

    Returns the annualised observed Sharpe, the expected-maximum null Sharpe and the
    probability that the true Sharpe exceeds zero given both corrections.
    """
    x = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 30 or x.std() == 0:
        return {"sharpe": 0.0, "sharpe0": 0.0, "dsr": 0.0, "n_trials": n_trials}

    sr = float(x.mean() / x.std())  # per-period
    skew = float(stats.skew(x))
    kurt = float(stats.kurtosis(x, fisher=False))

    variance = (
        float(np.var(trial_sharpes, ddof=1))
        if trial_sharpes is not None and len(trial_sharpes) > 1
        else 1.0 / n
    )
    gamma = 0.5772156649  # Euler-Mascheroni
    trials = max(n_trials, 2)
    sr0 = np.sqrt(variance) * (
        (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / trials)
        + gamma * stats.norm.ppf(1.0 - 1.0 / (trials * np.e))
    )

    denominator = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-9))
    dsr = float(stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / denominator))

    return {
        "sharpe": sr * np.sqrt(periods),
        "sharpe0": float(sr0 * np.sqrt(periods)),
        "dsr": dsr,
        "n_trials": trials,
        "skew": skew,
        "kurtosis": kurt,
    }


def probability_of_backtest_overfitting(strategy_returns: pd.DataFrame, n_splits: int = 10) -> dict:
    """PBO via combinatorially symmetric cross-validation.

    The return matrix is cut into ``n_splits`` contiguous blocks. For every way of
    splitting those blocks into equal in-sample and out-of-sample halves, the
    configuration with the best in-sample Sharpe is located and its out-of-sample
    rank recorded. PBO is the fraction of splits where that winner ended up in the
    bottom half out of sample -- i.e. the probability that selecting on the backtest
    buys nothing.
    """
    data = strategy_returns.dropna(how="all").fillna(0.0)
    if data.shape[1] < 2 or data.shape[0] < n_splits * 4:
        return {"pbo": float("nan"), "n_splits": n_splits, "n_configs": data.shape[1]}

    n_splits = n_splits if n_splits % 2 == 0 else n_splits - 1
    blocks = np.array_split(np.arange(len(data)), n_splits)
    logits: list[float] = []

    for combo in combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])

        is_sharpe = data.iloc[is_idx].apply(lambda c: _sharpe(c.to_numpy()))
        oos_sharpe = data.iloc[oos_idx].apply(lambda c: _sharpe(c.to_numpy()))

        best = is_sharpe.idxmax()
        rank = float(oos_sharpe.rank(pct=True)[best])
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(rank / (1.0 - rank))))

    return {
        "pbo": float(np.mean(np.asarray(logits) <= 0.0)),
        "n_splits": n_splits,
        "n_configs": int(data.shape[1]),
        "median_logit": float(np.median(logits)),
    }
