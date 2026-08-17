"""Walk-forward backtest engine.

Every strategy in this project -- the model, the teacher it learned from, and the
classical baselines -- reduces to the same interface: *given a date, return a vector
of weights*. The engine then does the identical thing to all of them (hold to the
next rebalance, let weights drift with prices, charge the same cost on traded
notional), which is the only way a comparison between a generative recommender and a
factor screen means anything.

The evaluation window starts after the validation cutoff. Nothing in the model, the
corpus threshold, or the row archetypes was chosen using data from that window.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gendesk.config import Config
from gendesk.evaluation.statistics import performance_summary, rolling_drawdown
from gendesk.features.store import FeatureStore
from gendesk.utils.logging import get_logger

log = get_logger(__name__)


#: A strategy is nothing more than a map from a date position to a symbol-indexed
#: weight vector summing to one. Model, teacher and baselines all satisfy it.
WeightFunction = Callable[[int], "pd.Series"]


@dataclass
class BacktestResult:
    """Daily returns and diagnostics for one strategy."""

    name: str
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    stats: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns.fillna(0.0)).cumprod()

    @property
    def drawdown(self) -> pd.Series:
        return rolling_drawdown(self.returns)

    def summary_row(self) -> dict:
        return {"strategy": self.name, **self.stats}


def rebalance_positions(
    store: FeatureStore, start: pd.Timestamp, end: pd.Timestamp, every: int
) -> list[int]:
    """Trading-day positions on which the book is rebuilt."""
    mask = (store.dates >= start) & (store.dates <= end) & store.warm
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return []
    return [int(p) for p in positions[::every]]


def run_backtest(
    name: str,
    weight_fn: WeightFunction,
    store: FeatureStore,
    config: Config,
    start: pd.Timestamp,
    end: pd.Timestamp,
    progress: Callable[[int, int], None] | None = None,
) -> BacktestResult:
    """Hold-to-next-rebalance backtest with drift and transaction costs."""
    cost_rate = config.backtest.cost_bps / 1e4
    positions = rebalance_positions(store, start, end, config.backtest.rebalance_days)
    if not positions:
        raise ValueError(f"no rebalance dates between {start.date()} and {end.date()}")

    returns = store.returns
    symbols = list(returns.columns)
    index = {s: i for i, s in enumerate(symbols)}
    returns_np = returns.to_numpy(dtype=np.float64)

    daily: list[float] = []
    daily_gross: list[float] = []
    dates: list[pd.Timestamp] = []
    turnover_records: dict[pd.Timestamp, float] = {}
    weight_records: dict[pd.Timestamp, pd.Series] = {}

    held = np.zeros(len(symbols), dtype=np.float64)
    last_position = int(store.dates.searchsorted(end, side="right")) - 1

    for k, position in enumerate(positions):
        target_series = weight_fn(position)
        target = np.zeros(len(symbols), dtype=np.float64)
        for symbol, weight in target_series.items():
            slot = index.get(str(symbol))
            if slot is not None:
                target[slot] = float(weight)
        total = target.sum()
        if total > 0:
            target /= total

        traded = float(np.abs(target - held).sum())
        cost = cost_rate * traded
        turnover_records[store.dates[position]] = traded / 2.0
        weight_records[store.dates[position]] = pd.Series(
            target[target > 0], index=[symbols[i] for i in np.flatnonzero(target > 0)]
        )

        held = target
        stop = positions[k + 1] if k + 1 < len(positions) else last_position

        for day in range(position + 1, stop + 1):
            step = np.nan_to_num(returns_np[day], nan=0.0)
            gross_return = float(held @ step)
            charge = cost if day == position + 1 else 0.0
            daily_gross.append(gross_return)
            daily.append(gross_return - charge)
            dates.append(store.dates[day])

            grown = held * (1.0 + step)
            total = grown.sum()
            held = grown / total if total > 0 else held

        if progress:
            progress(k + 1, len(positions))

    series = pd.Series(daily, index=pd.DatetimeIndex(dates), name=name)
    gross = pd.Series(daily_gross, index=pd.DatetimeIndex(dates), name=f"{name}_gross")
    weights = pd.DataFrame(weight_records).T.fillna(0.0)
    turnover = pd.Series(turnover_records, name="turnover")

    stats = performance_summary(series, config.backtest.risk_free)
    stats["avg_turnover"] = float(turnover.mean())
    stats["annual_turnover"] = float(
        turnover.mean() * (252 / max(config.backtest.rebalance_days, 1))
    )
    stats["cost_drag"] = float((gross.mean() - series.mean()) * 252)
    stats["n_rebalances"] = len(positions)
    stats["avg_names"] = float((weights > 0).sum(axis=1).mean()) if not weights.empty else 0.0

    log.info(
        "backtest_done",
        strategy=name,
        sharpe=round(stats["sharpe"], 3),
        cagr=round(stats["cagr"], 4),
        max_dd=round(stats["max_drawdown"], 4),
        turnover=round(stats["avg_turnover"], 3),
    )
    return BacktestResult(
        name=name,
        returns=series,
        gross_returns=gross,
        weights=weights,
        turnover=turnover,
        stats=stats,
    )


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    """Tabulate a set of backtests, best Sharpe first."""
    frame = pd.DataFrame([r.summary_row() for r in results]).set_index("strategy")
    return frame.sort_values("sharpe", ascending=False)


def returns_matrix(results: list[BacktestResult]) -> pd.DataFrame:
    """Aligned daily returns of every strategy, for PBO and correlation analysis."""
    return pd.concat({r.name: r.returns for r in results}, axis=1).dropna(how="all")
