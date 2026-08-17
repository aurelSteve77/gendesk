"""The reward model.

Netflix's reward is member satisfaction. The desk analogue is a *risk-adjusted,
benchmark-relative, cost-aware* outcome over the mandate's horizon. Three properties
matter for this to be a usable RL signal:

* **Benchmark-relative.** The market's own return is common to every candidate page,
  so leaving it in would drown the differences between pages in beta noise.
* **Risk-adjusted.** Active return is divided by the volatility budget, so a page
  cannot win by simply being more levered in disguise.
* **Penalised for path and turnover.** Drawdown inside the window and trading
  against the previous page both cost, which is what stops the policy from
  oscillating between equally attractive pages.

The same construction supplies per-slot rewards for weighted binary classification
post-training, so the two objectives are consistent by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gendesk.config import CorpusConfig, PersonaConfig
from gendesk.features.store import FeatureStore
from gendesk.portfolio.weights import page_weights, turnover
from gendesk.tokenization.page import Page

TRADING_DAYS = 252


@dataclass(frozen=True)
class PageReward:
    """Decomposition of a page's realised reward."""

    total: float
    active_return: float
    realised_vol: float
    max_drawdown: float
    turnover: float
    #: Effective number of independent bets, ``1 / sum(w^2)`` after correlation
    #: adjustment. Reported, never optimised -- see the emergence study.
    effective_bets: float
    components: dict[str, float] = field(default_factory=dict)


def _path_stats(port: np.ndarray, bench: np.ndarray) -> tuple[float, float, float]:
    """Total active return, annualised active volatility and max drawdown."""
    if port.size == 0:
        return 0.0, 0.0, 0.0
    total = float(np.expm1(np.log1p(port).sum()))
    bench_total = float(np.expm1(np.log1p(bench).sum()))
    active = total - bench_total

    active_daily = port - bench
    vol = float(np.std(active_daily) * np.sqrt(TRADING_DAYS)) if active_daily.size > 1 else 0.0

    equity = np.cumprod(1.0 + port)
    drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1.0))
    return active, vol, drawdown


def evaluate_page(
    page: Page,
    store: FeatureStore,
    position: int,
    persona: PersonaConfig,
    config: CorpusConfig,
    previous_weights: pd.Series | None = None,
    horizon: int | None = None,
) -> PageReward:
    """Score a page by what it actually earned over the forward window.

    Args:
        position: Index of the page's own date. All forward data is taken strictly
            after this index, which is what makes the reward a label rather than a
            feature.
    """
    horizon = horizon or config.reward_horizon
    weights = page_weights(page, store, position, persona)
    if weights.empty:
        return PageReward(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    path = store.returns.iloc[position + 1 : position + 1 + horizon]
    if path.empty:
        return PageReward(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    cols = [s for s in weights.index if s in path.columns]
    aligned = weights[cols].to_numpy(dtype=np.float64)
    aligned = aligned / aligned.sum() if aligned.sum() > 0 else aligned
    port = path[cols].to_numpy(dtype=np.float64) @ aligned
    bench = path[store_benchmark(store)].to_numpy(dtype=np.float64)

    active, vol, drawdown = _path_stats(port, bench)

    # Scale the horizon's active return into "volatility budget" units so pages with
    # different horizons and different market states are directly comparable.
    scale = config.vol_target * np.sqrt(horizon / TRADING_DAYS)
    reward = active / scale
    dd_cost = config.drawdown_penalty * abs(min(drawdown, 0.0)) / scale
    trn = turnover(previous_weights, weights)
    trn_cost = config.turnover_penalty * persona.turnover_penalty * trn

    effective = float(1.0 / np.sum(aligned**2)) if np.sum(aligned**2) > 0 else 0.0
    total = reward - dd_cost - trn_cost

    return PageReward(
        total=float(np.clip(total, -8.0, 8.0)),
        active_return=float(active),
        realised_vol=float(vol),
        max_drawdown=float(drawdown),
        turnover=float(trn),
        effective_bets=effective,
        components={
            "risk_adjusted_active": float(reward),
            "drawdown_cost": float(dd_cost),
            "turnover_cost": float(trn_cost),
        },
    )


def store_benchmark(store: FeatureStore) -> str:
    """Benchmark symbol available in the store's return frame."""
    for candidate in ("SPY", "IVV", "VOO"):
        if candidate in store.returns.columns:
            return candidate
    return str(store.returns.columns[0])


def slot_rewards(
    store: FeatureStore,
    position: int,
    horizon: int,
    clip: float = 3.0,
) -> np.ndarray:
    """Per-instrument reward used by weighted binary classification.

    Returns a vector over the catalog of *risk-scaled forward active returns*:
    the instrument's return over the window, minus the benchmark's, divided by its
    own volatility. Dividing by volatility is what stops the objective from simply
    learning "prefer high-beta names", which is the financial analogue of a
    recommender learning to prefer whatever is most popular.
    """
    forward = store.forward_return(position, horizon)
    bench_idx = store.symbols.index(store_benchmark(store))
    active = forward - forward[bench_idx]

    vol = store.vol.iloc[position].to_numpy(dtype=np.float64)
    vol = np.where(np.isfinite(vol) & (vol > 0.0), vol, np.nan)
    median_vol = float(np.nanmedian(vol)) if np.isfinite(vol).any() else 0.2
    vol = np.nan_to_num(vol, nan=median_vol)
    scale = np.maximum(vol, 0.05) * np.sqrt(horizon / TRADING_DAYS)

    scaled = np.asarray(active, dtype=np.float64) / scale
    available = store.available.iloc[position].to_numpy()
    scaled = np.where(available, scaled, 0.0)
    return np.clip(scaled, -clip, clip).astype(np.float32)
