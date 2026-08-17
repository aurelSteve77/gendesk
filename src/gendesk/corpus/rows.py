"""Row archetypes: the theses a desk page is built out of.

Each archetype is (a) a linear scoring rule over the standardised feature tensor,
(b) an eligibility rule over asset classes, and (c) a prior over macro regimes.

Two things to note about the role these play. First, they define the *teacher* that
writes the pretraining corpus -- the model is not restricted to them at generation
time and RL is free to populate a row with names the teacher would never have
picked. Second, the coefficients are deliberately textbook: the point of the project
is the generative page-construction layer, so the underlying signals are held to
well-documented, non-exotic risk premia to keep the comparison honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gendesk.data.universe import Catalog
from gendesk.features.cross_section import FEATURE_NAMES

#: Fund families that behave like macro hedges rather than equity exposure.
HEDGE_GROUPS = frozenset({"rates_credit", "real_assets", "currency_intl"})


@dataclass(frozen=True)
class RowArchetype:
    """A named thesis with a scoring rule and an eligibility rule."""

    name: str
    title: str
    thesis: str
    #: Feature name -> coefficient, applied to cross-sectional z-scores.
    weights: dict[str, float]
    #: Restrict the row to macro-hedge funds (True) or exclude them (False).
    hedge_only: bool = False
    allow_funds: bool = True
    #: Regime axis -> bucket -> multiplier on the probability the teacher selects
    #: this row. Unspecified buckets default to 1.0.
    regime_affinity: dict[str, dict[int, float]] = field(default_factory=dict)

    def coefficient_vector(self) -> np.ndarray:
        vec = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        for feature, coef in self.weights.items():
            vec[FEATURE_NAMES.index(feature)] = coef
        return vec

    def affinity(self, regimes: dict[str, int]) -> float:
        """Multiplicative prior on this row given the current regime."""
        score = 1.0
        for axis, table in self.regime_affinity.items():
            if axis in regimes:
                score *= table.get(int(regimes[axis]), 1.0)
        return score


ARCHETYPES: dict[str, RowArchetype] = {
    "MOMENTUM_LEADERS": RowArchetype(
        name="MOMENTUM_LEADERS",
        title="Momentum Leaders",
        thesis=(
            "Names the market has persistently paid for over the last year, "
            "excluding the most recent month to avoid short-term reversal."
        ),
        weights={
            "mom_12_1": 0.50,
            "mom_6_1": 0.30,
            "sharpe_126": 0.25,
            "downside_ratio": -0.15,
        },
        allow_funds=True,
        regime_affinity={
            "market_trend": {0: 0.5, 2: 1.6},
            "vol_level": {0: 1.3, 2: 0.6},
            "breadth": {0: 0.8, 2: 1.2},
        },
    ),
    "TREND_BREAKOUT": RowArchetype(
        name="TREND_BREAKOUT",
        title="Trend & Breakout",
        thesis=(
            "Instruments trading at the top of their 52-week range with a positive "
            "long-term slope: the classic time-series momentum expression."
        ),
        weights={
            "dist_52w_high": 0.45,
            "trend_200": 0.45,
            "mom_3m": 0.20,
            "vol_ratio": -0.10,
        },
        regime_affinity={
            "market_trend": {0: 0.5, 2: 1.5},
            "breadth": {0: 0.7, 2: 1.4},
        },
    ),
    "QUALITY_BALLAST": RowArchetype(
        name="QUALITY_BALLAST",
        title="Quality Ballast",
        thesis=(
            "Low-volatility, low-beta compounders with shallow drawdowns: the part "
            "of the page that is supposed to still be there after a shock."
        ),
        weights={
            "vol_63": -0.45,
            "beta": -0.25,
            "sharpe_126": 0.30,
            "drawdown_126": 0.20,
            "downside_ratio": -0.20,
        },
        regime_affinity={
            "vol_level": {0: 0.6, 2: 1.8},
            "market_trend": {0: 1.6, 2: 0.7},
            "vol_shock": {2: 1.5},
        },
    ),
    "MEAN_REVERSION": RowArchetype(
        name="MEAN_REVERSION",
        title="Short-Horizon Reversal",
        thesis=(
            "Recent losers inside an intact long-term uptrend: a one-week pullback "
            "in a name the market still likes."
        ),
        weights={
            "rev_5d": 0.60,
            "mom_1m": -0.35,
            "trend_200": 0.20,
            "idio_vol": 0.10,
        },
        regime_affinity={
            "vol_level": {2: 1.3},
            "market_trend": {1: 1.3},
            "dispersion": {2: 1.2},
        },
    ),
    "DISPERSION_HARVEST": RowArchetype(
        name="DISPERSION_HARVEST",
        title="Dispersion Harvest",
        thesis=(
            "High idiosyncratic volatility with low benchmark correlation: names "
            "whose outcome is about themselves, not about the index."
        ),
        weights={
            "idio_vol": 0.60,
            "corr_bench": -0.35,
            "mom_1m": 0.15,
        },
        regime_affinity={
            "dispersion": {0: 0.5, 2: 1.8},
            "implied_corr": {0: 1.5, 2: 0.6},
        },
    ),
    "HIGH_BETA_RISK_ON": RowArchetype(
        name="HIGH_BETA_RISK_ON",
        title="High-Beta Risk-On",
        thesis="Deliberate beta expression for mandates that want the market amplified.",
        weights={
            "beta": 0.55,
            "mom_3m": 0.30,
            "vol_63": 0.20,
        },
        regime_affinity={
            "market_trend": {0: 0.3, 2: 1.6},
            "vol_level": {0: 1.5, 2: 0.4},
        },
    ),
    "CROWDING_UNWIND": RowArchetype(
        name="CROWDING_UNWIND",
        title="Crowding Unwind",
        thesis=(
            "The other side of the momentum trade: last year's leaders that have "
            "already broken, bought after the unwind rather than into it."
        ),
        weights={
            "mom_12_1": -0.50,
            "rev_5d": 0.40,
            "drawdown_126": -0.25,
            "corr_bench": -0.15,
        },
        regime_affinity={
            "implied_corr": {2: 1.5},
            "vol_shock": {2: 1.4},
            "market_trend": {0: 1.3},
        },
    ),
    "MACRO_HEDGE": RowArchetype(
        name="MACRO_HEDGE",
        title="Macro Hedge",
        thesis=(
            "Duration, gold, the dollar and credit: instruments held because they "
            "are not equities, sized against the rest of the page."
        ),
        weights={
            "corr_bench": -0.50,
            "mom_3m": 0.30,
            "trend_200": 0.20,
        },
        hedge_only=True,
        regime_affinity={
            "vol_level": {0: 0.7, 2: 1.8},
            "curve_slope": {0: 1.3},
            "rate_impulse": {0: 1.3, 2: 0.8},
            "market_trend": {0: 1.5, 2: 0.7},
        },
    ),
}


def eligibility_matrix(catalog: Catalog) -> np.ndarray:
    """``(n_archetypes, n_instruments)`` boolean matrix of structural eligibility."""
    names = list(ARCHETYPES)
    matrix = np.zeros((len(names), len(catalog)), dtype=bool)
    for a, name in enumerate(names):
        arch = ARCHETYPES[name]
        for i, inst in enumerate(catalog):
            if arch.hedge_only:
                matrix[a, i] = inst.is_fund and inst.group in HEDGE_GROUPS
            elif inst.is_fund and inst.group in HEDGE_GROUPS:
                # Macro funds are reserved for the hedge row; letting them into an
                # equity row would let the teacher smuggle in duration exposure.
                matrix[a, i] = False
            elif inst.is_fund and not arch.allow_funds:
                matrix[a, i] = False
            else:
                matrix[a, i] = True
    return matrix


def archetype_scores(features: np.ndarray) -> np.ndarray:
    """Score every instrument under every archetype.

    Args:
        features: ``(n_instruments, n_features)`` cross-sectional z-scores for one date.

    Returns:
        ``(n_archetypes, n_instruments)`` scores in archetype-registry order.
    """
    coefs = np.stack([ARCHETYPES[name].coefficient_vector() for name in ARCHETYPES])
    return coefs @ features.T


def archetype_order() -> tuple[str, ...]:
    """Registry order, used to index the score and eligibility matrices."""
    return tuple(ARCHETYPES)
