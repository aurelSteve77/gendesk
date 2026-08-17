"""Page -> portfolio mapping.

The model chooses *what* is on the page; this module decides *how much*. Keeping
the two apart matters: it means an improvement in the backtest is attributable to
selection rather than to a cleverer optimiser, and it keeps the reward signal that
trains the model consistent with the portfolio that is actually traded.

Sizing is deliberately simple and non-optimising:

1. a risk budget per row, tilted by the mandate's risk appetite,
2. inverse-volatility weights inside each row,
3. a single-name cap, then renormalisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gendesk.config import PersonaConfig
from gendesk.features.store import FeatureStore
from gendesk.tokenization.page import Page

#: Rows whose purpose is protection rather than return. A defensive mandate over-
#: weights them; an aggressive one under-weights them.
DEFENSIVE_ROWS = frozenset({"MACRO_HEDGE", "QUALITY_BALLAST"})

#: Maximum weight of any single instrument, before renormalisation.
MAX_NAME_WEIGHT = 0.12

#: Volatility floor (annualised) so a near-riskless fund cannot absorb the book.
MIN_VOL = 0.03


def row_budgets(page: Page, persona: PersonaConfig) -> np.ndarray:
    """Risk budget per row, summing to one.

    Rows start equally weighted and are then tilted by the mandate: a ``low`` risk
    budget doubles the defensive rows' share, a ``high`` budget halves it.
    """
    tilt = {"low": 2.0, "medium": 1.0, "high": 0.5}[persona.risk_budget]
    raw = np.array(
        [tilt if row.archetype in DEFENSIVE_ROWS else 1.0 for row in page.rows],
        dtype=np.float64,
    )
    if raw.sum() <= 0:
        return np.full(len(page.rows), 1.0 / max(len(page.rows), 1))
    return raw / raw.sum()


def page_weights(
    page: Page,
    store: FeatureStore,
    position: int,
    persona: PersonaConfig,
    max_name_weight: float = MAX_NAME_WEIGHT,
) -> pd.Series:
    """Long-only, fully invested weights for a page, indexed by symbol.

    Volatility is read at ``position`` -- the page's own date -- so the sizing uses
    no information the desk would not have had when it traded.
    """
    if not page.rows:
        return pd.Series(dtype=np.float64)

    vol_row = store.vol.iloc[position]
    budgets = row_budgets(page, persona)
    weights: dict[str, float] = {}

    for budget, row in zip(budgets, page.rows, strict=True):
        symbols = [s for s in row.symbols if s in vol_row.index]
        if not symbols:
            continue
        vols = np.maximum(vol_row[symbols].to_numpy(dtype=np.float64), MIN_VOL)
        vols = np.where(np.isfinite(vols), vols, np.nanmedian(vols[np.isfinite(vols)]) or MIN_VOL)
        inv = 1.0 / vols
        inv = inv / inv.sum()
        for symbol, share in zip(symbols, inv, strict=True):
            # A symbol repeated across rows accumulates weight, which is the correct
            # reading of "the desk likes it for two different reasons".
            weights[symbol] = weights.get(symbol, 0.0) + float(budget * share)

    series = pd.Series(weights, dtype=np.float64)
    if series.empty:
        return series
    return cap_and_renormalise(series, max_name_weight)


def cap_and_renormalise(weights: pd.Series, cap: float, tolerance: float = 1e-12) -> pd.Series:
    """Enforce a single-name cap on a fully invested book.

    Clipping once and renormalising does *not* respect the cap -- renormalisation
    pushes the clipped names straight back above it. This is the standard
    water-filling fix: freeze the names at the cap and redistribute the remaining
    budget among the rest, repeating until nothing breaches.

    The cap is raised to ``1 / n`` when it is infeasible, since a fully invested
    book of ``n`` names cannot hold every name below ``1 / n``.
    """
    total = float(weights.sum())
    if total <= 0:
        return weights
    result = weights / total

    n = len(result)
    effective = max(cap, 1.0 / n)
    values = result.to_numpy(dtype=np.float64).copy()

    # Names that hit the cap stay frozen. Releasing them each pass -- which is what a
    # naive loop does -- makes the iteration oscillate instead of converge.
    frozen = np.zeros(n, dtype=bool)
    for _ in range(n):
        free = ~frozen
        if not free.any():
            break
        budget = 1.0 - effective * float(frozen.sum())
        pool = values[free]
        total_free = pool.sum()
        scaled = (
            pool / total_free * budget if total_free > 0 else np.full(pool.size, budget / pool.size)
        )
        newly = scaled > effective + tolerance
        if not newly.any():
            values[free] = scaled
            break
        frozen[np.flatnonzero(free)[newly]] = True
        values[frozen] = effective

    return pd.Series(values, index=result.index)


def align_weights(weights: pd.Series, symbols: tuple[str, ...]) -> np.ndarray:
    """Project a symbol-indexed weight vector onto the catalog ordering."""
    out = np.zeros(len(symbols), dtype=np.float64)
    index = {sym: i for i, sym in enumerate(symbols)}
    for symbol, weight in weights.items():
        pos = index.get(str(symbol))
        if pos is not None:
            out[pos] = float(weight)
    return out


def turnover(previous: pd.Series | None, current: pd.Series) -> float:
    """One-way turnover between two weight vectors (0 = no trade, 1 = full switch)."""
    if previous is None or previous.empty:
        return float(current.abs().sum())
    combined = previous.reindex(current.index.union(previous.index)).fillna(0.0)
    target = current.reindex(combined.index).fillna(0.0)
    return float((target - combined).abs().sum() / 2.0)
