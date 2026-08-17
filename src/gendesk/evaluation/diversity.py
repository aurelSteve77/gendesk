"""Page diversity.

The most striking claim in the GenPage post is that page diversity *rises* during
reinforcement learning even though nothing in the reward asks for it -- evidence
that the model is optimising the page as a whole rather than each slot in isolation.

That claim is testable here, and more sharply than at Netflix, because a portfolio
has a canonical definition of diversity. Three measures are tracked, none of which
appears anywhere in the reward:

* **Mean pairwise correlation** of the page's constituents over a trailing window.
* **Diversification ratio**, ``sum(w_i * sigma_i) / sigma_portfolio`` -- 1.0 for a
  single asset, rising as the book's risk decomposes into independent pieces.
* **Effective number of bets**, the inverse Herfindahl of the weights.

All three are computed from data available at the page's own date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gendesk.config import PersonaConfig
from gendesk.data.universe import FUND_SECTOR
from gendesk.features.store import FeatureStore
from gendesk.portfolio.weights import page_weights
from gendesk.tokenization.page import Page

DEFAULT_WINDOW = 126


@dataclass
class DiversityMetrics:
    mean_correlation: float
    diversification_ratio: float
    effective_bets: float
    n_sectors: int
    n_names: int
    feature_spread: float

    def as_dict(self) -> dict:
        return {
            "mean_correlation": self.mean_correlation,
            "diversification_ratio": self.diversification_ratio,
            "effective_bets": self.effective_bets,
            "n_sectors": self.n_sectors,
            "n_names": self.n_names,
            "feature_spread": self.feature_spread,
        }


def page_diversity(
    page: Page,
    store: FeatureStore,
    position: int,
    persona: PersonaConfig,
    window: int = DEFAULT_WINDOW,
) -> DiversityMetrics:
    """Diversity of a page, measured with trailing data only."""
    symbols = [s for s in dict.fromkeys(page.symbols) if s in store.returns.columns]
    if len(symbols) < 2:
        return DiversityMetrics(1.0, 1.0, float(len(symbols)), len(symbols), len(symbols), 0.0)

    start = max(0, position - window + 1)
    history = store.returns.iloc[start : position + 1][symbols]
    corr = history.corr().to_numpy()
    upper = corr[np.triu_indices_from(corr, k=1)]
    mean_corr = float(np.nanmean(upper)) if upper.size else 1.0

    weights = page_weights(page, store, position, persona)
    weights = weights[[s for s in weights.index if s in symbols]]
    w = weights.to_numpy(dtype=np.float64)
    if w.sum() <= 0:
        return DiversityMetrics(mean_corr, 1.0, 0.0, 0, len(symbols), 0.0)
    w = w / w.sum()

    order = [symbols.index(s) for s in weights.index]
    sub_corr = corr[np.ix_(order, order)]
    sigma = history[list(weights.index)].std().to_numpy(dtype=np.float64) * np.sqrt(252)
    sigma = np.nan_to_num(
        sigma, nan=float(np.nanmedian(sigma)) if np.isfinite(sigma).any() else 0.2
    )

    cov = np.outer(sigma, sigma) * np.nan_to_num(sub_corr, nan=0.0)
    port_vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
    div_ratio = float((w @ sigma) / port_vol) if port_vol > 0 else 1.0

    by_symbol = store.catalog.by_symbol
    sectors = {by_symbol[s].sector for s in symbols if s in by_symbol}
    sectors.discard(FUND_SECTOR)

    index = store.symbol_index
    feats = np.stack([store.values[position, index[s]] for s in symbols if s in index])
    centroid = feats.mean(axis=0)
    spread = float(np.mean(np.linalg.norm(feats - centroid, axis=1)))

    return DiversityMetrics(
        mean_correlation=mean_corr,
        diversification_ratio=div_ratio,
        effective_bets=float(1.0 / np.sum(w**2)),
        n_sectors=len(sectors),
        n_names=len(symbols),
        feature_spread=spread,
    )


def average_diversity(
    pages: list[Page],
    store: FeatureStore,
    position: int,
    persona: PersonaConfig,
    window: int = DEFAULT_WINDOW,
) -> dict:
    """Mean of :func:`page_diversity` over a batch of sampled pages."""
    metrics = [page_diversity(p, store, position, persona, window) for p in pages if p.rows]
    if not metrics:
        return DiversityMetrics(1.0, 1.0, 0.0, 0, 0, 0.0).as_dict()
    keys = metrics[0].as_dict().keys()
    return {k: float(np.mean([getattr(m, k) for m in metrics])) for k in keys}
