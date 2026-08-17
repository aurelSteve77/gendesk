"""The feature store: one object every downstream stage reads from.

Keeping the tensor, the regime frame, the risk inputs and the availability mask in
a single immutable container removes a whole class of alignment bugs -- there is
exactly one date index and one symbol order in the system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gendesk.config import Config
from gendesk.data.panel import PricePanel, load_panel
from gendesk.data.universe import Catalog
from gendesk.features.cross_section import FEATURE_NAMES, build_feature_tensor
from gendesk.features.regimes import REGIME_AXES, build_regimes
from gendesk.utils.hashing import hash_obj
from gendesk.utils.logging import get_logger
from gendesk.utils.paths import MANIFEST_DIR, PROCESSED_DIR, ensure_dirs

log = get_logger(__name__)


@dataclass(frozen=True)
class FeatureStore:
    """Aligned, point-in-time features for the whole catalog."""

    #: (dates, symbols, features) cross-sectional z-scores.
    values: np.ndarray
    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    regimes: pd.DataFrame
    #: Raw annualised volatility, used for inverse-volatility sizing.
    vol: pd.DataFrame
    #: Raw benchmark beta, used by the hedge row and the risk report.
    beta: pd.DataFrame
    #: Trailing median dollar volume, used by the liquidity constraint.
    adv: pd.DataFrame
    available: pd.DataFrame
    returns: pd.DataFrame
    #: Whether the trailing warm-up window is complete for a given date.
    warm: np.ndarray
    #: Instrument reference data, kept alongside the numerics so that sector caps
    #: and asset-class rules never need a second source of truth.
    catalog: Catalog

    def __post_init__(self) -> None:
        t, n, f = self.values.shape
        if (t, n) != (len(self.dates), len(self.symbols)):
            raise ValueError("feature tensor does not match its index")
        if f != len(self.feature_names):
            raise ValueError("feature tensor does not match its feature names")

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def symbol_index(self) -> dict[str, int]:
        return {sym: i for i, sym in enumerate(self.symbols)}

    def date_position(self, when: pd.Timestamp | str | date) -> int:
        """Index of the last session at or before ``when``."""
        ts = pd.Timestamp(when)
        pos = int(self.dates.searchsorted(ts, side="right")) - 1
        if pos < 0:
            raise KeyError(f"{ts.date()} precedes the sample")
        return pos

    def feature(self, name: str) -> np.ndarray:
        """A (dates, symbols) slice of one named feature."""
        return self.values[:, :, self.feature_names.index(name)]

    def eligible_dates(self, start: date | None = None, end: date | None = None) -> np.ndarray:
        """Positions of dates that are warm and have a usable cross-section."""
        mask = self.warm.copy()
        if start is not None:
            mask &= self.dates >= pd.Timestamp(start)
        if end is not None:
            mask &= self.dates <= pd.Timestamp(end)
        return np.flatnonzero(mask)

    def forward_return(self, position: int, horizon: int) -> np.ndarray:
        """Simple return of every symbol over ``(position, position + horizon]``.

        This is a *label*. It is never exposed to the model as an input; the corpus
        builder uses it to weight training examples and the reward model uses it to
        score sampled pages.
        """
        end = min(position + horizon, len(self.dates) - 1)
        if end <= position:
            return np.zeros(self.n_symbols, dtype=np.float32)
        window = cast(pd.DataFrame, np.log1p(self.returns.iloc[position + 1 : end + 1]))
        return np.expm1(window.sum(axis=0).to_numpy(dtype=np.float32))

    def forward_path(self, position: int, horizon: int) -> np.ndarray:
        """Daily return path of shape ``(horizon, n_symbols)`` after ``position``."""
        end = min(position + horizon, len(self.dates) - 1)
        return self.returns.iloc[position + 1 : end + 1].to_numpy(dtype=np.float32)

    # -- persistence ---------------------------------------------------------

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or PROCESSED_DIR
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "feature_values.npz",
            values=self.values,
            warm=self.warm,
        )
        self.regimes.to_parquet(directory / "regimes.parquet")
        self.vol.to_parquet(directory / "risk_vol.parquet")
        self.beta.to_parquet(directory / "risk_beta.parquet")
        self.adv.to_parquet(directory / "risk_adv.parquet")
        (directory / "feature_meta.json").write_text(
            json.dumps(
                {"feature_names": list(self.feature_names), "symbols": list(self.symbols)},
                indent=2,
            )
        )
        return directory


def load_features(directory: Path | None = None, panel: PricePanel | None = None) -> FeatureStore:
    """Load a previously built feature store."""
    directory = directory or PROCESSED_DIR
    panel = panel or load_panel(directory)

    payload = np.load(directory / "feature_values.npz")
    meta = json.loads((directory / "feature_meta.json").read_text())

    return FeatureStore(
        values=payload["values"],
        dates=pd.DatetimeIndex(panel.adj_close.index),
        symbols=tuple(meta["symbols"]),
        feature_names=tuple(meta["feature_names"]),
        regimes=pd.read_parquet(directory / "regimes.parquet"),
        vol=pd.read_parquet(directory / "risk_vol.parquet"),
        beta=pd.read_parquet(directory / "risk_beta.parquet"),
        adv=pd.read_parquet(directory / "risk_adv.parquet"),
        available=panel.available,
        returns=panel.returns,
        warm=payload["warm"],
        catalog=panel.catalog,
    )


def build_features(
    config: Config, panel: PricePanel | None = None, force: bool = False
) -> FeatureStore:
    """Build (or load from cache) the full feature store."""
    ensure_dirs()
    panel = panel or load_panel()

    manifest_path = MANIFEST_DIR / "features.json"
    fingerprint = hash_obj(
        {
            "symbols": panel.symbols,
            "dates": [str(panel.calendar.min()), str(panel.calendar.max()), len(panel.calendar)],
            "features": config.features.model_dump(mode="json"),
            "regimes": config.regimes.model_dump(mode="json"),
            "feature_names": FEATURE_NAMES,
            "regime_axes": REGIME_AXES,
        }
    )

    if manifest_path.exists() and not force:
        cached = json.loads(manifest_path.read_text())
        if cached.get("fingerprint") == fingerprint:
            log.info("features_cache_hit", fingerprint=fingerprint)
            return load_features(panel=panel)

    log.info("features_build_start", n_symbols=len(panel.symbols), n_dates=len(panel.calendar))
    values, raw = build_feature_tensor(panel, config.features)
    regimes = build_regimes(panel, config.regimes)

    warmup = config.features.min_warmup
    warm = np.zeros(len(panel.calendar), dtype=bool)
    warm[warmup:] = True
    # A date is only usable if enough of the catalog has a complete trailing window.
    coverage = raw["mom_12_1"].notna().sum(axis=1).to_numpy()
    warm &= coverage >= 50

    store = FeatureStore(
        values=values,
        dates=pd.DatetimeIndex(panel.adj_close.index),
        symbols=tuple(panel.symbols),
        feature_names=FEATURE_NAMES,
        regimes=regimes,
        vol=raw["vol_63"],
        beta=raw["beta"],
        adv=cast(pd.DataFrame, np.expm1(raw["turnover"])),
        available=panel.available,
        returns=panel.returns,
        warm=warm,
        catalog=panel.catalog,
    )
    store.save()

    manifest_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "built_at": pd.Timestamp.utcnow().isoformat(),
                "shape": list(values.shape),
                "features": list(FEATURE_NAMES),
                "regime_axes": list(REGIME_AXES),
                "first_warm_date": str(store.dates[int(np.argmax(warm))].date()),
            },
            indent=2,
            default=str,
        )
    )
    log.info(
        "features_built",
        shape=list(values.shape),
        first_warm=str(store.dates[int(np.argmax(warm))].date()),
    )
    return store
