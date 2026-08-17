"""Construction of the aligned price panel.

Everything downstream reads a :class:`PricePanel`: a set of date x symbol frames
sharing one trading calendar, plus an explicit *availability* mask. The mask is
what keeps the pipeline honest -- an instrument is only eligible on a date if it
was already listed, still listed, and liquid enough to trade at that date.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from gendesk.config import Config
from gendesk.data.providers import YahooChartProvider, fetch_many
from gendesk.data.universe import Catalog, load_catalog
from gendesk.utils.hashing import hash_obj
from gendesk.utils.logging import get_logger
from gendesk.utils.paths import CONFIG_DIR, MANIFEST_DIR, PROCESSED_DIR, ensure_dirs

log = get_logger(__name__)

#: Maximum number of consecutive sessions a stale price may be carried forward.
#: Longer gaps mark the instrument unavailable rather than inventing prices.
MAX_FFILL_DAYS = 3


@dataclass(frozen=True)
class PricePanel:
    """Aligned market data for the catalog."""

    adj_close: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    macro: pd.DataFrame
    #: True where the instrument was listed, priced and liquid on that date.
    available: pd.DataFrame
    catalog: Catalog

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.adj_close.index)

    @property
    def symbols(self) -> list[str]:
        return list(self.adj_close.columns)

    @property
    def returns(self) -> pd.DataFrame:
        """Simple daily total returns, zero-filled where unavailable."""
        rets = self.adj_close.pct_change(fill_method=None)
        return rets.where(self.available, 0.0).fillna(0.0)

    @property
    def dollar_volume(self) -> pd.DataFrame:
        return self.close * self.volume

    def slice(self, start: date | str | None = None, end: date | str | None = None) -> PricePanel:
        """Return a date-restricted view of the panel."""
        sl = slice(pd.Timestamp(start) if start else None, pd.Timestamp(end) if end else None)
        return PricePanel(
            adj_close=self.adj_close.loc[sl],
            close=self.close.loc[sl],
            volume=self.volume.loc[sl],
            macro=self.macro.loc[sl],
            available=self.available.loc[sl],
            catalog=self.catalog,
        )

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or PROCESSED_DIR
        directory.mkdir(parents=True, exist_ok=True)
        self.adj_close.to_parquet(directory / "adj_close.parquet")
        self.close.to_parquet(directory / "close.parquet")
        self.volume.to_parquet(directory / "volume.parquet")
        self.macro.to_parquet(directory / "macro.parquet")
        self.available.to_parquet(directory / "available.parquet")
        return directory


def load_panel(directory: Path | None = None, catalog: Catalog | None = None) -> PricePanel:
    """Load a previously built panel from disk."""
    directory = directory or PROCESSED_DIR
    required = ["adj_close", "close", "volume", "macro", "available"]
    missing = [name for name in required if not (directory / f"{name}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"panel files missing in {directory}: {missing}. Run `gendesk data build` first."
        )

    frames = {name: pd.read_parquet(directory / f"{name}.parquet") for name in required}
    full = catalog or load_catalog()
    return PricePanel(
        adj_close=frames["adj_close"],
        close=frames["close"],
        volume=frames["volume"],
        macro=frames["macro"],
        available=frames["available"].astype(bool),
        catalog=full.subset(list(frames["adj_close"].columns)),
    )


def _align(raw: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, field: str) -> pd.DataFrame:
    """Reindex one field of every symbol onto the shared calendar."""
    cols = {sym: frame[field].reindex(calendar) for sym, frame in raw.items()}
    return pd.DataFrame(cols, index=calendar)


def _availability(
    adj_close: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    window: int,
    min_dollar_volume: float,
) -> pd.DataFrame:
    """Boolean eligibility mask.

    An instrument is available on date ``t`` when (a) it has a price at ``t`` that is
    at most :data:`MAX_FFILL_DAYS` sessions stale, (b) ``t`` lies inside its listed
    life, and (c) its trailing median dollar volume clears the liquidity floor. All
    three tests use information available at ``t`` only.
    """
    priced = adj_close.notna()
    # Carry a price forward for a few sessions (exchange holidays, halts) but no more.
    carried = priced | (priced.rolling(MAX_FFILL_DAYS + 1, min_periods=1).sum() > 0).where(
        priced.cumsum() > 0, False
    )
    listed = (priced.cumsum() > 0) & (priced[::-1].cumsum()[::-1] > 0)

    liquid = (
        dollar_volume.rolling(window, min_periods=max(5, window // 4)).median() >= min_dollar_volume
    )
    return (carried & listed & liquid.fillna(False)).astype(bool)


def build_panel(config: Config, force: bool = False) -> PricePanel:
    """Download, clean, filter and cache the panel.

    Returns the cached panel when the manifest matches the current configuration
    unless ``force`` is set.
    """
    ensure_dirs()
    catalog = load_catalog(CONFIG_DIR / config.data.universe_file)

    manifest_path = MANIFEST_DIR / "panel.json"
    fingerprint = hash_obj(
        {
            "symbols": catalog.symbols,
            "macro": catalog.macro_series,
            "data": config.data.model_dump(mode="json"),
            "dollar_volume_window": config.features.dollar_volume_window,
        }
    )

    if manifest_path.exists() and not force:
        cached = json.loads(manifest_path.read_text())
        if cached.get("fingerprint") == fingerprint:
            log.info("panel_cache_hit", fingerprint=fingerprint, symbols=cached.get("n_symbols"))
            return load_panel(catalog=catalog)

    symbols = list(catalog.symbols) + list(catalog.macro_series)
    log.info("panel_download_start", n_symbols=len(symbols))

    with YahooChartProvider(
        timeout=config.data.request_timeout, max_retries=config.data.max_retries
    ) as provider:
        raw = fetch_many(
            provider,
            symbols,
            config.data.start,
            config.data.end,
            max_workers=config.data.max_workers,
        )

    if catalog.benchmark not in raw:
        raise RuntimeError(
            f"benchmark {catalog.benchmark} could not be downloaded; refusing to build a panel "
            "without a trading calendar anchor"
        )

    # The benchmark defines the trading calendar: no synthetic dates ever enter.
    calendar = pd.DatetimeIndex(raw[catalog.benchmark].index)
    calendar = calendar[
        (calendar >= pd.Timestamp(config.data.start)) & (calendar <= pd.Timestamp(config.data.end))
    ]

    tradables = {s: f for s, f in raw.items() if s in set(catalog.symbols)}
    adj_close = _align(tradables, calendar, "adj_close")
    close = _align(tradables, calendar, "close")
    volume = _align(tradables, calendar, "volume").fillna(0.0)

    macro_raw = {s: f for s, f in raw.items() if s in set(catalog.macro_series)}
    macro = _align(macro_raw, calendar, "adj_close").ffill()

    # --- quality filters -----------------------------------------------------
    n_obs = adj_close.notna().sum()
    life = adj_close.apply(
        lambda s: s.notna().cumsum().gt(0) & s[::-1].notna().cumsum()[::-1].gt(0)
    )
    missing_frac = (life & adj_close.isna()).sum() / life.sum().replace(0, np.nan)

    dollar_volume = close * volume
    median_dv = dollar_volume.median()

    keep = (
        (n_obs >= config.data.min_observations)
        & (missing_frac.fillna(1.0) <= config.data.max_missing_frac)
        & (median_dv.fillna(0.0) >= config.data.min_dollar_volume)
    )
    dropped = sorted(set(adj_close.columns) - set(keep[keep].index))
    log.info(
        "panel_filtered",
        kept=int(keep.sum()),
        dropped=len(dropped),
        dropped_symbols=dropped[:40],
    )

    cols = [c for c in catalog.symbols if c in set(keep[keep].index)]
    adj_close, close, volume = adj_close[cols], close[cols], volume[cols]

    available = _availability(
        adj_close,
        close * volume,
        config.features.dollar_volume_window,
        config.data.min_dollar_volume,
    )

    # Fill short gaps *after* the mask is computed so the mask stays truthful.
    adj_close = adj_close.ffill(limit=MAX_FFILL_DAYS)
    close = close.ffill(limit=MAX_FFILL_DAYS)

    panel = PricePanel(
        adj_close=adj_close,
        close=close,
        volume=volume,
        macro=macro,
        available=available,
        catalog=catalog.subset(cols),
    )
    panel.save()

    manifest = {
        "fingerprint": fingerprint,
        "built_at": pd.Timestamp.now("UTC").isoformat(),
        "n_symbols": len(cols),
        "n_dates": len(calendar),
        "start": str(calendar.min().date()),
        "end": str(calendar.max().date()),
        "dropped_symbols": dropped,
        "macro_series": list(macro.columns),
        "config": config.data.model_dump(mode="json"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    log.info("panel_built", **{k: manifest[k] for k in ("n_symbols", "n_dates", "start", "end")})
    return panel
