"""Price providers.

Only one provider is implemented (Yahoo's public chart endpoint) but the protocol
is deliberately narrow so an institutional feed can be dropped in without touching
anything downstream: return a daily OHLCV frame indexed by naive dates, with a
split/dividend-adjusted close column.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from typing import ClassVar, Protocol

import httpx
import numpy as np
import pandas as pd

from gendesk.utils.logging import get_logger

log = get_logger(__name__)

_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class PriceProvider(Protocol):
    """Minimal contract every price source must satisfy."""

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:  # pragma: no cover
        """Return a daily OHLCV frame for ``symbol`` over ``[start, end]``."""
        ...


def _to_epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


class YahooChartProvider:
    """Reader for ``query1.finance.yahoo.com/v8/finance/chart``.

    The endpoint is unauthenticated but rate limited, so requests are throttled by
    a shared token-bucket-ish delay and retried with jittered exponential backoff.
    """

    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    HEADERS: ClassVar[dict[str, str]] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 4,
        min_interval: float = 0.12,
    ) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            headers=self.HEADERS,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
        )
        self._max_retries = max_retries
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> YahooChartProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _throttle(self) -> None:
        """Serialise request starts so concurrent workers do not burst."""
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch one symbol. Returns an empty frame if the symbol is unknown."""
        params: dict[str, str] = {
            "period1": str(_to_epoch(start)),
            "period2": str(_to_epoch(end)),
            "interval": "1d",
            "includeAdjustedClose": "true",
            "events": "div|split",
        }
        url = self.BASE.format(symbol=symbol)

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            self._throttle()
            try:
                response = self._client.get(url, params=params)
                if response.status_code in (404, 422):
                    # Yahoo answers 404 both for genuinely unknown symbols and for
                    # throttled requests, so a 404 is only believed on the last
                    # attempt. Retrying costs one request and recovers ~4% of the
                    # catalog that would otherwise be silently dropped.
                    if attempt == self._max_retries - 1:
                        log.warning("symbol_not_found", symbol=symbol, status=response.status_code)
                        return pd.DataFrame(columns=_COLUMNS)
                    raise httpx.HTTPStatusError(
                        f"{symbol}: transient {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return self._parse(symbol, response.json())
            except Exception as exc:
                last_error = exc
                backoff = (2**attempt) * 0.75 + random.random() * 0.5
                log.warning(
                    "fetch_retry",
                    symbol=symbol,
                    attempt=attempt + 1,
                    backoff=round(backoff, 2),
                    error=str(exc)[:160],
                )
                time.sleep(backoff)

        log.error("fetch_failed", symbol=symbol, error=str(last_error)[:200])
        return pd.DataFrame(columns=_COLUMNS)

    @staticmethod
    def _parse(symbol: str, payload: dict) -> pd.DataFrame:
        chart = payload.get("chart") or {}
        if chart.get("error"):
            raise RuntimeError(f"{symbol}: {chart['error']}")
        results = chart.get("result") or []
        if not results:
            return pd.DataFrame(columns=_COLUMNS)

        result = results[0]
        timestamps = result.get("timestamp")
        if not timestamps:
            return pd.DataFrame(columns=_COLUMNS)

        quote = result["indicators"]["quote"][0]
        frame = pd.DataFrame(
            {
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            },
            index=pd.to_datetime(np.asarray(timestamps), unit="s", utc=True),
        )

        adj = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
        frame["adj_close"] = adj if adj is not None else frame["close"]

        # Yahoo returns exchange-local sessions stamped at market open in UTC; the
        # calendar date is all that matters downstream, so normalise to naive dates.
        index = pd.DatetimeIndex(frame.index)
        frame.index = index.tz_convert("America/New_York").normalize().tz_localize(None)
        frame.index.name = "date"

        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame[_COLUMNS].astype("float64")


def fetch_many(
    provider: PriceProvider,
    symbols: list[str] | tuple[str, ...],
    start: date,
    end: date,
    max_workers: int = 8,
) -> dict[str, pd.DataFrame]:
    """Fetch many symbols concurrently, returning only the non-empty results."""
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(provider.fetch, sym, start, end): sym for sym in symbols}
        for done, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                frame = future.result()
            except Exception as exc:
                log.error("fetch_exception", symbol=symbol, error=str(exc)[:200])
                continue
            if not frame.empty:
                out[symbol] = frame
            if done % 50 == 0:
                log.info("fetch_progress", done=done, total=len(futures), kept=len(out))
    return out
