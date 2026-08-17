"""The catalog: the fixed set of instruments that can receive an entity token.

The catalog is deliberately static reference data. It defines the *vocabulary* of
the model, so it must be stable across runs -- an instrument's token id is its
position in this list, and a reshuffle would invalidate every checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

from gendesk.utils.paths import CONFIG_DIR

#: Sector label assigned to every pooled fund. Funds are exempt from the
#: single-sector concentration cap because they are already diversified.
FUND_SECTOR = "Fund"


@dataclass(frozen=True, slots=True)
class Instrument:
    """One catalog entity."""

    symbol: str
    sector: str
    asset_class: str
    #: Sub-group for funds ("rates_credit", "sector", ...); empty for single names.
    group: str = ""

    @property
    def is_fund(self) -> bool:
        return self.asset_class == "fund"

    @property
    def is_hedge_candidate(self) -> bool:
        """Funds whose payoff is structurally decoupled from long equity beta."""
        return self.group in {"rates_credit", "real_assets", "currency_intl"}


@dataclass(frozen=True)
class Catalog:
    """An ordered, immutable collection of instruments."""

    instruments: tuple[Instrument, ...]
    benchmark: str
    macro_series: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.instruments)

    def __iter__(self) -> Iterator[Instrument]:
        return iter(self.instruments)

    def __getitem__(self, key: int | str) -> Instrument:
        if isinstance(key, int):
            return self.instruments[key]
        return self.by_symbol[key]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(inst.symbol for inst in self.instruments)

    @property
    def by_symbol(self) -> dict[str, Instrument]:
        return {inst.symbol: inst for inst in self.instruments}

    @property
    def sectors(self) -> tuple[str, ...]:
        """Unique sector labels in a stable order."""
        seen: dict[str, None] = {}
        for inst in self.instruments:
            seen.setdefault(inst.sector, None)
        return tuple(seen)

    def subset(self, symbols: list[str] | tuple[str, ...]) -> Catalog:
        """Return a catalog restricted to ``symbols``, preserving the original order."""
        keep = set(symbols)
        return Catalog(
            instruments=tuple(i for i in self.instruments if i.symbol in keep),
            benchmark=self.benchmark,
            macro_series=self.macro_series,
        )


def load_catalog(path: str | Path | None = None) -> Catalog:
    """Parse ``configs/universe.yaml`` into a :class:`Catalog`.

    Ordering is: equities grouped by sector in file order, then funds grouped by
    fund family in file order. Duplicates are rejected loudly -- a symbol appearing
    twice would silently collapse two token ids into one.
    """
    path = Path(path) if path is not None else CONFIG_DIR / "universe.yaml"
    raw = yaml.safe_load(path.read_text())

    instruments: list[Instrument] = []
    seen: set[str] = set()

    for sector, symbols in (raw.get("equities") or {}).items():
        for symbol in symbols:
            if symbol in seen:
                raise ValueError(f"duplicate symbol in universe file: {symbol}")
            seen.add(symbol)
            instruments.append(Instrument(symbol=symbol, sector=sector, asset_class="equity"))

    for group, symbols in (raw.get("funds") or {}).items():
        for symbol in symbols:
            if symbol in seen:
                raise ValueError(f"duplicate symbol in universe file: {symbol}")
            seen.add(symbol)
            instruments.append(
                Instrument(
                    symbol=symbol,
                    sector=FUND_SECTOR,
                    asset_class="fund",
                    group=group,
                )
            )

    macro = tuple(raw.get("macro_series") or ())
    benchmark = raw.get("benchmark", "SPY")
    if benchmark not in seen:
        raise ValueError(f"benchmark {benchmark!r} is not part of the catalog")

    return Catalog(instruments=tuple(instruments), benchmark=benchmark, macro_series=macro)
