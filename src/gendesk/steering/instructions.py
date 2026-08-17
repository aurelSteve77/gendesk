"""Plain-English steering.

GenRec's closing argument is that an LLM-native recommender opens the door to
"natural-language steering": you change what the system produces by *telling it*,
not by retraining it. Because GenDesk's context and constraints are both token-level,
that translates cleanly -- an instruction is compiled into (a) context tokens, which
shift the model's distribution, and (b) constraint masks, which are hard.

The parser is rule-based and deterministic by default. That is a deliberate choice
for a system that allocates capital: a mandate change must be inspectable and
reproducible, so the parser returns the exact set of edits it made, and the UI shows
them before the page is generated. An LLM backend is available for free-form phrasing
and produces the same typed :class:`Instruction` object, which is then applied through
the identical code path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from gendesk.config import Config, DecodeConfig, PersonaConfig
from gendesk.corpus.rows import ARCHETYPES
from gendesk.data.universe import Catalog

#: Phrase -> row archetype. Matched as whole words against a lower-cased instruction.
ROW_SYNONYMS: dict[str, str] = {
    "momentum": "MOMENTUM_LEADERS",
    "winners": "MOMENTUM_LEADERS",
    "trend": "TREND_BREAKOUT",
    "breakout": "TREND_BREAKOUT",
    "quality": "QUALITY_BALLAST",
    "defensive": "QUALITY_BALLAST",
    "defensives": "QUALITY_BALLAST",
    "ballast": "QUALITY_BALLAST",
    "low volatility": "QUALITY_BALLAST",
    "reversal": "MEAN_REVERSION",
    "pullback": "MEAN_REVERSION",
    "oversold": "MEAN_REVERSION",
    "dispersion": "DISPERSION_HARVEST",
    "idiosyncratic": "DISPERSION_HARVEST",
    "stock picking": "DISPERSION_HARVEST",
    "high beta": "HIGH_BETA_RISK_ON",
    "risk on": "HIGH_BETA_RISK_ON",
    "crowding": "CROWDING_UNWIND",
    "contrarian": "CROWDING_UNWIND",
    "hedge": "MACRO_HEDGE",
    "hedges": "MACRO_HEDGE",
    "duration": "MACRO_HEDGE",
    "gold": "MACRO_HEDGE",
    "protection": "MACRO_HEDGE",
}

#: Sector names as they appear in the catalog, keyed by the words people use.
SECTOR_SYNONYMS: dict[str, str] = {
    "tech": "Information Technology",
    "technology": "Information Technology",
    "software": "Information Technology",
    "semis": "Information Technology",
    "health": "Health Care",
    "healthcare": "Health Care",
    "pharma": "Health Care",
    "banks": "Financials",
    "financials": "Financials",
    "energy": "Energy",
    "oil": "Energy",
    "utilities": "Utilities",
    "staples": "Consumer Staples",
    "discretionary": "Consumer Discretionary",
    "retail": "Consumer Discretionary",
    "industrials": "Industrials",
    "materials": "Materials",
    "real estate": "Real Estate",
    "reits": "Real Estate",
    "communication": "Communication Services",
    "media": "Communication Services",
}

RISK_UP = ("aggressive", "risk on", "risk-on", "more risk", "maximise", "maximize")
RISK_DOWN = ("defensive", "cautious", "de-risk", "derisk", "reduce risk", "conservative", "safe")


@dataclass
class Instruction:
    """A parsed, inspectable set of edits to a mandate."""

    text: str
    pin_rows: list[str] = field(default_factory=list)
    ban_rows: list[str] = field(default_factory=list)
    exclude_sectors: list[str] = field(default_factory=list)
    exclude_symbols: list[str] = field(default_factory=list)
    risk_budget: str | None = None
    max_names_per_sector: int | None = None
    unmatched: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(
            [
                self.pin_rows,
                self.ban_rows,
                self.exclude_sectors,
                self.exclude_symbols,
                self.risk_budget,
                self.max_names_per_sector,
            ]
        )

    def describe(self) -> list[str]:
        """Human-readable summary of what will change, shown before generation."""
        out: list[str] = []
        for row in self.pin_rows:
            out.append(f"pin row {ARCHETYPES[row].title}")
        for row in self.ban_rows:
            out.append(f"remove row {ARCHETYPES[row].title}")
        for sector in self.exclude_sectors:
            out.append(f"exclude sector {sector}")
        if self.exclude_symbols:
            out.append(f"exclude {', '.join(self.exclude_symbols)}")
        if self.risk_budget:
            out.append(f"risk budget -> {self.risk_budget}")
        if self.max_names_per_sector is not None:
            out.append(f"max {self.max_names_per_sector} names per sector")
        return out


_NEGATION = re.compile(r"\b(no|not|without|avoid|less|reduce|cut|drop|remove|ban|exclude)\b")
_CLAUSE_BREAK = re.compile(r"[,;.]|\band\b|\bbut\b|\bwhile\b")


def _negated(text: str, phrase: str) -> bool:
    """True when ``phrase`` is preceded by a negation *inside its own clause*.

    Clause-scoping matters: in "cut energy exposure, be defensive" the negation
    belongs to energy, not to the defensive tilt. The window is therefore cut at the
    nearest preceding clause boundary rather than at a fixed character count.
    """
    for match in re.finditer(rf"\b{re.escape(phrase)}\b", text):
        window = text[: match.start()]
        breaks = list(_CLAUSE_BREAK.finditer(window))
        if breaks:
            window = window[breaks[-1].end() :]
        if _NEGATION.search(window):
            return True
    return False


def parse_instruction(
    text: str, catalog_symbols: tuple[str, ...] = (), sectors: tuple[str, ...] = ()
) -> Instruction:
    """Compile an instruction into typed edits. Never raises on unknown phrasing."""
    lowered = f" {text.lower().strip()} "
    instruction = Instruction(text=text.strip())

    for phrase, archetype in ROW_SYNONYMS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            target = instruction.ban_rows if _negated(lowered, phrase) else instruction.pin_rows
            if archetype not in target:
                target.append(archetype)

    known_sectors = set(sectors) or set(SECTOR_SYNONYMS.values())
    for phrase, sector in SECTOR_SYNONYMS.items():
        if sector not in known_sectors:
            continue
        matched = re.search(rf"\b{re.escape(phrase)}\b", lowered) and _negated(lowered, phrase)
        if matched and sector not in instruction.exclude_sectors:
            instruction.exclude_sectors.append(sector)

    for symbol in catalog_symbols:
        if re.search(rf"\b{re.escape(symbol.lower())}\b", lowered) and _negated(
            lowered, symbol.lower()
        ):
            instruction.exclude_symbols.append(symbol)

    if any(word in lowered for word in RISK_DOWN):
        instruction.risk_budget = "low"
    elif any(word in lowered for word in RISK_UP):
        instruction.risk_budget = "high"

    match = re.search(
        r"(?:max|at most|no more than)\s+(\d+)\s+(?:names?|positions?)\s+per\s+sector", lowered
    )
    if match:
        instruction.max_names_per_sector = int(match.group(1))

    # A row cannot be both pinned and banned; the explicit negation wins.
    instruction.pin_rows = [r for r in instruction.pin_rows if r not in instruction.ban_rows]

    if instruction.is_empty:
        instruction.unmatched = [text.strip()]
    return instruction


def apply_instruction(
    text: str,
    persona: PersonaConfig,
    config: Config,
    instruction: Instruction | None = None,
    catalog: Catalog | None = None,
) -> tuple[PersonaConfig, Config]:
    """Return a mandate and config with the instruction's edits applied.

    Row preferences become context and pinning (soft: the model still chooses what
    goes inside the row); sector and symbol exclusions become masks (hard: those
    tokens cannot be sampled at all).
    """
    catalog_symbols = catalog.symbols if catalog is not None else ()
    sectors = catalog.sectors if catalog is not None else ()
    instruction = instruction or parse_instruction(text, catalog_symbols, sectors)

    allowed = [r for r in persona.allowed_rows if r not in instruction.ban_rows]
    for row in instruction.pin_rows:
        if row not in allowed:
            allowed.insert(0, row)
    pinned = [r for r in persona.pinned_rows if r not in instruction.ban_rows]
    for row in instruction.pin_rows:
        if row not in pinned:
            pinned.append(row)
    pinned = pinned[: config.corpus.n_rows]

    excluded = list(persona.excluded_assets) + instruction.exclude_symbols
    if catalog is not None:
        excluded += sector_exclusions_to_symbols(instruction, catalog.by_symbol)

    updated_persona = persona.model_copy(
        update={
            "allowed_rows": tuple(allowed) or persona.allowed_rows,
            "pinned_rows": tuple(pinned),
            "excluded_assets": tuple(dict.fromkeys(excluded)),
            "risk_budget": instruction.risk_budget or persona.risk_budget,
        }
    )

    decode: DecodeConfig = config.decode
    if instruction.max_names_per_sector is not None:
        decode = decode.model_copy(
            update={"max_names_per_sector": instruction.max_names_per_sector}
        )
    updated_config = config.model_copy(update={"decode": decode})
    return updated_persona, updated_config


def sector_exclusions_to_symbols(instruction: Instruction, catalog_by_symbol: dict) -> list[str]:
    """Expand sector-level exclusions into concrete symbols for the constraint mask."""
    if not instruction.exclude_sectors:
        return []
    banned = set(instruction.exclude_sectors)
    return [sym for sym, inst in catalog_by_symbol.items() if inst.sector in banned]
