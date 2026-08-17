"""Business rules as token masks.

Every rule a desk actually has -- do not hold the same name twice, do not put more
than four technology names on one page, do not trade something you cannot get out
of, do not show a commodity fund to a long-only equity mandate, always include a
hedge row -- is expressible as "these token ids are illegal at this step".

The engine is batched: it tracks one constraint state per sampled sequence so a
whole GRPO group can be generated in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from gendesk.config import DecodeConfig, PersonaConfig
from gendesk.corpus.rows import ARCHETYPES, HEDGE_GROUPS, archetype_order
from gendesk.data.universe import FUND_SECTOR, Catalog
from gendesk.tokenization.vocab import Vocab


@dataclass
class ConstraintReport:
    """Which rules bound during a generation, for the audit trail in the UI."""

    duplicates_blocked: int = 0
    sector_blocked: int = 0
    liquidity_blocked: int = 0
    mandate_blocked: int = 0
    rows_forced: int = 0
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "duplicates_blocked": self.duplicates_blocked,
            "sector_blocked": self.sector_blocked,
            "liquidity_blocked": self.liquidity_blocked,
            "mandate_blocked": self.mandate_blocked,
            "rows_forced": self.rows_forced,
            **self.details,
        }


class ConstraintEngine:
    """Batched constraint tracker over a fixed catalog.

    One instance serves one (date, mandate) cell and ``batch`` parallel sequences.
    """

    def __init__(
        self,
        catalog: Catalog,
        vocab: Vocab,
        config: DecodeConfig,
        persona: PersonaConfig,
        available: np.ndarray,
        batch: int,
        device: torch.device,
    ) -> None:
        self.catalog = catalog
        self.vocab = vocab
        self.config = config
        self.persona = persona
        self.batch = batch
        self.device = device
        self.report = ConstraintReport()

        n = len(catalog)
        self.sectors = sorted({inst.sector for inst in catalog})
        sector_of = np.array([self.sectors.index(inst.sector) for inst in catalog])
        self.sector_of = torch.as_tensor(sector_of, device=device, dtype=torch.long)
        self.fund_sector = self.sectors.index(FUND_SECTOR) if FUND_SECTOR in self.sectors else -1

        base = torch.as_tensor(available.astype(bool), device=device)
        if config.enforce_liquidity is False:
            base = torch.ones_like(base)
        if persona.excluded_assets:
            excluded = torch.as_tensor(
                np.isin(np.asarray(catalog.symbols), np.asarray(persona.excluded_assets)),
                device=device,
            )
            self.report.mandate_blocked = int(excluded.sum())
            base = base & ~excluded
        self.base = base.unsqueeze(0).expand(batch, n).clone()

        # Structural eligibility per archetype (hedge funds reserved for hedge rows).
        self.archetypes = archetype_order()
        eligible = np.zeros((len(self.archetypes), n), dtype=bool)
        for a, name in enumerate(self.archetypes):
            arch = ARCHETYPES[name]
            for i, inst in enumerate(catalog):
                if arch.hedge_only:
                    eligible[a, i] = inst.is_fund and inst.group in HEDGE_GROUPS
                elif inst.is_fund and inst.group in HEDGE_GROUPS:
                    eligible[a, i] = False
                else:
                    eligible[a, i] = arch.allow_funds or not inst.is_fund
        self.eligible = torch.as_tensor(eligible, device=device)

        self.used = torch.zeros((batch, n), dtype=torch.bool, device=device)
        self.sector_counts = torch.zeros(
            (batch, len(self.sectors)), dtype=torch.long, device=device
        )
        self.rows_emitted: list[list[str]] = [[] for _ in range(batch)]

    # -- entity masking ------------------------------------------------------

    def entity_mask(self, archetype_ids: Tensor) -> Tensor:
        """``(batch, n_instruments)`` boolean mask of legal instruments.

        Args:
            archetype_ids: ``(batch,)`` index into the archetype registry for the row
                currently being written.
        """
        mask = self.base & self.eligible[archetype_ids]

        if self.config.enforce_dedup:
            blocked = (mask & self.used).sum()
            self.report.duplicates_blocked += int(blocked)
            mask = mask & ~self.used

        if self.config.enforce_sector_cap and self.config.max_names_per_sector > 0:
            capped = self.sector_counts >= self.config.max_names_per_sector
            if self.fund_sector >= 0:
                capped[:, self.fund_sector] = False
            per_instrument = capped.gather(1, self.sector_of.unsqueeze(0).expand(self.batch, -1))
            self.report.sector_blocked += int((mask & per_instrument).sum())
            mask = mask & ~per_instrument

        # Never hand back an all-false row: falling back to "any available, unused"
        # keeps generation alive, and the fallback is counted in the report.
        empty = ~mask.any(dim=1)
        if bool(empty.any()):
            fallback = self.base & ~self.used
            mask = torch.where(empty.unsqueeze(1), fallback, mask)
            self.report.details["fallbacks"] = self.report.details.get("fallbacks", 0) + int(
                empty.sum()
            )
        return mask

    def commit(self, entity_index: Tensor) -> None:
        """Record that each sequence selected ``entity_index`` (``(batch,)``)."""
        rows = torch.arange(self.batch, device=self.device)
        self.used[rows, entity_index] = True
        sectors = self.sector_of[entity_index]
        self.sector_counts[rows, sectors] += 1

    # -- row masking ---------------------------------------------------------

    def row_mask(self, rows_remaining: int) -> Tensor:
        """``(batch, n_archetypes)`` mask of legal row archetypes.

        Implements GenPage's row pinning: a mandate's pinned rows are forced as soon
        as the number of remaining slots equals the number of pinned rows still
        missing, so a compliant page is guaranteed without ever rejecting a sample.
        """
        allowed = torch.zeros(
            (self.batch, len(self.archetypes)), dtype=torch.bool, device=self.device
        )
        for b in range(self.batch):
            emitted = self.rows_emitted[b]
            missing = [r for r in self.persona.pinned_rows if r not in emitted]

            if self.config.enforce_row_pinning and len(missing) >= rows_remaining:
                choices = missing[:rows_remaining] or list(self.persona.allowed_rows)
                self.report.rows_forced += 1
            else:
                choices = [r for r in self.persona.allowed_rows if r not in emitted]
                if not choices:
                    choices = list(self.persona.allowed_rows)

            for name in choices:
                if name in ARCHETYPES:
                    allowed[b, self.archetypes.index(name)] = True
            if not bool(allowed[b].any()):
                allowed[b] = True
        return allowed

    def commit_row(self, archetype_ids: Tensor) -> None:
        for b, a in enumerate(archetype_ids.tolist()):
            self.rows_emitted[b].append(self.archetypes[int(a)])

    # -- verification --------------------------------------------------------

    def verify(self, symbols_per_sequence: list[list[str]]) -> list[dict]:
        """Post-hoc audit. Should always pass; used as a runtime assertion in tests."""
        by_symbol = self.catalog.by_symbol
        out = []
        for symbols in symbols_per_sequence:
            counts: dict[str, int] = {}
            for symbol in symbols:
                sector = by_symbol[symbol].sector
                if sector != FUND_SECTOR:
                    counts[sector] = counts.get(sector, 0) + 1
            out.append(
                {
                    "duplicates": len(symbols) - len(set(symbols)),
                    "max_sector": max(counts.values()) if counts else 0,
                    "excluded_hits": sum(s in set(self.persona.excluded_assets) for s in symbols),
                }
            )
        return out
