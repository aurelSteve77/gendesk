"""Constrained decoding.

The claim being tested is strong: a generated page is compliant *by construction*,
not by filtering afterwards. So the tests generate many pages from an untrained model
-- whose preferences are essentially random, which is the hardest case for the mask
-- and assert that no rule is ever violated.
"""

from __future__ import annotations

import torch

from gendesk.config import Config
from gendesk.data.universe import FUND_SECTOR
from gendesk.decoding.generate import PageGenerator
from gendesk.features.regimes import REGIME_AXES
from gendesk.features.store import FeatureStore
from gendesk.model.gendesk import GenDeskModel
from gendesk.tokenization.page import PageContext
from gendesk.tokenization.vocab import Vocab


def _generator(config: Config, store: FeatureStore, vocab: Vocab) -> PageGenerator:
    torch.manual_seed(0)
    model = GenDeskModel(config.model, vocab, store.n_features).eval()
    return PageGenerator(model, vocab, store, config, torch.device("cpu"))


def _context(config: Config, store: FeatureStore, position: int, persona) -> PageContext:
    return PageContext(
        persona=persona.name,
        risk_budget=persona.risk_budget,
        horizon_days=persona.horizon_days,
        regimes={axis: int(store.regimes[axis].iloc[position]) for axis in REGIME_AXES},
        history=(),
    )


def test_generated_pages_obey_every_rule(config: Config, store: FeatureStore, vocab: Vocab) -> None:
    generator = _generator(config, store, vocab)
    position = len(store.dates) - 5
    by_symbol = store.catalog.by_symbol

    for persona in config.personas:
        result = generator.generate(
            _context(config, store, position, persona),
            persona,
            position,
            n_samples=12,
            temperature=1.0,
            hybrid=False,
        )
        for page in result.pages:
            symbols = list(page.symbols)

            assert len(symbols) == len(set(symbols)), "duplicate instrument on a page"

            counts: dict[str, int] = {}
            for symbol in symbols:
                sector = by_symbol[symbol].sector
                if sector != FUND_SECTOR:
                    counts[sector] = counts.get(sector, 0) + 1
            assert max(counts.values(), default=0) <= config.decode.max_names_per_sector

            assert not set(symbols) & set(persona.excluded_assets)

            available = store.available.iloc[position]
            assert all(bool(available[s]) for s in symbols)

            for pinned in persona.pinned_rows:
                assert pinned in page.archetypes, f"{pinned} was not pinned onto the page"

            assert set(page.archetypes) <= set(persona.allowed_rows)


def test_hedge_row_only_holds_macro_funds(
    config: Config, store: FeatureStore, vocab: Vocab
) -> None:
    """A macro-hedge row must not quietly fill with equities."""
    generator = _generator(config, store, vocab)
    position = len(store.dates) - 5
    persona = config.personas[0]
    by_symbol = store.catalog.by_symbol

    result = generator.generate(
        _context(config, store, position, persona), persona, position, n_samples=8, temperature=1.0
    )
    seen = False
    for page in result.pages:
        for row in page.rows:
            if row.archetype != "MACRO_HEDGE":
                continue
            seen = True
            for symbol in row.symbols:
                instrument = by_symbol[symbol]
                assert instrument.is_fund and instrument.is_hedge_candidate
    assert seen, "the pinned hedge row never appeared"


def test_equity_rows_never_hold_macro_funds(
    config: Config, store: FeatureStore, vocab: Vocab
) -> None:
    generator = _generator(config, store, vocab)
    position = len(store.dates) - 5
    persona = config.personas[1]
    by_symbol = store.catalog.by_symbol

    result = generator.generate(
        _context(config, store, position, persona), persona, position, n_samples=8, temperature=1.0
    )
    for page in result.pages:
        for row in page.rows:
            if row.archetype == "MACRO_HEDGE":
                continue
            assert not any(by_symbol[s].is_hedge_candidate for s in row.symbols)


def test_hybrid_and_autoregressive_produce_the_same_shape(
    config: Config, store: FeatureStore, vocab: Vocab
) -> None:
    generator = _generator(config, store, vocab)
    position = len(store.dates) - 5
    persona = config.personas[1]
    context = _context(config, store, position, persona)

    auto = generator.generate(context, persona, position, n_samples=4, hybrid=False)
    hybrid = generator.generate(context, persona, position, n_samples=4, hybrid=True)

    assert auto.tokens.shape == hybrid.tokens.shape
    assert auto.step_tokens.shape == hybrid.step_tokens.shape
    # Hybrid decoding is only worth doing if it removes sequential calls.
    assert hybrid.model_calls < auto.model_calls


def test_step_masks_never_permit_an_illegal_token(
    config: Config, store: FeatureStore, vocab: Vocab
) -> None:
    """Every sampled token must have been legal under its own recorded mask."""
    generator = _generator(config, store, vocab)
    position = len(store.dates) - 5
    persona = config.personas[0]

    result = generator.generate(
        _context(config, store, position, persona), persona, position, n_samples=6, hybrid=False
    )
    chosen = result.step_tokens.unsqueeze(-1)
    legal = torch.gather(result.step_masks, 2, chosen).squeeze(-1)
    assert bool(legal.all())


def test_disabling_dedup_is_actually_a_choice(
    config: Config, store: FeatureStore, vocab: Vocab
) -> None:
    """The constraint switches must have an observable effect, or they are decoration."""
    relaxed = config.model_copy(
        update={
            "decode": config.decode.model_copy(
                update={"enforce_dedup": False, "enforce_sector_cap": False}
            )
        }
    )
    torch.manual_seed(0)
    model = GenDeskModel(config.model, vocab, store.n_features).eval()
    generator = PageGenerator(model, vocab, store, relaxed, torch.device("cpu"))

    position = len(store.dates) - 5
    persona = config.personas[1]
    result = generator.generate(
        _context(config, store, position, persona),
        persona,
        position,
        n_samples=24,
        temperature=1.5,
        hybrid=False,
    )
    duplicates = [len(p.symbols) - len(set(p.symbols)) for p in result.pages]
    assert max(duplicates) > 0, "relaxing dedup changed nothing, so it was never binding"
