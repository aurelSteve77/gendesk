"""Natural-language steering."""

from __future__ import annotations

from gendesk.config import Config
from gendesk.data.universe import Catalog
from gendesk.steering import apply_instruction, parse_instruction


def test_pins_a_row(catalog: Catalog) -> None:
    parsed = parse_instruction("I want more momentum", catalog.symbols, catalog.sectors)
    assert parsed.pin_rows == ["MOMENTUM_LEADERS"]
    assert not parsed.ban_rows


def test_negation_bans_a_row(catalog: Catalog) -> None:
    parsed = parse_instruction("no momentum for now", catalog.symbols, catalog.sectors)
    assert parsed.ban_rows == ["MOMENTUM_LEADERS"]
    assert not parsed.pin_rows


def test_negation_is_scoped_to_its_own_clause(catalog: Catalog) -> None:
    """'cut energy, be defensive' must not read as 'not defensive'."""
    parsed = parse_instruction(
        "cut energy exposure, be defensive", catalog.symbols, catalog.sectors
    )
    assert "QUALITY_BALLAST" in parsed.pin_rows
    assert "Energy" in parsed.exclude_sectors
    assert parsed.risk_budget == "low"


def test_symbol_exclusion(catalog: Catalog) -> None:
    symbol = catalog.symbols[0]
    parsed = parse_instruction(f"avoid {symbol}", catalog.symbols, catalog.sectors)
    assert parsed.exclude_symbols == [symbol]


def test_sector_cap_is_parsed(catalog: Catalog) -> None:
    parsed = parse_instruction(
        "at most 2 names per sector please", catalog.symbols, catalog.sectors
    )
    assert parsed.max_names_per_sector == 2


def test_unmatched_instruction_is_reported_not_guessed(catalog: Catalog) -> None:
    parsed = parse_instruction("do something clever", catalog.symbols, catalog.sectors)
    assert parsed.is_empty
    assert parsed.unmatched == ["do something clever"]


def test_apply_updates_mandate_and_constraints(config: Config, catalog: Catalog) -> None:
    persona = config.personas[1]  # 'pod', no pinned rows, no exclusions
    updated, updated_config = apply_instruction(
        "add duration hedges, no technology, max 1 name per sector",
        persona,
        config,
        catalog=catalog,
    )

    assert "MACRO_HEDGE" in updated.pinned_rows
    assert "MACRO_HEDGE" in updated.allowed_rows
    assert updated_config.decode.max_names_per_sector == 1

    tech = {i.symbol for i in catalog if i.sector == "Information Technology"}
    assert tech <= set(updated.excluded_assets)

    # The original objects are frozen and must be untouched.
    assert persona.pinned_rows == ()
    assert config.decode.max_names_per_sector == 2


def test_banned_row_is_removed_from_the_mandate(config: Config, catalog: Catalog) -> None:
    persona = config.personas[1]
    updated, _ = apply_instruction("no momentum at all", persona, config, catalog=catalog)
    assert "MOMENTUM_LEADERS" not in updated.allowed_rows
    assert "MOMENTUM_LEADERS" not in updated.pinned_rows


def test_steering_changes_the_generated_page(config: Config, store, vocab) -> None:
    """The instruction must actually reach the decoder, not merely the config object."""
    import torch

    from gendesk.decoding.generate import PageGenerator
    from gendesk.features.regimes import REGIME_AXES
    from gendesk.model.gendesk import GenDeskModel
    from gendesk.tokenization.page import PageContext

    torch.manual_seed(0)
    model = GenDeskModel(config.model, vocab, store.n_features).eval()
    persona = config.personas[1]
    position = len(store.dates) - 5

    steered, steered_config = apply_instruction(
        "no technology", persona, config, catalog=store.catalog
    )
    generator = PageGenerator(model, vocab, store, steered_config, torch.device("cpu"))
    context = PageContext(
        persona=steered.name,
        risk_budget=steered.risk_budget,
        horizon_days=steered.horizon_days,
        regimes={axis: int(store.regimes[axis].iloc[position]) for axis in REGIME_AXES},
        history=(),
    )
    result = generator.generate(context, steered, position, n_samples=8, temperature=1.0)

    by_symbol = store.catalog.by_symbol
    for page in result.pages:
        assert all(by_symbol[s].sector != "Information Technology" for s in page.symbols)
