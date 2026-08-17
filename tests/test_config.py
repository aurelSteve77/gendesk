"""Configuration loading and validation."""

from __future__ import annotations

from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from gendesk.config import Config, load_config
from gendesk.corpus.rows import ARCHETYPES
from gendesk.data.universe import load_catalog
from gendesk.utils.paths import CONFIG_DIR


def test_shipped_config_is_valid() -> None:
    config = load_config()
    assert config.personas
    assert config.page_slots == config.corpus.n_rows * config.corpus.row_size


def test_shipped_personas_reference_real_archetypes() -> None:
    config = load_config()
    for persona in config.personas:
        assert set(persona.allowed_rows) <= set(ARCHETYPES)
        assert set(persona.pinned_rows) <= set(persona.allowed_rows)
        assert len(persona.pinned_rows) <= config.corpus.n_rows


def test_shipped_personas_exclude_only_real_symbols() -> None:
    config = load_config()
    catalog_symbols = set(load_catalog().symbols)
    for persona in config.personas:
        unknown = set(persona.excluded_assets) - catalog_symbols
        assert not unknown, (
            f"{persona.name} excludes symbols that are not in the catalog: {unknown}"
        )


def test_config_is_frozen() -> None:
    config = load_config()
    with pytest.raises(ValidationError):
        config.run_name = "other"


def test_rejects_inverted_windows() -> None:
    with pytest.raises(ValidationError, match="train_end must precede"):
        Config.model_validate(
            {"backtest": {"train_end": date(2021, 1, 1), "valid_end": date(2020, 1, 1)}}
        )


def test_rejects_incoherent_attention_shapes() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        Config.model_validate({"model": {"d_model": 100, "n_heads": 8}})
    with pytest.raises(ValidationError, match="divisible"):
        Config.model_validate({"model": {"n_heads": 8, "n_kv_heads": 3}})


def test_rejects_unknown_keys() -> None:
    """A typo in a config file must fail loudly rather than be ignored."""
    with pytest.raises(ValidationError):
        Config.model_validate({"corpus": {"row_sizee": 6}})


def test_overrides_use_double_underscore(tmp_path) -> None:
    path = tmp_path / "conf.yaml"
    path.write_text(yaml.safe_dump({"run_name": "base"}))
    config = load_config(path, run_name="over", training__seed=99)
    assert config.run_name == "over"
    assert config.training.seed == 99


def test_universe_file_has_no_duplicates() -> None:
    catalog = load_catalog(CONFIG_DIR / "universe.yaml")
    assert len(set(catalog.symbols)) == len(catalog.symbols)
    assert catalog.benchmark in catalog.symbols
    assert all(inst.sector for inst in catalog)
