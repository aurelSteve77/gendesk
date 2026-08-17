"""Shared fixtures.

The whole suite runs against a synthetic market so it is fast, deterministic and
offline. The synthetic panel goes through the *real* feature, regime, tokenizer and
corpus code paths -- only the prices are fabricated -- so the tests exercise the
production logic rather than a parallel implementation of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gendesk.config import Config, PersonaConfig
from gendesk.data.panel import PricePanel
from gendesk.data.universe import Catalog, Instrument
from gendesk.features.cross_section import build_feature_tensor
from gendesk.features.regimes import build_regimes
from gendesk.features.store import FeatureStore

N_DAYS = 900
SECTORS = ("Information Technology", "Health Care", "Financials", "Energy")


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    """A small catalog: 16 single names across four sectors, plus six funds."""
    instruments: list[Instrument] = []
    for sector in SECTORS:
        for i in range(4):
            instruments.append(
                Instrument(symbol=f"{sector[:2].upper()}{i}", sector=sector, asset_class="equity")
            )
    for symbol, group in (
        ("SPY", "broad_equity"),
        ("QQQ", "broad_equity"),
        ("TLT", "rates_credit"),
        ("IEF", "rates_credit"),
        ("GLD", "real_assets"),
        ("UUP", "currency_intl"),
    ):
        instruments.append(
            Instrument(symbol=symbol, sector="Fund", asset_class="fund", group=group)
        )
    return Catalog(
        instruments=tuple(instruments),
        benchmark="SPY",
        macro_series=("^VIX", "^TNX", "^IRX"),
    )


@pytest.fixture(scope="session")
def panel(catalog: Catalog) -> PricePanel:
    """A synthetic panel with a common factor, so correlations are realistic."""
    rng = np.random.default_rng(20240101)
    dates = pd.bdate_range("2018-01-01", periods=N_DAYS, name="date")
    symbols = list(catalog.symbols)

    market = rng.normal(0.0004, 0.010, size=N_DAYS)
    frames = {}
    for symbol in symbols:
        beta = 0.3 + 1.4 * rng.random()
        idio = rng.normal(0.0, 0.008 + 0.004 * rng.random(), size=N_DAYS)
        drift = rng.normal(0.0002, 0.0003)
        rets = drift + beta * market + idio
        frames[symbol] = 100.0 * np.exp(np.cumsum(rets - 0.5 * rets.var()))

    adj_close = pd.DataFrame(frames, index=dates)
    close = adj_close.copy()
    volume = pd.DataFrame(
        rng.lognormal(14.5, 0.4, size=(N_DAYS, len(symbols))), index=dates, columns=symbols
    )

    macro = pd.DataFrame(
        {
            "^VIX": 14
            + 8 * np.abs(pd.Series(market, index=dates).rolling(21, min_periods=1).std() * 50),
            "^TNX": 2.5 + np.cumsum(rng.normal(0, 0.01, N_DAYS)),
            "^IRX": 1.5 + np.cumsum(rng.normal(0, 0.01, N_DAYS)),
        },
        index=dates,
    )

    available = pd.DataFrame(True, index=dates, columns=symbols)
    # One name lists late, so availability is exercised rather than assumed.
    available.iloc[:120, 3] = False
    adj_close.iloc[:120, 3] = np.nan

    return PricePanel(
        adj_close=adj_close,
        close=close,
        volume=volume,
        macro=macro,
        available=available,
        catalog=catalog,
    )


@pytest.fixture(scope="session")
def config(catalog: Catalog) -> Config:
    """A small but structurally complete configuration."""
    personas = (
        PersonaConfig(
            name="core",
            risk_budget="low",
            horizon_days=21,
            max_names=12,
            max_sector_weight=0.5,
            allowed_rows=("QUALITY_BALLAST", "MACRO_HEDGE", "MOMENTUM_LEADERS", "MEAN_REVERSION"),
            pinned_rows=("MACRO_HEDGE",),
            excluded_assets=("GLD",),
        ),
        PersonaConfig(
            name="pod",
            risk_budget="high",
            horizon_days=21,
            max_names=12,
            max_sector_weight=0.6,
            allowed_rows=(
                "MOMENTUM_LEADERS",
                "MEAN_REVERSION",
                "DISPERSION_HARVEST",
                "TREND_BREAKOUT",
            ),
        ),
    )
    return Config.model_validate(
        {
            "run_name": "test",
            "corpus": {
                "n_rows": 3,
                "row_size": 3,
                "stride_days": 21,
                "reward_horizon": 10,
                "history_pages": 2,
                "candidates_per_cell": 3,
            },
            "model": {
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 2,
                "d_ff": 128,
                "max_seq_len": 128,
            },
            "training": {
                "device": "cpu",
                "pretrain": {"epochs": 1, "batch_size": 8},
                "wbc": {"epochs": 1, "batch_size": 8},
                "rl": {"steps": 2, "prompts_per_step": 1, "group_size": 3},
            },
            "decode": {"max_names_per_sector": 2, "autoregressive_slots": 1},
            "backtest": {
                "train_end": "2020-06-30",
                "valid_end": "2020-12-31",
                "rebalance_days": 21,
                "bootstrap_samples": 100,
            },
            "features": {"min_warmup": 300},
            "personas": [p.model_dump() for p in personas],
        }
    )


@pytest.fixture(scope="session")
def store(panel: PricePanel, config: Config) -> FeatureStore:
    """A feature store built by the production code path from synthetic prices."""
    values, raw = build_feature_tensor(panel, config.features)
    regimes = build_regimes(panel, config.regimes)

    warm = np.zeros(len(panel.calendar), dtype=bool)
    warm[config.features.min_warmup :] = True

    return FeatureStore(
        values=values,
        dates=pd.DatetimeIndex(panel.adj_close.index),
        symbols=tuple(panel.symbols),
        feature_names=tuple(store_feature_names()),
        regimes=regimes,
        vol=raw["vol_63"],
        beta=raw["beta"],
        adv=np.expm1(raw["turnover"]),
        available=panel.available,
        returns=panel.returns,
        warm=warm,
        catalog=panel.catalog,
    )


def store_feature_names() -> tuple[str, ...]:
    from gendesk.features.cross_section import FEATURE_NAMES

    return FEATURE_NAMES


@pytest.fixture(scope="session")
def vocab(store: FeatureStore, config: Config):
    from gendesk.tokenization.vocab import build_vocab

    return build_vocab(store.catalog, tuple(p.name for p in config.personas))
