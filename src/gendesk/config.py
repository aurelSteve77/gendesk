"""Typed configuration for the whole pipeline.

A single YAML file (``configs/default.yaml``) is the source of truth. Every stage
receives a validated :class:`Config` instance rather than loose keyword arguments,
which keeps the data / corpus / model / backtest contracts explicit and makes runs
self-describing: the config is serialised next to every artifact it produced.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from gendesk.utils.paths import CONFIG_DIR


class FrozenModel(BaseModel):
    """Immutable base so a config cannot be mutated mid-run."""

    model_config = {"frozen": True, "extra": "forbid"}


# --------------------------------------------------------------------------- data


class DataConfig(FrozenModel):
    """Market data acquisition and panel construction."""

    start: date = date(2005, 1, 1)
    end: date = date(2026, 8, 1)
    provider: Literal["yahoo"] = "yahoo"
    universe_file: str = "universe.yaml"
    #: Minimum number of observations for an instrument to enter the catalog.
    min_observations: int = 1500
    #: Maximum fraction of missing closes tolerated over an instrument's own life.
    max_missing_frac: float = 0.02
    #: Median 60-day dollar volume floor (USD) for tradability.
    min_dollar_volume: float = 5e6
    request_timeout: float = 30.0
    max_retries: int = 4
    #: Concurrent requests against the price API.
    max_workers: int = 8


# ----------------------------------------------------------------------- features


class FeatureConfig(FrozenModel):
    """Point-in-time cross-sectional and macro feature construction."""

    #: Trailing windows (trading days) for momentum-style features.
    momentum_windows: tuple[int, ...] = (21, 63, 126, 252)
    #: Skip window for 12-1 momentum (avoids the short-term reversal effect).
    momentum_skip: int = 21
    reversal_window: int = 5
    vol_window: int = 63
    beta_window: int = 252
    corr_window: int = 126
    dollar_volume_window: int = 60
    #: Half-life for the exponentially weighted covariance used in risk budgeting.
    ewma_halflife: int = 63
    #: Minimum trailing observations before a feature row is considered valid.
    min_warmup: int = 260
    #: Winsorisation quantiles applied to every cross-sectional feature.
    winsor: tuple[float, float] = (0.01, 0.99)


class RegimeConfig(FrozenModel):
    """Discretisation of the macro state into a small vocabulary of tokens."""

    #: Macro series pulled from the price API and used only as conditioning.
    macro_symbols: tuple[str, ...] = ("^VIX", "^TNX", "^IRX", "^GSPC")
    #: Number of buckets per regime axis (terciles by default).
    n_buckets: int = 3
    #: Trailing window used to rank the current macro reading against its own past.
    rank_window: int = 504


# ------------------------------------------------------------------------- corpus


class PersonaConfig(FrozenModel):
    """A synthetic allocator mandate. Personas are the 'members' of the system."""

    name: str
    risk_budget: Literal["low", "medium", "high"]
    horizon_days: int
    max_names: int
    max_sector_weight: float
    #: Row archetypes this mandate is allowed to receive, in preference order.
    allowed_rows: tuple[str, ...]
    #: Rows that must appear on every page for this mandate (GenPage "row pinning").
    pinned_rows: tuple[str, ...] = ()
    #: Instruments excluded outright (e.g. commodity funds for a long-only equity book).
    excluded_assets: tuple[str, ...] = ()
    turnover_penalty: float = 1.0


class CorpusConfig(FrozenModel):
    """Construction of the tokenized page corpus used for pretraining."""

    #: Trading-day spacing between consecutive page snapshots.
    stride_days: int = 5
    #: Number of instruments per row.
    row_size: int = 6
    #: Number of rows per page.
    n_rows: int = 5
    #: Forward horizon (trading days) over which a page's reward is realised.
    reward_horizon: int = 21
    #: Quantile of the page-reward distribution below which pages are discarded.
    #: This is the analogue of Netflix pretraining only on positive impressions.
    positive_quantile: float = 0.55
    #: Number of teacher-sampled candidate pages per (date, persona) cell.
    candidates_per_cell: int = 6
    #: Temperature of the teacher's softmax over instrument scores.
    teacher_temperature: float = 0.35
    #: Length of the verbalised interaction history (previous page rows) in tokens.
    history_pages: int = 3
    #: Annualised volatility target used to normalise page rewards.
    vol_target: float = 0.12
    #: Penalty coefficient on realised max drawdown inside the reward window.
    drawdown_penalty: float = 0.5
    #: Penalty coefficient on page turnover relative to the previous page.
    turnover_penalty: float = 0.25
    seed: int = 17


# -------------------------------------------------------------------------- model


class ModelConfig(FrozenModel):
    """Decoder-only backbone shared by the page and scoring heads."""

    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 704
    dropout: float = 0.1
    max_seq_len: int = 320
    rope_theta: float = 10_000.0
    #: GenPage explicitly unties the input embedding from the output projection so
    #: the same backbone can serve next-token prediction and sigmoid scoring.
    tie_embeddings: bool = False
    #: Blend weight for content-based embeddings of cold-start instruments.
    semantic_fusion: bool = True
    semantic_dim: int = 16
    init_std: float = 0.02


# ----------------------------------------------------------------------- training


class PretrainConfig(FrozenModel):
    epochs: int = 8
    batch_size: int = 64
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    #: Loss weight of context tokens relative to page tokens (0 = page tokens only).
    context_loss_weight: float = 0.1
    amp: bool = True
    log_every: int = 50


class WBCConfig(FrozenModel):
    """Weighted binary classification post-training (GenPage 'Option A')."""

    epochs: int = 3
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    #: Negatives sampled per slot. 0 means "score the entire eligible catalog in one
    #: pass", which is affordable at this catalog size and removes sampling noise
    #: from the objective; sampling exists for catalogs where it is not.
    negatives_per_slot: int = 0
    #: Keeps the language-modelling objective alive during post-training, exactly as
    #: GenRec mixes an LM loss into its ranking objective.
    lm_loss_weight: float = 0.2
    #: Clip on the per-token reward weight to stop a single tail event dominating.
    reward_clip: float = 3.0
    amp: bool = True
    log_every: int = 50


class RLConfig(FrozenModel):
    """Dr. GRPO page-level reinforcement learning (GenPage 'Option B')."""

    steps: int = 300
    #: Prompts (date x persona cells) per optimisation step.
    prompts_per_step: int = 8
    #: Sampled pages per prompt -- the GRPO group.
    group_size: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.0
    grad_clip: float = 0.5
    temperature: float = 1.0
    top_k: int = 32
    #: KL coefficient against the frozen post-trained reference policy.
    kl_coef: float = 0.02
    #: PPO-style ratio clip. Dr. GRPO keeps the clip but drops the std normalisation
    #: of advantages and the length normalisation of the loss.
    clip_eps: float = 0.2
    inner_epochs: int = 1
    #: Evaluate the diversity of generated pages every N steps (emergence study).
    diversity_every: int = 20
    log_every: int = 10
    amp: bool = False


class TrainingConfig(FrozenModel):
    pretrain: PretrainConfig = PretrainConfig()
    wbc: WBCConfig = WBCConfig()
    rl: RLConfig = RLConfig()
    device: Literal["auto", "cuda", "cpu"] = "auto"
    seed: int = 1337
    num_workers: int = 2


# ----------------------------------------------------------------------- decoding


class DecodeConfig(FrozenModel):
    """Constrained and hybrid decoding at serving time."""

    temperature: float = 0.7
    top_k: int = 24
    #: GenPage hybrid decoding: autoregress the first ``autoregressive_slots``
    #: entities of a row, then score the remaining slots in a single forward pass.
    autoregressive_slots: int = 2
    hybrid: bool = True
    #: Business rules enforced as token masks.
    enforce_dedup: bool = True
    enforce_sector_cap: bool = True
    enforce_liquidity: bool = True
    enforce_row_pinning: bool = True
    #: Maximum instruments from one sector on a single page.
    max_names_per_sector: int = 4


# ----------------------------------------------------------------------- backtest


class BacktestConfig(FrozenModel):
    """Walk-forward out-of-sample evaluation."""

    #: Last date of the training window. Everything after is strictly out-of-sample.
    train_end: date = date(2019, 12, 31)
    #: Validation window used for model selection only.
    valid_end: date = date(2021, 12, 31)
    #: Gap (trading days) between train and test to purge label overlap.
    embargo_days: int = 21
    rebalance_days: int = 21
    #: One-way transaction cost in basis points, applied to traded notional.
    cost_bps: float = 5.0
    #: Risk-free proxy used for Sharpe (annualised decimal, constant approximation).
    risk_free: float = 0.02
    #: Number of block-bootstrap resamples for Sharpe inference.
    bootstrap_samples: int = 2000
    bootstrap_block: int = 21
    #: Number of independent trials attempted, used by the deflated Sharpe ratio.
    n_trials: int = 24
    seed: int = 7


# -------------------------------------------------------------------------- top


class Config(FrozenModel):
    """Root configuration object."""

    run_name: str = "gendesk"
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    regimes: RegimeConfig = RegimeConfig()
    corpus: CorpusConfig = CorpusConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    decode: DecodeConfig = DecodeConfig()
    backtest: BacktestConfig = BacktestConfig()
    personas: tuple[PersonaConfig, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _check_windows(self) -> Config:
        if self.backtest.train_end >= self.backtest.valid_end:
            raise ValueError("backtest.train_end must precede backtest.valid_end")
        if self.data.start >= self.data.end:
            raise ValueError("data.start must precede data.end")
        if self.model.d_model % self.model.n_heads:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if self.model.n_heads % self.model.n_kv_heads:
            raise ValueError("model.n_heads must be divisible by model.n_kv_heads")
        return self

    @property
    def page_slots(self) -> int:
        """Number of instrument slots on a page."""
        return self.corpus.n_rows * self.corpus.row_size

    def dump(self) -> dict[str, Any]:
        """JSON-safe dictionary, used to stamp artifacts with their provenance."""
        return self.model_dump(mode="json")


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load and validate the configuration.

    Args:
        path: YAML file. Defaults to ``configs/default.yaml``.
        **overrides: Dotted-path overrides, e.g. ``training__seed=1``. Nested keys
            use a double underscore so they can be passed from a shell.
    """
    path = Path(path) if path is not None else CONFIG_DIR / "default.yaml"
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}

    for dotted, value in overrides.items():
        node = raw
        parts = dotted.split("__")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    return Config.model_validate(raw)
