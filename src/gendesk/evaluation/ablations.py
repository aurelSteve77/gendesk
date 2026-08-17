"""Ablation grid.

The headline offline finding in the GenPage post is that *enriching the prompt buys
more than scaling the model*: roughly a 6.9% loss reduction from context enrichment
against 1.3% from a 7.5x parameter increase. That is a claim about where the
information lives, and it is worth testing rather than repeating.

Three families are run, all on the same corpus, the same validation split and the
same number of epochs, so the only thing that differs is the thing being ablated:

1. **Context ladder** -- progressively add mandate, regime and history blocks to the
   prompt at fixed capacity.
2. **Capacity ladder** -- scale the backbone from ~0.9M to ~19M parameters at fixed,
   full context.
3. **Design choices** -- semantic feature fusion on/off, row-archetype tokens on/off,
   and the outcome filter on/off.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from gendesk.config import Config, ModelConfig
from gendesk.corpus.build import PageCorpus, load_corpus
from gendesk.features.store import FeatureStore, load_features
from gendesk.tokenization.page import ContextSpec
from gendesk.tokenization.vocab import build_vocab
from gendesk.training.pretrain import pretrain
from gendesk.utils.logging import get_logger
from gendesk.utils.paths import REPORT_DIR, ensure_dirs

log = get_logger(__name__)

FULL_CONTEXT = ContextSpec(
    persona=True, risk=True, horizon=True, regimes=True, history=True, row_tokens=True
)

#: (label, spec) -- each rung adds one block to the prompt.
CONTEXT_LADDER: tuple[tuple[str, ContextSpec], ...] = (
    (
        "ctx0_rows_only",
        ContextSpec(persona=False, risk=False, horizon=False, regimes=False, history=False),
    ),
    (
        "ctx1_mandate",
        ContextSpec(persona=True, risk=True, horizon=True, regimes=False, history=False),
    ),
    (
        "ctx2_mandate_regime",
        ContextSpec(persona=True, risk=True, horizon=True, regimes=True, history=False),
    ),
    ("ctx3_full", FULL_CONTEXT),
)

#: (label, overrides) -- capacity at fixed full context.
CAPACITY_LADDER: tuple[tuple[str, dict], ...] = (
    ("cap0_tiny", {"d_model": 96, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2, "d_ff": 256}),
    ("cap1_small", {"d_model": 160, "n_layers": 6, "n_heads": 8, "n_kv_heads": 4, "d_ff": 448}),
    ("cap2_base", {}),
    ("cap3_large", {"d_model": 384, "n_layers": 10, "n_heads": 12, "n_kv_heads": 6, "d_ff": 1024}),
)


def _subsample(corpus: PageCorpus, fraction: float, seed: int) -> PageCorpus:
    """Deterministically thin the *training* split.

    The ablation grid is compute-bounded, and what it needs to measure is the gap
    between cells, not the absolute level. Every cell sees the identical subsample,
    and the validation split is never touched.
    """
    if fraction >= 1.0:
        return corpus
    rng = np.random.default_rng(seed)
    train = [ex for ex in corpus.examples if ex.split == "train"]
    other = [ex for ex in corpus.examples if ex.split != "train"]
    keep = rng.choice(len(train), size=max(1, int(len(train) * fraction)), replace=False)
    kept = [train[int(i)] for i in sorted(keep)]
    return PageCorpus(examples=kept + other, meta={**corpus.meta, "subsample": fraction})


def _run_one(
    label: str,
    family: str,
    config: Config,
    store: FeatureStore,
    corpus: PageCorpus,
    spec: ContextSpec,
    model_overrides: dict,
    epochs: int,
    use_outcome_filter: bool = True,
    subsample: float = 1.0,
) -> dict:
    """Train one ablation cell and report its validation metrics."""
    model_config = ModelConfig.model_validate({**config.model.model_dump(), **model_overrides})
    cell_config = config.model_copy(
        update={
            "model": model_config,
            "training": config.training.model_copy(
                update={"pretrain": config.training.pretrain.model_copy(update={"epochs": epochs})}
            ),
        }
    )
    vocab = build_vocab(store.catalog, tuple(p.name for p in config.personas))

    model, metrics = pretrain(
        cell_config,
        store,
        _subsample(corpus, subsample, config.training.seed),
        vocab,
        spec=spec,
        checkpoint_name=f"ablation_{label}",
        log_run=False,
        use_outcome_filter=use_outcome_filter,
        save=False,
    )
    row = {
        "family": family,
        "cell": label,
        "context": spec.name,
        "params_m": round(model.n_parameters / 1e6, 3),
        "epochs": epochs,
        **metrics.as_dict(),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("ablation_cell_done", **{k: v for k, v in row.items() if k != "n_slots"})
    return row


def run_ablations(
    config: Config,
    epochs: int = 3,
    store: FeatureStore | None = None,
    corpus: PageCorpus | None = None,
    save: bool = True,
    subsample: float = 0.6,
) -> pd.DataFrame:
    """Run the full grid and return a tidy frame of results."""
    ensure_dirs()
    store = store or load_features()
    corpus = corpus or load_corpus()

    rows: list[dict] = []

    for label, spec in CONTEXT_LADDER:
        rows.append(
            _run_one(label, "context", config, store, corpus, spec, {}, epochs, subsample=subsample)
        )

    for label, overrides in CAPACITY_LADDER:
        rows.append(
            _run_one(
                label,
                "capacity",
                config,
                store,
                corpus,
                FULL_CONTEXT,
                overrides,
                epochs,
                subsample=subsample,
            )
        )

    rows.append(
        _run_one(
            "design_no_semantic_fusion",
            "design",
            config,
            store,
            corpus,
            FULL_CONTEXT,
            {"semantic_fusion": False},
            epochs,
            subsample=subsample,
        )
    )
    rows.append(
        _run_one(
            "design_no_row_tokens",
            "design",
            config,
            store,
            corpus,
            replace(FULL_CONTEXT, row_tokens=False),
            {},
            epochs,
            subsample=subsample,
        )
    )
    rows.append(
        _run_one(
            "design_no_outcome_filter",
            "design",
            config,
            store,
            corpus,
            FULL_CONTEXT,
            {},
            epochs,
            use_outcome_filter=False,
            subsample=subsample,
        )
    )

    frame = pd.DataFrame(rows)

    # Improvements are quoted against the weakest rung of each ladder, which is how
    # the GenPage comparison is framed: what does adding X buy, from a fixed base?
    frame["loss_vs_family_base"] = frame.groupby("family")["loss"].transform(
        lambda s: 1.0 - s / s.iloc[0]
    )
    frame["mrr_vs_family_base"] = frame.groupby("family")["mrr"].transform(
        lambda s: s / s.iloc[0] - 1.0
    )

    if save:
        frame.to_csv(REPORT_DIR / "ablations.csv", index=False)
        (REPORT_DIR / "ablations.json").write_text(
            json.dumps(frame.to_dict(orient="records"), indent=2, default=str)
        )
        log.info("ablations_saved", path=str(REPORT_DIR / "ablations.csv"), n_cells=len(frame))
    return frame


def summarise_headline(frame: pd.DataFrame) -> dict:
    """Reduce the grid to the single comparison the GenPage post makes."""
    context = frame[frame.family == "context"]
    capacity = frame[frame.family == "capacity"]
    if context.empty or capacity.empty:
        return {}

    return {
        "context_loss_reduction": float(context["loss_vs_family_base"].iloc[-1]),
        "context_mrr_gain": float(context["mrr_vs_family_base"].iloc[-1]),
        "capacity_loss_reduction": float(capacity["loss_vs_family_base"].iloc[-1]),
        "capacity_mrr_gain": float(capacity["mrr_vs_family_base"].iloc[-1]),
        "capacity_multiple": float(
            capacity["params_m"].iloc[-1] / max(capacity["params_m"].iloc[0], 1e-9)
        ),
    }
