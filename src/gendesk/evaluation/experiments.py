"""Experiment orchestration: the full out-of-sample study and the latency benchmark.

Everything here runs strictly after the validation cutoff. The model, the corpus
threshold, the row archetypes and the sizing rule were all fixed before this window
opens, so the numbers it produces are the ones worth arguing about.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gendesk.config import Config
from gendesk.evaluation.backtest import run_backtest
from gendesk.evaluation.baselines import build_baselines
from gendesk.evaluation.statistics import (
    block_bootstrap_difference,
    block_bootstrap_sharpe,
    deflated_sharpe_ratio,
    performance_summary,
    probability_of_backtest_overfitting,
)
from gendesk.evaluation.strategies import GenDeskStrategy
from gendesk.features.regimes import REGIME_AXES
from gendesk.features.store import FeatureStore, load_features
from gendesk.tokenization.page import PageContext
from gendesk.tokenization.vocab import Vocab, build_vocab
from gendesk.training.checkpoint import checkpoint_exists, load_checkpoint
from gendesk.training.schedule import resolve_device
from gendesk.utils.logging import get_logger
from gendesk.utils.paths import REPORT_DIR, ensure_dirs

log = get_logger(__name__)

#: Which checkpoints become tradable strategies, and which head drives selection.
MODEL_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("gendesk_pretrain", "pretrain", "lm"),
    ("gendesk_wbc", "wbc", "value"),
    ("gendesk_rl", "rl", "lm"),
)

#: Baselines that do not depend on the mandate, so they are run once.
MANDATE_FREE = ("benchmark_spy", "equal_weight", "momentum_12_1", "low_volatility", "risk_parity")


def evaluation_window(
    config: Config, store: FeatureStore, window: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start and end of the requested evaluation window, with the embargo applied."""
    embargo = pd.Timedelta(days=int(config.backtest.embargo_days * 1.5))
    if window == "valid":
        return pd.Timestamp(config.backtest.train_end) + embargo, pd.Timestamp(
            config.backtest.valid_end
        )
    return pd.Timestamp(config.backtest.valid_end) + embargo, pd.Timestamp(store.dates[-1])


def _aggregate(results: dict[str, list[pd.Series]]) -> dict[str, pd.Series]:
    """Equal-weight the per-mandate return series of each strategy."""
    out: dict[str, pd.Series] = {}
    for name, series_list in results.items():
        frame = pd.concat(series_list, axis=1)
        out[name] = frame.mean(axis=1).rename(name)
    return out


def run_backtest_suite(
    config: Config,
    window: str = "test",
    store: FeatureStore | None = None,
    vocab: Vocab | None = None,
    save: bool = True,
) -> dict:
    """Run every model variant and every baseline across every mandate."""
    ensure_dirs()
    store = store or load_features()
    vocab = vocab or build_vocab(store.catalog, tuple(p.name for p in config.personas))
    device = resolve_device(config.training.device)

    start, end = evaluation_window(config, store, window)
    log.info("backtest_suite_start", window=window, start=str(start.date()), end=str(end.date()))

    models: dict[str, tuple[Any, str]] = {}
    for label, checkpoint, head in MODEL_VARIANTS:
        if checkpoint_exists(checkpoint):
            model, _ = load_checkpoint(checkpoint, vocab, device=device)
            models[label] = (model.eval(), head)
        else:
            log.warning("checkpoint_missing", checkpoint=checkpoint)

    per_strategy: dict[str, list[pd.Series]] = {}
    per_persona_rows: list[dict] = []
    diagnostics: dict[str, dict] = {}
    saved_pages: dict[str, dict] = {}

    # -- mandate-free baselines, run once ------------------------------------
    reference_persona = config.personas[0]
    shared = build_baselines(store, config, reference_persona)
    for name in MANDATE_FREE:
        result = run_backtest(name, shared[name], store, config, start, end)
        per_strategy[name] = [result.returns]
        per_persona_rows.append({"persona": "-", **result.summary_row()})

    # -- mandate-specific strategies -----------------------------------------
    for persona in config.personas:
        baselines = build_baselines(store, config, persona)
        for name in ("pipeline_multistage", "teacher_book"):
            result = run_backtest(f"{name}", baselines[name], store, config, start, end)
            per_strategy.setdefault(name, []).append(result.returns)
            per_persona_rows.append({"persona": persona.name, **result.summary_row()})

        for label, (model, head) in models.items():
            strategy = GenDeskStrategy(
                model=model,
                vocab=vocab,
                store=store,
                config=config,
                persona=persona,
                head=head,
                temperature=0.0,
            )
            result = run_backtest(label, strategy, store, config, start, end)
            per_strategy.setdefault(label, []).append(result.returns)
            per_persona_rows.append({"persona": persona.name, **result.summary_row()})

            if label == "gendesk_rl" or (label == "gendesk_wbc" and "gendesk_rl" not in models):
                saved_pages[persona.name] = {
                    str(date.date()): [
                        {"archetype": row.archetype, "symbols": list(row.symbols)}
                        for row in page.rows
                    ]
                    for date, page in strategy.pages.items()
                }
                diagnostics[persona.name] = {
                    **strategy.latency_summary(),
                    "archetype_mix": strategy.archetype_mix().to_dict(),
                    "diversity": strategy.diversity_frame().mean().to_dict()
                    if not strategy.diversity_frame().empty
                    else {},
                }

    aggregated = _aggregate(per_strategy)
    returns_frame = pd.DataFrame(aggregated).dropna(how="all")

    summary = []
    for name, series in aggregated.items():
        stats = performance_summary(series, config.backtest.risk_free)
        stats["strategy"] = name
        summary.append(stats)
    summary_frame = (
        pd.DataFrame(summary).set_index("strategy").sort_values("sharpe", ascending=False)
    )

    # -- inference ------------------------------------------------------------
    trial_sharpes = summary_frame["sharpe"].to_numpy() / np.sqrt(252)
    inference: dict[str, dict] = {}
    for name, series in aggregated.items():
        test = block_bootstrap_sharpe(
            series,
            config.backtest.bootstrap_samples,
            config.backtest.bootstrap_block,
            config.backtest.seed,
        )
        dsr = deflated_sharpe_ratio(series, config.backtest.n_trials, trial_sharpes)
        inference[name] = {"sharpe_test": test.as_dict(), "deflated": dsr}

    comparisons: dict[str, dict] = {}
    headline = "gendesk_rl" if "gendesk_rl" in aggregated else next(iter(models), None)
    if headline and headline in aggregated:
        for reference in ("pipeline_multistage", "teacher_book", "benchmark_spy", "equal_weight"):
            if reference in aggregated:
                comparisons[f"{headline}_vs_{reference}"] = block_bootstrap_difference(
                    aggregated[headline],
                    aggregated[reference],
                    config.backtest.bootstrap_samples,
                    config.backtest.bootstrap_block,
                    config.backtest.seed,
                ).as_dict()

    pbo = probability_of_backtest_overfitting(returns_frame)

    report: dict[str, Any] = {
        "window": window,
        "start": str(start.date()),
        "end": str(end.date()),
        "summary": summary_frame.reset_index().to_dict(orient="records"),
        "per_persona": per_persona_rows,
        "inference": inference,
        "comparisons": comparisons,
        "pbo": pbo,
        "diagnostics": diagnostics,
        "config": config.dump(),
    }

    if save:
        (REPORT_DIR / f"backtest_{window}.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
        returns_frame.to_parquet(REPORT_DIR / f"returns_{window}.parquet")
        pd.DataFrame(per_persona_rows).to_csv(REPORT_DIR / f"per_persona_{window}.csv", index=False)
        (REPORT_DIR / f"pages_{window}.json").write_text(
            json.dumps(saved_pages, indent=2, default=str)
        )
        log.info("backtest_suite_saved", path=str(REPORT_DIR))

    report["returns"] = returns_frame
    return report


def run_latency_study(
    config: Config,
    repeats: int = 20,
    store: FeatureStore | None = None,
    checkpoint: str = "rl",
) -> list[dict]:
    """Time hybrid row decoding against full autoregression.

    Reports both wall-clock latency and the number of sequential model invocations,
    because the second is the hardware-independent quantity: hybrid decoding removes
    ``row_size - autoregressive_slots`` sequential steps from every row.
    """
    from gendesk.decoding.generate import PageGenerator

    store = store or load_features()
    vocab = build_vocab(store.catalog, tuple(p.name for p in config.personas))
    device = resolve_device(config.training.device)

    name = checkpoint if checkpoint_exists(checkpoint) else "pretrain"
    model, _ = load_checkpoint(name, vocab, device=device)
    model.eval()

    generator = PageGenerator(model, vocab, store, config, device)
    persona = config.personas[1] if len(config.personas) > 1 else config.personas[0]
    position = len(store.dates) - 2
    context = PageContext(
        persona=persona.name,
        risk_budget=persona.risk_budget,
        horizon_days=persona.horizon_days,
        regimes={axis: int(store.regimes[axis].iloc[position]) for axis in REGIME_AXES},
        history=(),
    )

    rows: list[dict] = []
    for mode, hybrid in (("autoregressive", False), ("hybrid_row", True)):
        # Warm-up: the first call pays CUDA context and allocator costs.
        for _ in range(3):
            generator.generate(context, persona, position, n_samples=1, hybrid=hybrid)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings, calls = [], []
        for _ in range(repeats):
            result = generator.generate(context, persona, position, n_samples=1, hybrid=hybrid)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(result.latency_ms)
            calls.append(result.model_calls)

        rows.append(
            {
                "mode": mode,
                "median_ms": float(np.median(timings)),
                "p95_ms": float(np.percentile(timings, 95)),
                "sequential_model_calls": float(np.mean(calls)),
                "repeats": repeats,
            }
        )

    if len(rows) == 2:
        base, fast = rows[0], rows[1]
        rows.append(
            {
                "mode": "reduction",
                "median_ms": 1.0 - fast["median_ms"] / max(base["median_ms"], 1e-9),
                "p95_ms": 1.0 - fast["p95_ms"] / max(base["p95_ms"], 1e-9),
                "sequential_model_calls": 1.0
                - fast["sequential_model_calls"] / max(base["sequential_model_calls"], 1e-9),
                "repeats": repeats,
            }
        )

    path: Path = REPORT_DIR / "latency.json"
    path.write_text(json.dumps(rows, indent=2))
    return rows
