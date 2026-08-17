"""End-to-end smoke test.

Runs every stage -- corpus, pretraining, WBC post-training, RL, generation, backtest
-- on the synthetic market at miniature scale. It does not check that the model is
any *good*; it checks that the stages compose, that the artifacts they exchange have
the shapes they promise, and that nothing regresses into an exception.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gendesk.config import Config
from gendesk.corpus.build import build_corpus
from gendesk.evaluation.backtest import run_backtest
from gendesk.evaluation.strategies import GenDeskStrategy
from gendesk.features.store import FeatureStore
from gendesk.tokenization.vocab import Vocab
from gendesk.training.pretrain import pretrain
from gendesk.training.rl import PromptBook, train_rl
from gendesk.training.wbc import train_wbc


@pytest.fixture(scope="module")
def corpus(config: Config, store: FeatureStore, tmp_path_factory):
    """Build a miniature corpus into a temporary directory."""
    import gendesk.corpus.build as build_module

    directory = tmp_path_factory.mktemp("corpus")
    original_processed = build_module.PROCESSED_DIR
    original_manifest = build_module.MANIFEST_DIR
    build_module.PROCESSED_DIR = directory
    build_module.MANIFEST_DIR = directory
    try:
        yield build_corpus(config, store, force=True)
    finally:
        build_module.PROCESSED_DIR = original_processed
        build_module.MANIFEST_DIR = original_manifest


def test_corpus_has_all_splits_and_a_threshold(corpus, config: Config) -> None:
    counts = corpus.meta["counts"]
    assert counts["train"] > 0
    assert counts["test"] > 0
    assert len(corpus) == sum(counts.values())

    # Exactly one deterministic book per (persona, date) cell.
    books = [ex for ex in corpus.examples if ex.is_book]
    keys = {(ex.persona, ex.position) for ex in books}
    assert len(books) == len(keys)

    kept = [ex for ex in corpus.examples if ex.split == "train" and ex.keep_pretrain]
    assert 0 < len(kept) < counts["train"], "the outcome filter kept everything or nothing"
    assert min(ex.reward for ex in kept) >= corpus.meta["threshold"] - 1e-9


def test_history_only_contains_earlier_pages(corpus) -> None:
    """A page's context must never contain a page from its own date or later."""
    by_cell = {}
    for example in corpus.examples:
        if example.is_book:
            by_cell.setdefault(example.persona, {})[example.position] = {
                sym for row in example.rows for sym in row[1]
            }

    for example in corpus.examples:
        earlier = [p for p in by_cell[example.persona] if p < example.position]
        allowed = set()
        for position in earlier:
            allowed |= by_cell[example.persona][position]
        for page in example.history:
            assert set(page) <= allowed


def test_full_training_and_evaluation_composes(
    config: Config, store: FeatureStore, vocab: Vocab, corpus
) -> None:
    torch.manual_seed(0)

    model, slot_metrics = pretrain(
        config, store, corpus, vocab, checkpoint_name="smoke", log_run=False, save=False
    )
    assert 0.0 < slot_metrics.mrr <= 1.0
    assert slot_metrics.n_slots > 0
    assert np.isfinite(slot_metrics.loss)

    model, wbc_metrics = train_wbc(
        config, store, corpus, vocab, model, checkpoint_name="smoke", log_run=False
    )
    assert np.isfinite(wbc_metrics.wbc_loss)
    assert 0.0 <= wbc_metrics.auc <= 1.0

    prompts = PromptBook(corpus, config, split="train")
    assert len(prompts) > 0

    model, trace = train_rl(
        config, store, corpus, vocab, model, checkpoint_name="smoke", log_run=False
    )
    assert len(trace) == config.training.rl.steps
    assert all(np.isfinite(record["mean_reward"]) for record in trace)
    assert all(record["kl"] >= -1e-6 for record in trace)

    # The trained model must be usable as a strategy end to end.
    persona = config.personas[0]
    strategy = GenDeskStrategy(
        model=model, vocab=vocab, store=store, config=config, persona=persona, temperature=0.0
    )
    result = run_backtest(
        "gendesk_smoke",
        strategy,
        store,
        config,
        pd.Timestamp(store.dates[700]),
        pd.Timestamp(store.dates[-1]),
    )
    assert result.returns.notna().all()
    assert np.isfinite(result.stats["sharpe"])
    assert (result.weights.sum(axis=1) - 1.0).abs().max() < 1e-6
    assert strategy.pages

    diversity = strategy.diversity_frame()
    assert not diversity.empty
    assert (diversity["mean_correlation"].abs() <= 1.0).all()
    assert (diversity["diversification_ratio"] >= 1.0 - 1e-6).all()


def test_strategy_history_is_its_own_output(
    config: Config, store: FeatureStore, vocab: Vocab
) -> None:
    """At inference the model conditions on the pages it generated, not on a teacher."""
    from gendesk.model.gendesk import GenDeskModel

    torch.manual_seed(0)
    model = GenDeskModel(config.model, vocab, store.n_features).eval()
    persona = config.personas[0]
    strategy = GenDeskStrategy(
        model=model, vocab=vocab, store=store, config=config, persona=persona, temperature=0.0
    )

    strategy(700)
    first = next(iter(strategy.pages.values()))
    context = strategy.context(721)
    assert context.history[-1] == first.symbols
