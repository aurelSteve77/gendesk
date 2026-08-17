"""Vocabulary and page encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gendesk.config import Config
from gendesk.data.universe import load_catalog
from gendesk.features.regimes import REGIME_AXES
from gendesk.tokenization.page import ContextSpec, Page, PageContext, PageSequence, Row
from gendesk.tokenization.vocab import ROW_ARCHETYPES, SPECIALS, Vocab, build_vocab


def test_entity_block_is_contiguous_and_last(vocab: Vocab) -> None:
    """Generation only samples inside the entity block, so it must be one slice."""
    assert vocab.entity_offset + vocab.n_instruments == vocab.size
    for i, symbol in enumerate(vocab.symbols):
        assert vocab.entity(symbol) == vocab.entity_offset + i
        assert vocab.is_entity(vocab.entity(symbol))
    assert not vocab.is_entity(vocab.entity_offset - 1)


def test_index_round_trip(vocab: Vocab) -> None:
    for i in range(vocab.n_instruments):
        assert vocab.entity_index(vocab.index_to_token(i)) == i
    with pytest.raises(ValueError):
        vocab.entity_index(0)


def test_specials_and_rows_are_present(vocab: Vocab) -> None:
    for token in SPECIALS:
        assert vocab.tokens[vocab.id(token)] == token
    for archetype in ROW_ARCHETYPES:
        token_id = vocab.row(archetype)
        assert vocab.is_row(token_id)
        assert vocab.row_archetype(token_id) == archetype


def test_unknown_token_maps_to_unk(vocab: Vocab) -> None:
    assert vocab.id("NOT_A_SYMBOL") == vocab.id("<unk>")


def test_fingerprint_changes_with_the_catalog(config: Config) -> None:
    catalog = load_catalog()
    names = tuple(p.name for p in config.personas)
    full = build_vocab(catalog, names)
    trimmed = build_vocab(catalog.subset(list(catalog.symbols)[:50]), names)
    assert full.fingerprint() != trimmed.fingerprint()


def _page_and_context(vocab: Vocab) -> tuple[Page, PageContext]:
    symbols = list(vocab.symbols)
    page = Page(
        date=pd.Timestamp("2021-06-30"),
        persona="core",
        rows=(
            Row("MOMENTUM_LEADERS", tuple(symbols[:3])),
            Row("MACRO_HEDGE", tuple(symbols[3:6])),
        ),
    )
    context = PageContext(
        persona="core",
        risk_budget="low",
        horizon_days=21,
        regimes=dict.fromkeys(REGIME_AXES, 1),
        history=(tuple(symbols[6:9]),),
    )
    return page, context


def test_encode_decode_round_trip(vocab: Vocab) -> None:
    page, context = _page_and_context(vocab)
    sequence = PageSequence(vocab)
    encoded = sequence.encode(page, context)
    decoded = sequence.decode(encoded.tokens, page.date, page.persona)

    assert decoded.archetypes == page.archetypes
    assert decoded.symbols == page.symbols


def test_encoded_index_structure_is_consistent(vocab: Vocab) -> None:
    page, context = _page_and_context(vocab)
    encoded = PageSequence(vocab).encode(page, context)

    assert encoded.entity_positions.size == len(page.symbols)
    assert encoded.slot_row.tolist() == [0, 0, 0, 1, 1, 1]
    assert encoded.page_mask[: encoded.prompt_len].sum() == 0
    assert encoded.page_mask[encoded.prompt_len :].all()

    for position, catalog_index in zip(encoded.entity_positions, encoded.slot_entity, strict=True):
        assert vocab.entity_index(int(encoded.tokens[position])) == catalog_index


def test_context_spec_controls_prompt_length(vocab: Vocab) -> None:
    _, context = _page_and_context(vocab)
    bare = PageSequence(
        vocab, ContextSpec(persona=False, risk=False, horizon=False, regimes=False, history=False)
    )
    full = PageSequence(vocab, ContextSpec())

    assert len(bare.encode_prompt(context)) < len(full.encode_prompt(context))
    # The bare prompt is still a valid prompt: bos, ctx, page.
    assert bare.encode_prompt(context)[-1] == vocab.page


def test_decode_tolerates_a_truncated_page(vocab: Vocab) -> None:
    page, context = _page_and_context(vocab)
    encoded = PageSequence(vocab).encode(page, context)
    truncated = encoded.tokens[:-4]
    decoded = PageSequence(vocab).decode(truncated, page.date, page.persona)
    assert len(decoded.rows) >= 1
    assert all(len(row.symbols) > 0 for row in decoded.rows)


def test_pad_batch_shapes(vocab: Vocab) -> None:
    from gendesk.tokenization.page import pad_batch

    page, context = _page_and_context(vocab)
    sequence = PageSequence(vocab)
    short = sequence.encode(Page(page.date, page.persona, page.rows[:1]), context)
    long = sequence.encode(page, context)

    batch = pad_batch([short, long], vocab.pad, max_len=256)
    assert batch.tokens.shape[0] == 2
    assert batch.tokens.shape[1] == len(long)
    assert np.all(batch.tokens[0, batch.lengths[0] :] == vocab.pad)
    assert batch.attention_mask[0].sum() == batch.lengths[0]
