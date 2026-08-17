"""Model mechanics: untied heads, feature fusion, cold start, cache correctness."""

from __future__ import annotations

import torch

from gendesk.config import Config
from gendesk.features.store import FeatureStore
from gendesk.model.gendesk import GenDeskModel
from gendesk.model.transformer import KVCache
from gendesk.tokenization.vocab import Vocab


def _model(config: Config, store: FeatureStore, vocab: Vocab) -> GenDeskModel:
    torch.manual_seed(0)
    return GenDeskModel(config.model, vocab, store.n_features).eval()


def _inputs(config: Config, store: FeatureStore, vocab: Vocab, batch: int = 3, length: int = 24):
    torch.manual_seed(1)
    tokens = torch.randint(0, vocab.size, (batch, length))
    features = torch.as_tensor(store.values[500:503], dtype=torch.float32)
    return tokens, features


def test_forward_shapes(config: Config, store: FeatureStore, vocab: Vocab) -> None:
    model = _model(config, store, vocab)
    tokens, features = _inputs(config, store, vocab)
    out = model(tokens, features, want_logits=True, want_scores=True)

    assert out.hidden.shape == (3, 24, config.model.d_model)
    assert out.logits.shape == (3, 24, vocab.size)
    assert out.entity_scores.shape == (3, 24, vocab.n_instruments)


def test_output_projection_is_untied_from_the_input_embedding(
    config: Config, store: FeatureStore, vocab: Vocab
) -> None:
    """GenPage unties them so one trunk can serve generation and scoring."""
    model = _model(config, store, vocab)
    assert model.token_embed.weight.data_ptr() != model.structural_head.weight.data_ptr()
    assert (
        model.input_entities.id_embed.weight.data_ptr()
        != model.output_entities.id_embed.weight.data_ptr()
    )
    assert (
        model.output_entities.id_embed.weight.data_ptr()
        != model.value_entities.id_embed.weight.data_ptr()
    )


def test_entity_logits_move_with_market_state(
    config: Config, store: FeatureStore, vocab: Vocab
) -> None:
    """The catalog is time-varying: the same instrument in a different state scores differently."""
    model = _model(config, store, vocab)
    tokens, features = _inputs(config, store, vocab)

    base = model(tokens, features).logits
    shifted = model(tokens, features + 1.0).logits

    entity_block = slice(vocab.entity_offset, None)
    assert not torch.allclose(base[..., entity_block], shifted[..., entity_block])


def test_semantic_fusion_can_be_disabled(config: Config, store: FeatureStore, vocab: Vocab) -> None:
    frozen = config.model.model_copy(update={"semantic_fusion": False})
    torch.manual_seed(0)
    model = GenDeskModel(frozen, vocab, store.n_features).eval()
    tokens, features = _inputs(config, store, vocab)

    base = model(tokens, features).logits
    shifted = model(tokens, features * 3.0).logits
    torch.testing.assert_close(base, shifted)


def test_cold_start_instrument_still_has_a_representation(
    config: Config, store: FeatureStore, vocab: Vocab
) -> None:
    """Zeroing an instrument's learned id must not zero its score.

    This is the cold-start property: a name with no trained identity is still
    representable through its current market state.
    """
    model = _model(config, store, vocab)
    tokens, features = _inputs(config, store, vocab)
    target = 5

    with torch.no_grad():
        model.output_entities.id_embed.weight[target] = 0.0
    logits = model(tokens, features).logits[..., vocab.entity_offset + target]
    assert torch.isfinite(logits).all()
    assert float(logits.abs().max()) > 0.0


def test_kv_cache_matches_a_full_forward(config: Config, store: FeatureStore, vocab: Vocab) -> None:
    """Incremental decoding must reproduce the dense forward pass."""
    model = _model(config, store, vocab)
    tokens, features = _inputs(config, store, vocab, batch=2, length=12)
    features = features[:2]

    dense = model.backbone(model.embed(tokens, features))

    cache = KVCache.empty(config.model.n_layers)
    prefix = model.backbone(model.embed(tokens[:, :8], features), cache=cache, offset=0)
    torch.testing.assert_close(prefix, dense[:, :8], rtol=2e-4, atol=2e-4)

    for step in range(8, 12):
        out = model.backbone(
            model.embed(tokens[:, step : step + 1], features), cache=cache, offset=cache.length
        )
        torch.testing.assert_close(out[:, 0], dense[:, step], rtol=2e-4, atol=2e-4)


def test_loss_ignores_zero_weighted_positions(
    config: Config, store: FeatureStore, vocab: Vocab
) -> None:
    model = _model(config, store, vocab)
    tokens, features = _inputs(config, store, vocab)

    weights = torch.ones_like(tokens, dtype=torch.float32)
    weights[:, 10:] = 0.0
    loss_a, _ = model.loss_next_token(tokens, features, weights)

    altered = tokens.clone()
    altered[:, 12:] = (altered[:, 12:] + 7) % vocab.size
    weights_b = weights.clone()
    loss_b, _ = model.loss_next_token(altered, features, weights_b)

    # Changing only zero-weighted *targets* must not change the loss; positions before
    # the mask still condition on the same prefix.
    assert abs(float(loss_a) - float(loss_b)) < 1e-4


def test_checkpoint_round_trip_rejects_a_different_vocabulary(
    config: Config, store: FeatureStore, vocab: Vocab
) -> None:
    import pytest

    from gendesk.tokenization.vocab import build_vocab

    model = _model(config, store, vocab)
    payload = model.checkpoint()

    restored = GenDeskModel.from_checkpoint(payload, vocab)
    assert restored.n_parameters == model.n_parameters

    other = build_vocab(store.catalog.subset(list(store.symbols)[:10]), vocab.persona_names)
    with pytest.raises(ValueError, match="different vocabulary"):
        GenDeskModel.from_checkpoint(payload, other)
