"""The GenDesk model: one backbone, two heads, time-varying entity representations.

Three design decisions carry most of the weight here.

**Untied embeddings.** GenPage unties the input embedding from the output projection
so the same trunk can serve next-token prediction and sigmoid scoring. The same
applies here, and it is what allows the WBC stage to reshape the scoring geometry
without destroying the generative distribution.

**Semantic fusion, adapted to a non-stationary catalog.** Netflix fuses an item's ID
embedding with a content embedding derived from its synopsis and cast, so a brand-new
title is representable before anyone has watched it. An instrument's "content" is not
static: what NVDA *is*, for the purposes of a page, is its current momentum,
volatility and correlation. So the entity representation is

``E[i] = id_embedding[i] + W_fuse @ features[i, t]``

which solves the cold-start problem the same way (an instrument with no trained ID
still has a representation) and additionally makes the catalog time-varying, which a
static recommender vocabulary is not.

**A catalog-aware output head.** Because entity representations move with the market,
the output projection must too. Logits over entities are an inner product between the
hidden state and the *current* entity representation rather than a fixed weight row.
This is the mechanism that lets a single forward pass score the entire catalog --
GenRec's "prefill-only" inference -- and it is what the value head reuses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gendesk.config import ModelConfig
from gendesk.model.transformer import KVCache, TransformerBackbone
from gendesk.tokenization.vocab import Vocab


@dataclass
class ModelOutput:
    """Everything a forward pass can produce; heads are computed lazily."""

    hidden: Tensor
    logits: Tensor | None = None
    entity_scores: Tensor | None = None


class EntityRepresentation(nn.Module):
    """Time-varying representation of every catalog instrument.

    ``id_embed`` is what the model learns about an instrument's identity;
    ``fuse`` is what it learns about instruments *in a given state*. An unseen
    instrument keeps the second half.
    """

    def __init__(self, n_instruments: int, n_features: int, d_model: int, config: ModelConfig):
        super().__init__()
        self.id_embed = nn.Embedding(n_instruments, d_model)
        self.enabled = config.semantic_fusion
        self.fuse = nn.Sequential(
            nn.Linear(n_features, config.semantic_dim * 4),
            nn.GELU(),
            nn.Linear(config.semantic_dim * 4, d_model),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        nn.init.normal_(self.id_embed.weight, std=config.init_std)

    def forward(self, features: Tensor) -> Tensor:
        """``features``: ``(batch, n_instruments, n_features)`` -> ``(batch, n, d)``."""
        base = self.id_embed.weight.unsqueeze(0)
        if not self.enabled:
            return base.expand(features.shape[0], -1, -1)
        return base + self.scale * self.fuse(features)


class GenDeskModel(nn.Module):
    """Decoder-only page generator with a catalog-wide scoring head."""

    def __init__(self, config: ModelConfig, vocab: Vocab, n_features: int) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab.size
        self.entity_offset = vocab.entity_offset
        self.n_instruments = vocab.n_instruments
        self.n_features = n_features
        self.vocab_fingerprint = vocab.fingerprint()

        d = config.d_model
        # Input side: structural/context tokens have plain embeddings; entity tokens
        # additionally receive their current market state.
        self.token_embed = nn.Embedding(vocab.size, d)
        self.input_entities = EntityRepresentation(vocab.n_instruments, n_features, d, config)
        self.embed_drop = nn.Dropout(config.dropout)

        self.backbone = TransformerBackbone(config)

        # Output side, untied from the input side.
        self.structural_head = nn.Linear(d, vocab.entity_offset, bias=True)
        self.output_entities = EntityRepresentation(vocab.n_instruments, n_features, d, config)
        self.entity_bias = nn.Parameter(torch.zeros(vocab.n_instruments))

        # Scoring head: its own projection and its own entity geometry, so post-
        # training can move it without dragging the generative head along.
        self.value_proj = nn.Linear(d, d, bias=False)
        self.value_entities = EntityRepresentation(vocab.n_instruments, n_features, d, config)
        self.value_bias = nn.Parameter(torch.zeros(vocab.n_instruments))

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.init_std)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- embedding -----------------------------------------------------------

    def embed(self, tokens: Tensor, features: Tensor) -> Tensor:
        """Embed a token sequence, fusing market state into entity positions.

        Args:
            tokens: ``(batch, seq)`` token ids.
            features: ``(batch, n_instruments, n_features)`` standardised features
                for each example's own date.
        """
        x = self.token_embed(tokens)
        if not self.input_entities.enabled:
            return self.embed_drop(x)

        is_entity = tokens >= self.entity_offset
        index = (tokens - self.entity_offset).clamp(0, self.n_instruments - 1)
        gathered = torch.gather(features, 1, index.unsqueeze(-1).expand(-1, -1, self.n_features))
        fused = self.input_entities.fuse(gathered) * self.input_entities.scale
        return self.embed_drop(x + fused * is_entity.unsqueeze(-1))

    # -- heads ---------------------------------------------------------------

    def logits_from_hidden(self, hidden: Tensor, features: Tensor) -> Tensor:
        """Full-vocabulary logits with a catalog-aware entity block."""
        structural = self.structural_head(hidden)
        entities = self.output_entities(features)
        entity_logits = torch.einsum("bld,bnd->bln", hidden, entities) + self.entity_bias
        return torch.cat([structural, entity_logits], dim=-1)

    def scores_from_hidden(self, hidden: Tensor, features: Tensor) -> Tensor:
        """Pre-sigmoid catalog scores for every position.

        One matrix product covers the entire catalog, which is what makes the
        prefill-only serving path viable: score every instrument for every slot in a
        single pass, with no token-by-token decoding.
        """
        query = self.value_proj(hidden)
        entities = self.value_entities(features)
        return torch.einsum("bld,bnd->bln", query, entities) + self.value_bias

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        tokens: Tensor,
        features: Tensor,
        cache: KVCache | None = None,
        offset: int = 0,
        want_logits: bool = True,
        want_scores: bool = False,
    ) -> ModelOutput:
        hidden = self.backbone(self.embed(tokens, features), cache=cache, offset=offset)
        return ModelOutput(
            hidden=hidden,
            logits=self.logits_from_hidden(hidden, features) if want_logits else None,
            entity_scores=self.scores_from_hidden(hidden, features) if want_scores else None,
        )

    # -- convenience ---------------------------------------------------------

    def entity_logits(self, hidden: Tensor, features: Tensor) -> Tensor:
        """Logits restricted to the entity block (generation only ever needs these)."""
        entities = self.output_entities(features)
        return torch.einsum("bld,bnd->bln", hidden, entities) + self.entity_bias

    def structural_logits(self, hidden: Tensor) -> Tensor:
        return self.structural_head(hidden)

    def loss_next_token(
        self,
        tokens: Tensor,
        features: Tensor,
        weights: Tensor,
        label_smoothing: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        """Weighted next-token cross-entropy.

        Args:
            weights: ``(batch, seq)`` per-position loss weight, already aligned with
                ``tokens``. Positions with weight zero (padding, prompt) drop out.

        Returns:
            ``(loss, per_token_logprob)`` where the log-probabilities are detached
            and used by the RL stage as the behaviour policy reference.
        """
        output = self.forward(tokens[:, :-1], features, want_logits=True)
        assert output.logits is not None
        logits = output.logits
        target = tokens[:, 1:]
        weight = weights[:, 1:]

        flat_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="none",
            label_smoothing=label_smoothing,
        ).view_as(target)

        denom = weight.sum().clamp(min=1.0)
        loss = (flat_loss * weight).sum() / denom
        return loss, (-flat_loss).detach()

    # -- persistence ---------------------------------------------------------

    def checkpoint(self) -> dict:
        return {
            "state_dict": self.state_dict(),
            "config": self.config.model_dump(mode="json"),
            "vocab_fingerprint": self.vocab_fingerprint,
            "n_features": self.n_features,
            "entity_offset": self.entity_offset,
            "n_instruments": self.n_instruments,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict, vocab: Vocab) -> GenDeskModel:
        config = ModelConfig.model_validate(payload["config"])
        if payload["vocab_fingerprint"] != vocab.fingerprint():
            raise ValueError(
                "checkpoint was trained with a different vocabulary; rebuild the corpus "
                "or restore the matching universe file"
            )
        model = cls(config, vocab, payload["n_features"])
        model.load_state_dict(payload["state_dict"])
        return model
