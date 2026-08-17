"""Page generation.

Two decoding modes share one code path:

* **Autoregressive.** Every slot is sampled conditioned on the slot before it. This
  is what the RL stage optimises, because the sampling distribution is a plain
  product of per-step categoricals and the log-probability is exactly recoverable.

* **Hybrid row decoding.** Only the first ``autoregressive_slots`` instruments of a
  row are sampled step by step; the rest of the row is filled from a *single*
  additional forward pass, by sampling without replacement from that one
  distribution. This is GenPage's latency trick, and it is the reason a generative
  page can be cheaper to serve than a multi-stage ranking pipeline: the number of
  sequential model calls stops scaling with the number of slots on the page.

Both modes return everything the RL stage needs -- the decision positions, the mask
that was in force at each one, and the sampled token -- so log-probabilities can be
recomputed exactly under the constrained distribution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd
import torch
from torch import Tensor

from gendesk.config import Config, PersonaConfig
from gendesk.corpus.rows import archetype_order
from gendesk.decoding.constraints import ConstraintEngine, ConstraintReport
from gendesk.features.store import FeatureStore
from gendesk.model.gendesk import GenDeskModel
from gendesk.model.transformer import KVCache
from gendesk.tokenization.page import Page, PageContext, PageSequence, Row
from gendesk.tokenization.vocab import Vocab


@dataclass
class GenerationResult:
    """A batch of pages sampled from one prompt."""

    pages: list[Page]
    #: ``(batch, seq)`` full token sequences, prompt included.
    tokens: Tensor
    #: ``(n_steps,)`` positions in ``tokens`` where a token was sampled.
    step_positions: Tensor
    #: ``(batch, n_steps)`` sampled token ids.
    step_tokens: Tensor
    #: ``(batch, n_steps, vocab)`` mask in force at each decision.
    step_masks: Tensor
    #: ``(batch, n_steps)`` log-probability of each sampled token.
    step_logprobs: Tensor
    prompt_len: int
    report: ConstraintReport
    #: Number of sequential model invocations, the quantity hybrid decoding reduces.
    model_calls: int = 0
    latency_ms: float = 0.0
    extras: dict = field(default_factory=dict)

    @property
    def batch(self) -> int:
        return int(self.tokens.shape[0])


class PageGenerator:
    """Constrained page decoder for a single (date, mandate) cell at a time."""

    def __init__(
        self,
        model: GenDeskModel,
        vocab: Vocab,
        store: FeatureStore,
        config: Config,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.store = store
        self.config = config
        self.device = device or next(model.parameters()).device
        self.archetypes = archetype_order()
        self._row_token_ids = torch.as_tensor(
            [vocab.row(name) for name in self.archetypes], device=self.device
        )
        lookup = torch.full((vocab.size,), -1, dtype=torch.long, device=self.device)
        lookup[self._row_token_ids] = torch.arange(len(self.archetypes), device=self.device)
        self._token_to_archetype = lookup

    # -- helpers -------------------------------------------------------------

    def features_for(self, position: int, batch: int) -> Tensor:
        values = self.store.values[position]
        tensor = torch.as_tensor(values, device=self.device, dtype=torch.float32)
        return tensor.unsqueeze(0).expand(batch, -1, -1)

    def _sample(self, logits: Tensor, mask: Tensor, temperature: float, top_k: int) -> Tensor:
        """Masked, temperature-scaled, optionally top-k sampling."""
        logits = logits.masked_fill(~mask, float("-inf"))
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        if top_k and top_k < logits.shape[-1]:
            kth = logits.topk(min(top_k, int(mask.sum(dim=-1).max())), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = torch.where(probs.sum(-1, keepdim=True) > 0, probs, mask.float())
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _entity_logits(self, hidden: Tensor, features: Tensor, head: str) -> Tensor:
        """Entity scores from either the generative head or the WBC value head."""
        if head == "value":
            return self.model.scores_from_hidden(hidden.unsqueeze(1), features)[:, 0]
        return self.model.entity_logits(hidden.unsqueeze(1), features)[:, 0]

    @staticmethod
    def _logprob(logits: Tensor, mask: Tensor, chosen: Tensor, temperature: float) -> Tensor:
        scaled = logits.masked_fill(~mask, float("-inf")) / max(temperature, 1e-6)
        return torch.log_softmax(scaled, dim=-1).gather(1, chosen.unsqueeze(-1)).squeeze(-1)

    # -- main entry point ----------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        context: PageContext,
        persona: PersonaConfig,
        position: int,
        n_samples: int = 1,
        temperature: float | None = None,
        top_k: int | None = None,
        hybrid: bool | None = None,
        spec_sequence: PageSequence | None = None,
        head: str = "lm",
    ) -> GenerationResult:
        """Sample ``n_samples`` pages for one mandate on one date.

        Args:
            head: ``"lm"`` scores entities with the generative output projection;
                ``"value"`` scores them with the WBC head, which is GenPage's
                "generation as value prediction" serving mode. Row archetypes are
                always chosen by the structural head -- only the value head's
                entity geometry differs.
        """
        cfg = self.config.decode
        temperature = cfg.temperature if temperature is None else temperature
        top_k = cfg.top_k if top_k is None else top_k
        hybrid = cfg.hybrid if hybrid is None else hybrid

        sequence = spec_sequence or PageSequence(self.vocab)
        prompt = sequence.encode_prompt(context)
        prompt_len = len(prompt)

        vocab_size = self.vocab.size
        offset = self.vocab.entity_offset
        engine = ConstraintEngine(
            catalog=self.store.catalog,
            vocab=self.vocab,
            config=cfg,
            persona=persona,
            available=self.store.available.iloc[position].to_numpy().astype(bool),
            batch=n_samples,
            device=self.device,
        )

        features = self.features_for(position, n_samples)
        tokens = torch.as_tensor(prompt, device=self.device).unsqueeze(0).repeat(n_samples, 1)

        cache = KVCache.empty(self.model.config.n_layers)
        started = time.perf_counter()
        hidden = self.model.backbone(self.model.embed(tokens, features), cache=cache, offset=0)[
            :, -1
        ]
        model_calls = 1

        step_positions: list[int] = []
        step_tokens: list[Tensor] = []
        step_masks: list[Tensor] = []
        step_logprobs: list[Tensor] = []

        n_rows = self.config.corpus.n_rows
        row_size = self.config.corpus.row_size
        rows_symbols: list[list[list[str]]] = [[] for _ in range(n_samples)]
        rows_archetypes: list[list[str]] = [[] for _ in range(n_samples)]

        for row_idx in range(n_rows):
            # ---- row-type decision -------------------------------------------
            allowed_rows = engine.row_mask(n_rows - row_idx)
            row_logits = self.model.structural_logits(hidden.unsqueeze(1))[:, 0]
            full_mask = torch.zeros((n_samples, vocab_size), dtype=torch.bool, device=self.device)
            full_mask[:, self._row_token_ids] = allowed_rows
            padded_logits = torch.full((n_samples, vocab_size), float("-inf"), device=self.device)
            padded_logits[:, : row_logits.shape[-1]] = row_logits

            chosen_row_token = self._sample(padded_logits, full_mask, temperature, 0)
            logprob = self._logprob(padded_logits, full_mask, chosen_row_token, temperature)

            step_positions.append(int(tokens.shape[1]))
            step_tokens.append(chosen_row_token)
            step_masks.append(full_mask)
            step_logprobs.append(logprob)

            archetype_ids = self._token_to_archetype[chosen_row_token].clamp(min=0)
            engine.commit_row(archetype_ids)
            for b, a in enumerate(archetype_ids.tolist()):
                rows_archetypes[b].append(self.archetypes[int(a)])
                rows_symbols[b].append([])

            tokens = torch.cat([tokens, chosen_row_token.unsqueeze(1)], dim=1)
            hidden = self.model.backbone(
                self.model.embed(chosen_row_token.unsqueeze(1), features),
                cache=cache,
                offset=cache.length,
            )[:, -1]
            model_calls += 1

            # ---- entity slots -------------------------------------------------
            n_sequential = row_size if not hybrid else min(cfg.autoregressive_slots, row_size)

            for _ in range(n_sequential):
                entity_mask = engine.entity_mask(archetype_ids)
                entity_logits = self._entity_logits(hidden, features, head)
                full_mask = torch.zeros(
                    (n_samples, vocab_size), dtype=torch.bool, device=self.device
                )
                full_mask[:, offset:] = entity_mask
                padded_logits = torch.full(
                    (n_samples, vocab_size), float("-inf"), device=self.device
                )
                padded_logits[:, offset:] = entity_logits

                chosen = self._sample(padded_logits, full_mask, temperature, top_k)
                logprob = self._logprob(padded_logits, full_mask, chosen, temperature)

                step_positions.append(int(tokens.shape[1]))
                step_tokens.append(chosen)
                step_masks.append(full_mask)
                step_logprobs.append(logprob)

                engine.commit(chosen - offset)
                for b, tok in enumerate(chosen.tolist()):
                    rows_symbols[b][row_idx].append(self.vocab.tokens[int(tok)])

                tokens = torch.cat([tokens, chosen.unsqueeze(1)], dim=1)
                hidden = self.model.backbone(
                    self.model.embed(chosen.unsqueeze(1), features),
                    cache=cache,
                    offset=cache.length,
                )[:, -1]
                model_calls += 1

            remaining = row_size - n_sequential
            if remaining > 0:
                filled, fill_logprobs, fill_masks = self._fill_row(
                    hidden, features, engine, archetype_ids, remaining, temperature, top_k, head
                )
                for slot in range(remaining):
                    step_positions.append(int(tokens.shape[1]) + slot)
                step_tokens.extend(filled.unbind(dim=1))
                step_masks.extend(fill_masks)
                step_logprobs.extend(fill_logprobs.unbind(dim=1))

                for b in range(n_samples):
                    for tok in filled[b].tolist():
                        rows_symbols[b][row_idx].append(self.vocab.tokens[int(tok)])

                tokens = torch.cat([tokens, filled], dim=1)
                # One batched pass keeps the cache consistent for the next row, so
                # the whole tail of the row costs a single sequential call.
                hidden = self.model.backbone(
                    self.model.embed(filled, features), cache=cache, offset=cache.length
                )[:, -1]
                model_calls += 1

            eor = torch.full((n_samples, 1), self.vocab.eor, device=self.device, dtype=torch.long)
            tokens = torch.cat([tokens, eor], dim=1)
            hidden = self.model.backbone(
                self.model.embed(eor, features), cache=cache, offset=cache.length
            )[:, -1]
            model_calls += 1

        eop = torch.full((n_samples, 1), self.vocab.eop, device=self.device, dtype=torch.long)
        tokens = torch.cat([tokens, eop], dim=1)
        latency_ms = (time.perf_counter() - started) * 1000.0

        pages = [
            Page(
                date=pd.Timestamp(self.store.dates[position]),
                persona=persona.name,
                rows=tuple(
                    Row(arch, tuple(syms))
                    for arch, syms in zip(rows_archetypes[b], rows_symbols[b], strict=True)
                    if syms
                ),
            )
            for b in range(n_samples)
        ]

        return GenerationResult(
            pages=pages,
            tokens=tokens,
            step_positions=torch.as_tensor(step_positions, device=self.device),
            step_tokens=torch.stack(step_tokens, dim=1),
            step_masks=torch.stack(step_masks, dim=1),
            step_logprobs=torch.stack(step_logprobs, dim=1),
            prompt_len=prompt_len,
            report=engine.report,
            model_calls=model_calls,
            latency_ms=latency_ms,
        )

    # -- hybrid fill ---------------------------------------------------------

    def _fill_row(
        self,
        hidden: Tensor,
        features: Tensor,
        engine: ConstraintEngine,
        archetype_ids: Tensor,
        remaining: int,
        temperature: float,
        top_k: int,
        head: str = "lm",
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Fill the tail of a row from one distribution, sampling without replacement.

        The log-probability of the resulting ordered set is the Plackett-Luce
        product: each pick is drawn from the same logits renormalised over what is
        still legal. That keeps the hybrid path exactly scorable for RL, rather than
        an approximation that would bias the gradient.
        """
        n_samples = hidden.shape[0]
        vocab_size = self.vocab.size
        offset = self.vocab.entity_offset

        entity_logits = self._entity_logits(hidden, features, head)
        picks: list[Tensor] = []
        logprobs: list[Tensor] = []
        masks: list[Tensor] = []

        for _ in range(remaining):
            entity_mask = engine.entity_mask(archetype_ids)
            full_mask = torch.zeros((n_samples, vocab_size), dtype=torch.bool, device=self.device)
            full_mask[:, offset:] = entity_mask
            padded_logits = torch.full((n_samples, vocab_size), float("-inf"), device=self.device)
            padded_logits[:, offset:] = entity_logits

            chosen = self._sample(padded_logits, full_mask, temperature, top_k)
            logprobs.append(self._logprob(padded_logits, full_mask, chosen, temperature))
            masks.append(full_mask)
            engine.commit(chosen - offset)
            picks.append(chosen)

        return torch.stack(picks, dim=1), torch.stack(logprobs, dim=1), masks
