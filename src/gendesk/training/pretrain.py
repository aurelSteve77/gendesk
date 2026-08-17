"""Stage 1: next-token pretraining on outcome-filtered pages.

The model learns the *language* of desk pages: which archetype follows which in a
given regime, which instruments co-occur inside a row, how a mandate's page differs
from another's. No forward return is visible at this stage -- selection pressure was
already applied when the corpus discarded the candidates that did badly.

The offline metric is deliberately GenRec's: mean reciprocal rank of the held-out
instrument among the eligible catalog at each slot. It measures the thing that
actually matters for a recommender -- can the model put the right entity near the
top of a 362-way choice -- and it is comparable across the ablation grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from gendesk.config import Config
from gendesk.corpus.build import PageCorpus
from gendesk.features.store import FeatureStore
from gendesk.model.gendesk import GenDeskModel
from gendesk.tokenization.page import ContextSpec
from gendesk.tokenization.vocab import Vocab
from gendesk.training.checkpoint import RunLogger, save_checkpoint
from gendesk.training.dataset import FeatureBank, PageBatch, PageDataset, collate_pages
from gendesk.training.schedule import (
    build_optimizer,
    clip_gradients,
    cosine_schedule,
    resolve_device,
)
from gendesk.utils.logging import get_logger
from gendesk.utils.seed import set_seed

log = get_logger(__name__)


@dataclass
class SlotMetrics:
    """Retrieval quality at instrument slots."""

    loss: float
    mrr: float
    hit_at_1: float
    hit_at_5: float
    hit_at_20: float
    n_slots: int

    def as_dict(self) -> dict:
        return {
            "loss": self.loss,
            "mrr": self.mrr,
            "hit@1": self.hit_at_1,
            "hit@5": self.hit_at_5,
            "hit@20": self.hit_at_20,
            "n_slots": self.n_slots,
        }


def _loader(
    dataset: PageDataset, batch_size: int, pad_id: int, shuffle: bool, workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=partial(collate_pages, pad_id=pad_id),
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_slots(
    model: GenDeskModel,
    loader: DataLoader,
    bank: FeatureBank,
    device: torch.device,
    vocab: Vocab,
    max_batches: int | None = None,
) -> SlotMetrics:
    """Rank the held-out instrument at every slot against the eligible catalog."""
    model.eval()
    total_loss, total_weight = 0.0, 0.0
    ranks: list[np.ndarray] = []

    for i, raw in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch: PageBatch = raw.to(device)
        features = bank.gather(batch.positions)

        loss, _ = model.loss_next_token(batch.tokens, features, batch.loss_weights)
        weight = float(batch.loss_weights[:, 1:].sum())
        total_loss += float(loss) * weight
        total_weight += weight

        hidden = model.backbone(model.embed(batch.tokens[:, :-1], features))
        entity_logits = model.entity_logits(hidden, features)

        target = batch.tokens[:, 1:]
        slots = batch.slot_mask[:, 1:]
        if not bool(slots.any()):
            continue

        eligible = bank.eligibility(batch.positions).unsqueeze(1)
        masked = entity_logits.masked_fill(~eligible, float("-inf"))

        target_index = (target - vocab.entity_offset).clamp(min=0)
        true_score = masked.gather(2, target_index.unsqueeze(-1)).squeeze(-1)
        rank = (masked > true_score.unsqueeze(-1)).sum(dim=-1) + 1
        ranks.append(rank[slots].detach().cpu().numpy())

    model.train()
    flat = np.concatenate(ranks) if ranks else np.array([1.0])
    return SlotMetrics(
        loss=total_loss / max(total_weight, 1e-9),
        mrr=float(np.mean(1.0 / flat)),
        hit_at_1=float(np.mean(flat <= 1)),
        hit_at_5=float(np.mean(flat <= 5)),
        hit_at_20=float(np.mean(flat <= 20)),
        n_slots=int(flat.size),
    )


def pretrain(
    config: Config,
    store: FeatureStore,
    corpus: PageCorpus,
    vocab: Vocab,
    spec: ContextSpec | None = None,
    model: GenDeskModel | None = None,
    checkpoint_name: str = "pretrain",
    log_run: bool = True,
    use_outcome_filter: bool = True,
    save: bool = True,
) -> tuple[GenDeskModel, SlotMetrics]:
    """Run stage-1 pretraining and return the model plus its validation metrics.

    Args:
        use_outcome_filter: When False, pretrain on every teacher candidate rather
            than only the ones that earned their keep. This is the ablation that
            isolates what Netflix's "positive impressions only" rule is worth.
    """
    cfg = config.training.pretrain
    device = resolve_device(config.training.device)
    set_seed(config.training.seed)

    spec = spec or ContextSpec(history_pages=config.corpus.history_pages)
    train_examples = corpus.split("train", pretrain_only=use_outcome_filter)
    valid_examples = corpus.split("valid", pretrain_only=True)
    if not train_examples:
        raise RuntimeError("no training examples survived the positive-outcome filter")

    train_ds = PageDataset(train_examples, vocab, config, spec)
    valid_ds = PageDataset(valid_examples or train_examples[:512], vocab, config, spec)

    train_loader = _loader(train_ds, cfg.batch_size, vocab.pad, True, config.training.num_workers)
    valid_loader = _loader(valid_ds, cfg.batch_size, vocab.pad, False, 0)

    model = model or GenDeskModel(config.model, vocab, store.n_features)
    model.to(device)
    bank = FeatureBank(store, device)

    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = cosine_schedule(optimizer, total_steps, cfg.warmup_ratio, cfg.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    runner = RunLogger(config.run_name, checkpoint_name, config) if log_run else None
    log.info(
        "pretrain_start",
        n_train=len(train_ds),
        n_valid=len(valid_ds),
        steps=total_steps,
        params=model.n_parameters,
        device=str(device),
        context=spec.name,
    )

    step = 0
    best = float("inf")
    best_metrics: SlotMetrics | None = None

    for epoch in range(cfg.epochs):
        model.train()
        for raw in train_loader:
            batch: PageBatch = raw.to(device)
            features = bank.gather(batch.positions)

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss, _ = model.loss_next_token(
                    batch.tokens, features, batch.loss_weights, cfg.label_smoothing
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = clip_gradients(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            if step % cfg.log_every == 0:
                record = {
                    "stage": "pretrain",
                    "epoch": epoch,
                    "step": step,
                    "loss": float(loss),
                    "grad_norm": grad_norm,
                    "lr": scheduler.get_last_lr()[0],
                }
                if runner:
                    runner.log(**record)
                log.info(
                    "pretrain_step",
                    **{k: round(v, 5) if isinstance(v, float) else v for k, v in record.items()},
                )

        metrics = evaluate_slots(model, valid_loader, bank, device, vocab)
        log.info(
            "pretrain_eval",
            epoch=epoch,
            **{k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.as_dict().items()},
        )
        if runner:
            runner.log(stage="pretrain_eval", epoch=epoch, **metrics.as_dict())

        if metrics.loss < best:
            best = metrics.loss
            best_metrics = metrics
            if save:
                save_checkpoint(model, checkpoint_name, config, metrics.as_dict())

    assert best_metrics is not None
    if runner:
        runner.summary(
            {"best": best_metrics.as_dict(), "context": spec.name, "params": model.n_parameters}
        )
    return model, best_metrics
