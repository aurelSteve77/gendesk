"""Stage 2: weighted binary classification post-training.

GenPage's first post-training option reframes generation as *value prediction*:
every token carries a reward derived from user feedback, and the model is trained
with a weighted binary objective rather than pure imitation.

The financial translation is direct. At every instrument slot on a page, the label
for instrument ``i`` is whether ``i`` went on to beat the benchmark over the mandate's
horizon, and the weight is how much, scaled by ``i``'s own volatility. Volatility
scaling is the important detail: without it the objective collapses into "prefer
high-beta names", which is the exact analogue of a recommender learning to prefer
whatever is most popular.

Two objectives are mixed, following GenRec:

* the weighted BCE over the whole eligible catalog at every slot, and
* the original next-token loss at a small weight, which keeps the generative
  distribution intact so the model can still *write a page* rather than merely
  score one.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gendesk.config import Config
from gendesk.corpus.build import PageCorpus
from gendesk.features.store import FeatureStore
from gendesk.model.gendesk import GenDeskModel
from gendesk.tokenization.page import ContextSpec
from gendesk.tokenization.vocab import Vocab
from gendesk.training.checkpoint import RunLogger, save_checkpoint
from gendesk.training.dataset import (
    FeatureBank,
    PageBatch,
    PageDataset,
    RewardBank,
    collate_pages,
)
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
class WBCMetrics:
    """Quality of the value head as a ranker of forward outcomes."""

    wbc_loss: float
    lm_loss: float
    #: Rank correlation between predicted score and realised risk-scaled return.
    rank_ic: float
    #: Mean realised reward of the top decile by predicted score, minus the mean of all.
    decile_spread: float
    auc: float

    def as_dict(self) -> dict:
        return {
            "wbc_loss": self.wbc_loss,
            "lm_loss": self.lm_loss,
            "rank_ic": self.rank_ic,
            "decile_spread": self.decile_spread,
            "auc": self.auc,
        }


def _weighted_bce(
    scores: torch.Tensor,
    rewards: torch.Tensor,
    eligible: torch.Tensor,
    slot_mask: torch.Tensor,
    negatives_per_slot: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Reward-weighted BCE over the eligible catalog at every instrument slot.

    Args:
        scores: ``(B, L, N)`` pre-sigmoid catalog scores.
        rewards: ``(B, N)`` risk-scaled forward active returns.
        eligible: ``(B, N)`` availability mask.
        slot_mask: ``(B, L)`` True at positions whose *next* token is an instrument.
    """
    labels = (rewards > 0).float()
    weights = rewards.abs() * eligible.float()

    if negatives_per_slot > 0 and negatives_per_slot < scores.shape[-1]:
        keep = torch.zeros_like(weights, dtype=torch.bool)
        idx = torch.randint(
            0,
            scores.shape[-1],
            (scores.shape[0], negatives_per_slot),
            device=scores.device,
            generator=generator,
        )
        keep.scatter_(1, idx, True)
        weights = weights * keep.float()

    per_instrument = F.binary_cross_entropy_with_logits(
        scores, labels.unsqueeze(1).expand_as(scores), reduction="none"
    )
    weighted = per_instrument * weights.unsqueeze(1)
    per_position = weighted.sum(-1) / weights.sum(-1, keepdim=True).clamp(min=1e-6)

    mask = slot_mask.float()
    return (per_position * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate_wbc(
    model: GenDeskModel,
    loader: DataLoader,
    bank: FeatureBank,
    rewards: RewardBank,
    device: torch.device,
    config: Config,
    max_batches: int | None = 20,
) -> WBCMetrics:
    """Measure how well the value head orders the forward cross-section."""
    model.eval()
    wbc_total, lm_total, batches = 0.0, 0.0, 0
    ics: list[float] = []
    spreads: list[float] = []
    aucs: list[float] = []

    for i, raw in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch: PageBatch = raw.to(device)
        features = bank.gather(batch.positions)
        reward = rewards.gather(batch.positions)
        eligible = bank.eligibility(batch.positions)

        hidden = model.backbone(model.embed(batch.tokens[:, :-1], features))
        scores = model.scores_from_hidden(hidden, features)
        slot_mask = batch.slot_mask[:, 1:]

        wbc_total += float(
            _weighted_bce(
                scores, reward, eligible, slot_mask, config.training.wbc.negatives_per_slot
            )
        )
        lm_loss, _ = model.loss_next_token(batch.tokens, features, batch.loss_weights)
        lm_total += float(lm_loss)
        batches += 1

        # Rank quality is measured on the first slot of the page, where the model
        # has the context but has not yet committed to anything.
        first_slot = slot_mask.float().argmax(dim=1)
        picked = scores[torch.arange(scores.shape[0], device=device), first_slot]
        for b in range(picked.shape[0]):
            mask = eligible[b]
            if int(mask.sum()) < 30:
                continue
            pred = picked[b][mask].float().cpu().numpy()
            actual = reward[b][mask].float().cpu().numpy()
            if np.std(pred) < 1e-9 or np.std(actual) < 1e-9:
                continue
            pred_rank = np.argsort(np.argsort(pred))
            actual_rank = np.argsort(np.argsort(actual))
            ics.append(float(np.corrcoef(pred_rank, actual_rank)[0, 1]))

            top = pred_rank >= (len(pred_rank) - max(1, len(pred_rank) // 10))
            spreads.append(float(actual[top].mean() - actual.mean()))

            positive = actual > 0
            if 0 < positive.sum() < len(positive):
                order = np.argsort(pred)
                ranks = np.empty(len(pred), dtype=float)
                ranks[order] = np.arange(1, len(pred) + 1)
                n_pos = int(positive.sum())
                n_neg = len(pred) - n_pos
                aucs.append(
                    float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
                )

    model.train()
    return WBCMetrics(
        wbc_loss=wbc_total / max(batches, 1),
        lm_loss=lm_total / max(batches, 1),
        rank_ic=float(np.mean(ics)) if ics else 0.0,
        decile_spread=float(np.mean(spreads)) if spreads else 0.0,
        auc=float(np.mean(aucs)) if aucs else 0.5,
    )


def train_wbc(
    config: Config,
    store: FeatureStore,
    corpus: PageCorpus,
    vocab: Vocab,
    model: GenDeskModel,
    spec: ContextSpec | None = None,
    checkpoint_name: str = "wbc",
    log_run: bool = True,
) -> tuple[GenDeskModel, WBCMetrics]:
    """Run stage-2 post-training starting from a pretrained model."""
    cfg = config.training.wbc
    device = resolve_device(config.training.device)
    set_seed(config.training.seed + 1)

    spec = spec or ContextSpec(history_pages=config.corpus.history_pages)

    # Post-training uses *all* candidate pages, not just the ones that worked: the
    # objective needs the losers as much as the winners in order to learn an ordering.
    train_examples = corpus.split("train")
    valid_examples = corpus.split("valid") or train_examples[:1024]

    train_ds = PageDataset(train_examples, vocab, config, spec)
    valid_ds = PageDataset(valid_examples, vocab, config, spec)
    collate = partial(collate_pages, pad_id=vocab.pad)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )

    model.to(device)
    bank = FeatureBank(store, device)
    positions = [ex.position for ex in train_examples + valid_examples]
    rewards = RewardBank(store, positions, config.corpus.reward_horizon, cfg.reward_clip, device)

    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = cosine_schedule(optimizer, total_steps, cfg.warmup_ratio, 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    runner = RunLogger(config.run_name, checkpoint_name, config) if log_run else None
    log.info("wbc_start", n_train=len(train_ds), steps=total_steps, device=str(device))

    step = 0
    best = -float("inf")
    best_metrics: WBCMetrics | None = None

    for epoch in range(cfg.epochs):
        model.train()
        for raw in train_loader:
            batch: PageBatch = raw.to(device)
            features = bank.gather(batch.positions)
            reward = rewards.gather(batch.positions)
            eligible = bank.eligibility(batch.positions)

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                hidden = model.backbone(model.embed(batch.tokens[:, :-1], features))
                scores = model.scores_from_hidden(hidden, features)
                wbc_loss = _weighted_bce(
                    scores,
                    reward,
                    eligible,
                    batch.slot_mask[:, 1:],
                    cfg.negatives_per_slot,
                )
                logits = model.logits_from_hidden(hidden, features)
                lm_loss = (
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        batch.tokens[:, 1:].reshape(-1),
                        reduction="none",
                    ).view_as(batch.tokens[:, 1:])
                    * batch.loss_weights[:, 1:]
                ).sum() / batch.loss_weights[:, 1:].sum().clamp(min=1.0)
                loss = wbc_loss + cfg.lm_loss_weight * lm_loss

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
                    "stage": "wbc",
                    "epoch": epoch,
                    "step": step,
                    "wbc_loss": float(wbc_loss),
                    "lm_loss": float(lm_loss),
                    "grad_norm": grad_norm,
                    "lr": scheduler.get_last_lr()[0],
                }
                if runner:
                    runner.log(**record)
                log.info(
                    "wbc_step",
                    **{k: round(v, 5) if isinstance(v, float) else v for k, v in record.items()},
                )

        metrics = evaluate_wbc(model, valid_loader, bank, rewards, device, config)
        log.info("wbc_eval", epoch=epoch, **{k: round(v, 5) for k, v in metrics.as_dict().items()})
        if runner:
            runner.log(stage="wbc_eval", epoch=epoch, **metrics.as_dict())

        # Selection is on validation rank IC: the value head's job is ordering the
        # cross-section, and the BCE level is not comparable across epochs once the
        # reward weights shift the effective sample.
        if metrics.rank_ic > best:
            best = metrics.rank_ic
            best_metrics = metrics
            save_checkpoint(model, checkpoint_name, config, metrics.as_dict())

    assert best_metrics is not None
    if runner:
        runner.summary({"best": best_metrics.as_dict()})
    return model, best_metrics
