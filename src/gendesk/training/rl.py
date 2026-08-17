"""Stage 3: page-level reinforcement learning with Dr. GRPO.

GenPage's second post-training option optimises the *whole page* against a reward
model, which is what lets interactions between rows -- overlap, redundancy, hedging
-- enter the objective at all. A per-slot loss cannot see them.

The algorithm is Dr. GRPO (GRPO Done Right): sample a group of pages from the same
prompt, use the group mean as the baseline, and drop the two normalisations that bias
vanilla GRPO -- dividing the advantage by the group standard deviation, and dividing
the loss by sequence length. The first inflates the gradient on prompts where every
candidate scored about the same, which is precisely where the signal is noise; the
second biases against long sequences. Every page here has the same number of decision
steps, so the length term would be harmless, but the std term is not: quiet markets
produce exactly the low-dispersion groups it over-weights.

Sampling must be purely autoregressive during RL. Hybrid decoding fills a row's tail
from a single hidden state, so a teacher-forced recomputation would not reproduce the
sampling distribution and the importance ratio would be wrong. The generator is
therefore called with ``hybrid=False`` here, and hybrid decoding is a serving-time
choice measured separately in the latency study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gendesk.config import Config, PersonaConfig
from gendesk.corpus.build import PageCorpus
from gendesk.decoding.generate import GenerationResult, PageGenerator
from gendesk.evaluation.diversity import average_diversity
from gendesk.features.regimes import REGIME_AXES
from gendesk.features.store import FeatureStore
from gendesk.model.gendesk import GenDeskModel
from gendesk.portfolio.reward import evaluate_page
from gendesk.portfolio.weights import page_weights
from gendesk.tokenization.page import Page, PageContext
from gendesk.tokenization.vocab import Vocab
from gendesk.training.checkpoint import RunLogger, save_checkpoint
from gendesk.training.schedule import build_optimizer, clip_gradients, resolve_device
from gendesk.utils.logging import get_logger
from gendesk.utils.seed import set_seed

log = get_logger(__name__)

#: Steps averaged before the reward is trusted enough to select a checkpoint on.
SMOOTHING_WINDOW = 20
#: Minimum improvement in the smoothed reward that justifies writing a checkpoint.
CHECKPOINT_TOLERANCE = 1e-3


@dataclass
class RLMetrics:
    mean_reward: float
    reward_std: float
    kl: float
    entropy: float
    diversity: dict

    def as_dict(self) -> dict:
        return {
            "mean_reward": self.mean_reward,
            "reward_std": self.reward_std,
            "kl": self.kl,
            "entropy": self.entropy,
            **{f"div_{k}": v for k, v in self.diversity.items()},
        }


class PromptBook:
    """Prompts to train on, plus the book each prompt is trading away from.

    The turnover term in the reward needs a previous holding. Using the teacher's
    deterministic book at the previous rebalance keeps the RL reward on the same
    scale as the corpus reward, and -- because the book is chosen ex ante -- keeps
    the forward window of one page out of the context of the next.
    """

    def __init__(self, corpus: PageCorpus, config: Config, split: str = "train") -> None:
        self.personas = {p.name: p for p in config.personas}
        books: dict[tuple[str, int], Page] = {}
        for example in corpus.examples:
            if example.is_book:
                books[(example.persona, example.position)] = example.to_page()

        self.entries: list[tuple[str, int, dict, list[list[str]], Page | None]] = []
        by_persona: dict[str, list[int]] = {}
        for persona, position in sorted(books, key=lambda k: (k[0], k[1])):
            by_persona.setdefault(persona, []).append(position)

        wanted = {
            (example.persona, example.position): example
            for example in corpus.examples
            if example.split == split and example.is_book
        }
        for (persona, position), example in sorted(wanted.items(), key=lambda kv: kv[0][1]):
            history = example.history
            positions = by_persona[persona]
            rank = positions.index(position)
            previous = books.get((persona, positions[rank - 1])) if rank > 0 else None
            self.entries.append((persona, position, example.regimes, history, previous))

    def __len__(self) -> int:
        return len(self.entries)

    def sample(self, rng: np.random.Generator, n: int) -> list[tuple]:
        idx = rng.choice(len(self.entries), size=min(n, len(self.entries)), replace=False)
        return [self.entries[int(i)] for i in idx]


def _recompute_logprobs(
    model: GenDeskModel,
    result: GenerationResult,
    features: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probabilities of the sampled tokens under ``model``.

    Returns ``(logprobs, entropy)`` at the recorded decision positions, with the
    same masks that were in force during sampling reapplied, so the distribution is
    identical to the one that generated the sample.
    """
    hidden = model.backbone(model.embed(result.tokens[:, :-1], features))
    logits = model.logits_from_hidden(hidden, features)

    decision_inputs = result.step_positions - 1
    selected = logits.index_select(1, decision_inputs)
    masked = selected.masked_fill(~result.step_masks, float("-inf")) / max(temperature, 1e-6)

    log_probs = torch.log_softmax(masked, dim=-1)
    chosen = log_probs.gather(2, result.step_tokens.unsqueeze(-1)).squeeze(-1)

    probs = log_probs.exp()
    entropy = -(probs * torch.nan_to_num(log_probs, neginf=0.0)).sum(-1)
    return chosen, entropy


def train_rl(
    config: Config,
    store: FeatureStore,
    corpus: PageCorpus,
    vocab: Vocab,
    model: GenDeskModel,
    checkpoint_name: str = "rl",
    log_run: bool = True,
) -> tuple[GenDeskModel, list[dict]]:
    """Run Dr. GRPO page-level optimisation. Returns the model and its metric trace."""
    cfg = config.training.rl
    device = resolve_device(config.training.device)
    set_seed(config.training.seed + 2)
    rng = np.random.default_rng(config.training.seed + 2)

    # Eval mode throughout: the sampled tokens came from a dropout-free forward pass,
    # so recomputing their log-probabilities *with* dropout would make the importance
    # ratio differ from 1 even at the first inner epoch, and would inject a spurious KL
    # against a reference that is, at step 0, the identical network. Gradients still flow.
    model.to(device).eval()
    reference = GenDeskModel(config.model, vocab, store.n_features).to(device)
    reference.load_state_dict(model.state_dict())
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    generator = PageGenerator(model, vocab, store, config, device)
    prompts = PromptBook(corpus, config, split="train")
    if not prompts.entries:
        raise RuntimeError("no training prompts available for RL")

    personas: dict[str, PersonaConfig] = {p.name: p for p in config.personas}
    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)
    runner = RunLogger(config.run_name, checkpoint_name, config) if log_run else None

    log.info("rl_start", n_prompts=len(prompts), steps=cfg.steps, device=str(device))
    trace: list[dict] = []
    best_reward = -float("inf")

    for step in range(cfg.steps):
        batch_rewards: list[float] = []
        step_losses: list[float] = []
        step_kl: list[float] = []
        step_entropy: list[float] = []
        diversity_samples: list[dict] = []

        optimizer.zero_grad(set_to_none=True)

        for persona_name, position, regimes, history, previous in prompts.sample(
            rng, cfg.prompts_per_step
        ):
            persona = personas[persona_name]
            context = PageContext(
                persona=persona_name,
                risk_budget=persona.risk_budget,
                horizon_days=persona.horizon_days,
                regimes={axis: int(regimes[axis]) for axis in REGIME_AXES},
                history=tuple(tuple(h) for h in history),
            )

            with torch.no_grad():
                result = generator.generate(
                    context,
                    persona,
                    position,
                    n_samples=cfg.group_size,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    hybrid=False,
                )

            previous_weights = (
                page_weights(previous, store, position, persona) if previous else None
            )
            rewards = np.array(
                [
                    evaluate_page(
                        page,
                        store,
                        position,
                        persona,
                        config.corpus,
                        previous_weights=previous_weights,
                    ).total
                    for page in result.pages
                ],
                dtype=np.float64,
            )
            batch_rewards.extend(rewards.tolist())

            # Dr. GRPO: centre on the group mean, do NOT divide by the group std.
            advantage = torch.as_tensor(
                rewards - rewards.mean(), dtype=torch.float32, device=device
            ).unsqueeze(1)

            features = generator.features_for(position, cfg.group_size)
            new_logprobs, entropy = _recompute_logprobs(model, result, features, cfg.temperature)
            with torch.no_grad():
                ref_logprobs, _ = _recompute_logprobs(reference, result, features, cfg.temperature)

            ratio = torch.exp(new_logprobs - result.step_logprobs)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantage
            policy_loss = -torch.min(unclipped, clipped)

            # k3 estimator: unbiased, non-negative, low variance.
            delta = ref_logprobs - new_logprobs
            kl = torch.exp(delta) - delta - 1.0

            # Constant normaliser (group size x steps), not per-sequence length.
            denom = float(cfg.group_size * new_logprobs.shape[1] * cfg.prompts_per_step)
            loss = (policy_loss + cfg.kl_coef * kl).sum() / denom
            loss.backward()

            step_losses.append(float(loss))
            step_kl.append(float(kl.mean()))
            step_entropy.append(float(entropy.mean()))

            if cfg.diversity_every and step % cfg.diversity_every == 0:
                diversity_samples.append(average_diversity(result.pages, store, position, persona))

        grad_norm = clip_gradients(model.parameters(), cfg.grad_clip)
        optimizer.step()

        diversity = (
            {
                key: float(np.mean([d[key] for d in diversity_samples]))
                for key in diversity_samples[0]
            }
            if diversity_samples
            else {}
        )
        metrics = RLMetrics(
            mean_reward=float(np.mean(batch_rewards)),
            reward_std=float(np.std(batch_rewards)),
            kl=float(np.mean(step_kl)),
            entropy=float(np.mean(step_entropy)),
            diversity=diversity,
        )
        record = {
            "stage": "rl",
            "step": step,
            "loss": float(np.mean(step_losses)),
            "grad_norm": grad_norm,
            **metrics.as_dict(),
        }
        trace.append(record)
        if runner:
            runner.log(**record)
        if step % cfg.log_every == 0:
            log.info(
                "rl_step",
                **{k: round(v, 5) if isinstance(v, float) else v for k, v in record.items()},
            )

        # Checkpoint on a smoothed reward: a single group is far too noisy to select on.
        # The improvement has to be material, otherwise a 25 MB write lands on almost
        # every step and the run becomes I/O bound rather than compute bound.
        window = [r["mean_reward"] for r in trace[-SMOOTHING_WINDOW:]]
        if len(trace) >= SMOOTHING_WINDOW:
            smoothed = float(np.mean(window))
            if smoothed > best_reward + CHECKPOINT_TOLERANCE:
                best_reward = smoothed
                save_checkpoint(model, checkpoint_name, config, {"mean_reward": best_reward})

    if best_reward == -float("inf"):
        # Too few steps for the smoothed criterion to fire; keep the final policy.
        save_checkpoint(
            model, checkpoint_name, config, {"mean_reward": float(np.mean(batch_rewards))}
        )
    if runner:
        runner.summary({"best_mean_reward": best_reward, "n_steps": cfg.steps})
    return model, trace


__all__ = ["PromptBook", "RLMetrics", "train_rl"]
