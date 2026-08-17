# ADR 0003: Dr. GRPO, not vanilla GRPO, for page-level optimisation

**Status:** accepted

## Context

Stage 3 optimises the whole page against a portfolio reward. GenPage uses a Dr. GRPO
variant. Group-relative policy optimisation is a natural fit: sample a group of pages
from one prompt, use the group mean as the baseline, and skip the value network
entirely.

Vanilla GRPO applies two normalisations that Dr. GRPO removes:

1. the advantage is divided by the group's standard deviation, and
2. the loss is divided by sequence length.

## Decision

Use Dr. GRPO: centre advantages on the group mean, do **not** divide by the group
standard deviation, and normalise the loss by a constant (group size x decision steps x
prompts) rather than by sequence length.

## Consequences

**The std normalisation is actively harmful in this domain.** It scales up the gradient
on prompts where every sampled page scored about the same. In markets those are the
quiet regimes -- narrow dispersion, everything correlated, the differences between
candidate pages are noise. Vanilla GRPO would weight exactly those prompts most
heavily. Removing the division means a low-dispersion group contributes a small
gradient, which is the correct behaviour.

**The length normalisation is a non-issue here but is dropped anyway.** Every page has
the same number of decision steps (5 rows x (1 archetype + 6 instruments)), so the term
is a constant. Keeping it would be harmless; removing it keeps the implementation
faithful to the algorithm as published, and it would start to matter the moment pages
became variable-length.

**A KL penalty against the frozen post-trained policy is retained** (k3 estimator:
unbiased, non-negative, low variance). Without it the policy drifts away from the
generative distribution that pretraining and WBC established and starts emitting pages
that are legal but incoherent.

**Sampling must be purely autoregressive during RL.** Hybrid row decoding fills a row's
tail from a single hidden state, so a teacher-forced recomputation would not reproduce
the sampling distribution and the importance ratio would be silently wrong. The RL
stage therefore calls the generator with `hybrid=False`; hybrid decoding is a
serving-time choice, benchmarked separately.

**Masks are recorded at sampling time and reapplied at recomputation.** The behaviour
policy is the *constrained* distribution, not the raw softmax. Scoring the sample under
the unconstrained distribution would make every ratio wrong by the mass the constraint
removed. `tests/test_constraints.py::test_step_masks_never_permit_an_illegal_token`
guards the recording path.

## Alternatives considered

*PPO with a learned value head.* A page-level scalar reward with no intermediate
signal makes a value function hard to fit, and it doubles the parameter count of a
model whose whole point is to be small. Group-relative baselines need no critic.

*Plain REINFORCE with a moving-average baseline.* Simpler, but the baseline is stale
across regimes -- the same page is worth very different amounts in different market
states -- and a group sampled from the *same* prompt controls for that exactly.
