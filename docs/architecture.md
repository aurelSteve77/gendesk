# Architecture

## The translation

Netflix's GenRec and GenPage posts describe replacing a multi-stage recommendation
pipeline with a single decoder-only transformer that writes the whole homepage. The
translation to a research desk is close to literal, and the parts that do *not*
translate are the interesting ones.

| Netflix | GenDesk | Note |
|---|---|---|
| Member | Mandate (a persona: pension, endowment, pod, macro overlay, ...) | Six synthetic allocators with different risk budgets, horizons and constraints |
| Title (movie, show, game) | Instrument (equity or fund) | 362 US listings, one token each |
| Homepage | Desk page: five themed rows of six instruments | |
| Row type ("Because you watched", "Top 10") | Row archetype (Momentum Leaders, Macro Hedge, ...) | Eight theses, each a scoring rule and an eligibility rule |
| Viewing history | The desk's own previous pages | At inference this is the model's own output, not a teacher's |
| Request context | Macro regime: volatility, curve, breadth, dispersion, correlation | Eight axes, terciled against a trailing window |
| Positive impressions | Teacher pages that beat a reward quantile | Selection pressure applied before the model ever sees the data |
| Member satisfaction | Risk-adjusted, benchmark-relative, turnover-penalised forward return | |
| Semantic embedding fusion (synopsis, cast) | Fusion of the instrument's *current market state* | The one place the analogy improves in translation -- see below |
| Constrained decoding (dedup, row pinning) | Mandate rules as token masks (dedup, sector caps, exclusions, pinned rows) | |
| Hybrid row decoding | Same idea, same purpose | |
| Dr. GRPO whole-page RL | Same algorithm, portfolio reward | |

## The pipeline

```
configs/universe.yaml                 configs/default.yaml
        |                                     |
        v                                     v
  data/panel.py  --->  features/  --->  corpus/  --->  training/  --->  decoding/
   (Yahoo EOD)        cross_section     teacher +      pretrain          constraints
   362 x 5,428        + regimes         outcome        -> WBC            + hybrid
   aligned panel      point-in-time     filter         -> Dr. GRPO       row decoding
        |                                     |                |
        +------------------> portfolio/ <-----+                |
                             weights + reward                   |
                                     |                          |
                                     v                          v
                              evaluation/backtest  <----  evaluation/strategies
                              baselines, statistics,      (the model as a
                              ablations, diversity         weight function)
```

## The three things that had to change

### 1. The catalog is not stationary

A movie is the same object next week. A stock is not: what `NVDA` *is*, for the
purpose of deciding whether it belongs in a Momentum Leaders row, is its current
momentum, volatility, beta and correlation. Netflix's semantic fusion exists to solve
cold start -- give a new title a representation before anyone has watched it. Here the
same mechanism additionally solves non-stationarity:

```
E[i, t] = id_embedding[i] + W_fuse @ features[i, t]
```

The identity term is what the model learns about the instrument; the fusion term is
what it learns about instruments *in a given state*. Cold start falls out for free (an
instrument with no trained identity keeps the second term), and the ablation grid
measures what the fusion term is worth.

Because entity representations move with the market, the **output** projection has to
move with them too. Logits over instruments are an inner product between the hidden
state and the *current* entity representation rather than a fixed row of a weight
matrix. That is also what makes the whole catalog scorable in a single forward pass,
which is GenRec's prefill-only serving mode.

### 2. There is no production log

Netflix pretrains on impressions that were actually served. There is no served desk
here, so the corpus is manufactured in the same spirit: a stochastic, regime-aware
factor screen proposes several candidate pages per (date, mandate) cell, each is
scored by what it went on to earn, and the ones below a reward quantile are discarded.

The teacher never sees forward returns. Selection pressure enters only through the
filter. This matters for interpretation: the model is not distilling an oracle, it is
learning the language of pages that worked, and the RL stage is free to leave the
teacher behind. The `teacher_book` baseline in the backtest is exactly the screen the
corpus was written by, so the gap between it and GenDesk is what the generative layer
added on top of its own teacher.

### 3. The reward has to be risk-adjusted or the whole thing degenerates

A recommender trained on raw engagement learns to recommend whatever is popular. A
desk trained on raw forward return learns to buy whatever has the highest beta, which
looks like skill until it does not. Every reward in this system is therefore

* **benchmark-relative** -- the market's own return is common to every candidate page
  and would otherwise drown the differences between them,
* **volatility-scaled** -- an instrument's forward active return is divided by its own
  volatility, and a page's by its volatility budget,
* **penalised for path and turnover** -- drawdown inside the window and trading against
  the previous book both cost.

The same construction supplies both the page-level reward for RL and the per-slot
rewards for weighted binary classification, so the two objectives cannot disagree.

## Training stages

| Stage | Objective | What it produces |
|---|---|---|
| 1. Pretraining | Weighted next-token cross-entropy on outcome-filtered pages | A model that writes plausible, regime-appropriate, mandate-appropriate pages |
| 2. WBC post-training | Reward-weighted BCE over the eligible catalog at every slot, mixed with a small language-modelling loss | A value head that ranks the forward cross-section, without destroying the generative distribution |
| 3. Dr. GRPO RL | Group-relative policy optimisation on the page-level reward | Whole-page optimisation: interactions between rows enter the objective |

Stage 3 uses Dr. GRPO rather than vanilla GRPO. Vanilla divides the advantage by the
group standard deviation, which inflates the gradient exactly where the candidates all
scored about the same -- in markets, that is the quiet regimes where the differences
are noise. It also normalises the loss by sequence length; every page here has the
same number of decision steps, so that term is harmless, but there is no reason to
keep it.

## Serving

Generation is constrained: business rules are token masks, not a post-hoc filter, so a
generated page is compliant by construction and nothing is ever rejected and retried.
`tests/test_constraints.py` asserts this against an *untrained* model, whose
preferences are essentially random -- the hardest case for the mask.

Hybrid row decoding autoregresses the first two instruments of each row and fills the
rest from a single forward pass, sampling without replacement from that one
distribution (a Plackett-Luce draw, so the path stays exactly scorable). The number of
sequential model invocations therefore stops scaling with the number of slots.

RL always samples with hybrid decoding **off**: the fill step conditions several slots
on one hidden state, so a teacher-forced recomputation would not reproduce the sampling
distribution and the importance ratio would be wrong. Hybrid decoding is a serving
choice, measured separately.

## Module map

| Module | Responsibility |
|---|---|
| `gendesk.data` | Catalog definition, Yahoo EOD acquisition, aligned panel with an availability mask |
| `gendesk.features` | Point-in-time cross-sectional features and macro regime buckets |
| `gendesk.tokenization` | The vocabulary and the page <-> sequence encoding |
| `gendesk.corpus` | Row archetypes, the teacher policy, corpus construction and purged splits |
| `gendesk.model` | Backbone, entity representations, generative and value heads |
| `gendesk.training` | The three stages, datasets, checkpointing |
| `gendesk.decoding` | Constraint engine and the page generator |
| `gendesk.portfolio` | Page -> weights, and the reward model |
| `gendesk.evaluation` | Backtest engine, baselines, statistics, ablations, diversity, reports |
| `gendesk.steering` | Plain-English instructions -> context tokens and masks |
