# GenDesk

**LLM-native, end-to-end generative construction of an institutional research desk page.**

A decoder-only transformer that writes an entire investment page -- themed rows of
instruments, sized into a portfolio -- in one autoregressive pass, conditioned on the
mandate asking and the macro regime it is asking in. It replaces the classical
retrieve -> rank -> diversify pipeline with a single model, and it is trained the way
Netflix trains GenRec and GenPage: pretraining on outcome-filtered pages, weighted
binary classification post-training, then whole-page reinforcement learning.

Built from the two Netflix Tech Blog posts on
[GenRec](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3)
and
[GenPage](https://netflixtechblog.com/genpage-towards-end-to-end-generative-homepage-construction-at-netflix-77146fba8a08),
transposed to financial markets -- including the parts of the analogy that break.

---

## Why this is not just a recommender with tickers in it

Three things do not survive the translation, and dealing with them is most of the work:

**1. The catalog is not stationary.** A movie is the same object next week. Whether
`NVDA` belongs in a "Momentum Leaders" row depends on this month's momentum, not on
the fact that it is NVDA. So the entity representation is
`E[i,t] = id_embedding[i] + W @ features[i,t]` -- Netflix's semantic fusion, but where
the "content" is the instrument's *current market state*. This solves cold start the
same way it does at Netflix, and additionally makes the vocabulary time-varying, which
a recommender's never is. It also forces the output projection to be an inner product
against the current representation rather than a fixed weight row -- which is exactly
what makes the whole catalog scorable in a single prefill pass.

**2. There is no production log.** Netflix pretrains on impressions members actually
responded to. So the corpus is manufactured in the same spirit: a stochastic,
regime-aware factor screen proposes candidate pages, each is scored by what it went on
to earn, and everything below a reward quantile is discarded. The teacher never sees
forward returns; selection pressure enters only through the filter. The teacher's own
screen is a baseline in the backtest, so the gap between it and GenDesk is what the
generative layer added on top of its own teacher.

**3. Engagement is not the reward.** A recommender trained on raw engagement learns to
recommend whatever is popular. A desk trained on raw forward return learns to buy
whatever has the highest beta -- which looks like skill until it does not. Every reward
here is benchmark-relative, volatility-scaled, and penalised for drawdown and turnover.

---

## What it does

```bash
gendesk generate --persona macro_overlay --instruction "add duration hedges and cut energy exposure"
```

```
                    Macro Overlay - 2026-07-31
Row                  Instruments                              Weight
Macro Hedge          TLT, IEF, GLD, TIP, LQD, UUP              31.6%
Quality Ballast      CHD, KO, PG, WEC, AEP, ED                 24.1%
Trend & Breakout     AVGO, GE, ORCL, WMT, RTX, LIN             15.9%
Crowding Unwind      ...
```

Five rows, six instruments each, subject to the mandate's rules -- and the rules are
enforced as **token masks during generation**, so a page is compliant by construction
rather than filtered afterwards. Nothing is rejected and retried, which matters because
rejection sampling would silently distort the RL gradient.

The Streamlit app (`make app`) shows the generated page with its evidence, the regime
it was written in, the constraint audit, the out-of-sample results, the ablation grid,
the RL diversification study, and the raw token stream the model actually sees.

---

## Architecture

| Stage | What happens |
|---|---|
| **Tokenizer** | 415 tokens: 8 structural, 6 mandates, 3 risk budgets, 4 horizons, 24 regime buckets, 8 row archetypes, **362 instruments at one token each**. A page is ~50 tokens. |
| **Backbone** | Decoder-only, 8 layers, d=256, GQA, RoPE, SwiGLU. 6.4M parameters. Input embedding untied from output projection, as GenPage specifies. |
| **Stage 1** | Next-token pretraining on outcome-filtered pages. Metric: MRR of the held-out instrument among the eligible catalog. |
| **Stage 2** | Weighted binary classification over the whole catalog at every slot, per-slot rewards from volatility-scaled forward active returns, mixed with a small LM loss. |
| **Stage 3** | Dr. GRPO on the page-level portfolio reward. Group-relative baseline, no std normalisation, KL to the frozen stage-2 policy. |
| **Decoding** | Constraint masks (dedup, sector caps, mandate exclusions, liquidity, row pinning) + hybrid row decoding. |
| **Evaluation** | Walk-forward out-of-sample, seven baselines, block-bootstrap inference, deflated Sharpe, PBO. |

Full detail: [`docs/architecture.md`](docs/architecture.md) ·
[`docs/methodology.md`](docs/methodology.md) ·
[decision records](docs/adr/)

### Why Dr. GRPO and not GRPO

Vanilla GRPO divides the advantage by the group's standard deviation. That scales *up*
the gradient on prompts where every sampled page scored about the same -- in markets,
precisely the quiet regimes where the differences between pages are noise. Dr. GRPO
drops that division. ([ADR 0003](docs/adr/0003-dr-grpo-for-page-level-rl.md))

### Why constraints are masks

A mandate that forbids commodity funds is not a page that gets rejected; it is a token
that is never sampled. Because the instrument block of the vocabulary is contiguous,
every rule is an elementwise operation on a `(batch, 362)` boolean tensor, batched
across a whole GRPO group. `tests/test_constraints.py` asserts compliance against an
*untrained* model -- whose preferences are essentially random, the hardest case for the
mask. ([ADR 0005](docs/adr/0005-constraints-as-token-masks.md))

---

## Data

362 US instruments (equities across all eleven GICS sectors, plus broad, style, sector,
rates, credit, real-asset and currency funds), **5,428 sessions from 2005-01-03 to
2026-07-31**, split- and dividend-adjusted, from Yahoo's public daily endpoint.

The panel carries an explicit **availability mask** -- listed, priced within three
sessions, and above a liquidity floor -- computed *before* any forward fill, so filling
short gaps cannot make the mask lie. Every stage reads it.

Splits are **purged**: an example joins the training split only if its entire 21-session
reward window closes before the cutoff. `tests/test_leakage.py` multiplies every price
after a cut date by 3x and asserts that no feature and no regime bucket at or before the
cut moves -- and that the perturbation *does* change values after it, so the test cannot
pass for the wrong reason.

---

## Results

Four claims from the Netflix posts were worth testing rather than repeating.
**Two replicated, one did not, and one produced a clean null** -- all four are written
up in [`docs/findings.md`](docs/findings.md), including the two that make the project
look worse:

- **The generative page beats the multi-stage pipeline it replaces** on out-of-sample
  Sharpe, drawdown and turnover -- but every paired bootstrap interval contains zero, so
  the honest claim is *competitive with*, not *better than*.
- **Diversification did not emerge under RL.** Reward improved over 300 steps while every
  diversity measure fell. The likely reason -- this reward is already volatility-scaled,
  so it pays for diversification directly and leaves no slack for it to emerge into -- is
  a testable hypothesis, stated as one.
- **The value head learns nothing.** Validation rank IC is ~0 for 21-day forward returns
  from sixteen price features, and the WBC variant is the worst strategy in the table.
  What the model is good at is *composing a page*, not forecasting.

Tables below are generated by `gendesk eval report`; nothing here is transcribed by hand.
Read [`docs/limitations.md`](docs/limitations.md) first -- the constraints on what any of
this means are listed there before the numbers.

<!-- RESULTS:BEGIN -->
*Run `make pipeline` to populate this section.*
<!-- RESULTS:END -->

---

## Quickstart

```bash
make setup          # venv + CUDA torch + package
make data           # download and cache the price panel (~1 min)
make features       # point-in-time features and regime buckets
make corpus         # teacher pages, scored and filtered
make train          # stage 1 + stage 2
make rl             # stage 3
make backtest       # walk-forward out-of-sample vs every baseline
make app            # the Streamlit desk
```

Or the whole thing: `scripts/run_pipeline.sh`.

Everything runs on one consumer GPU (developed on a 6 GB GTX 1660 Ti) and falls back
to CPU.

## Repository layout

```
src/gendesk/
  data/          catalog, Yahoo acquisition, aligned panel + availability mask
  features/      point-in-time cross-sectional features and regime buckets
  tokenization/  the vocabulary and the page <-> sequence encoding
  corpus/        row archetypes, teacher policy, purged corpus construction
  model/         backbone, entity representations, generative + value heads
  training/      pretraining, WBC post-training, Dr. GRPO
  decoding/      constraint engine and page generator
  portfolio/     page -> weights, and the reward model
  evaluation/    backtest, baselines, statistics, ablations, diversity, reports
  steering/      plain-English instructions -> context tokens and masks
app/             Streamlit front end
tests/           leakage, constraints, tokenizer, model, reward, backtest, end-to-end
docs/            architecture, methodology, limitations, decision records
```

## Development

```bash
make lint    # ruff
make type    # mypy
make test    # pytest (offline: the whole suite runs on a synthetic market)
```

CI runs lint, format check, mypy and the offline test suite on Python 3.11 and 3.12.

---

## Not investment advice

A research prototype built to explore an architecture, on public end-of-day data, with
a flat cost model and a survivorship-biased universe. It is not a trading system and
carries no claim that the strategy would make money after real-world frictions.
