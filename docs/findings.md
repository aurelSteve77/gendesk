# Findings

Four claims from the Netflix posts were worth testing rather than repeating. Two
replicated, one did not, and one produced a clean null. This document states all four,
including the two that make the project look worse.

Numbers come from `artifacts/reports/`; regenerate with `scripts/run_pipeline.sh`.
Read [`limitations.md`](limitations.md) alongside this -- in particular the fact that a
single 4.5-year out-of-sample window is roughly one macro regime, not 1,130 independent
observations.

---

## 1. Does the generative page beat the pipeline it replaces?

**Directionally yes; statistically not demonstrated.**

Out-of-sample, 2022-01-31 to 2026-07-31, equal-weighted across six mandates, net of
5 bp one-way costs, rebalanced every 21 sessions:

| Strategy | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| 12-1 momentum | 18.4% | 20.9% | 0.82 | -20.4% |
| **GenDesk (RL)** | **12.9%** | **13.6%** | **0.81** | **-14.1%** |
| GenDesk (pretrained) | 13.1% | 14.0% | 0.81 | -15.0% |
| Teacher screen | 12.1% | 14.2% | 0.74 | -16.2% |
| S&P 500 ETF | 13.5% | 17.5% | 0.70 | -22.1% |
| Low volatility | 10.4% | 12.4% | 0.70 | -16.2% |
| Multi-stage pipeline | 10.2% | 12.4% | 0.69 | -13.3% |
| Equal weight | 11.5% | 15.8% | 0.64 | -18.9% |
| GenDesk (WBC head) | 9.3% | 14.6% | 0.54 | -18.1% |

GenDesk (RL) beats the multi-stage pipeline it is designed to replace (+0.12 Sharpe),
the teacher screen it learned from (+0.09), and the benchmark (+0.15) -- while running
at three-quarters of the market's volatility and roughly two-thirds of its drawdown.

**And none of those differences is significant.** Paired block-bootstrap tests
(21-session blocks, both legs resampled on the same indices):

| Comparison | Sharpe difference | 95% CI | p |
|---|---|---|---|
| vs multi-stage pipeline | +0.12 | [-0.31, +0.56] | 0.60 |
| vs teacher screen | +0.09 | [-0.10, +0.28] | 0.37 |
| vs benchmark | +0.15 | [-0.32, +0.62] | 0.52 |
| vs equal weight | +0.20 | [-0.16, +0.54] | 0.27 |

Every interval contains zero. The correct reading is that **the architecture is
competitive with, not proven superior to, the pipeline it replaces**, on this window.

The probability of backtest overfitting across the ten strategies is **88%** -- high,
and worth interpreting rather than just quoting. PBO asks how often the in-sample best
configuration lands below the out-of-sample median. When the candidates are ten
long-only equity books with 0.85+ pairwise correlation, their ranking *is* mostly
noise, so a high PBO is the expected reading and is a statement about the comparison
being underpowered rather than about GenDesk specifically.

Where the result is more solid: GenDesk (RL) achieves the highest Sharpe of any
mandate-aware strategy with the second-lowest drawdown, and it does so with lower
turnover than the teacher screen it was distilled from.

## 2. Does prompt content beat parameter count?

See the generated table in [`../artifacts/reports/RESULTS.md`](../artifacts/reports/RESULTS.md).
The grid holds corpus, validation split and epochs fixed and moves one variable at a
time: a four-rung context ladder at fixed capacity, and a four-rung capacity ladder
spanning ~0.9M to ~19M parameters at fixed full context.

## 3. Does diversification emerge from whole-page RL?

**No -- it went the other way, and the reason is instructive.**

Over 300 Dr. GRPO steps the mean group reward improved from **-0.49 to -0.11**, so the
policy was learning. Meanwhile every diversity measure moved *against* the GenPage
finding:

| Measure | Start | End | Direction |
|---|---|---|---|
| Mean pairwise correlation | 0.142 | 0.209 | less diversified |
| Diversification ratio | 2.14 | 1.91 | less diversified |
| Effective number of bets | 26.3 | 25.3 | less diversified |
| Distinct sectors per page | 9.7 | 9.4 | less diversified |

Policy entropy was flat (2.33 to 2.35), so this is not collapse -- the model kept
exploring and simply chose to concentrate.

The most likely explanation is that **the reward already pays for diversification, so
there is no slack for it to emerge into.** Netflix's engagement reward contains nothing
about diversity, which leaves room for it to appear as a free byproduct of whole-page
optimisation. This reward is volatility-scaled and drawdown-penalised: a concentrated
page is punished through its realised volatility before any diversity metric sees it.
Once that pressure is already priced in, the remaining way to increase reward is to
concentrate into higher-conviction names, and that is what the policy did.

That is a claim about *this* reward, not a refutation of the GenPage result. The
testable follow-up is to strip the volatility scaling from the reward and re-run: if
diversity then rises, the mechanism above is confirmed.

## 4. Does the value head learn to rank forward returns?

**No, and that is the honest answer rather than a bug.**

The WBC stage's validation rank IC is approximately **zero** (0.0005 at the selected
epoch, AUC 0.49) while its training loss falls steadily -- textbook overfitting. As a
tradable strategy the WBC head is the *worst* variant in the table above (Sharpe 0.54).

This is what should be expected: predicting 21-day forward cross-sectional returns from
sixteen price and volume features, on 25k training pages, is close to the hardest
version of a problem that is barely solvable with far richer data. Reporting it as a
finding rather than tuning until it looked better is the point.

It also explains the ordering in the backtest. The pretrained and RL variants, which
select instruments through the *generative* head, both work; the variant that selects
through the head trained to forecast returns does not. What the model is actually good
at is composing a coherent, regime-appropriate, well-diversified page -- not forecasting.

## 5. Does hybrid row decoding pay for itself?

See the generated latency table in
[`../artifacts/reports/RESULTS.md`](../artifacts/reports/RESULTS.md). The
hardware-independent quantity is the count of sequential model invocations per page,
which hybrid decoding reduces by construction: `row_size - autoregressive_slots` fewer
sequential steps per row.

---

## What I would do next

1. **Rolling-origin retraining.** One contiguous test window is the weakest part of the
   protocol. Refit annually and step forward; it multiplies compute by the number of
   folds but is the single biggest credibility improvement available.
2. **Strip volatility scaling from the RL reward** and re-measure diversity, to test the
   mechanism proposed in section 3.
3. **A survivorship-free universe.** Point-in-time index membership would remove the one
   bias that no amount of relative comparison fully neutralises.
4. **Richer context before a bigger model** -- if the ablation ladder shows what GenPage
   reports, the next marginal unit of effort belongs in the prompt, not the parameters.
