# Methodology

This document states exactly what was done, in enough detail to disagree with.

## 1. Data

**Source.** Yahoo Finance's public daily chart endpoint, split- and dividend-adjusted
closes. End-of-day only; no intraday, no fundamentals, no analyst data, no news.

**Catalog.** A fixed list in `configs/universe.yaml`: liquid US single names across the
eleven GICS sectors, plus broad, style, sector, rates, credit, real-asset and currency
funds. After quality filters, **362 instruments over 5,428 sessions (2005-01-03 to
2026-07-31)**.

**Filters.** An instrument enters the catalog if it has at least 1,500 observations,
misses at most 2% of the closes inside its own listed life, and clears a $5M median
daily dollar-volume floor.

**Calendar.** The benchmark's own trading days define the calendar. No synthetic dates
are ever created; a price may be carried forward at most three sessions (exchange
holidays, halts) and a longer gap marks the instrument unavailable rather than
inventing a price.

**Availability mask.** A separate boolean frame, `available[t, i]`, true only when the
instrument was listed, priced within three sessions, and liquid at `t`. It is computed
*before* any forward fill, so filling short gaps cannot make the mask lie. Everything
downstream -- teacher, decoder, baselines, backtest -- reads it.

## 2. Features

Sixteen cross-sectional features, each a well-documented risk-premium or
microstructure signal, each computed from a trailing window ending at the observation
date:

| Family | Features |
|---|---|
| Momentum / trend | 12-1 momentum, 6-1 momentum, 3-month, 1-month, distance to 52-week high, 200-day trend |
| Reversal | 5-day reversal |
| Risk | 63-day volatility, volatility ratio, 252-day beta, one-factor residual volatility, 126-day drawdown, 126-day Sharpe, benchmark correlation, downside ratio |
| Liquidity | log trailing median dollar volume |

Each is winsorised at the 1st/99th percentile **within each date** and standardised
across the available cross-section on that date, then clipped to +/-4. Standardising
per date is what makes a z-score of +2 mean the same thing in 2008 and in 2024, which
is a prerequisite for the tokenizer's discretisation to be stable.

## 3. Regimes

Eight macro axes, each terciled against its own trailing 504-session distribution:
volatility level, volatility shock, curve slope, rate impulse, market trend, breadth,
cross-sectional dispersion, and an average-correlation proxy. Ranking against a
*trailing* window (rather than the full sample) is what keeps the discretisation
point-in-time -- the tercile boundaries at date `t` use data up to `t` only.

The correlation proxy uses the fact that an equally weighted index is only as volatile
as its constituents when they move together: the ratio of index variance to mean
constituent variance rises with average pairwise correlation, and is de-biased for the
number of names.

## 4. Corpus

**Grid.** Every 5th trading day x 6 mandates, from the first fully warm date. 1,030
dates, 6 mandates, 6 candidate pages per cell: **37,080 pages**.

**Teacher.** For each cell it picks five row archetypes -- pinned rows first, then
sampled with a regime-dependent prior -- and fills each row by sampling instruments
sequentially from a softmax over that archetype's score, at temperature 0.35. Sector
caps and deduplication are applied *during* selection, so the corpus obeys exactly the
rules the decoder later enforces.

**Book.** One deterministic (greedy) page per cell is the desk's actual holding. It
defines the interaction history and the turnover baseline. It is chosen ex ante --
selecting the book by realised reward would leak the forward window of one page into
the context of the next.

**Outcome filter.** Each candidate is scored by its realised reward over the following
21 sessions; the pretraining set keeps those above the 55th percentile of the training
split's reward distribution. About 11,300 pages survive.

## 5. Splits

Purged, and purged in the direction that matters. An example belongs to the training
split only if its **entire 21-session reward window closes before** `train_end`
(2019-12-31). A page dated one day before the cutoff carries a label computed from
three weeks of future data and is discarded, not reassigned. The same purge is applied
at the validation boundary (2021-12-31).

The out-of-sample window used for every headline number starts one embargo period
after the validation cutoff and runs to the end of the sample.

## 6. Reward

For a page `P` held by mandate `m` at date `t` over horizon `h`:

```
active      = total return of P over (t, t+h]  -  benchmark total return
scale       = vol_target * sqrt(h / 252)
reward      = active / scale
              - drawdown_penalty * |max drawdown of P inside the window| / scale
              - turnover_penalty * mandate_penalty * turnover(previous book, P)
```

clipped to +/-8. Per-slot rewards for the WBC stage use the same logic at the
instrument level: forward active return divided by the instrument's own volatility,
clipped, and zeroed where unavailable.

## 7. Portfolio construction

Deliberately simple and non-optimising, so that a backtest difference is attributable
to *selection* rather than to a cleverer optimiser:

1. equal risk budget per row, tilted by the mandate's risk appetite (a low-risk budget
   doubles the defensive rows' share, a high one halves it),
2. inverse-volatility weights inside each row,
3. a 12% single-name cap enforced by water-filling (clip-then-renormalise silently
   breaches the cap -- see `cap_and_renormalise`), then normalisation to fully invested.

Long only, no leverage, no shorting.

## 8. Backtest

Every strategy -- the model, the teacher, and all baselines -- is a function from a
date to a weight vector, and the engine treats them identically: rebalance every 21
sessions, hold to the next rebalance with weights drifting on realised returns, charge
5 bp of traded notional on each rebalance. Turnover is reported one-way (half the
traded notional).

At inference the model's context contains **the pages it itself generated** at previous
rebalances. There is no oracle history and no teacher assistance.

## 9. Inference

* **Stationary block bootstrap** (21-session blocks, 2,000 resamples) for Sharpe
  confidence intervals and p-values. Paired comparisons resample both legs with the
  *same* block indices, preserving their contemporaneous correlation.
* **Deflated Sharpe ratio** (Bailey and Lopez de Prado) discounting for the number of
  configurations tried and for the skew and kurtosis of the realised returns.
* **Probability of backtest overfitting** via combinatorially symmetric
  cross-validation across the strategy set.
* **Newey-West** HAC t-statistics where autocorrelation matters.

## 10. What is held fixed

The row archetype coefficients, the reward parameters, the sizing rule, the model
architecture and the corpus threshold were all fixed before the out-of-sample window
was opened. The ablation grid trains on the same corpus and reports validation-split
metrics only; it never touches the test window.
