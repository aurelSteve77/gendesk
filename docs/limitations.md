# Limitations

Every one of these is a real constraint on what the results mean. They are listed
first because a research prototype that hides them is worth less than one that does
not.

## Data

**Survivorship bias.** The catalog is a fixed snapshot of instruments that exist
today. Names that were delisted or acquired during the sample return no history and
are dropped rather than back-filled, so the universe is biased upward in absolute
terms. The mitigation used here is that *every* headline result is a relative
comparison against baselines drawn from the same catalog and traded through the same
engine -- the bias is common to all of them. It is a mitigation, not a fix. A
point-in-time index-membership database would be the fix.

**End-of-day only.** No intraday data, no order book, no fundamentals, no earnings
dates, no analyst revisions, no news, no positioning or flow data. The feature set is
price and volume. A real desk's edge usually is not.

**Adjusted prices.** Yahoo's adjusted closes are restated as corporate actions occur,
so a historical adjusted price today is not the price a system would have seen then.
This is small for liquid US listings but not zero.

**One venue, one currency, one asset class.** US listings only, USD only, cash equities
and ETFs only. No futures, options, credit or FX.

## Execution

**Costs are a flat 5 bp of traded notional.** No market impact, no spread modelling, no
participation constraints, no borrow, no financing. At the turnover levels here that
is defensible for liquid large caps and optimistic for anything else. The backtest
reports cost drag explicitly so the sensitivity is visible.

**Close-to-close execution.** Weights are set on the signal date and returns accrue
from the next session, which is the right side of the boundary, but it assumes a fill
at the close with no slippage.

**No capacity analysis.** Nothing here estimates how much capital the strategy would
absorb before its own trading moved the prices it depends on.

## Modelling

**The teacher is synthetic.** Netflix pretrains on impressions that were actually
served to members. There is no served desk here, so the corpus comes from a factor
screen plus an outcome filter. The model therefore inherits the screen's blind spots
in its prior. The RL stage can move away from it and the `teacher_book` baseline
measures whether it did, but the starting distribution is not a production log.

**Mandates are invented.** The six personas are plausible, not observed. Their risk
budgets, horizons and constraints were chosen to span a sensible space, not fitted to
real allocators.

**Row archetypes are hand-specified.** The eight theses and their coefficients are
textbook, chosen before any results were looked at, and deliberately not tuned. That
keeps the comparison honest but means the underlying signals are weak by modern
standards -- the contribution being tested is the generative page layer, not the alpha.

**Small model, small corpus.** ~6.4M parameters on ~11k pretraining pages. The scaling
behaviour observed in the ablation grid is real for this regime and should not be
extrapolated to the sizes Netflix operates at.

**Single market history.** There is one realisation of 2005-2026. Twenty-one years is
about six independent macro regimes, not 5,428 independent observations, which is why
every inference here uses block bootstrapping and reports a deflated Sharpe ratio.
Even so, the effective sample is small and the confidence intervals are wide.

## Evaluation

**One out-of-sample window.** The test period is a single contiguous stretch. A
rolling-origin retrain (refit the model every year and step forward) would be a
stronger protocol and is the obvious next step; it was not run here because the
training budget is a single consumer GPU.

**Multiple comparisons.** Several model variants and baselines are compared. The
deflated Sharpe ratio corrects for the number of configurations attempted and the PBO
statistic measures how much of the ranking survives resampling, but the honest reading
is still that a Sharpe difference inside its bootstrap interval has not been
demonstrated.

**The ablation grid is compute-bounded.** Cells are trained for three epochs on a 60%
subsample of the training split. Every cell sees the identical subsample and the same
validation split, so the *gaps* are comparable, but the absolute levels are lower than
a full run's.

## Results that did not come out well

Listed here rather than left for a reader to discover:

* **The value head has no measurable predictive power.** Validation rank IC is ~0 and
  the WBC variant is the worst strategy in the backtest. See
  [`findings.md`](findings.md) section 4.
* **Diversification fell under RL** rather than rising as GenPage reports. See
  [`findings.md`](findings.md) section 3 for the proposed mechanism and the experiment
  that would test it.
* **No performance difference is statistically significant.** Every paired bootstrap
  interval against every baseline contains zero, and the probability of backtest
  overfitting across the strategy set is high. The architecture is competitive with the
  pipeline it replaces; it has not been shown to beat it.

## Scope

This is a research prototype built to explore an architecture. It is not investment
advice, not a trading system, and not backed by any claim that the strategy would make
money after real-world frictions.
