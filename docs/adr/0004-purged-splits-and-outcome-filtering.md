# ADR 0004: Purged splits, and an outcome filter that never touches the model's input

**Status:** accepted

## Context

Netflix pretrains on positive production impressions. Reproducing that requires two
things that are easy to get subtly wrong:

1. deciding which pages count as "positive" without leaking the future into the
   model's inputs, and
2. splitting a time series whose labels span 21 sessions.

## Decision

**Outcome filtering.** The teacher proposes candidate pages with no knowledge of
forward returns. Each is scored after the fact, and candidates below the 55th
percentile of the training split's reward distribution are dropped from the
pretraining set. The model sees only the surviving *pages*; it never sees a reward.

**The book is chosen ex ante.** Each (date, mandate) cell has one deterministic
"book" -- the teacher's greedy page -- which defines the interaction history in the
next cell's context, and the previous holding for the turnover penalty. It is chosen by
teacher score, never by realised reward.

**Purged splits.** An example belongs to the training split only if its entire reward
window closes before `train_end`. Examples straddling a boundary are assigned to a
`purged` split and used by nothing.

## Consequences

**The obvious version of this is wrong.** Selecting the book by realised reward would
be the natural thing to do -- it is "what the desk would have chosen with hindsight" --
and it leaks. The book at date `d-5` has a reward window running to `d+16`, so
conditioning the page at `d` on it puts three weeks of the future into the prompt.
Choosing the book ex ante removes the entire class of bug.

**About 1% of examples are purged**, which is cheap insurance.

**The filter is a hyperparameter with real teeth.** `positive_quantile` controls how
selective pretraining is. Setting it to 0 (train on everything) is an ablation cell in
the grid: it isolates what Netflix's "positive impressions only" rule buys.

**Post-training deliberately uses everything.** WBC needs the losers as much as the
winners -- an ordering cannot be learned from positives alone -- so stage 2 reads the
unfiltered training split.

**A leakage test suite enforces the invariants.** `tests/test_leakage.py` multiplies
every price after a cut date by 3x and asserts that no feature, and no regime bucket,
at or before the cut moves by more than floating-point noise. It also asserts that the
perturbation *does* change values after the cut, so the test cannot pass for the wrong
reason.
