# ADR 0005: Business rules are token masks, not a post-hoc filter

**Status:** accepted

## Context

A desk page must satisfy hard rules: no instrument twice, at most four names from one
sector, nothing the mandate excludes, nothing illiquid, and certain rows always
present. There are three places to enforce them -- in the training data, after
generation, or during generation.

GenPage enforces them during generation: "business rules directly translate to
token-level masks".

## Decision

Enforce during generation. A `ConstraintEngine` tracks per-sequence state (which
instruments are used, per-sector counts, which archetypes have been emitted) and
returns a boolean mask over the instrument block at every decision step. The generator
samples only inside the mask.

Row pinning is handled the same way: a mandate's pinned rows are forced as soon as the
number of remaining row slots equals the number of pinned rows still missing.

## Consequences

**A generated page is compliant by construction.** Nothing is ever rejected and
retried, which matters for RL -- rejection sampling would silently change the
distribution being optimised.

**The probability mass is redistributed, not discarded.** Masking before the softmax
means the model's preference ordering among *legal* instruments is preserved exactly.
Filtering after generation would instead leave a hole.

**The mask is the behaviour policy.** Log-probabilities for RL must be computed under
the masked distribution, which is why masks are recorded at sampling time and replayed
at recomputation (ADR 0003).

**The constraint state is batched.** One engine serves a whole GRPO group in parallel,
as `(batch, n_instruments)` boolean tensors, so generating eight pages costs the same
number of sequential steps as generating one.

**There is an escape hatch, and it is counted.** If a mask ever goes all-false (a
mandate so restrictive that nothing is legal), the engine falls back to "any available,
unused instrument" and increments a counter in the constraint report, which the UI
displays. Silent failure is the thing to avoid; a visible fallback is not.

**The same rules bind the baselines.** The `pipeline_multistage` baseline applies the
identical sector cap and exclusion list in its diversification stage, so the comparison
is about *where* the rules are applied, not *whether*.

## Alternatives considered

*Teach the rules through data alone.* The corpus does obey the rules, so the model
learns them approximately. It is not enough: an approximate sector cap is a breached
sector cap, and a mandate that changes at inference (natural-language steering) has no
training data at all.

*Reject and resample.* Correct but unboundedly slow under tight mandates, and it
distorts the RL gradient.
