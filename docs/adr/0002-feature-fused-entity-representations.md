# ADR 0002: Entity representations fuse identity with current market state

**Status:** accepted

## Context

GenPage handles cold start with "semantic embedding fusion": a new title is
represented as a blend of its ID embedding and a content embedding derived from its
synopsis, cast and genres, so it is representable before anyone has watched it.

Financial instruments have a harder version of the same problem, and a second one on
top of it:

1. **Cold start.** A newly listed instrument has no learned identity.
2. **Non-stationarity.** An instrument's identity is not what determines its place on a
   page. Whether `NVDA` belongs in a Momentum Leaders row depends on this month's
   momentum, not on the fact that it is NVDA. A static ID embedding would encode the
   average NVDA across twenty-one years, which describes no particular date.

A conventional recommender does not face (2): a movie is the same object next week.

## Decision

The entity representation is

```
E[i, t] = id_embedding[i] + scale * MLP(features[i, t])
```

applied on **both** the input side (when an instrument token is embedded) and the
output side (when instruments are scored), with independent parameters for each, and a
third independent copy for the value head.

Consequently the output projection over instruments is not a fixed weight matrix but
an inner product against the current representation:

```
logits[b, l, i] = hidden[b, l] . E_out[i, t_b] + bias[i]
```

## Consequences

**The catalog is time-varying.** The same instrument on two different dates produces
different logits, which is the behaviour the problem requires.
`tests/test_model.py::test_entity_logits_move_with_market_state` asserts it.

**Cold start falls out.** Zeroing an instrument's learned identity leaves a finite,
non-zero score driven entirely by its state.
`test_cold_start_instrument_still_has_a_representation` asserts it.

**The whole catalog is scorable in one forward pass.** Scoring is a single matrix
product against `E_out[:, t]`, so ranking the catalog needs no token-by-token decoding.
This is GenRec's prefill-only serving mode, and the WBC stage uses it directly.

**Cost: one extra MLP per forward pass over `(batch, 362, 16)`.** Measurable but small
relative to the attention stack -- and it only registers at all because the backbone is
deliberately tiny.

**Risk: the model can lean entirely on features and ignore identity**, collapsing into
a linear factor screen with extra steps. The ablation grid tests the opposite direction
(`design_no_semantic_fusion` removes the fusion term); the identity term's contribution
shows up as the gap between GenDesk and the `pipeline_multistage` baseline, which is a
pure function of the same features.

## Alternatives considered

*Feature tokens appended to the sequence.* Discretising each feature into buckets and
emitting them as tokens would let attention read them directly, but sixteen features x
362 instruments is not representable in any reasonable sequence budget. It is feasible
only for instruments already on the page, which is the wrong set -- the model needs
state for everything it might choose.

*Static ID embeddings plus a separate re-ranking model.* That is the multi-stage
pipeline this project argues against, and it is included as a baseline.
