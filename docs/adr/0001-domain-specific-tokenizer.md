# ADR 0001: One token per instrument, not subword text

**Status:** accepted

## Context

GenRec verbalises user history as natural language and feeds it to an adapted LLM.
GenPage instead builds a domain-specific tokenizer where each entity and row type is a
single token, and reports compressing a 16-token GPT-5 representation into 4.

Both are defensible. The choice matters here because the sequence budget determines
how much history and context a page can carry, and because it determines whether
constrained decoding is cheap or expensive.

## Decision

A fixed, domain-specific vocabulary:

```
[8 specials] [6 personas] [3 risk budgets] [4 horizons] [24 regime buckets] [8 row archetypes] [362 instruments]
```

415 tokens total, with the instrument block **contiguous and last**.

## Consequences

**A page is ~50 tokens.** Prompt plus five rows of six instruments plus structure. A
text verbalisation of the same page -- tickers, sector names, factor values, mandate
description -- would be several hundred, and most of it would be spent re-tokenising
the same strings on every example.

**Constrained decoding becomes a mask over 362 booleans**, not a search over a subword
lattice. Because the instrument block is one contiguous slice, the decoder never has
to reason about the rest of the vocabulary: dedup, sector caps and mandate exclusions
are elementwise operations on a fixed-width boolean tensor, batched across a whole
GRPO group.

**Checkpoints are bound to a vocabulary.** Adding an instrument shifts every token id
after it. `Vocab.fingerprint()` is stamped into every checkpoint and
`GenDeskModel.from_checkpoint` refuses to load against a different one, which turns a
silent catastrophe into a loud error.

**No transfer from a pretrained language model.** This is the real cost. A verbalised
system inherits an LLM's world knowledge -- that a semiconductor company and a
semiconductor equipment maker are related, that an energy name is exposed to oil.
Here, all of that must be learned from co-occurrence in ~11k pages, or supplied
explicitly through the feature vector. For a 362-instrument catalog with sixteen
engineered features that is an acceptable trade; for a catalog of tens of thousands of
instruments with rich text descriptions it would not be.

**Natural-language steering still works**, because instructions are compiled into
context tokens and masks rather than fed to the model as text (ADR 0005).

## Alternatives considered

*Verbalised prompts into a small pretrained LLM.* Rejected on compute: the smallest
usable open model is 100x this backbone, and the ablation grid needs eleven training
runs on one consumer GPU.

*Semantic IDs (RQ-VAE over item embeddings).* Attractive for a large catalog, since it
makes the vocabulary sublinear in the number of items. Unnecessary at 362 instruments,
and it would add a quantisation stage between the features and the model that the
feature-fusion approach (ADR 0002) achieves more directly.
