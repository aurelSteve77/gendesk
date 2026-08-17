"""GenDesk: LLM-native generative construction of a financial research desk page.

The package mirrors the two-model architecture described in Netflix's GenRec and
GenPage posts, transposed to financial markets:

* ``gendesk.tokenization`` -- a domain-specific tokenizer where every tradable
  instrument, row archetype, regime bucket and mandate attribute is a single token.
* ``gendesk.model`` -- a decoder-only transformer with untied input embeddings and
  output projection, exposing both an autoregressive head (page generation) and a
  sigmoid value head (catalog-wide scoring in a single prefill pass).
* ``gendesk.training`` -- three stages: next-token pretraining on outcome-filtered
  pages, weighted binary classification post-training, and Dr. GRPO page-level RL.
* ``gendesk.decoding`` -- constrained decoding (business rules as token masks) and
  hybrid row decoding (autoregress a few slots, score the rest in one pass).
* ``gendesk.portfolio`` / ``gendesk.evaluation`` -- page-to-portfolio mapping and a
  walk-forward, purged out-of-sample evaluation harness.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
