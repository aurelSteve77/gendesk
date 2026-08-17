"""Domain-specific tokenization.

GenPage's central efficiency argument is that a general-purpose text tokenizer
wastes sequence length on entities that the system already knows as atoms. The same
holds here: ``NVDA`` is four BPE pieces to a language model and one token to a desk.

This package defines the fixed vocabulary -- structural markers, mandate attributes,
regime buckets, row archetypes and one token per catalog instrument -- and the
encoder that lays a page out as a sequence.
"""

from gendesk.tokenization.page import Page, PageSequence, Row
from gendesk.tokenization.vocab import ROW_ARCHETYPES, Vocab, build_vocab

__all__ = [
    "ROW_ARCHETYPES",
    "Page",
    "PageSequence",
    "Row",
    "Vocab",
    "build_vocab",
]
