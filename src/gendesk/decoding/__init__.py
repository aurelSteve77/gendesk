"""Constrained and hybrid decoding.

Business rules are enforced as token masks rather than as a post-hoc filter, which
is the single most practical idea in the GenPage post: a mandate that forbids
commodity funds is not a page that gets rejected, it is a token that is never
sampled. A generated page is therefore *always* compliant by construction.
"""

from gendesk.decoding.constraints import ConstraintEngine, ConstraintReport
from gendesk.decoding.generate import GenerationResult, PageGenerator

__all__ = ["ConstraintEngine", "ConstraintReport", "GenerationResult", "PageGenerator"]
