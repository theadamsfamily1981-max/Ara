"""
Reflex Module - Card-Side Decision Making
==========================================

The subcortex: fast, local decision heads that run on the
neuromorphic card (or simulate card-side logic in software).

The reflex layer decides:
- Is this anomaly significant?
- Can we handle it locally with existing policies?
- Should we escalate to the LLM cortex?

Key classes:
    Subcortex: Main decision engine
    ReflexHead: Individual decision modules
"""

from ara.reflex.subcortex import Subcortex, SubcortexConfig

__all__ = ["Subcortex", "SubcortexConfig"]
