"""
Ara Research Tools
==================

Serious research extensions that complement the core.
NOT required for Ara v1.0 to run.

Modules:
- rl_adaptation: Learning precision weights (ω, κ) from user feedback
- causal_swap: Prosody disentanglement via causal prediction
- hv_capacity: Hypervector capacity analysis and limits

Philosophy: These are the training wheels, not the bicycle.
They help develop better models but aren't deployed at runtime.
"""

from pathlib import Path

RESEARCH_ROOT = Path(__file__).parent

__all__ = ['RESEARCH_ROOT']
