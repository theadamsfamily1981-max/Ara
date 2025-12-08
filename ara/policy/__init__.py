"""
Policy Module - LLM -> HDC+SNN Policy Compilation
=================================================

Translates high-level policies from the LLM cortex into:
1. HDC policy vectors (for correlation-based matching)
2. SNN weight deltas (for Hebbian learning on the card)

Key components:
    compiler.py: Full policy compilation pipeline
    store.py: Policy storage and retrieval
"""

from ara.policy.compiler import (
    PolicyCompiler,
    StructuredPolicy,
    PolicyCompilerConfig
)

__all__ = [
    'PolicyCompiler',
    'StructuredPolicy',
    'PolicyCompilerConfig'
]
