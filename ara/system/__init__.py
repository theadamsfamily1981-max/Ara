"""
Ara System - Core Infrastructure
================================

Contains global state management and cross-layer communication.
"""

from .axis import AxisMundi, LayerSlot, stack_alignment

__all__ = [
    "AxisMundi",
    "LayerSlot",
    "stack_alignment",
]
