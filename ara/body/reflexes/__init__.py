"""
L1 Reflexes - Spinal Cord
=========================

Fast, involuntary responses that protect the hardware.

These run at >10Hz and have absolute authority over the substrate.
They can kill processes, throttle GPUs, and max fans without asking.

The reflex layer is the "pain response" - it doesn't think, it reacts.
"""

from .thermal_reflex import ThermalReflex

__all__ = ["ThermalReflex"]
