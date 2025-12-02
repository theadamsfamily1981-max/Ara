"""
Ara Hardware Module - FPGA and Neuromorphic Accelerators

Includes:
- Forest Kitten 33 SNN Fabric
- Hardware abstraction layer
- Emulation support for development
"""

from ara.hardware.kitten import (
    ForestKitten33,
    KittenConfig,
    KittenState,
    KittenEmulator,
    create_kitten,
)

__all__ = [
    "ForestKitten33",
    "KittenConfig",
    "KittenState",
    "KittenEmulator",
    "create_kitten",
]
