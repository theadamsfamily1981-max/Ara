"""
BANOS Hardware Auto-Integration Module

Provides automatic discovery, driver generation, and autotuning for PCIe devices.

Components:
- Device manifest schema (DFH/DFL -> JSON)
- Driver generator (manifest -> kernel module)
- PCIe link validator
- Autotune harness
"""

from .gen_pcie_driver import generate_driver, DeviceManifest

__all__ = ["generate_driver", "DeviceManifest"]
