"""
BANOS PCIe Link Validator

Validates PCIe link status, speed negotiation, and lane configuration.
Supports Gen1-Gen6, x1-x32 configurations.

Usage:
    from banos.pci_validator import PCIeLinkValidator

    validator = PCIeLinkValidator()
    result = validator.validate("0000:03:00.0")
"""

from .validator import (
    PCIeLinkValidator,
    PCIeGeneration,
    PCIeLinkStatus,
    ValidationResult,
)

__all__ = [
    "PCIeLinkValidator",
    "PCIeGeneration",
    "PCIeLinkStatus",
    "ValidationResult",
]
