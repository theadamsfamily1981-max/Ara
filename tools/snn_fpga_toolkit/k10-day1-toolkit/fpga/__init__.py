"""
K10 Day 1 Toolkit - FPGA Management Module

FPGA bitstream loading and algorithm management for Stratix 10 SoC.

Features:
- Sysfs-based FPGA manager interface
- Algorithm switching automation
- ISO extraction and deployment
- Bitstream verification
"""

__version__ = "1.0.0"

from .bitstream_manager import BitstreamManager
from .algorithm_switcher import AlgorithmSwitcher
from .iso_extractor import ISOExtractor

__all__ = [
    'BitstreamManager',
    'AlgorithmSwitcher',
    'ISOExtractor',
]
