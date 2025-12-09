"""
K10 Day 1 Toolkit - Network Module

Network protocol analysis and automation for K10/P2 miners.

Features:
- UDP discovery protocol emulation
- Management API interface
- Mass configuration automation
- Network scanning and enumeration
"""

__version__ = "1.0.0"

from .discovery import DiscoveryProtocol
from .management_api import ManagementAPI
from .mass_config import MassConfigurator

__all__ = [
    'DiscoveryProtocol',
    'ManagementAPI',
    'MassConfigurator',
]
