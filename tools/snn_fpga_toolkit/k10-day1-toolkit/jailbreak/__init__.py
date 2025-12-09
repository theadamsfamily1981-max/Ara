"""
K10 Day 1 Toolkit - Jailbreak Module

Automated privilege escalation for K10/ColEngine P2 miners.

Methods:
- Method A: Network-based credential exploitation (non-invasive)
- Method B: Offline shadow file modification (requires physical access)
- Method C: U-Boot environment injection (requires UART console)
"""

__version__ = "1.0.0"

from .method_a_network import NetworkJailbreak
from .method_b_shadow import ShadowEditor
from .method_c_uboot import UBootInjector
from .auto_jailbreak import AutoJailbreak

__all__ = [
    'NetworkJailbreak',
    'ShadowEditor',
    'UBootInjector',
    'AutoJailbreak',
]
