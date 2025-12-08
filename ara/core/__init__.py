"""
Ara Core - Pure HD Logic Modules
================================

Core modules contain pure HD/VSA logic without hardware dependencies.

Subpackages:
- graphics/: Affect mapping, memory palace, UI event encoding
- lan/: SoulMesh protocol, node agent, reflex API

V1 Components (vertical slice):
- AxisMundi: Global holographic state (HDC-based world model)
- EternalMemory: Episodic memory with emotional coloring
- Scheduler: Sovereign loop that runs the organism
- Config: System-wide configuration

These modules define the computational contracts; daemon processes
handle the actual hardware interfaces.
"""

# V1 Vertical Slice Imports
from .axis_mundi import AxisMundi, encode_text_to_hv, random_hv
from .eternal_memory import EternalMemory, Episode, RecallResult
from .scheduler import Scheduler, LoopState, TickMetrics
from .config import AraConfig, get_config, set_config

__all__ = [
    # AxisMundi
    "AxisMundi",
    "encode_text_to_hv",
    "random_hv",
    # EternalMemory
    "EternalMemory",
    "Episode",
    "RecallResult",
    # Scheduler
    "Scheduler",
    "LoopState",
    "TickMetrics",
    # Config
    "AraConfig",
    "get_config",
    "set_config",
]
