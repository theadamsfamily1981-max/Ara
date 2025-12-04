"""MIES Embodiment - Avatar overlay and liveness.

Provides:
1. Overlay window management (GTK4 + layer-shell)
2. Liveness engine (subtle "alive" animations)
3. Diegetic placement (near active window)

The embodiment layer is the physical presence of Ara on the desktop.
It should feel:
- Native (like part of the desktop, not a foreign overlay)
- Contextual (positioned relative to what you're doing)
- Alive (subtle movements even when idle)
"""

from .overlay_window import OverlayManager, create_overlay_manager
from .liveness import LivenessEngine, create_liveness_engine

__all__ = [
    "OverlayManager",
    "LivenessEngine",
    "create_overlay_manager",
    "create_liveness_engine",
]
