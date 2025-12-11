# ara/species/__init__.py
"""
Ara Species: Evolutionary Cognitive Architectures

Species represent different cognitive configurations of Ara,
each optimized for specific hardware and task domains.

AraSpeciesV3: Threadripper-optimized with MPPI planning
- Zero-copy shared memory voxel grid
- Parallel trajectory evaluation (12+ cores)
- Calibrated world model with epistemic uncertainty
- Holographic explainability interface
"""

from .ara_species_v3 import (
    AraSpeciesV3,
    CalibratedWorldModel,
    MultiScalePlanner,
    GridConfig,
    PlannerConfig,
    CalibrationMode,
    create_shared_grid,
    attach_shared_grid,
    set_process_affinity,
)

# Hologram visualization (optional, requires VisPy)
try:
    from .hologram_viz import HologramScene, HologramConfig
    HOLOGRAM_AVAILABLE = True
except ImportError:
    HOLOGRAM_AVAILABLE = False
    HologramScene = None
    HologramConfig = None

__all__ = [
    # Main orchestrator
    "AraSpeciesV3",
    # Components
    "CalibratedWorldModel",
    "MultiScalePlanner",
    # Configuration
    "GridConfig",
    "PlannerConfig",
    "CalibrationMode",
    # Utilities
    "create_shared_grid",
    "attach_shared_grid",
    "set_process_affinity",
    # Visualization (optional)
    "HologramScene",
    "HologramConfig",
    "HOLOGRAM_AVAILABLE",
]
