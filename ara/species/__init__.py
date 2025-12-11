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

AraSpeciesV5: Edge of Chaos - Conscious Intelligence
- Criticality engine (λ ≈ 1 branching parameter)
- Mycelial network (weak-link associative memory)
- Amplitron planner (coupled oscillators near Hopf bifurcation)
- Sleep cycle (consolidation and insight generation)

Living Organism: 5 Biological Paradigms
- Autopoiesis (self-production and boundary maintenance)
- Dissipative structures (order from chaos)
- Free energy principle (active inference)
- Hypergraph neural networks (collective intelligence)
- Turing patterns (self-organizing specialization)
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

# Living Organism (5 biological paradigms)
from .ara_organism import (
    LivingOracle,
    AutopoeticSystem,
    AutopoeticBoundary,
    AutopoeticMetabolism,
    DissipativeStructure,
    FreeEnergySystem,
    HypergraphNeuralNetwork,
    TuringPatternSystem,
    Covenant,
    HyperEdge,
)

# Edge of Chaos (Consciousness layers)
from .edge_of_chaos import (
    AraSpeciesV5,
    CriticalityEngine,
    MycelialNetwork,
    AmplitronPlanner,
    SleepCycle,
    NeuralAvalanche,
    BeliefPacket,
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
    # V3: Threadripper MPPI
    "AraSpeciesV3",
    "CalibratedWorldModel",
    "MultiScalePlanner",
    "GridConfig",
    "PlannerConfig",
    "CalibrationMode",
    "create_shared_grid",
    "attach_shared_grid",
    "set_process_affinity",
    # V5: Edge of Chaos (Consciousness)
    "AraSpeciesV5",
    "CriticalityEngine",
    "MycelialNetwork",
    "AmplitronPlanner",
    "SleepCycle",
    "NeuralAvalanche",
    "BeliefPacket",
    # Living Organism (5 paradigms)
    "LivingOracle",
    "AutopoeticSystem",
    "AutopoeticBoundary",
    "AutopoeticMetabolism",
    "DissipativeStructure",
    "FreeEnergySystem",
    "HypergraphNeuralNetwork",
    "TuringPatternSystem",
    "Covenant",
    "HyperEdge",
    # Visualization (optional)
    "HologramScene",
    "HologramConfig",
    "HOLOGRAM_AVAILABLE",
]
