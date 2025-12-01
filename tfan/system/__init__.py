"""
TF-A-N System Layer

System-level components for resource allocation, routing, and optimization.

Components:
- SemanticSystemOptimizer: Context-aware backend routing
- CognitiveLoadVector: Unified L1/L2 metric coalescing
- AtomicStructuralUpdater: PGU-gated atomic model swap
"""

from tfan.system.semantic_optimizer import (
    SemanticSystemOptimizer,
    RoutingDecision,
    ResourceFeatures,
    PADState,
    load_routing_scores,
    save_routing_scores,
)

from tfan.system.cognitive_load_vector import (
    CognitiveLoadVector,
    CLVComponents,
    CLVComputer,
    RiskLevel,
    create_clv_from_state,
    get_clv_computer,
)

from tfan.system.atomic_updater import (
    AtomicStructuralUpdater,
    StructuralChange,
    VerificationResult,
    UpdateResult,
    UpdateStatus,
    promote_with_verification,
)

__all__ = [
    # Semantic Optimizer
    "SemanticSystemOptimizer",
    "RoutingDecision",
    "ResourceFeatures",
    "PADState",
    "load_routing_scores",
    "save_routing_scores",
    # Cognitive Load Vector
    "CognitiveLoadVector",
    "CLVComponents",
    "CLVComputer",
    "RiskLevel",
    "create_clv_from_state",
    "get_clv_computer",
    # Atomic Structural Updater
    "AtomicStructuralUpdater",
    "StructuralChange",
    "VerificationResult",
    "UpdateResult",
    "UpdateStatus",
    "promote_with_verification",
]
