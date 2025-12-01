"""
Adaptive Geometry: Task-Optimized Hyperbolic Curvature

This module implements adaptive geometry selection, allowing the system
to choose optimal manifold curvature for different task types.

Key Insight:
- Hierarchical tasks (planning, tree search) → Higher curvature (Poincaré)
- Flat retrieval tasks → Lower curvature (Euclidean-like)
- The system learns which geometry works best for each workload

Curvature Parameter (c):
- c → 0: Euclidean space (flat)
- c = 1: Standard Poincaré ball
- c > 1: Higher curvature (steeper hierarchy)
- c < 1: Lower curvature (flatter space)

Integration:
- L5 MetaLearner can tune curvature_c
- SemanticOptimizer uses curvature in routing decisions
- PGU verifies topological constraints hold under curvature changes
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging
import math

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("tfan.geometry")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


# =============================================================================
# GEOMETRY TYPES
# =============================================================================

class GeometryType(str, Enum):
    """Types of geometric spaces."""
    EUCLIDEAN = "euclidean"           # c → 0
    HYPERBOLIC_LOW = "hyperbolic_low"  # c ∈ (0, 0.5)
    HYPERBOLIC_STD = "hyperbolic_std"  # c ∈ [0.5, 1.5]
    HYPERBOLIC_HIGH = "hyperbolic_high" # c > 1.5
    SPHERICAL = "spherical"           # c < 0 (future)


class TaskGeometryHint(str, Enum):
    """Hints for geometry selection based on task type."""
    HIERARCHICAL = "hierarchical"     # Tree-like, planning → high curvature
    FLAT_RETRIEVAL = "flat_retrieval" # Dense similarity → low curvature
    SEQUENTIAL = "sequential"         # Chain reasoning → medium curvature
    CLUSTERING = "clustering"         # Group similarity → adaptive
    GENERAL = "general"               # No preference


# =============================================================================
# HYPERBOLIC OPERATIONS
# =============================================================================

@dataclass
class HyperbolicConfig:
    """Configuration for hyperbolic operations."""
    curvature: float = 1.0  # c parameter
    epsilon: float = 1e-7   # Numerical stability
    max_norm: float = 1.0 - 1e-5  # Ball boundary

    def effective_curvature(self) -> float:
        """Get effective curvature (always positive for Poincaré)."""
        return abs(self.curvature) + self.epsilon


class HyperbolicMath:
    """
    Hyperbolic geometry operations in Poincaré ball model.

    All operations are parameterized by curvature c.
    """

    def __init__(self, config: Optional[HyperbolicConfig] = None):
        self.config = config or HyperbolicConfig()

    @property
    def c(self) -> float:
        """Curvature parameter."""
        return self.config.effective_curvature()

    def mobius_add(self, x: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
        """
        Möbius addition in Poincaré ball.

        x ⊕_c y = ((1 + 2c⟨x,y⟩ + c‖y‖²)x + (1 - c‖x‖²)y) / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²)
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required for hyperbolic operations")

        c = self.c
        x_norm_sq = np.sum(x ** 2, axis=-1, keepdims=True)
        y_norm_sq = np.sum(y ** 2, axis=-1, keepdims=True)
        xy_dot = np.sum(x * y, axis=-1, keepdims=True)

        num = (1 + 2 * c * xy_dot + c * y_norm_sq) * x + (1 - c * x_norm_sq) * y
        denom = 1 + 2 * c * xy_dot + c ** 2 * x_norm_sq * y_norm_sq

        return num / (denom + self.config.epsilon)

    def exp_map(self, x: "np.ndarray", v: "np.ndarray") -> "np.ndarray":
        """
        Exponential map: tangent vector to manifold point.

        exp_x^c(v) = x ⊕_c (tanh(√c ‖v‖ / (2(1-c‖x‖²))) * v / (√c‖v‖))
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required for hyperbolic operations")

        c = self.c
        sqrt_c = math.sqrt(c)

        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
        x_norm_sq = np.sum(x ** 2, axis=-1, keepdims=True)

        # Conformal factor
        lambda_x = 2 / (1 - c * x_norm_sq + self.config.epsilon)

        # Scaled tangent vector
        tanh_arg = sqrt_c * lambda_x * v_norm / 2
        v_scaled = np.tanh(tanh_arg) * v / (sqrt_c * v_norm + self.config.epsilon)

        return self.mobius_add(x, v_scaled)

    def log_map(self, x: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
        """
        Logarithmic map: manifold point to tangent vector.

        log_x^c(y) = (2 / (√c λ_x)) * arctanh(√c ‖-x ⊕_c y‖) * (-x ⊕_c y / ‖-x ⊕_c y‖)
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required for hyperbolic operations")

        c = self.c
        sqrt_c = math.sqrt(c)

        x_norm_sq = np.sum(x ** 2, axis=-1, keepdims=True)
        lambda_x = 2 / (1 - c * x_norm_sq + self.config.epsilon)

        # Möbius subtraction: -x ⊕_c y
        neg_x = -x
        diff = self.mobius_add(neg_x, y)
        diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True)

        # Log map
        coef = 2 / (sqrt_c * lambda_x) * np.arctanh(sqrt_c * diff_norm + self.config.epsilon)

        return coef * diff / (diff_norm + self.config.epsilon)

    def distance(self, x: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
        """
        Hyperbolic distance in Poincaré ball.

        d_c(x, y) = (2/√c) * arctanh(√c ‖-x ⊕_c y‖)
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy required for hyperbolic operations")

        c = self.c
        sqrt_c = math.sqrt(c)

        # Möbius subtraction
        neg_x = -x
        diff = self.mobius_add(neg_x, y)
        diff_norm = np.linalg.norm(diff, axis=-1)

        return 2 / sqrt_c * np.arctanh(np.minimum(sqrt_c * diff_norm, 1 - self.config.epsilon))

    def project_to_ball(self, x: "np.ndarray") -> "np.ndarray":
        """Project point back onto Poincaré ball if outside."""
        if not NUMPY_AVAILABLE:
            return x

        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        max_norm = self.config.max_norm

        # Scale down if norm exceeds boundary
        scale = np.where(norm > max_norm, max_norm / norm, 1.0)
        return x * scale


# =============================================================================
# ADAPTIVE GEOMETRY SELECTOR
# =============================================================================

@dataclass
class GeometrySelection:
    """Result of geometry selection."""
    geometry_type: GeometryType
    curvature: float
    confidence: float
    reason: str


class AdaptiveGeometrySelector:
    """
    Selects optimal geometry based on task characteristics.

    Learns from past performance to improve selection.
    """

    def __init__(
        self,
        default_curvature: float = 1.0,
        min_curvature: float = 0.1,
        max_curvature: float = 3.0,
    ):
        """
        Initialize geometry selector.

        Args:
            default_curvature: Default c value
            min_curvature: Minimum allowed c
            max_curvature: Maximum allowed c
        """
        self.default_curvature = default_curvature
        self.min_curvature = min_curvature
        self.max_curvature = max_curvature

        # Performance history by geometry type
        self.performance_history: Dict[GeometryType, List[float]] = {
            gt: [] for gt in GeometryType
        }

        # Task-type to geometry mapping (learned)
        self.task_geometry_map: Dict[TaskGeometryHint, float] = {
            TaskGeometryHint.HIERARCHICAL: 1.5,    # High curvature
            TaskGeometryHint.FLAT_RETRIEVAL: 0.3,  # Low curvature
            TaskGeometryHint.SEQUENTIAL: 1.0,      # Standard
            TaskGeometryHint.CLUSTERING: 0.8,      # Slightly lower
            TaskGeometryHint.GENERAL: 1.0,         # Default
        }

        logger.info(f"AdaptiveGeometrySelector initialized with c={default_curvature}")

    def select_geometry(
        self,
        task_hint: TaskGeometryHint,
        hierarchy_depth: Optional[int] = None,
        embedding_dim: int = 64,
    ) -> GeometrySelection:
        """
        Select optimal geometry for task.

        Args:
            task_hint: Task type hint
            hierarchy_depth: Expected hierarchy depth (if known)
            embedding_dim: Embedding dimension

        Returns:
            GeometrySelection with recommended curvature
        """
        # Base curvature from task type
        base_c = self.task_geometry_map.get(task_hint, self.default_curvature)

        # Adjust for hierarchy depth
        if hierarchy_depth is not None:
            # Deeper hierarchies benefit from higher curvature
            depth_factor = min(2.0, 1.0 + 0.2 * (hierarchy_depth - 3))
            base_c *= depth_factor

        # Clamp to valid range
        curvature = max(self.min_curvature, min(self.max_curvature, base_c))

        # Determine geometry type
        if curvature < 0.3:
            geometry_type = GeometryType.HYPERBOLIC_LOW
        elif curvature < 1.2:
            geometry_type = GeometryType.HYPERBOLIC_STD
        else:
            geometry_type = GeometryType.HYPERBOLIC_HIGH

        return GeometrySelection(
            geometry_type=geometry_type,
            curvature=curvature,
            confidence=0.8,  # Could be learned
            reason=f"Task={task_hint.value}, depth={hierarchy_depth}"
        )

    def update_from_reward(
        self,
        geometry_type: GeometryType,
        curvature: float,
        reward: float,
        task_hint: TaskGeometryHint,
    ):
        """
        Update geometry preferences from reward signal.

        Used by L5 meta-learner to improve selection.
        """
        # Record performance
        self.performance_history[geometry_type].append(reward)

        # Keep last 100 entries
        if len(self.performance_history[geometry_type]) > 100:
            self.performance_history[geometry_type] = self.performance_history[geometry_type][-100:]

        # Update task mapping if reward is good
        if reward > 0.7:
            # Blend current estimate with new curvature
            current = self.task_geometry_map.get(task_hint, self.default_curvature)
            self.task_geometry_map[task_hint] = 0.9 * current + 0.1 * curvature

        logger.debug(f"Updated geometry for {task_hint.value}: c={self.task_geometry_map[task_hint]:.3f}")

    def get_best_geometry(self, task_hint: TaskGeometryHint) -> float:
        """Get best curvature for task type."""
        return self.task_geometry_map.get(task_hint, self.default_curvature)


# =============================================================================
# GEOMETRY-AWARE EMBEDDING LAYER
# =============================================================================

class HyperbolicEmbedding:
    """
    Embedding layer with adaptive hyperbolic geometry.

    Embeddings live on Poincaré ball with learnable curvature.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        curvature: float = 1.0,
    ):
        """
        Initialize hyperbolic embedding.

        Args:
            num_embeddings: Number of embedding vectors
            embedding_dim: Dimension of each embedding
            curvature: Initial curvature parameter
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.config = HyperbolicConfig(curvature=curvature)
        self.math = HyperbolicMath(self.config)

        # Initialize embeddings in tangent space at origin
        if NUMPY_AVAILABLE:
            # Small random initialization (will be projected to ball)
            self.embeddings = np.random.randn(num_embeddings, embedding_dim) * 0.1
            self.embeddings = self.math.project_to_ball(self.embeddings)
        else:
            self.embeddings = None

        logger.info(f"HyperbolicEmbedding: {num_embeddings}x{embedding_dim}, c={curvature}")

    def set_curvature(self, curvature: float):
        """Update curvature parameter."""
        self.config.curvature = curvature
        self.math = HyperbolicMath(self.config)

        # Re-project embeddings to new ball
        if self.embeddings is not None:
            self.embeddings = self.math.project_to_ball(self.embeddings)

    def get_embedding(self, idx: int) -> "np.ndarray":
        """Get embedding vector."""
        if self.embeddings is None:
            raise RuntimeError("NumPy required")
        return self.embeddings[idx]

    def distance(self, idx1: int, idx2: int) -> float:
        """Compute hyperbolic distance between embeddings."""
        if self.embeddings is None:
            raise RuntimeError("NumPy required")

        x = self.embeddings[idx1]
        y = self.embeddings[idx2]
        return float(self.math.distance(x, y))

    def nearest_neighbors(self, idx: int, k: int = 5) -> List[Tuple[int, float]]:
        """Find k nearest neighbors in hyperbolic space."""
        if self.embeddings is None:
            return []

        x = self.embeddings[idx]
        distances = []

        for i in range(self.num_embeddings):
            if i != idx:
                d = float(self.math.distance(x, self.embeddings[i]))
                distances.append((i, d))

        distances.sort(key=lambda t: t[1])
        return distances[:k]


# =============================================================================
# INTEGRATION WITH SEMANTIC OPTIMIZER
# =============================================================================

@dataclass
class GeometricRoutingDecision:
    """Routing decision with geometry information."""
    curvature: float
    geometry_type: GeometryType
    task_hint: TaskGeometryHint
    confidence: float
    backend_preference: str  # sparse/dense/fpga


def compute_geometric_routing(
    task_hint: TaskGeometryHint,
    valence: float,
    arousal: float,
    hierarchy_depth: Optional[int] = None,
    selector: Optional[AdaptiveGeometrySelector] = None,
) -> GeometricRoutingDecision:
    """
    Compute routing decision with geometry selection.

    Integrates with SemanticSystemOptimizer.
    """
    if selector is None:
        selector = AdaptiveGeometrySelector()

    # Select geometry
    geo = selector.select_geometry(task_hint, hierarchy_depth)

    # Backend preference based on geometry
    if geo.geometry_type == GeometryType.HYPERBOLIC_HIGH:
        # High curvature needs more compute, prefer dense/FPGA
        backend = "fpga" if arousal > 0.7 else "dense"
    elif geo.geometry_type == GeometryType.HYPERBOLIC_LOW:
        # Low curvature is simpler, sparse is efficient
        backend = "sparse"
    else:
        # Standard curvature, depends on valence
        backend = "sparse" if valence > 0 else "pgu_verified"

    return GeometricRoutingDecision(
        curvature=geo.curvature,
        geometry_type=geo.geometry_type,
        task_hint=task_hint,
        confidence=geo.confidence,
        backend_preference=backend,
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global selector instance
_geometry_selector: Optional[AdaptiveGeometrySelector] = None


def get_geometry_selector() -> AdaptiveGeometrySelector:
    """Get or create global geometry selector."""
    global _geometry_selector
    if _geometry_selector is None:
        _geometry_selector = AdaptiveGeometrySelector()
    return _geometry_selector


def select_geometry_for_task(
    task_type: str,
    hierarchy_depth: Optional[int] = None,
) -> Dict[str, Any]:
    """
    High-level geometry selection.

    Returns dict with curvature, type, and confidence.
    """
    # Map task type string to hint
    hint_map = {
        "planning": TaskGeometryHint.HIERARCHICAL,
        "retrieval": TaskGeometryHint.FLAT_RETRIEVAL,
        "reasoning": TaskGeometryHint.SEQUENTIAL,
        "clustering": TaskGeometryHint.CLUSTERING,
    }

    hint = hint_map.get(task_type, TaskGeometryHint.GENERAL)
    selector = get_geometry_selector()
    selection = selector.select_geometry(hint, hierarchy_depth)

    return {
        "curvature": selection.curvature,
        "geometry_type": selection.geometry_type.value,
        "confidence": selection.confidence,
        "reason": selection.reason,
    }


# =============================================================================
# L8: COGNITIVE PHASE TRANSITIONS
# =============================================================================

class CognitivePhase(str, Enum):
    """
    Cognitive phases based on geometric regime.

    The network shifts between phases like cognitive states,
    each with different reasoning characteristics.
    """
    FLAT_LOCAL = "flat_local"           # c ≈ 0: Local, similarity-based
    TRANSITIONAL = "transitional"       # c ∈ (0.3, 0.7): Mixed reasoning
    HIERARCHICAL = "hierarchical"       # c ∈ [0.7, 1.5]: Tree-like structure
    DEEP_ABSTRACT = "deep_abstract"     # c > 1.5: Deep hierarchy, abstraction


@dataclass
class PhaseTransitionState:
    """State of cognitive phase transition."""
    current_phase: CognitivePhase
    previous_phase: CognitivePhase
    curvature: float
    transition_in_progress: bool
    transition_progress: float  # 0-1 during transition
    stability: float            # How stable the current phase is
    recommended_mode: str       # L6 reasoning mode recommendation


class CognitivePhaseController:
    """
    Controls cognitive phase transitions based on geometry.

    This is the L8 component that treats curvature changes as
    cognitive state shifts, not just parameter tweaks.
    """

    def __init__(
        self,
        initial_curvature: float = 1.0,
        transition_rate: float = 0.1,  # Max change per step
        stability_threshold: float = 0.8,  # Require stability before transition
    ):
        """
        Initialize phase controller.

        Args:
            initial_curvature: Starting curvature
            transition_rate: Maximum curvature change per update
            stability_threshold: Required stability to complete transition
        """
        self.current_curvature = initial_curvature
        self.target_curvature = initial_curvature
        self.transition_rate = transition_rate
        self.stability_threshold = stability_threshold

        # Phase boundaries (must be defined before _curvature_to_phase)
        self.phase_boundaries = {
            CognitivePhase.FLAT_LOCAL: (0.0, 0.3),
            CognitivePhase.TRANSITIONAL: (0.3, 0.7),
            CognitivePhase.HIERARCHICAL: (0.7, 1.5),
            CognitivePhase.DEEP_ABSTRACT: (1.5, 5.0),
        }

        # L6 mode recommendations per phase
        self.phase_to_mode = {
            CognitivePhase.FLAT_LOCAL: "KG_ASSISTED",
            CognitivePhase.TRANSITIONAL: "HYBRID",
            CognitivePhase.HIERARCHICAL: "PGU_VERIFIED",
            CognitivePhase.DEEP_ABSTRACT: "FORMAL_FIRST",
        }

        # Phase state (after phase_boundaries is defined)
        self._current_phase = self._curvature_to_phase(initial_curvature)
        self._previous_phase = self._current_phase
        self._transition_in_progress = False
        self._transition_progress = 1.0
        self._stability = 1.0

        # History for analysis
        self._phase_history: List[Tuple[float, CognitivePhase]] = []
        self._max_history = 100

        logger.info(f"CognitivePhaseController: c={initial_curvature}, phase={self._current_phase.value}")

    def _curvature_to_phase(self, c: float) -> CognitivePhase:
        """Determine cognitive phase from curvature."""
        for phase, (low, high) in self.phase_boundaries.items():
            if low <= c < high:
                return phase
        return CognitivePhase.DEEP_ABSTRACT if c >= 1.5 else CognitivePhase.FLAT_LOCAL

    def request_phase_transition(self, target_phase: CognitivePhase) -> bool:
        """
        Request transition to a target phase.

        Args:
            target_phase: Desired cognitive phase

        Returns:
            True if transition started, False if blocked
        """
        if target_phase == self._current_phase:
            return True  # Already there

        # Check stability before allowing transition
        if self._stability < self.stability_threshold:
            logger.warning(
                f"Phase transition blocked: stability {self._stability:.2f} "
                f"< threshold {self.stability_threshold}"
            )
            return False

        # Set target curvature (middle of target phase range)
        low, high = self.phase_boundaries[target_phase]
        self.target_curvature = (low + high) / 2

        self._previous_phase = self._current_phase
        self._transition_in_progress = True
        self._transition_progress = 0.0

        logger.info(
            f"Phase transition: {self._current_phase.value} → {target_phase.value} "
            f"(c: {self.current_curvature:.2f} → {self.target_curvature:.2f})"
        )
        return True

    def request_curvature(self, target_c: float) -> bool:
        """
        Request specific curvature (may trigger phase transition).

        Args:
            target_c: Target curvature value

        Returns:
            True if request accepted
        """
        target_phase = self._curvature_to_phase(target_c)
        self.target_curvature = max(0.01, target_c)  # Keep positive

        if target_phase != self._current_phase:
            if self._stability < self.stability_threshold:
                logger.warning("Curvature change may destabilize: low stability")
            self._previous_phase = self._current_phase
            self._transition_in_progress = True
            self._transition_progress = 0.0

        return True

    def update(self, stability_signal: float = 1.0) -> PhaseTransitionState:
        """
        Update phase controller state.

        Args:
            stability_signal: Current stability (0-1), e.g. from L7 or CLV

        Returns:
            Current phase transition state
        """
        # Update stability (smoothed)
        self._stability = 0.8 * self._stability + 0.2 * stability_signal

        # Progress transition
        if self._transition_in_progress:
            delta = self.target_curvature - self.current_curvature
            max_step = self.transition_rate * self._stability  # Slower if unstable

            if abs(delta) <= max_step:
                # Transition complete
                self.current_curvature = self.target_curvature
                self._transition_in_progress = False
                self._transition_progress = 1.0
            else:
                # Gradual transition
                step = max_step if delta > 0 else -max_step
                self.current_curvature += step
                progress = 1.0 - abs(delta - step) / abs(self.target_curvature - self.current_curvature + 1e-6)
                self._transition_progress = max(0, min(1, progress))

            # Update phase
            new_phase = self._curvature_to_phase(self.current_curvature)
            if new_phase != self._current_phase:
                self._previous_phase = self._current_phase
                self._current_phase = new_phase

                # Record transition
                import time
                self._phase_history.append((time.time(), new_phase))
                if len(self._phase_history) > self._max_history:
                    self._phase_history.pop(0)

                logger.info(f"Phase transition complete: {new_phase.value} (c={self.current_curvature:.2f})")

        return PhaseTransitionState(
            current_phase=self._current_phase,
            previous_phase=self._previous_phase,
            curvature=self.current_curvature,
            transition_in_progress=self._transition_in_progress,
            transition_progress=self._transition_progress,
            stability=self._stability,
            recommended_mode=self.phase_to_mode[self._current_phase],
        )

    def get_current_phase(self) -> CognitivePhase:
        """Get current cognitive phase."""
        return self._current_phase

    def get_recommended_l6_mode(self) -> str:
        """Get recommended L6 reasoning mode for current phase."""
        return self.phase_to_mode[self._current_phase]

    def get_state(self) -> Dict[str, Any]:
        """Get controller state for monitoring."""
        return {
            "current_phase": self._current_phase.value,
            "previous_phase": self._previous_phase.value,
            "curvature": self.current_curvature,
            "target_curvature": self.target_curvature,
            "transition_in_progress": self._transition_in_progress,
            "transition_progress": self._transition_progress,
            "stability": self._stability,
            "recommended_mode": self.phase_to_mode[self._current_phase],
            "phase_history_length": len(self._phase_history),
        }


def select_phase_for_task(
    task_type: str,
    complexity: float = 0.5,
    hierarchy_depth: Optional[int] = None,
) -> CognitivePhase:
    """
    Select appropriate cognitive phase for task.

    Args:
        task_type: Type of task (planning, retrieval, reasoning, etc.)
        complexity: Task complexity 0-1
        hierarchy_depth: Optional hierarchy depth hint

    Returns:
        Recommended cognitive phase
    """
    # Base selection from task type
    task_phase_map = {
        "planning": CognitivePhase.HIERARCHICAL,
        "retrieval": CognitivePhase.FLAT_LOCAL,
        "reasoning": CognitivePhase.TRANSITIONAL,
        "abstraction": CognitivePhase.DEEP_ABSTRACT,
        "navigation": CognitivePhase.HIERARCHICAL,
        "similarity": CognitivePhase.FLAT_LOCAL,
        "classification": CognitivePhase.TRANSITIONAL,
    }

    base_phase = task_phase_map.get(task_type, CognitivePhase.TRANSITIONAL)

    # Adjust for complexity
    if complexity > 0.8 and base_phase != CognitivePhase.DEEP_ABSTRACT:
        # High complexity → deeper phase
        phases = list(CognitivePhase)
        idx = phases.index(base_phase)
        if idx < len(phases) - 1:
            base_phase = phases[idx + 1]

    # Adjust for hierarchy depth
    if hierarchy_depth is not None:
        if hierarchy_depth > 5:
            base_phase = CognitivePhase.DEEP_ABSTRACT
        elif hierarchy_depth > 3:
            base_phase = CognitivePhase.HIERARCHICAL

    return base_phase


# Global phase controller
_phase_controller: Optional[CognitivePhaseController] = None


def get_phase_controller() -> CognitivePhaseController:
    """Get or create global phase controller."""
    global _phase_controller
    if _phase_controller is None:
        _phase_controller = CognitivePhaseController()
    return _phase_controller


def get_cognitive_phase() -> str:
    """Get current cognitive phase name."""
    return get_phase_controller().get_current_phase().value


def transition_to_phase(phase_name: str) -> bool:
    """Request transition to named phase."""
    phase_map = {p.value: p for p in CognitivePhase}
    if phase_name not in phase_map:
        logger.warning(f"Unknown phase: {phase_name}")
        return False
    return get_phase_controller().request_phase_transition(phase_map[phase_name])


__all__ = [
    "GeometryType",
    "TaskGeometryHint",
    "HyperbolicConfig",
    "HyperbolicMath",
    "GeometrySelection",
    "AdaptiveGeometrySelector",
    "HyperbolicEmbedding",
    "GeometricRoutingDecision",
    "compute_geometric_routing",
    "get_geometry_selector",
    "select_geometry_for_task",
    # L8 Phase Transitions
    "CognitivePhase",
    "PhaseTransitionState",
    "CognitivePhaseController",
    "select_phase_for_task",
    "get_phase_controller",
    "get_cognitive_phase",
    "transition_to_phase",
]
