"""
Semantic System Optimizer for TF-A-N

Intelligent resource allocation layer that upgrades simple backend routing
to a dynamically weighted scoring system based on:
1. PAD (Pleasure-Arousal-Dominance) state from L2 Appraisal
2. Resource features from CXL/FPGA benchmarks
3. PGU verification status
4. Historical routing success scores

This module bridges the gap between emotional state and hardware routing,
implementing context-aware resource allocation.

Control Law (L3 → Resource):
    If Valence < threshold:
        → Prefer PGU-verified backend (safety over latency)
    If Arousal high:
        → Prefer low-latency FPGA path
    If Dominance low (uncertainty):
        → Prefer redundant/fallback backends

Usage:
    from tfan.system.semantic_optimizer import SemanticSystemOptimizer, PADState

    optimizer = SemanticSystemOptimizer()

    pad = PADState(valence=-0.3, arousal=0.8, dominance=0.5)
    decision = optimizer.recommend_route(
        pad_state=pad,
        resource_features={"fpga_available": True, "gpu_util": 0.7},
    )

    print(f"Recommended backend: {decision.backend}")
    print(f"Confidence: {decision.confidence:.2f}")
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from datetime import datetime

import yaml

logger = logging.getLogger("tfan.system.semantic_optimizer")

# Routing scores persistence path
DEFAULT_SCORES_PATH = Path("configs/routing_scores.yaml")


class Backend(str, Enum):
    """Available compute backends."""
    GPU_DENSE = "gpu_dense"        # Standard GPU dense computation
    GPU_SPARSE = "gpu_sparse"      # GPU with sparse masks
    FPGA_SNN = "fpga_snn"          # FPGA neuromorphic accelerator
    CPU_FALLBACK = "cpu_fallback"  # CPU fallback (always available)
    PGU_VERIFIED = "pgu_verified"  # PGU-verified safe path


@dataclass
class PADState:
    """Pleasure-Arousal-Dominance emotional state from L2 Appraisal."""
    valence: float = 0.0      # [-1, 1] Pleasure/displeasure
    arousal: float = 0.5      # [0, 1] Activation level
    dominance: float = 0.5    # [0, 1] Control/certainty
    stability_gap: float = 0.0  # Topological stability metric
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PADState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ResourceFeatures:
    """Hardware resource state from CXL/FPGA benchmarks."""
    # GPU state
    gpu_available: bool = True
    gpu_utilization: float = 0.0    # [0, 1]
    gpu_memory_free_gb: float = 8.0
    gpu_temperature_c: float = 50.0

    # FPGA state
    fpga_available: bool = False
    fpga_utilization: float = 0.0
    fpga_temperature_c: float = 40.0
    fpga_threshold: float = 1.0     # Current v_th setting

    # CXL state
    cxl_available: bool = False
    cxl_latency_ns: float = 200.0
    cxl_bandwidth_gbps: float = 32.0

    # PGU state
    pgu_cache_hit_rate: float = 0.5
    pgu_avg_latency_ms: float = 50.0

    # System state
    cpu_utilization: float = 0.0
    memory_pressure: float = 0.0    # [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResourceFeatures":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RoutingDecision:
    """Output of semantic routing decision."""
    backend: Backend
    confidence: float               # [0, 1] Decision confidence
    scores: Dict[str, float]        # Per-backend scores
    reasoning: List[str]            # Human-readable reasoning
    pgu_required: bool              # Whether PGU verification needed
    fallback_backend: Backend       # Backup if primary fails
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["backend"] = self.backend.value
        d["fallback_backend"] = self.fallback_backend.value
        return d


@dataclass
class RoutingScores:
    """Persistent learned routing scores."""
    # Per-backend success rates
    success_rates: Dict[str, float] = field(default_factory=lambda: {
        Backend.GPU_DENSE.value: 0.9,
        Backend.GPU_SPARSE.value: 0.85,
        Backend.FPGA_SNN.value: 0.8,
        Backend.CPU_FALLBACK.value: 0.95,
        Backend.PGU_VERIFIED.value: 0.99,
    })

    # Per-backend latency history (EMA)
    latency_ema: Dict[str, float] = field(default_factory=lambda: {
        Backend.GPU_DENSE.value: 10.0,
        Backend.GPU_SPARSE.value: 8.0,
        Backend.FPGA_SNN.value: 2.0,
        Backend.CPU_FALLBACK.value: 50.0,
        Backend.PGU_VERIFIED.value: 15.0,
    })

    # Context-specific adjustments
    low_valence_penalty: Dict[str, float] = field(default_factory=lambda: {
        Backend.GPU_DENSE.value: 0.2,
        Backend.GPU_SPARSE.value: 0.1,
        Backend.FPGA_SNN.value: 0.05,
        Backend.CPU_FALLBACK.value: 0.0,
        Backend.PGU_VERIFIED.value: -0.2,  # Bonus for PGU when valence low
    })

    high_arousal_bonus: Dict[str, float] = field(default_factory=lambda: {
        Backend.GPU_DENSE.value: 0.0,
        Backend.GPU_SPARSE.value: 0.1,
        Backend.FPGA_SNN.value: 0.3,  # FPGA preferred for speed
        Backend.CPU_FALLBACK.value: -0.3,
        Backend.PGU_VERIFIED.value: 0.0,
    })

    # Update tracking
    total_decisions: int = 0
    total_successes: int = 0
    last_updated: str = ""
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoutingScores":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_routing_scores(path: Optional[Path] = None) -> RoutingScores:
    """Load routing scores from YAML file."""
    path = path or DEFAULT_SCORES_PATH

    if not path.exists():
        logger.info(f"No routing scores found at {path}, using defaults")
        return RoutingScores()

    with open(path) as f:
        data = yaml.safe_load(f)

    scores = RoutingScores.from_dict(data)
    logger.info(f"Loaded routing scores from {path}")
    return scores


def save_routing_scores(scores: RoutingScores, path: Optional[Path] = None):
    """Save routing scores to YAML file."""
    path = path or DEFAULT_SCORES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    scores.last_updated = datetime.utcnow().isoformat()

    with open(path, 'w') as f:
        yaml.dump(scores.to_dict(), f, default_flow_style=False)

    logger.info(f"Saved routing scores to {path}")


class SemanticSystemOptimizer:
    """
    Intelligent resource allocation using semantic context.

    Combines emotional state (PAD) with hardware resources to make
    context-aware backend selection decisions.

    Architecture:
        PAD State (L2) ─┐
                        ├─→ Scoring Engine ─→ Backend Selection
        Resources ──────┘

        Learned Scores (persistent) ←──┘ (feedback loop)
    """

    def __init__(
        self,
        scores_path: Optional[Path] = None,
        auto_persist: bool = True,
        safety_first: bool = True,
    ):
        """
        Initialize optimizer.

        Args:
            scores_path: Path to persistent scores file
            auto_persist: Whether to auto-save scores after updates
            safety_first: If True, always prefer PGU-verified when uncertain
        """
        self.scores_path = scores_path or DEFAULT_SCORES_PATH
        self.auto_persist = auto_persist
        self.safety_first = safety_first

        # Load persistent scores
        self.scores = load_routing_scores(self.scores_path)

        # Thresholds for context-aware decisions
        self.valence_safety_threshold = -0.3  # Below this → prefer safe backends
        self.arousal_speed_threshold = 0.7    # Above this → prefer fast backends
        self.dominance_certainty_threshold = 0.4  # Below this → prefer verified
        self.stability_gap_threshold = 0.3    # Above this → prefer conservative

        logger.info("SemanticSystemOptimizer initialized")

    def recommend_route(
        self,
        pad_state: PADState,
        resource_features: Optional[ResourceFeatures] = None,
        workload_hint: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Recommend compute backend based on semantic context.

        This is the main entry point for context-aware resource allocation.

        Args:
            pad_state: Current emotional state from L2 Appraisal
            resource_features: Hardware resource availability
            workload_hint: Optional hint ("latency_critical", "throughput", "safe")

        Returns:
            RoutingDecision with recommended backend and reasoning
        """
        resources = resource_features or ResourceFeatures()
        reasoning = []

        # Step 1: Compute base scores for each backend
        scores = self._compute_base_scores(resources)
        reasoning.append(f"Base scores: {self._format_scores(scores)}")

        # Step 2: Apply PAD-based adjustments
        scores = self._apply_pad_adjustments(scores, pad_state, reasoning)

        # Step 3: Apply stability-based adjustments
        scores = self._apply_stability_adjustments(scores, pad_state, reasoning)

        # Step 4: Apply resource availability constraints
        scores = self._apply_availability_constraints(scores, resources, reasoning)

        # Step 5: Apply workload hint
        if workload_hint:
            scores = self._apply_workload_hint(scores, workload_hint, reasoning)

        # Step 6: Select best backend
        best_backend, confidence = self._select_best(scores)

        # Step 7: Determine if PGU verification needed
        pgu_required = self._needs_pgu_verification(pad_state, resources)
        if pgu_required:
            reasoning.append("PGU verification required (low valence/dominance)")

        # Step 8: Select fallback
        fallback = self._select_fallback(scores, best_backend)

        decision = RoutingDecision(
            backend=best_backend,
            confidence=confidence,
            scores={k.value: v for k, v in scores.items()},
            reasoning=reasoning,
            pgu_required=pgu_required,
            fallback_backend=fallback,
            timestamp=datetime.utcnow().isoformat(),
        )

        logger.info(f"Route decision: {best_backend.value} (conf={confidence:.2f})")

        return decision

    def _compute_base_scores(
        self,
        resources: ResourceFeatures,
    ) -> Dict[Backend, float]:
        """Compute base scores from learned success rates and latency."""
        scores = {}

        for backend in Backend:
            # Base = success_rate - normalized_latency
            success = self.scores.success_rates.get(backend.value, 0.5)
            latency = self.scores.latency_ema.get(backend.value, 50.0)

            # Normalize latency to [0, 1] (lower is better)
            latency_score = max(0, 1 - latency / 100.0)

            # Weighted combination
            scores[backend] = 0.6 * success + 0.4 * latency_score

        return scores

    def _apply_pad_adjustments(
        self,
        scores: Dict[Backend, float],
        pad: PADState,
        reasoning: List[str],
    ) -> Dict[Backend, float]:
        """Apply PAD-based score adjustments."""

        # Low valence → penalize risky backends, bonus safe ones
        if pad.valence < self.valence_safety_threshold:
            reasoning.append(f"Low valence ({pad.valence:.2f}) → safety preference")
            for backend in scores:
                penalty = self.scores.low_valence_penalty.get(backend.value, 0.0)
                scores[backend] -= penalty

        # High arousal → bonus for fast backends
        if pad.arousal > self.arousal_speed_threshold:
            reasoning.append(f"High arousal ({pad.arousal:.2f}) → speed preference")
            for backend in scores:
                bonus = self.scores.high_arousal_bonus.get(backend.value, 0.0)
                scores[backend] += bonus

        # Low dominance → prefer verified/safe
        if pad.dominance < self.dominance_certainty_threshold:
            reasoning.append(f"Low dominance ({pad.dominance:.2f}) → verified preference")
            scores[Backend.PGU_VERIFIED] += 0.2
            scores[Backend.CPU_FALLBACK] += 0.1

        return scores

    def _apply_stability_adjustments(
        self,
        scores: Dict[Backend, float],
        pad: PADState,
        reasoning: List[str],
    ) -> Dict[Backend, float]:
        """Apply topological stability-based adjustments."""

        if pad.stability_gap > self.stability_gap_threshold:
            reasoning.append(f"High stability gap ({pad.stability_gap:.2f}) → conservative")
            # Penalize aggressive backends
            scores[Backend.GPU_DENSE] -= 0.15
            scores[Backend.GPU_SPARSE] -= 0.1
            # Bonus conservative paths
            scores[Backend.PGU_VERIFIED] += 0.15
            scores[Backend.CPU_FALLBACK] += 0.1

        return scores

    def _apply_availability_constraints(
        self,
        scores: Dict[Backend, float],
        resources: ResourceFeatures,
        reasoning: List[str],
    ) -> Dict[Backend, float]:
        """Apply hardware availability constraints."""

        # FPGA availability
        if not resources.fpga_available:
            scores[Backend.FPGA_SNN] = -1.0  # Unavailable
            reasoning.append("FPGA unavailable")
        elif resources.fpga_temperature_c > 80:
            scores[Backend.FPGA_SNN] -= 0.3
            reasoning.append("FPGA thermal throttling")

        # GPU availability
        if not resources.gpu_available:
            scores[Backend.GPU_DENSE] = -1.0
            scores[Backend.GPU_SPARSE] = -1.0
            reasoning.append("GPU unavailable")
        elif resources.gpu_utilization > 0.9:
            scores[Backend.GPU_DENSE] -= 0.2
            scores[Backend.GPU_SPARSE] -= 0.1
            reasoning.append("GPU heavily loaded")

        # Memory pressure
        if resources.memory_pressure > 0.8:
            scores[Backend.GPU_DENSE] -= 0.2
            reasoning.append("High memory pressure")

        # PGU cache performance
        if resources.pgu_cache_hit_rate < 0.3:
            scores[Backend.PGU_VERIFIED] -= 0.1
            reasoning.append("Low PGU cache hits")

        return scores

    def _apply_workload_hint(
        self,
        scores: Dict[Backend, float],
        hint: str,
        reasoning: List[str],
    ) -> Dict[Backend, float]:
        """Apply workload-specific adjustments."""

        if hint == "latency_critical":
            reasoning.append("Workload hint: latency_critical")
            scores[Backend.FPGA_SNN] += 0.2
            scores[Backend.GPU_SPARSE] += 0.1
            scores[Backend.CPU_FALLBACK] -= 0.3

        elif hint == "throughput":
            reasoning.append("Workload hint: throughput")
            scores[Backend.GPU_DENSE] += 0.2
            scores[Backend.FPGA_SNN] -= 0.1

        elif hint == "safe":
            reasoning.append("Workload hint: safe")
            scores[Backend.PGU_VERIFIED] += 0.3
            scores[Backend.CPU_FALLBACK] += 0.1

        return scores

    def _select_best(
        self,
        scores: Dict[Backend, float],
    ) -> Tuple[Backend, float]:
        """Select best backend and compute confidence."""
        # Filter out unavailable
        available = {k: v for k, v in scores.items() if v >= 0}

        if not available:
            return Backend.CPU_FALLBACK, 0.5

        # Sort by score
        sorted_backends = sorted(available.items(), key=lambda x: -x[1])
        best = sorted_backends[0]

        # Confidence = gap between best and second-best
        if len(sorted_backends) > 1:
            second = sorted_backends[1]
            confidence = min(1.0, 0.5 + (best[1] - second[1]))
        else:
            confidence = 0.8

        return best[0], confidence

    def _select_fallback(
        self,
        scores: Dict[Backend, float],
        primary: Backend,
    ) -> Backend:
        """Select fallback backend."""
        available = {k: v for k, v in scores.items() if v >= 0 and k != primary}

        if not available:
            return Backend.CPU_FALLBACK

        # Prefer safe fallbacks
        if Backend.CPU_FALLBACK in available:
            return Backend.CPU_FALLBACK
        if Backend.PGU_VERIFIED in available:
            return Backend.PGU_VERIFIED

        return max(available, key=lambda k: available[k])

    def _needs_pgu_verification(
        self,
        pad: PADState,
        resources: ResourceFeatures,
    ) -> bool:
        """Determine if PGU verification is required."""
        if self.safety_first:
            # Require PGU when uncertain or stressed
            if pad.valence < self.valence_safety_threshold:
                return True
            if pad.dominance < self.dominance_certainty_threshold:
                return True
            if pad.stability_gap > self.stability_gap_threshold:
                return True

        return False

    def _format_scores(self, scores: Dict[Backend, float]) -> str:
        """Format scores for logging."""
        return ", ".join(f"{k.value}={v:.2f}" for k, v in scores.items())

    def update_from_feedback(
        self,
        backend: Backend,
        success: bool,
        latency_ms: float,
        pad_state: Optional[PADState] = None,
    ):
        """
        Update routing scores from execution feedback.

        This enables online learning of routing preferences.

        Args:
            backend: Backend that was used
            success: Whether execution succeeded
            latency_ms: Actual execution latency
            pad_state: PAD state when decision was made (for context learning)
        """
        alpha = 0.1  # EMA smoothing factor

        # Update success rate
        current_success = self.scores.success_rates.get(backend.value, 0.5)
        new_success = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_success
        self.scores.success_rates[backend.value] = new_success

        # Update latency EMA
        current_latency = self.scores.latency_ema.get(backend.value, 50.0)
        new_latency = alpha * latency_ms + (1 - alpha) * current_latency
        self.scores.latency_ema[backend.value] = new_latency

        # Update context-specific scores if PAD provided
        if pad_state is not None and success:
            # Learn that this backend works well in this context
            if pad_state.valence < self.valence_safety_threshold:
                penalty = self.scores.low_valence_penalty.get(backend.value, 0.0)
                self.scores.low_valence_penalty[backend.value] = penalty * 0.95

        # Track totals
        self.scores.total_decisions += 1
        if success:
            self.scores.total_successes += 1

        # Auto-persist
        if self.auto_persist:
            save_routing_scores(self.scores, self.scores_path)

        logger.debug(f"Updated scores for {backend.value}: success={new_success:.2f}, latency={new_latency:.1f}ms")

    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status."""
        return {
            "total_decisions": self.scores.total_decisions,
            "total_successes": self.scores.total_successes,
            "success_rate": (
                self.scores.total_successes / self.scores.total_decisions
                if self.scores.total_decisions > 0 else 0.0
            ),
            "success_rates": self.scores.success_rates,
            "latency_ema": self.scores.latency_ema,
            "safety_first": self.safety_first,
            "last_updated": self.scores.last_updated,
        }


def create_semantic_optimizer(
    config_path: Optional[Path] = None,
) -> SemanticSystemOptimizer:
    """
    Factory function to create SemanticSystemOptimizer.

    Args:
        config_path: Path to scores persistence file

    Returns:
        Configured optimizer
    """
    return SemanticSystemOptimizer(scores_path=config_path)


# Exports
__all__ = [
    "SemanticSystemOptimizer",
    "PADState",
    "ResourceFeatures",
    "RoutingDecision",
    "RoutingScores",
    "Backend",
    "load_routing_scores",
    "save_routing_scores",
    "create_semantic_optimizer",
]
