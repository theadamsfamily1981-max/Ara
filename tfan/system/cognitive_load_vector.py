"""
Cognitive Load Vector (CLV) - Unified L1/L2 Metric Coalescing

Consolidates all L1 Homeostatic and L2 Appraisal metrics into a single,
structured vector that L3 Metacontrol can act upon. This eliminates
"signal spam" over D-Bus/WebSocket and provides a coherent risk assessment.

CLV Components:
    CLV_Instability: EPR-CV ⊕ topo_gap → Primary risk signal
    CLV_Resource:    Jerk ⊕ CXL_latency → System overhead/effort
    CLV_Structural:  r ⊕ k → Current structural fingerprint

Integration:
    L1 Homeostasis ─┐
                    ├─→ CLV ─→ L3 Metacontrol ─→ AEPO ─→ PGU
    L2 Appraisal ───┘

D-Bus Signal:
    org.ara.metacontrol.CLVUpdated(instability, resource, structural, timestamp)

Usage:
    from tfan.system.cognitive_load_vector import (
        CognitiveLoadVector,
        CLVComputer,
        create_clv_from_state,
    )

    clv = create_clv_from_state(
        epr_cv=0.12,
        topo_gap=0.25,
        jerk=0.05,
        valence=-0.3,
        arousal=0.7,
    )

    print(f"Risk level: {clv.risk_level}")
    print(f"Instability: {clv.instability:.3f}")
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path

logger = logging.getLogger("tfan.system.clv")


class RiskLevel(str, Enum):
    """Risk level classification from CLV."""
    NOMINAL = "nominal"       # System operating normally
    ELEVATED = "elevated"     # Minor concern, increased monitoring
    WARNING = "warning"       # Significant concern, conservative policy
    CRITICAL = "critical"     # Immediate intervention needed
    EMERGENCY = "emergency"   # System unstable, fail-safe mode


@dataclass
class CLVComponents:
    """Individual CLV component values."""
    # Instability component (EPR-CV ⊕ topo_gap)
    epr_cv: float = 0.0           # [0, 1] Epistemic uncertainty
    topo_gap: float = 0.0         # [0, 1] Topological stability gap
    stability_trend: float = 0.0  # [-1, 1] Rate of change

    # Resource component (Jerk ⊕ CXL latency)
    jerk: float = 0.0             # [0, 1] PAD rate of change
    cxl_latency_ns: float = 200.0 # CXL memory latency
    gpu_utilization: float = 0.0  # [0, 1]
    memory_pressure: float = 0.0  # [0, 1]

    # Structural component (r ⊕ k)
    keep_ratio: float = 1.0       # [0, 1] Sparsity keep ratio
    spectral_gap: float = 0.1     # λ₂ Fiedler value
    beta1: int = 0                # Loop count
    connectivity: int = 1         # Connected components

    # PAD state
    valence: float = 0.0          # [-1, 1]
    arousal: float = 0.5          # [0, 1]
    dominance: float = 0.5        # [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CognitiveLoadVector:
    """
    Unified Cognitive Load Vector.

    Coalesces all L1/L2 metrics into three scalar dimensions + risk level.
    This single vector replaces multiple individual signals for L3 control.
    """
    # Primary dimensions (each [0, 1])
    instability: float = 0.0      # Combined EPR-CV + topo_gap risk
    resource: float = 0.0         # Combined system overhead
    structural: float = 0.0       # Combined structural health

    # Derived properties
    risk_level: RiskLevel = RiskLevel.NOMINAL
    risk_score: float = 0.0       # [0, 1] Overall risk

    # Control hints (derived from risk analysis)
    recommend_conservative: bool = False
    recommend_pgu_check: bool = False
    recommend_structural_change: bool = False

    # Metadata
    components: CLVComponents = field(default_factory=CLVComponents)
    timestamp: str = ""
    computation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_dbus_tuple(self) -> Tuple[float, float, float, str, str]:
        """Format for D-Bus signal emission."""
        return (
            self.instability,
            self.resource,
            self.structural,
            self.risk_level.value,
            self.timestamp,
        )


class CLVComputer:
    """
    Computes Cognitive Load Vector from raw metrics.

    Implements the fusion logic for combining L1/L2 signals into
    the unified CLV representation.
    """

    # Weight matrices for component fusion
    INSTABILITY_WEIGHTS = {
        "epr_cv": 0.4,          # EPR-CV contribution
        "topo_gap": 0.35,       # Topological gap contribution
        "neg_valence": 0.15,    # Negative valence contribution
        "trend": 0.1,           # Trend contribution
    }

    RESOURCE_WEIGHTS = {
        "jerk": 0.3,            # PAD jerk contribution
        "cxl_latency": 0.2,     # CXL latency contribution
        "gpu_util": 0.25,       # GPU utilization
        "memory": 0.25,         # Memory pressure
    }

    STRUCTURAL_WEIGHTS = {
        "sparsity": 0.3,        # 1 - keep_ratio
        "spectral": 0.3,        # Spectral gap quality
        "loops": 0.2,           # β₁ loop redundancy
        "connectivity": 0.2,    # Single component = good
    }

    # Risk thresholds
    RISK_THRESHOLDS = {
        RiskLevel.NOMINAL: 0.2,
        RiskLevel.ELEVATED: 0.4,
        RiskLevel.WARNING: 0.6,
        RiskLevel.CRITICAL: 0.8,
        RiskLevel.EMERGENCY: 1.0,
    }

    def __init__(
        self,
        epr_cv_target: float = 0.15,
        topo_gap_target: float = 0.1,
        cxl_latency_target_ns: float = 200.0,
    ):
        """
        Initialize CLV computer.

        Args:
            epr_cv_target: Target EPR-CV (hard gate ≤ 0.15)
            topo_gap_target: Target topological gap
            cxl_latency_target_ns: Target CXL latency
        """
        self.epr_cv_target = epr_cv_target
        self.topo_gap_target = topo_gap_target
        self.cxl_latency_target_ns = cxl_latency_target_ns

        # History for trend computation
        self._history: List[CLVComponents] = []
        self._max_history = 10

        # Callbacks for CLV updates
        self._callbacks: List[Callable[[CognitiveLoadVector], None]] = []

    def compute(
        self,
        epr_cv: float = 0.0,
        topo_gap: float = 0.0,
        jerk: float = 0.0,
        cxl_latency_ns: float = 200.0,
        gpu_utilization: float = 0.0,
        memory_pressure: float = 0.0,
        keep_ratio: float = 1.0,
        spectral_gap: float = 0.1,
        beta1: int = 0,
        connectivity: int = 1,
        valence: float = 0.0,
        arousal: float = 0.5,
        dominance: float = 0.5,
    ) -> CognitiveLoadVector:
        """
        Compute CLV from raw metrics.

        This is the main fusion function that combines all signals.

        Args:
            epr_cv: Epistemic uncertainty [0, 1]
            topo_gap: Topological stability gap [0, 1]
            jerk: PAD rate of change [0, 1]
            cxl_latency_ns: CXL memory latency
            gpu_utilization: GPU utilization [0, 1]
            memory_pressure: Memory pressure [0, 1]
            keep_ratio: Sparsity keep ratio [0, 1]
            spectral_gap: λ₂ Fiedler value
            beta1: Loop count (β₁ Betti number)
            connectivity: Connected components (β₀)
            valence: PAD valence [-1, 1]
            arousal: PAD arousal [0, 1]
            dominance: PAD dominance [0, 1]

        Returns:
            CognitiveLoadVector
        """
        start = time.perf_counter()

        # Store components
        components = CLVComponents(
            epr_cv=epr_cv,
            topo_gap=topo_gap,
            jerk=jerk,
            cxl_latency_ns=cxl_latency_ns,
            gpu_utilization=gpu_utilization,
            memory_pressure=memory_pressure,
            keep_ratio=keep_ratio,
            spectral_gap=spectral_gap,
            beta1=beta1,
            connectivity=connectivity,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
        )

        # Compute stability trend
        trend = self._compute_trend(components)
        components.stability_trend = trend

        # Update history
        self._history.append(components)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Compute instability dimension
        instability = self._compute_instability(components, trend)

        # Compute resource dimension
        resource = self._compute_resource(components)

        # Compute structural dimension
        structural = self._compute_structural(components)

        # Compute overall risk
        risk_score = self._compute_risk_score(instability, resource, structural)
        risk_level = self._classify_risk(risk_score)

        # Derive control hints
        recommend_conservative = risk_level in [RiskLevel.WARNING, RiskLevel.CRITICAL, RiskLevel.EMERGENCY]
        recommend_pgu_check = valence < -0.3 or topo_gap > 0.3
        recommend_structural_change = structural > 0.5 and risk_level == RiskLevel.WARNING

        elapsed = (time.perf_counter() - start) * 1000

        clv = CognitiveLoadVector(
            instability=instability,
            resource=resource,
            structural=structural,
            risk_level=risk_level,
            risk_score=risk_score,
            recommend_conservative=recommend_conservative,
            recommend_pgu_check=recommend_pgu_check,
            recommend_structural_change=recommend_structural_change,
            components=components,
            timestamp=datetime.utcnow().isoformat(),
            computation_time_ms=elapsed,
        )

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(clv)
            except Exception as e:
                logger.error(f"CLV callback error: {e}")

        return clv

    def _compute_instability(self, c: CLVComponents, trend: float) -> float:
        """Compute instability dimension."""
        w = self.INSTABILITY_WEIGHTS

        # Normalize EPR-CV relative to target
        epr_norm = min(1.0, c.epr_cv / self.epr_cv_target) if self.epr_cv_target > 0 else 0

        # Normalize topo_gap relative to target
        topo_norm = min(1.0, c.topo_gap / self.topo_gap_target) if self.topo_gap_target > 0 else 0

        # Negative valence contributes to instability
        neg_valence = max(0, -c.valence)

        # Combine with weights
        instability = (
            w["epr_cv"] * epr_norm +
            w["topo_gap"] * topo_norm +
            w["neg_valence"] * neg_valence +
            w["trend"] * max(0, trend)  # Positive trend = worsening
        )

        return min(1.0, max(0.0, instability))

    def _compute_resource(self, c: CLVComponents) -> float:
        """Compute resource dimension."""
        w = self.RESOURCE_WEIGHTS

        # Normalize CXL latency relative to target
        cxl_norm = min(1.0, c.cxl_latency_ns / (self.cxl_latency_target_ns * 2))

        # Combine with weights
        resource = (
            w["jerk"] * c.jerk +
            w["cxl_latency"] * cxl_norm +
            w["gpu_util"] * c.gpu_utilization +
            w["memory"] * c.memory_pressure
        )

        return min(1.0, max(0.0, resource))

    def _compute_structural(self, c: CLVComponents) -> float:
        """Compute structural dimension."""
        w = self.STRUCTURAL_WEIGHTS

        # Sparsity: lower keep_ratio = more sparse = potentially less stable
        sparsity_risk = 1.0 - c.keep_ratio

        # Spectral: low spectral gap = poor connectivity
        spectral_risk = max(0, 1.0 - c.spectral_gap * 10)  # Normalize to [0, 1]

        # Loops: more loops = more redundancy = better
        loops_quality = min(1.0, c.beta1 / 20.0)  # Normalize to [0, 1]
        loops_risk = 1.0 - loops_quality

        # Connectivity: single component = good
        connectivity_risk = min(1.0, (c.connectivity - 1) / 5.0)

        # Combine with weights
        structural = (
            w["sparsity"] * sparsity_risk +
            w["spectral"] * spectral_risk +
            w["loops"] * loops_risk +
            w["connectivity"] * connectivity_risk
        )

        return min(1.0, max(0.0, structural))

    def _compute_risk_score(
        self,
        instability: float,
        resource: float,
        structural: float,
    ) -> float:
        """Compute overall risk score from dimensions."""
        # Weighted combination with instability having highest priority
        risk = 0.5 * instability + 0.3 * resource + 0.2 * structural

        # Apply high-arousal amplification
        if self._history:
            arousal = self._history[-1].arousal
            if arousal > 0.7:
                risk *= 1.0 + (arousal - 0.7)

        return min(1.0, max(0.0, risk))

    def _classify_risk(self, risk_score: float) -> RiskLevel:
        """Classify risk score into level."""
        for level in [RiskLevel.NOMINAL, RiskLevel.ELEVATED, RiskLevel.WARNING, RiskLevel.CRITICAL]:
            if risk_score <= self.RISK_THRESHOLDS[level]:
                return level
        return RiskLevel.EMERGENCY

    def _compute_trend(self, current: CLVComponents) -> float:
        """Compute stability trend from history."""
        if len(self._history) < 2:
            return 0.0

        # Compare current EPR-CV + topo_gap vs recent average
        recent = self._history[-3:] if len(self._history) >= 3 else self._history

        recent_instability = sum(
            h.epr_cv + h.topo_gap for h in recent
        ) / len(recent)

        current_instability = current.epr_cv + current.topo_gap

        # Positive trend = worsening
        trend = current_instability - recent_instability

        return max(-1.0, min(1.0, trend * 5))  # Scale and clamp

    def on_clv_update(self, callback: Callable[[CognitiveLoadVector], None]):
        """Register callback for CLV updates."""
        self._callbacks.append(callback)

    def get_history_summary(self) -> Dict[str, Any]:
        """Get summary of recent CLV history."""
        if not self._history:
            return {"count": 0}

        return {
            "count": len(self._history),
            "avg_epr_cv": sum(h.epr_cv for h in self._history) / len(self._history),
            "avg_topo_gap": sum(h.topo_gap for h in self._history) / len(self._history),
            "avg_valence": sum(h.valence for h in self._history) / len(self._history),
            "trend": self._history[-1].stability_trend if self._history else 0.0,
        }


def create_clv_from_state(
    epr_cv: float = 0.0,
    topo_gap: float = 0.0,
    jerk: float = 0.0,
    valence: float = 0.0,
    arousal: float = 0.5,
    dominance: float = 0.5,
    **kwargs,
) -> CognitiveLoadVector:
    """
    Convenience function to create CLV from state values.

    Args:
        epr_cv: Epistemic uncertainty
        topo_gap: Topological stability gap
        jerk: PAD rate of change
        valence: PAD valence
        arousal: PAD arousal
        dominance: PAD dominance
        **kwargs: Additional metrics

    Returns:
        CognitiveLoadVector
    """
    computer = CLVComputer()
    return computer.compute(
        epr_cv=epr_cv,
        topo_gap=topo_gap,
        jerk=jerk,
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        **kwargs,
    )


# D-Bus integration (if available)
DBUS_AVAILABLE = False
try:
    import dbus
    import dbus.service
    DBUS_AVAILABLE = True
except ImportError:
    pass


class CLVDBusService:
    """D-Bus service for CLV signal emission."""

    DBUS_BUS_NAME = "org.ara.metacontrol"
    DBUS_PATH = "/org/ara/metacontrol/CLV"
    DBUS_INTERFACE = "org.ara.metacontrol.CLVInterface"

    def __init__(self):
        self._bus = None
        self._connected = False

        if DBUS_AVAILABLE:
            try:
                self._bus = dbus.SessionBus()
                self._connected = True
                logger.info("CLV D-Bus service connected")
            except Exception as e:
                logger.warning(f"D-Bus connection failed: {e}")

    def emit_clv_updated(self, clv: CognitiveLoadVector):
        """Emit CLVUpdated D-Bus signal."""
        if not self._connected or self._bus is None:
            return

        try:
            # Emit signal with CLV tuple
            # Signal: CLVUpdated(instability, resource, structural, risk_level, timestamp)
            pass  # D-Bus signal emission would go here
        except Exception as e:
            logger.debug(f"D-Bus emit failed: {e}")

    def is_connected(self) -> bool:
        return self._connected


# Global CLV computer instance
_global_clv_computer: Optional[CLVComputer] = None


def get_clv_computer() -> CLVComputer:
    """Get global CLV computer instance."""
    global _global_clv_computer
    if _global_clv_computer is None:
        _global_clv_computer = CLVComputer()
    return _global_clv_computer


# Exports
__all__ = [
    "CognitiveLoadVector",
    "CLVComponents",
    "CLVComputer",
    "RiskLevel",
    "create_clv_from_state",
    "get_clv_computer",
    "CLVDBusService",
    "DBUS_AVAILABLE",
]
