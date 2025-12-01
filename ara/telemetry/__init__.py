"""
ARA Pulse Telemetry Module

Prometheus metrics and telemetry for L3 Metacontrol and Pulse.
Exposes metrics at /metrics endpoint for monitoring.

Metrics exposed:
- ara_pulse_pad_valence: Current PAD valence (pleasure)
- ara_pulse_pad_arousal: Current PAD arousal
- ara_pulse_pad_dominance: Current PAD dominance
- ara_pulse_evt_flagged_total: Counter of EVT threshold exceeded events
- ara_metacontrol_temperature_mult: Current temperature multiplier
- ara_metacontrol_memory_mult: Current memory write multiplier
- ara_metacontrol_mode: Current workspace mode (gauge with labels)
- ara_process_turn_seconds: Histogram of turn processing time
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field
import logging
import json

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("ara.telemetry")

# Try to import prometheus_client
PROMETHEUS_AVAILABLE = False
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not installed. Metrics will not be exported.")


@dataclass
class MetricSnapshot:
    """A snapshot of all metrics at a point in time."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # PAD state
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.5
    confidence: float = 1.0

    # Metacontrol
    workspace_mode: str = "default"
    temperature_multiplier: float = 1.0
    memory_write_multiplier: float = 1.0
    attention_gain: float = 1.0
    effective_weight: float = 1.0

    # Derived
    effective_temperature: float = 0.7
    effective_memory_p: float = 0.5

    # Flags
    evt_flagged: bool = False
    p_anom: float = 0.0

    # Processing
    processing_time_ms: float = 0.0
    turn_number: int = 0
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "pad": {
                "valence": self.valence,
                "arousal": self.arousal,
                "dominance": self.dominance,
                "confidence": self.confidence,
            },
            "metacontrol": {
                "workspace_mode": self.workspace_mode,
                "temperature_multiplier": self.temperature_multiplier,
                "memory_write_multiplier": self.memory_write_multiplier,
                "attention_gain": self.attention_gain,
                "effective_weight": self.effective_weight,
            },
            "effective": {
                "temperature": self.effective_temperature,
                "memory_p": self.effective_memory_p,
            },
            "flags": {
                "evt_flagged": self.evt_flagged,
                "p_anom": self.p_anom,
            },
            "processing": {
                "time_ms": self.processing_time_ms,
                "turn_number": self.turn_number,
                "session_id": self.session_id,
            },
        }


class PulseTelemetry:
    """
    Pulse telemetry collector with optional Prometheus export.

    Collects and exposes metrics for:
    - PAD state (valence, arousal, dominance)
    - L3 Metacontrol (temperature/memory multipliers)
    - Processing performance (turn latency)
    - Anomaly detection (EVT flags)
    """

    def __init__(
        self,
        registry: Optional["CollectorRegistry"] = None,
        namespace: str = "ara",
        enable_prometheus: bool = True,
    ):
        self.namespace = namespace
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE

        # History buffer
        self._snapshots: List[MetricSnapshot] = []
        self._max_history = 1000

        # Workspace mode counts for tracking
        self._mode_counts: Dict[str, int] = {}

        # Initialize Prometheus metrics if available
        if self.enable_prometheus:
            self._init_prometheus_metrics(registry)
        else:
            self._metrics = {}

    def _init_prometheus_metrics(self, registry: Optional["CollectorRegistry"]):
        """Initialize Prometheus metric collectors."""
        kwargs = {}
        if registry is not None:
            kwargs["registry"] = registry

        # PAD state gauges
        self._pad_valence = Gauge(
            f"{self.namespace}_pulse_pad_valence",
            "Current PAD valence (pleasure) [-1, 1]",
            **kwargs
        )
        self._pad_arousal = Gauge(
            f"{self.namespace}_pulse_pad_arousal",
            "Current PAD arousal [0, 1]",
            **kwargs
        )
        self._pad_dominance = Gauge(
            f"{self.namespace}_pulse_pad_dominance",
            "Current PAD dominance [0, 1]",
            **kwargs
        )
        self._pad_confidence = Gauge(
            f"{self.namespace}_pulse_confidence",
            "Confidence in PAD estimate [0, 1]",
            **kwargs
        )

        # Metacontrol gauges
        self._mc_temp_mult = Gauge(
            f"{self.namespace}_metacontrol_temperature_mult",
            "L3 Metacontrol temperature multiplier",
            **kwargs
        )
        self._mc_mem_mult = Gauge(
            f"{self.namespace}_metacontrol_memory_mult",
            "L3 Metacontrol memory write multiplier",
            **kwargs
        )
        self._mc_attention = Gauge(
            f"{self.namespace}_metacontrol_attention_gain",
            "L3 Metacontrol attention gain",
            **kwargs
        )
        self._mc_weight = Gauge(
            f"{self.namespace}_metacontrol_effective_weight",
            "L3 Metacontrol effective weight",
            **kwargs
        )

        # Workspace mode (as labeled gauge)
        self._workspace_mode = Gauge(
            f"{self.namespace}_workspace_mode",
            "Current workspace mode",
            ["mode"],
            **kwargs
        )

        # Effective values
        self._eff_temp = Gauge(
            f"{self.namespace}_effective_temperature",
            "Final effective LLM temperature",
            **kwargs
        )
        self._eff_mem_p = Gauge(
            f"{self.namespace}_effective_memory_p",
            "Final effective memory write probability",
            **kwargs
        )

        # Counters
        self._evt_flagged = Counter(
            f"{self.namespace}_pulse_evt_flagged_total",
            "Total EVT threshold exceeded events",
            **kwargs
        )
        self._turns_processed = Counter(
            f"{self.namespace}_turns_processed_total",
            "Total conversation turns processed",
            **kwargs
        )

        # Histogram for processing time
        self._processing_time = Histogram(
            f"{self.namespace}_process_turn_seconds",
            "Turn processing time in seconds",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            **kwargs
        )

        # Info metric
        self._info = Info(
            f"{self.namespace}_telemetry",
            "Telemetry module information",
            **kwargs
        )
        self._info.info({
            "version": "0.1.0",
            "namespace": self.namespace,
        })

    def record(self, snapshot: MetricSnapshot):
        """Record a metric snapshot and update Prometheus gauges."""
        # Store in history
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_history:
            self._snapshots = self._snapshots[-self._max_history:]

        # Track mode usage
        mode = snapshot.workspace_mode
        self._mode_counts[mode] = self._mode_counts.get(mode, 0) + 1

        # Update Prometheus metrics if enabled
        if self.enable_prometheus:
            self._update_prometheus(snapshot)

        logger.debug(f"Telemetry recorded: mode={mode}, temp_mult={snapshot.temperature_multiplier:.3f}")

    def _update_prometheus(self, snapshot: MetricSnapshot):
        """Update Prometheus metric values."""
        # PAD state
        self._pad_valence.set(snapshot.valence)
        self._pad_arousal.set(snapshot.arousal)
        self._pad_dominance.set(snapshot.dominance)
        self._pad_confidence.set(snapshot.confidence)

        # Metacontrol
        self._mc_temp_mult.set(snapshot.temperature_multiplier)
        self._mc_mem_mult.set(snapshot.memory_write_multiplier)
        self._mc_attention.set(snapshot.attention_gain)
        self._mc_weight.set(snapshot.effective_weight)

        # Workspace mode (set current mode to 1, others to 0)
        for mode in ["work", "relax", "creative", "support", "default"]:
            self._workspace_mode.labels(mode=mode).set(
                1.0 if mode == snapshot.workspace_mode else 0.0
            )

        # Effective values
        self._eff_temp.set(snapshot.effective_temperature)
        self._eff_mem_p.set(snapshot.effective_memory_p)

        # Counters
        if snapshot.evt_flagged:
            self._evt_flagged.inc()
        self._turns_processed.inc()

        # Processing time histogram (convert ms to seconds)
        self._processing_time.observe(snapshot.processing_time_ms / 1000.0)

    def record_from_turn(self, turn_result: Dict[str, Any]):
        """
        Record metrics from a ProcessedTurn result dict.

        Args:
            turn_result: Result from AraOrchestrator.process_turn().to_dict()
        """
        affect = turn_result.get("affect", {})
        pad = affect.get("pad", {})
        metacontrol = turn_result.get("metacontrol", {})

        snapshot = MetricSnapshot(
            valence=pad.get("pleasure", 0.0),
            arousal=pad.get("arousal", 0.0),
            dominance=pad.get("dominance", 0.5),
            confidence=affect.get("confidence", 1.0),
            workspace_mode=turn_result.get("workspace_mode", "default"),
            temperature_multiplier=metacontrol.get("temperature_multiplier", 1.0),
            memory_write_multiplier=metacontrol.get("memory_write_multiplier", 1.0),
            attention_gain=metacontrol.get("attention_gain", 1.0),
            effective_weight=metacontrol.get("effective_weight", 1.0),
            effective_temperature=turn_result.get("effective_temperature", 0.7),
            effective_memory_p=turn_result.get("effective_memory_p", 0.5),
            evt_flagged=affect.get("evt_flagged", False),
            p_anom=affect.get("p_anom", 0.0),
            processing_time_ms=turn_result.get("processing_time_ms", 0.0),
            turn_number=turn_result.get("turn_number", 0),
            session_id=turn_result.get("session_id"),
        )

        self.record(snapshot)

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metric snapshots."""
        return [s.to_dict() for s in self._snapshots[-limit:]]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._snapshots:
            return {"error": "No data collected"}

        recent = self._snapshots[-100:]

        return {
            "total_turns": len(self._snapshots),
            "mode_distribution": self._mode_counts.copy(),
            "avg_processing_ms": sum(s.processing_time_ms for s in recent) / len(recent),
            "avg_valence": sum(s.valence for s in recent) / len(recent),
            "avg_arousal": sum(s.arousal for s in recent) / len(recent),
            "avg_temp_mult": sum(s.temperature_multiplier for s in recent) / len(recent),
            "evt_flagged_count": sum(1 for s in recent if s.evt_flagged),
        }

    def export_prometheus(self) -> bytes:
        """Export metrics in Prometheus format."""
        if not self.enable_prometheus:
            return b"# Prometheus metrics not enabled\n"
        return generate_latest()


# Global telemetry instance
_telemetry: Optional[PulseTelemetry] = None


def get_telemetry() -> PulseTelemetry:
    """Get or create the global telemetry instance."""
    global _telemetry
    if _telemetry is None:
        _telemetry = PulseTelemetry()
    return _telemetry


def record_turn_metrics(turn_result: Dict[str, Any]):
    """Convenience function to record metrics from a turn result."""
    telemetry = get_telemetry()
    telemetry.record_from_turn(turn_result)


def get_telemetry_summary() -> Dict[str, Any]:
    """Get telemetry summary."""
    telemetry = get_telemetry()
    return telemetry.get_summary()


def create_telemetry_router():
    """
    Create FastAPI router with telemetry endpoints.

    Returns router with:
    - GET /metrics - Prometheus metrics export
    - GET /telemetry/history - Recent metric snapshots
    - GET /telemetry/summary - Summary statistics
    """
    try:
        from fastapi import APIRouter
        from fastapi.responses import Response
    except ImportError:
        logger.warning("FastAPI not available, telemetry router not created")
        return None

    router = APIRouter(tags=["telemetry"])
    telemetry = get_telemetry()

    @router.get("/metrics")
    async def prometheus_metrics():
        """Export Prometheus metrics."""
        content = telemetry.export_prometheus()
        return Response(
            content=content,
            media_type="text/plain; charset=utf-8",
        )

    @router.get("/telemetry/history")
    async def telemetry_history(limit: int = 100):
        """Get recent metric snapshots."""
        return telemetry.get_history(limit=limit)

    @router.get("/telemetry/summary")
    async def telemetry_summary():
        """Get telemetry summary statistics."""
        return telemetry.get_summary()

    return router


__all__ = [
    "MetricSnapshot",
    "PulseTelemetry",
    "get_telemetry",
    "record_turn_metrics",
    "get_telemetry_summary",
    "create_telemetry_router",
    "PROMETHEUS_AVAILABLE",
]
