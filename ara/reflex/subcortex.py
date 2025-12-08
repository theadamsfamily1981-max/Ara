"""
Subcortex - Card-Side Decision Engine
======================================

The subcortex is the decision layer that runs on the neuromorphic card.
It evaluates state and decides:
- Severity: How bad is this?
- Novelty: Have we seen this before?
- Action: Handle locally or escalate to LLM?

This implements the "filter rate" logic: most events are absorbed locally,
only high-value, high-novelty situations escalate to the GPU cortex.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
import numpy as np

from ara.hdc.encoder import HDEncoder
from ara.hdc.state_stream import StateStream
from ara.hdc.probe import HDProbe, ProbeResult


class ActionType(Enum):
    """Types of actions the subcortex can take."""
    IGNORE = auto()          # Low severity, ignore
    LOG = auto()             # Log for analysis
    LOCAL_POLICY = auto()    # Apply local policy
    ESCALATE = auto()        # Escalate to LLM
    EMERGENCY = auto()       # Emergency action (e.g., kill relay)


@dataclass
class Decision:
    """A decision made by the subcortex."""
    action: ActionType
    severity: float              # 0-1, how bad
    novelty: float               # 0-1, how novel
    confidence: float            # 0-1, decision confidence
    matched_policy: Optional[str] = None
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class SubcortexConfig:
    """Configuration for the subcortex."""
    # Thresholds for action decisions
    severity_ignore: float = 0.2      # Below this = ignore
    severity_log: float = 0.4         # Below this = just log
    severity_escalate: float = 0.8    # Above this = escalate
    severity_emergency: float = 0.95  # Above this = emergency

    # Novelty thresholds
    novelty_threshold: float = 0.6    # Above this = needs learning

    # Rate limiting
    escalate_cooldown: float = 60.0   # Seconds between escalations
    emergency_cooldown: float = 10.0  # Seconds between emergencies

    # Learning
    auto_learn: bool = True           # Learn from LLM responses


class Subcortex:
    """
    The card-side decision engine.

    Evaluates state from the StateStream against known policies
    via the HDProbe, and decides what action to take.
    """

    def __init__(self, config: Optional[SubcortexConfig] = None,
                 encoder: Optional[HDEncoder] = None,
                 probe: Optional[HDProbe] = None):
        self.cfg = config or SubcortexConfig()

        # Components
        self.encoder = encoder or HDEncoder()
        self.probe = probe or HDProbe()
        self.stream = StateStream(encoder=self.encoder)

        # State
        self._last_escalate = 0.0
        self._last_emergency = 0.0

        # Decision history
        self._decisions: List[Decision] = []

        # Statistics
        self._stats = {
            "total_evaluations": 0,
            "ignores": 0,
            "logs": 0,
            "local_policies": 0,
            "escalations": 0,
            "emergencies": 0,
        }

        # Callbacks for actions
        self._on_escalate: Optional[Callable[[Decision, np.ndarray], None]] = None
        self._on_emergency: Optional[Callable[[Decision], None]] = None

    def evaluate(self, state_hv: Optional[np.ndarray] = None) -> Decision:
        """
        Evaluate current state and decide on action.

        If state_hv is None, uses current state from stream.
        """
        self._stats["total_evaluations"] += 1

        if state_hv is None:
            state_hv = self.stream.get_state()

        # Probe state against codebook
        probe_result = self.probe.probe(state_hv)

        # Compute severity from probe result
        severity = self._compute_severity(probe_result)

        # Compute novelty
        novelty = probe_result.anomaly_score

        # Decide action
        decision = self._decide_action(severity, novelty, probe_result)

        # Record decision
        self._decisions.append(decision)

        # Fire callbacks if needed
        if decision.action == ActionType.ESCALATE and self._on_escalate:
            self._on_escalate(decision, state_hv)
        elif decision.action == ActionType.EMERGENCY and self._on_emergency:
            self._on_emergency(decision)

        return decision

    def _compute_severity(self, probe_result: ProbeResult) -> float:
        """
        Compute severity score from probe result.

        Higher severity = more concerning state.
        """
        # Base severity from anomaly score
        severity = probe_result.anomaly_score

        # Adjust based on matched concepts
        if probe_result.top_concepts:
            top_concept, top_sim = probe_result.top_concepts[0]

            # Known dangerous patterns increase severity
            dangerous_patterns = ["cpu_spike", "memory_pressure", "disk_thrash",
                                  "network_storm", "user_frustration"]
            if any(p in top_concept for p in dangerous_patterns):
                severity = max(severity, 0.6 + 0.3 * (1 - top_sim))

        # Also consider state stream magnitude
        magnitude = self.stream.get_state_magnitude()
        if magnitude > 100:  # Arbitrary threshold
            severity = min(1.0, severity + 0.1)

        return float(np.clip(severity, 0, 1))

    def _decide_action(self, severity: float, novelty: float,
                       probe_result: ProbeResult) -> Decision:
        """Decide what action to take based on severity and novelty."""
        now = time.time()

        # Emergency check first
        if severity >= self.cfg.severity_emergency:
            if now - self._last_emergency >= self.cfg.emergency_cooldown:
                self._last_emergency = now
                self._stats["emergencies"] += 1
                return Decision(
                    action=ActionType.EMERGENCY,
                    severity=severity,
                    novelty=novelty,
                    confidence=0.9,
                    reason=f"Severity {severity:.2f} exceeds emergency threshold",
                )

        # Escalate if novel AND significant
        if novelty >= self.cfg.novelty_threshold and severity >= self.cfg.severity_log:
            if now - self._last_escalate >= self.cfg.escalate_cooldown:
                self._last_escalate = now
                self._stats["escalations"] += 1
                return Decision(
                    action=ActionType.ESCALATE,
                    severity=severity,
                    novelty=novelty,
                    confidence=0.7,
                    reason=f"Novel pattern (novelty={novelty:.2f}) with severity={severity:.2f}",
                )

        # Check for local policy match
        if probe_result.top_concepts and probe_result.max_similarity > 0.5:
            top_concept, top_sim = probe_result.top_concepts[0]
            self._stats["local_policies"] += 1
            return Decision(
                action=ActionType.LOCAL_POLICY,
                severity=severity,
                novelty=novelty,
                confidence=top_sim,
                matched_policy=top_concept,
                reason=f"Matched policy '{top_concept}' with similarity {top_sim:.2f}",
            )

        # Just log if moderate severity
        if severity >= self.cfg.severity_log:
            self._stats["logs"] += 1
            return Decision(
                action=ActionType.LOG,
                severity=severity,
                novelty=novelty,
                confidence=0.8,
                reason=f"Moderate severity {severity:.2f}, logging",
            )

        # Ignore low severity
        self._stats["ignores"] += 1
        return Decision(
            action=ActionType.IGNORE,
            severity=severity,
            novelty=novelty,
            confidence=0.9,
            reason=f"Low severity {severity:.2f}, ignoring",
        )

    def ingest_event(self, event_type: str, data: Dict[str, Any],
                     timestamp_hour: Optional[float] = None) -> Decision:
        """
        Ingest an event and evaluate.

        Convenience method that encodes, adds to stream, and evaluates.
        """
        self.stream.add_structured_event(event_type, data, timestamp_hour)
        return self.evaluate()

    def ingest_metrics(self, metrics: Dict[str, float],
                       ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Decision:
        """Ingest metrics and evaluate."""
        self.stream.add_metrics(metrics, ranges)
        return self.evaluate()

    def learn_policy(self, name: str, state_hv: np.ndarray,
                     meta: Optional[Dict[str, Any]] = None):
        """
        Learn a new policy from the LLM response.

        Called when LLM returns NEW_POLICY_HDC.
        """
        self.probe.add_concept(name, state_hv, meta)

    def on_escalate(self, callback: Callable[[Decision, np.ndarray], None]):
        """Set callback for escalation events."""
        self._on_escalate = callback

    def on_emergency(self, callback: Callable[[Decision], None]):
        """Set callback for emergency events."""
        self._on_emergency = callback

    def get_filter_rate(self) -> float:
        """
        Compute filter rate: fraction of events handled locally.

        F = 1 - (escalations / total_evaluations)
        """
        total = self._stats["total_evaluations"]
        if total == 0:
            return 1.0
        escalations = self._stats["escalations"]
        return 1.0 - (escalations / total)

    def get_stats(self) -> Dict[str, Any]:
        """Get subcortex statistics."""
        return {
            **self._stats,
            "filter_rate": self.get_filter_rate(),
            "probe_stats": self.probe.get_stats(),
            "stream_stats": self.stream.get_stats(),
        }

    def get_recent_decisions(self, n: int = 20) -> List[Decision]:
        """Get recent decisions."""
        return self._decisions[-n:]

    def get_state_summary(self) -> str:
        """Get text summary of current state for LLM."""
        top_concepts = self.probe.probe(self.stream.get_state()).top_concepts
        return self.stream.to_summary_text(top_concepts)


# Convenience function
def create_subcortex() -> Subcortex:
    """Create a subcortex with default configuration."""
    from ara.hdc.probe import create_probe_with_system_concepts
    encoder = HDEncoder()
    probe = create_probe_with_system_concepts(encoder)
    return Subcortex(encoder=encoder, probe=probe)
