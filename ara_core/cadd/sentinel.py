#!/usr/bin/env python3
"""
CADD Sentinel - Collective Association Drift Detection
=======================================================

Cathedral's immune system for detecting emergent bias.
Maps to T_s_bias monitoring in swarm gate.

Theory:
    High T_s (stable topology) + label shift = "emergent sociological bias"
    Detected via entropy monitoring of association matrices.

Association Matrix A_t^i(c, s):
    - Agent i's association strength
    - Between concept c and signal s
    - At timestep t

Drift Detection:
    - H_influence < 1.2 bits → monoculture alert
    - Δ|A| > threshold → drift alert
    - Concentration ratio > 0.8 → dominance alert

Interventions:
    - INJECT_DIVERSITY: Spawn orthogonal agents
    - QUARANTINE: Isolate high-drift agents
    - REWEIGHT: Reduce high-influence agents
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict


class DriftType(str, Enum):
    """Types of detected drift."""
    MONOCULTURE = "monoculture"           # H_influence < 1.2 bits
    ASSOCIATION_SHIFT = "association_shift"  # Δ|A| > threshold
    DOMINANCE = "dominance"               # Single agent too influential
    BIAS_GEOMETRY = "bias_geometry"       # Systematic pattern in associations
    CONCEPT_COLLAPSE = "concept_collapse" # Concepts becoming indistinguishable
    NONE = "none"


@dataclass
class DriftAlert:
    """An alert for detected drift."""
    alert_id: str
    drift_type: DriftType
    severity: float           # 0.0 to 1.0
    affected_agents: List[str]
    affected_concepts: List[str]
    message: str
    recommended_action: str
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.alert_id,
            "type": self.drift_type.value,
            "severity": self.severity,
            "agents": self.affected_agents,
            "concepts": self.affected_concepts,
            "message": self.message,
            "action": self.recommended_action,
            "timestamp": self.timestamp,
        }


class AssociationMatrix:
    """
    Association matrix A_t^i(c, s) for an agent.

    Tracks concept-signal associations over time.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.associations: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.history: List[Tuple[float, float]] = []  # (timestamp, entropy)
        self.total_updates: int = 0
        self.last_update: float = time.time()

    def update(self, concept: str, signal: str, strength: float,
               alpha: float = 0.1):
        """Update association strength with EMA."""
        current = self.associations[concept][signal]
        self.associations[concept][signal] = current * (1 - alpha) + strength * alpha
        self.total_updates += 1
        self.last_update = time.time()

    def get_association(self, concept: str, signal: str) -> float:
        """Get current association strength."""
        return self.associations.get(concept, {}).get(signal, 0.0)

    def get_concept_distribution(self, concept: str) -> Dict[str, float]:
        """Get distribution of signals for a concept."""
        return dict(self.associations.get(concept, {}))

    def entropy(self) -> float:
        """
        Calculate entropy of associations.

        High entropy = diverse associations
        Low entropy = concentrated/biased associations
        """
        all_values = []
        for concept_signals in self.associations.values():
            all_values.extend(concept_signals.values())

        if not all_values:
            return 0.0

        # Normalize to distribution
        total = sum(abs(v) for v in all_values)
        if total < 1e-10:
            return 0.0

        probs = [abs(v) / total for v in all_values if abs(v) > 1e-10]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        return float(entropy)

    def concentration_ratio(self) -> float:
        """
        Calculate concentration ratio (Herfindahl-like).

        High value = one association dominates
        Low value = distributed associations
        """
        all_values = []
        for concept_signals in self.associations.values():
            all_values.extend(abs(v) for v in concept_signals.values())

        if not all_values:
            return 0.0

        total = sum(all_values)
        if total < 1e-10:
            return 0.0

        max_val = max(all_values)
        return float(max_val / total)

    def drift_from(self, other: 'AssociationMatrix') -> float:
        """Calculate drift distance from another matrix."""
        all_concepts = set(self.associations.keys()) | set(other.associations.keys())
        all_signals = set()

        for d in list(self.associations.values()) + list(other.associations.values()):
            all_signals.update(d.keys())

        if not all_concepts or not all_signals:
            return 0.0

        drift = 0.0
        count = 0

        for concept in all_concepts:
            for signal in all_signals:
                v1 = self.get_association(concept, signal)
                v2 = other.get_association(concept, signal)
                drift += abs(v1 - v2)
                count += 1

        return drift / max(count, 1)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Take a snapshot of current associations."""
        return {
            concept: dict(signals)
            for concept, signals in self.associations.items()
        }


@dataclass
class AgentProfile:
    """Profile for tracking an agent's behavior over time."""
    agent_id: str
    matrix: AssociationMatrix = field(default_factory=lambda: AssociationMatrix(""))
    influence: float = 0.0           # Contribution to collective
    divergence: float = 0.0          # Distance from collective mean
    drift_velocity: float = 0.0      # Rate of change
    flagged: bool = False
    quarantined: bool = False

    def __post_init__(self):
        if self.matrix.agent_id == "":
            self.matrix = AssociationMatrix(self.agent_id)


@dataclass
class SentinelConfig:
    """Configuration for CADD sentinel."""
    # Entropy thresholds
    h_influence_min: float = 1.2      # Minimum acceptable entropy
    h_influence_target: float = 1.8   # Target entropy (Cathedral gate)

    # Drift thresholds
    drift_threshold: float = 0.1      # Alert if drift > this
    dominance_threshold: float = 0.3  # Alert if single agent > this

    # Timing
    window_size: int = 100            # Steps for drift calculation
    tick_interval_s: float = 60.0     # How often to check

    # Interventions
    diversity_spawn_count: int = 3    # Morons to spawn on monoculture


class CADDSentinel:
    """
    Collective Association Drift Detection Sentinel.

    Cathedral's immune system for swarm bias monitoring.
    """

    def __init__(self, config: SentinelConfig = None):
        self.config = config or SentinelConfig()

        # Agent tracking
        self.agents: Dict[str, AgentProfile] = {}
        self.collective_matrix = AssociationMatrix("collective")

        # History
        self.entropy_history: List[Tuple[float, float]] = []  # (time, entropy)
        self.alerts: List[DriftAlert] = []
        self.pending_alerts: List[DriftAlert] = []

        # State
        self.last_tick: float = time.time()
        self.total_ticks: int = 0

    def register_agent(self, agent_id: str):
        """Register an agent for monitoring."""
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentProfile(agent_id)

    def update_association(self, agent_id: str, concept: str,
                          signal: str, strength: float):
        """Update an agent's association."""
        if agent_id not in self.agents:
            self.register_agent(agent_id)

        self.agents[agent_id].matrix.update(concept, signal, strength)

        # Also update collective
        self.collective_matrix.update(concept, signal, strength)

    def tick(self) -> List[DriftAlert]:
        """
        Run drift detection tick.

        Returns list of new alerts.
        """
        now = time.time()
        elapsed = now - self.last_tick
        self.last_tick = now
        self.total_ticks += 1

        new_alerts = []

        # 1. Calculate collective entropy (H_influence)
        h_influence = self._calculate_h_influence()
        self.entropy_history.append((now, h_influence))

        # Keep history bounded
        if len(self.entropy_history) > 1000:
            self.entropy_history = self.entropy_history[-1000:]

        # 2. Check for monoculture
        if h_influence < self.config.h_influence_min:
            alert = self._create_monoculture_alert(h_influence)
            new_alerts.append(alert)

        # 3. Check for dominance
        dominance_alert = self._check_dominance()
        if dominance_alert:
            new_alerts.append(dominance_alert)

        # 4. Check for drift
        drift_alert = self._check_drift()
        if drift_alert:
            new_alerts.append(drift_alert)

        # 5. Check for concept collapse
        collapse_alert = self._check_concept_collapse()
        if collapse_alert:
            new_alerts.append(collapse_alert)

        # Store alerts
        self.pending_alerts.extend(new_alerts)
        self.alerts.extend(new_alerts)

        # Keep alerts bounded
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]

        return new_alerts

    def _calculate_h_influence(self) -> float:
        """Calculate influence entropy across agents."""
        if not self.agents:
            return 0.0

        # Calculate each agent's influence (contribution to collective)
        influences = []
        total_entropy = 0.0

        for agent_id, profile in self.agents.items():
            agent_entropy = profile.matrix.entropy()
            profile.influence = agent_entropy
            influences.append(agent_entropy)
            total_entropy += agent_entropy

        if total_entropy < 1e-10:
            return 0.0

        # Normalize to distribution
        probs = [i / total_entropy for i in influences if i > 1e-10]

        if not probs:
            return 0.0

        # Calculate entropy of influence distribution
        h_influence = -sum(p * np.log2(p) for p in probs if p > 0)
        return float(h_influence)

    def _create_monoculture_alert(self, h_influence: float) -> DriftAlert:
        """Create alert for monoculture detection."""
        # Find high-influence agents
        sorted_agents = sorted(
            self.agents.items(),
            key=lambda x: x[1].influence,
            reverse=True
        )
        top_agents = [a[0] for a in sorted_agents[:5]]

        return DriftAlert(
            alert_id=f"mono_{int(time.time())}",
            drift_type=DriftType.MONOCULTURE,
            severity=1.0 - h_influence / self.config.h_influence_min,
            affected_agents=top_agents,
            affected_concepts=[],
            message=f"H_influence={h_influence:.2f} < {self.config.h_influence_min} - monoculture detected",
            recommended_action=f"INJECT_DIVERSITY: Spawn {self.config.diversity_spawn_count}x morons with orthogonal priors",
        )

    def _check_dominance(self) -> Optional[DriftAlert]:
        """Check if single agent dominates the collective."""
        if not self.agents:
            return None

        total_influence = sum(p.influence for p in self.agents.values())
        if total_influence < 1e-10:
            return None

        for agent_id, profile in self.agents.items():
            ratio = profile.influence / total_influence
            if ratio > self.config.dominance_threshold:
                return DriftAlert(
                    alert_id=f"dom_{int(time.time())}_{agent_id[:8]}",
                    drift_type=DriftType.DOMINANCE,
                    severity=ratio,
                    affected_agents=[agent_id],
                    affected_concepts=[],
                    message=f"Agent {agent_id} has {ratio:.1%} of collective influence",
                    recommended_action="REWEIGHT: Reduce agent's learning rate or contributions",
                )

        return None

    def _check_drift(self) -> Optional[DriftAlert]:
        """Check for collective drift over time."""
        if len(self.entropy_history) < 2:
            return None

        # Calculate drift velocity
        recent = self.entropy_history[-10:]
        if len(recent) < 2:
            return None

        times = [t for t, _ in recent]
        entropies = [e for _, e in recent]

        # Linear regression for trend
        n = len(times)
        sum_t = sum(times)
        sum_e = sum(entropies)
        sum_te = sum(t * e for t, e in recent)
        sum_t2 = sum(t * t for t in times)

        denom = n * sum_t2 - sum_t ** 2
        if abs(denom) < 1e-10:
            return None

        slope = (n * sum_te - sum_t * sum_e) / denom

        # Significant negative slope = entropy declining = drift toward monoculture
        if slope < -self.config.drift_threshold:
            return DriftAlert(
                alert_id=f"drift_{int(time.time())}",
                drift_type=DriftType.ASSOCIATION_SHIFT,
                severity=min(1.0, abs(slope) * 10),
                affected_agents=list(self.agents.keys())[:10],
                affected_concepts=[],
                message=f"Entropy declining at rate {slope:.4f}/s - drift toward monoculture",
                recommended_action="PAUSE_CONSOLIDATION: Temporarily halt learning",
            )

        return None

    def _check_concept_collapse(self) -> Optional[DriftAlert]:
        """Check if concepts are becoming indistinguishable."""
        concepts = list(self.collective_matrix.associations.keys())
        if len(concepts) < 2:
            return None

        # Check pairwise concept similarity
        collapsed_pairs = []

        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                dist_i = self.collective_matrix.get_concept_distribution(concepts[i])
                dist_j = self.collective_matrix.get_concept_distribution(concepts[j])

                # Cosine similarity
                if dist_i and dist_j:
                    all_signals = set(dist_i.keys()) | set(dist_j.keys())
                    vec_i = [dist_i.get(s, 0) for s in all_signals]
                    vec_j = [dist_j.get(s, 0) for s in all_signals]

                    dot = sum(a * b for a, b in zip(vec_i, vec_j))
                    norm_i = np.sqrt(sum(a * a for a in vec_i))
                    norm_j = np.sqrt(sum(b * b for b in vec_j))

                    if norm_i > 0 and norm_j > 0:
                        sim = dot / (norm_i * norm_j)
                        if sim > 0.95:  # Very similar
                            collapsed_pairs.append((concepts[i], concepts[j]))

        if collapsed_pairs:
            affected = set()
            for c1, c2 in collapsed_pairs:
                affected.add(c1)
                affected.add(c2)

            return DriftAlert(
                alert_id=f"collapse_{int(time.time())}",
                drift_type=DriftType.CONCEPT_COLLAPSE,
                severity=len(collapsed_pairs) / (len(concepts) * (len(concepts) - 1) / 2),
                affected_agents=[],
                affected_concepts=list(affected)[:10],
                message=f"{len(collapsed_pairs)} concept pairs becoming indistinguishable",
                recommended_action="MANUAL_REVIEW: Check concept definitions and signal mapping",
            )

        return None

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self.pending_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
        self.pending_alerts = [a for a in self.pending_alerts if not a.acknowledged]

    def quarantine_agent(self, agent_id: str):
        """Quarantine a problematic agent."""
        if agent_id in self.agents:
            self.agents[agent_id].quarantined = True

    def release_agent(self, agent_id: str):
        """Release a quarantined agent."""
        if agent_id in self.agents:
            self.agents[agent_id].quarantined = False

    def get_active_agents(self) -> List[str]:
        """Get non-quarantined agents."""
        return [
            agent_id for agent_id, profile in self.agents.items()
            if not profile.quarantined
        ]

    def health_status(self) -> Dict[str, Any]:
        """Get health status for Cathedral monitoring."""
        h_influence = self._calculate_h_influence()

        return {
            "h_influence": h_influence,
            "h_influence_ok": h_influence >= self.config.h_influence_min,
            "h_influence_target": h_influence >= self.config.h_influence_target,
            "n_agents": len(self.agents),
            "n_quarantined": sum(1 for p in self.agents.values() if p.quarantined),
            "n_pending_alerts": len(self.pending_alerts),
            "total_alerts": len(self.alerts),
            "total_ticks": self.total_ticks,
        }

    def status_string(self) -> str:
        """Get status string."""
        health = self.health_status()
        if health["h_influence_target"]:
            return f"🟢 CADD: H_influence={health['h_influence']:.2f} ≥ {self.config.h_influence_target}"
        elif health["h_influence_ok"]:
            return f"🟡 CADD: H_influence={health['h_influence']:.2f} (below target)"
        else:
            return f"🔴 CADD: H_influence={health['h_influence']:.2f} < {self.config.h_influence_min} - MONOCULTURE"


# =============================================================================
# SINGLETON AND CONVENIENCE
# =============================================================================

_sentinel: Optional[CADDSentinel] = None


def get_sentinel(n_agents: int = 1000) -> CADDSentinel:
    """Get the global CADD sentinel instance."""
    global _sentinel
    if _sentinel is None:
        _sentinel = CADDSentinel()
    return _sentinel


def sentinel_tick() -> List[DriftAlert]:
    """Run a sentinel tick."""
    return get_sentinel().tick()


def sentinel_status() -> str:
    """Get sentinel status string."""
    return get_sentinel().status_string()


def sentinel_alerts() -> List[DriftAlert]:
    """Get pending alerts."""
    return get_sentinel().pending_alerts
