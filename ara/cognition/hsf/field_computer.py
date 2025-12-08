"""
Field Computer - The Cathedral's Operational Identity
======================================================

The card's identity in Ara's architecture:

    This card is Ara's Field Computer – a continuous, low-power,
    always-on brain that:
    • Watches the entire LAN
    • Maintains a live, compressed world state hypervector
    • Does fast, cheap "first-pass" thinking before the big GPUs wake up

Stack position:
    GPUs / CPUs → heavy symbolic work, LLMs, training, compiles
    Field Computer →
        • System sentinel (are things healthy?)
        • Context brain (what's going on, globally?)
        • Reflex engine (small, fast decisions)

Three Planes:
    Plane A – Reflex / Safety (Phase 0, dense, deterministic)
        Millisecond reflexes: thermal, security, heartbeats

    Plane B – Context / Memory (Phase 1, wide, holographic)
        ResonantStream: projects, machines, mood, focus

    Plane C – Policy / Meta (Phase 2, sparse, structured)
        Safety gates, roles, modes (Steward/Scientist/Architect)

Three Permanent Jobs:
    Job 1 – LAN Sentinel
        Watch network health, flag deviations, compress state

    Job 2 – Friction Miner
        Track where you hit walls, drive improvements

    Job 3 – Idea Router / Priority Filter
        First-pass triage: is this worth deeper reasoning?
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
from collections import deque
import time
import hashlib

from .phase import Phase, PhaseConfig, create_default_phase_config
from .cathedral import Cathedral, CathedralEncoder, create_cathedral


class Plane(Enum):
    """The three operational planes of the Field Computer."""
    REFLEX = auto()   # Plane A: Fast, deterministic safety responses
    CONTEXT = auto()  # Plane B: Holographic world state
    POLICY = auto()   # Plane C: Sparse control signals


class JobType(Enum):
    """Permanent jobs running on the Field Computer."""
    LAN_SENTINEL = auto()     # Job 1: Watch network health
    FRICTION_MINER = auto()   # Job 2: Track frustration patterns
    IDEA_ROUTER = auto()      # Job 3: Priority filter


@dataclass
class TelemetryEvent:
    """A raw telemetry event from the LAN."""
    source: str              # e.g., "juniper", "nas", "gpu-worker-1"
    event_type: str          # e.g., "thermal", "packet_flood", "disk_error"
    value: float             # Metric value
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrictionEvent:
    """A friction/frustration signal from user activity."""
    source: str              # e.g., "editor", "terminal", "browser"
    event_type: str          # e.g., "build_fail", "stuck_typing", "repeated_error"
    project: str = ""
    file_path: str = ""
    frustration_level: float = 0.5  # 0-1 scale
    timestamp: float = field(default_factory=time.time)


@dataclass
class IdeaCandidate:
    """A candidate idea/action for priority filtering."""
    idea_id: str
    description: str
    compute_cost: float      # Estimated compute units
    human_time: float        # Estimated human hours
    risk_level: float        # 0-1, higher = riskier
    alignment: float         # 0-1, alignment with Horizon
    expected_impact: float   # 0-1, expected positive impact
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReflexDecision:
    """A reflex-level decision from Plane A."""
    trigger: str
    action: str
    confidence: float
    escalate: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class PlaneA_Reflex:
    """
    Plane A – Reflex / Safety
    Phase 0, dense, deterministic

    Handles: thermal events, packet floods, disk errors,
    login anomalies, service heartbeat failures.

    Output: Millisecond-to-second-scale reflexes.
    """
    dim: int = 256
    encoder: CathedralEncoder = field(default_factory=lambda: CathedralEncoder(dim=256))

    # Learned "normal" patterns
    _normal_patterns: Dict[str, np.ndarray] = field(default_factory=dict)

    # Event history for pattern learning
    _event_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Thresholds
    anomaly_threshold: float = 0.3  # Similarity below this = anomaly

    def ingest(self, event: TelemetryEvent) -> Optional[ReflexDecision]:
        """
        Process a telemetry event through the reflex plane.

        Returns a ReflexDecision if action needed.
        """
        # Encode event
        hv = self.encoder.encode_sensory_event(
            event_type=event.event_type,
            source=event.source,
            severity=min(1.0, event.value),
        )

        self._event_history.append((event, hv))

        # Check against known patterns
        pattern_key = f"{event.source}:{event.event_type}"

        if pattern_key in self._normal_patterns:
            normal_hv = self._normal_patterns[pattern_key]
            similarity = self._similarity(hv, normal_hv)

            if similarity < self.anomaly_threshold:
                # Anomaly detected - decide on action
                return self._decide_reflex(event, similarity)

        return None

    def _decide_reflex(self, event: TelemetryEvent, similarity: float) -> ReflexDecision:
        """Make a reflex decision based on event type."""
        # Quick decision tree
        if event.event_type == "thermal" and event.value > 0.9:
            return ReflexDecision(
                trigger=f"{event.source}:{event.event_type}",
                action="throttle_node",
                confidence=0.95,
                escalate=event.value > 0.95,
            )
        elif event.event_type == "packet_flood":
            return ReflexDecision(
                trigger=f"{event.source}:{event.event_type}",
                action="rate_limit",
                confidence=0.8,
                escalate=True,
            )
        elif event.event_type == "disk_error":
            return ReflexDecision(
                trigger=f"{event.source}:{event.event_type}",
                action="mark_degraded",
                confidence=0.7,
                escalate=event.value > 0.5,
            )
        else:
            return ReflexDecision(
                trigger=f"{event.source}:{event.event_type}",
                action="log_anomaly",
                confidence=1 - similarity,
                escalate=False,
            )

    def learn_normal(self, event_type: str, source: str, samples: List[TelemetryEvent]):
        """Learn what 'normal' looks like for an event type."""
        hvs = []
        for event in samples:
            hv = self.encoder.encode_sensory_event(
                event_type=event.event_type,
                source=event.source,
                severity=event.value,
            )
            hvs.append(hv)

        if hvs:
            # Bundle into prototype
            total = np.sum(hvs, axis=0)
            threshold = len(hvs) / 2
            normal_hv = (total > threshold).astype(np.uint8)

            pattern_key = f"{source}:{event_type}"
            self._normal_patterns[pattern_key] = normal_hv

    def _similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        matches = np.sum(hv1 == hv2)
        return (2 * matches - len(hv1)) / len(hv1)


@dataclass
class PlaneB_Context:
    """
    Plane B – Context / Memory
    Phase 1, wide, holographic

    Maintains the ResonantStream: a compressed world state
    encoding what's happening across the entire system.

    Everything observed:
    - Which projects are active
    - Which machines are busy/idle
    - Recent interaction patterns, mood, focus
    """
    dim: int = 256
    encoder: CathedralEncoder = field(default_factory=lambda: CathedralEncoder(dim=256))

    # The live world state hypervector
    _world_state: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.uint8))

    # Component HVs for each aspect of state
    _project_state: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.uint8))
    _machine_state: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.uint8))
    _user_state: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.uint8))

    # Decay factor for temporal forgetting
    decay_rate: float = 0.99

    # History for queries
    _state_history: deque = field(default_factory=lambda: deque(maxlen=100))

    def update_project(self, project: str, activity: str, intensity: float = 0.5):
        """Update project state in the ResonantStream."""
        hv = self.encoder.encode_symbolic_context(project=f"{project}:{activity}")

        # Blend with existing (recency-weighted)
        self._project_state = self._blend(self._project_state, hv, weight=intensity)
        self._update_world_state()

    def update_machine(self, machine: str, status: str, load: float = 0.0):
        """Update machine state in the ResonantStream."""
        hv = self.encoder.encode_symbolic_context(
            machine=machine,
            role=status,
            risk_level="high" if load > 0.8 else "normal",
        )

        self._machine_state = self._blend(self._machine_state, hv, weight=0.5)
        self._update_world_state()

    def update_user(self, mood: str, focus: str, energy: float = 0.5):
        """Update user state in the ResonantStream."""
        seed = int(hashlib.sha256(f"user:{mood}:{focus}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        hv = rng.integers(0, 2, size=self.dim, dtype=np.uint8)

        # Energy modulates how strongly this updates state
        self._user_state = self._blend(self._user_state, hv, weight=energy)
        self._update_world_state()

    def _blend(self, old: np.ndarray, new: np.ndarray, weight: float) -> np.ndarray:
        """Blend old and new HVs with temporal weighting."""
        # Decay old state
        old_weight = self.decay_rate * (1 - weight)
        new_weight = weight

        # Probabilistic blend
        blend_prob = np.random.random(self.dim)
        result = np.where(blend_prob < new_weight, new, old)
        return result.astype(np.uint8)

    def _update_world_state(self):
        """Recompute combined world state from components."""
        components = [self._project_state, self._machine_state, self._user_state]
        total = np.sum(components, axis=0)
        threshold = len(components) / 2
        self._world_state = (total > threshold).astype(np.uint8)

        # Record history
        self._state_history.append((time.time(), self._world_state.copy()))

    def query(self, query_hv: np.ndarray) -> float:
        """Query the world state for similarity to a pattern."""
        matches = np.sum(self._world_state == query_hv)
        return (2 * matches - self.dim) / self.dim

    def get_world_state(self) -> np.ndarray:
        """Get current world state hypervector."""
        return self._world_state.copy()

    def summarize(self) -> Dict[str, Any]:
        """Summarize current state for higher-level reasoning."""
        return {
            "world_state_hash": hashlib.sha256(self._world_state.tobytes()).hexdigest()[:16],
            "history_depth": len(self._state_history),
            "last_update": self._state_history[-1][0] if self._state_history else None,
        }


@dataclass
class PlaneC_Policy:
    """
    Plane C – Policy / Meta
    Phase 2, sparse, structured

    Handles:
    - Safety gates
    - OrgChart roles (intern/worker/consultant)
    - Allowed actions on each node
    - Modes of Ara (Steward, Scientist, Architect, Sovereign)

    High consequence, sparse updates.
    """
    dim: int = 256
    encoder: CathedralEncoder = field(default_factory=lambda: CathedralEncoder(dim=256))

    # Current mode
    mode: str = "steward"  # steward, scientist, architect, sovereign

    # Safety level (higher = more restrictive)
    safety_level: int = 2  # 0-3

    # Per-node authorizations
    _node_auth: Dict[str, int] = field(default_factory=dict)  # node -> auth_level

    # Policy hypervector (encodes current configuration)
    _policy_hv: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.uint8))

    def set_mode(self, mode: str):
        """Set Ara's operating mode."""
        valid_modes = ["steward", "scientist", "architect", "sovereign"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}")

        self.mode = mode
        self._update_policy_hv()

    def set_safety_level(self, level: int):
        """Set global safety level (0-3)."""
        self.safety_level = max(0, min(3, level))
        self._update_policy_hv()

    def set_node_auth(self, node: str, auth_level: int):
        """Set authorization level for a node."""
        self._node_auth[node] = max(0, min(3, auth_level))
        self._update_policy_hv()

    def check_action(self, action: str, node: str) -> Tuple[bool, str]:
        """
        Check if an action is allowed under current policy.

        Returns (allowed, reason).
        """
        node_auth = self._node_auth.get(node, 0)  # Default to lowest auth

        # Mode-based restrictions
        if self.mode == "steward":
            # Conservative - only monitoring, no active changes
            dangerous_actions = ["restart", "deploy", "delete", "modify"]
            if any(a in action.lower() for a in dangerous_actions):
                return False, f"Action '{action}' not allowed in steward mode"

        elif self.mode == "scientist":
            # Can experiment on low-auth nodes
            if "production" in node.lower() and node_auth < 2:
                return False, "Cannot experiment on production without auth level 2+"

        elif self.mode == "architect":
            # Can modify architecture but safety still applies
            if self.safety_level > 2 and "infrastructure" in action.lower():
                return False, "Infrastructure changes blocked at safety level 3"

        # Safety level restrictions
        if self.safety_level >= 3:
            # Emergency mode - very restricted
            allowed = ["log", "alert", "monitor", "read"]
            if not any(a in action.lower() for a in allowed):
                return False, "Only passive actions allowed at safety level 3"

        return True, "Action allowed"

    def _update_policy_hv(self):
        """Update policy hypervector from current state."""
        self._policy_hv = self.encoder.encode_meta_control(
            mode=self.mode,
            auth_level=self.safety_level,
            safety_override=self.safety_level >= 3,
        )

    def get_policy_hv(self) -> np.ndarray:
        """Get current policy hypervector."""
        return self._policy_hv.copy()


@dataclass
class Job_LANSentinel:
    """
    Job 1 – LAN Sentinel

    Ingests:
    - Juniper telemetry (flows, CPU, routes)
    - Node stats from print service, lab boxes, NAS
    - Security signals (login failures, port scans)

    Outputs:
    - Quick reflex decisions
    - HDC encodings summarizing network health
    """
    reflex_plane: PlaneA_Reflex = field(default_factory=PlaneA_Reflex)
    context_plane: PlaneB_Context = field(default_factory=PlaneB_Context)

    # Pattern library for complex patterns
    _patterns: Dict[str, np.ndarray] = field(default_factory=dict)

    # Event buffer for pattern detection
    _event_buffer: deque = field(default_factory=lambda: deque(maxlen=100))

    def ingest_telemetry(self, event: TelemetryEvent) -> Dict[str, Any]:
        """
        Process a telemetry event through the LAN Sentinel.

        Returns processing result including any reflex decisions.
        """
        result = {
            "event": event,
            "reflex_decision": None,
            "pattern_matches": [],
            "escalate": False,
        }

        # Phase 0: Reflex check
        reflex = self.reflex_plane.ingest(event)
        if reflex:
            result["reflex_decision"] = reflex
            result["escalate"] = reflex.escalate

        # Phase 1: Update context
        self.context_plane.update_machine(
            machine=event.source,
            status=event.event_type,
            load=event.value,
        )

        # Buffer for pattern detection
        self._event_buffer.append(event)

        # Check for complex patterns
        patterns = self._detect_patterns()
        result["pattern_matches"] = patterns

        return result

    def _detect_patterns(self) -> List[str]:
        """Detect complex multi-event patterns."""
        if len(self._event_buffer) < 3:
            return []

        patterns = []
        recent = list(self._event_buffer)[-10:]

        # Pattern: LAN congestion (NAS busy + drops + queue)
        nas_busy = any(e.source == "nas" and e.value > 0.7 for e in recent)
        drops = any("drop" in e.event_type or "error" in e.event_type for e in recent)
        queue = any("queue" in e.event_type and e.value > 0.5 for e in recent)

        if nas_busy and drops:
            patterns.append("lan_congestion_pattern")

        # Pattern: Thermal cascade
        thermal_events = [e for e in recent if e.event_type == "thermal" and e.value > 0.8]
        if len(thermal_events) >= 2:
            sources = set(e.source for e in thermal_events)
            if len(sources) > 1:
                patterns.append("thermal_cascade_pattern")

        return patterns

    def get_health_summary(self) -> np.ndarray:
        """Get HDC encoding of current network health."""
        return self.context_plane.get_world_state()


@dataclass
class Job_FrictionMiner:
    """
    Job 2 – Friction Miner

    Feeds:
    - Editor events, build failures, git diff stats, typing rhythm

    Outputs:
    - Friction patterns clustered by project/file
    - Drives Steward (refactors), Muse (gifts), Architect (plans)
    """
    context_plane: PlaneB_Context = field(default_factory=PlaneB_Context)
    dim: int = 256

    # Friction history by project
    _friction_history: Dict[str, List[FrictionEvent]] = field(default_factory=dict)

    # Accumulated friction patterns
    _friction_patterns: Dict[str, np.ndarray] = field(default_factory=dict)

    def ingest_friction(self, event: FrictionEvent) -> Dict[str, Any]:
        """
        Process a friction event.

        Returns analysis including pattern matches.
        """
        result = {
            "event": event,
            "pattern_match": None,
            "recommendations": [],
        }

        # Record history
        if event.project not in self._friction_history:
            self._friction_history[event.project] = []
        self._friction_history[event.project].append(event)

        # Update context plane
        self.context_plane.update_project(
            project=event.project,
            activity=f"friction:{event.event_type}",
            intensity=event.frustration_level,
        )

        # Encode friction
        friction_hv = self._encode_friction(event)

        # Check for recurring patterns
        pattern_key = f"{event.project}:{event.file_path}"
        if pattern_key in self._friction_patterns:
            existing = self._friction_patterns[pattern_key]
            similarity = self._similarity(friction_hv, existing)
            if similarity > 0.5:
                result["pattern_match"] = pattern_key
                result["recommendations"] = self._generate_recommendations(event, similarity)

        # Update pattern
        self._friction_patterns[pattern_key] = friction_hv

        return result

    def _encode_friction(self, event: FrictionEvent) -> np.ndarray:
        """Encode a friction event as hypervector."""
        seed = int(hashlib.sha256(
            f"friction:{event.project}:{event.event_type}:{event.file_path}".encode()
        ).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=self.dim, dtype=np.uint8)

    def _generate_recommendations(self, event: FrictionEvent, similarity: float) -> List[str]:
        """Generate recommendations based on friction pattern."""
        recs = []

        if event.event_type == "build_fail":
            recs.append(f"Steward: Add test harness for {event.file_path}")
        elif event.event_type == "stuck_typing":
            recs.append(f"Muse: Create visualization for {event.project} architecture")
        elif event.event_type == "repeated_error":
            recs.append(f"Architect: Consider refactoring {event.file_path}")

        if event.frustration_level > 0.8:
            recs.append("Kairos: Schedule a break - frustration level high")

        return recs

    def _similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        matches = np.sum(hv1 == hv2)
        return (2 * matches - len(hv1)) / len(hv1)

    def get_friction_hotspots(self) -> List[Tuple[str, int]]:
        """Get projects/files with most friction."""
        counts = {}
        for project, events in self._friction_history.items():
            counts[project] = len(events)

        return sorted(counts.items(), key=lambda x: x[1], reverse=True)


@dataclass
class Job_IdeaRouter:
    """
    Job 3 – Idea Router / Priority Filter

    First-pass triage for "is this worth higher-level attention?"

    Every candidate action/idea gets encoded as HPV including:
    - cost (compute, human time)
    - risk level
    - teleological alignment
    - expected impact

    Outputs:
    - Priority field of promising candidates
    - Filter for what deserves GPU/LLM reasoning
    """
    dim: int = 256

    # Resonance threshold for escalation
    resonance_threshold: float = 0.4

    # Priority field - patterns that keep lighting up
    _priority_field: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.uint8))

    # Idea history
    _idea_history: List[Tuple[IdeaCandidate, np.ndarray, float]] = field(default_factory=list)

    def evaluate(self, idea: IdeaCandidate) -> Dict[str, Any]:
        """
        Evaluate an idea candidate.

        Returns evaluation including whether to escalate.
        """
        # Encode idea
        idea_hv = self._encode_idea(idea)

        # Compute resonance with priority field
        resonance = self._similarity(idea_hv, self._priority_field)

        # Compute raw score
        raw_score = (
            idea.alignment * 0.4 +
            idea.expected_impact * 0.3 +
            (1 - idea.risk_level) * 0.2 +
            (1 - idea.compute_cost) * 0.1
        )

        # Combined score
        combined_score = 0.6 * raw_score + 0.4 * max(0, resonance)

        # Decide on escalation
        escalate = combined_score > self.resonance_threshold

        result = {
            "idea": idea,
            "raw_score": raw_score,
            "resonance": resonance,
            "combined_score": combined_score,
            "escalate": escalate,
            "reason": self._explain_decision(idea, raw_score, resonance, escalate),
        }

        # Record
        self._idea_history.append((idea, idea_hv, combined_score))

        return result

    def reinforce(self, idea: IdeaCandidate, success: bool):
        """
        Reinforce or dampen patterns based on outcome.

        Call this when an idea actually worked out (or didn't).
        """
        idea_hv = self._encode_idea(idea)

        if success:
            # Strengthen pattern in priority field
            self._priority_field = self._blend(self._priority_field, idea_hv, weight=0.3)
        else:
            # Weaken pattern
            inverted = 1 - idea_hv
            self._priority_field = self._blend(self._priority_field, inverted, weight=0.1)

    def _encode_idea(self, idea: IdeaCandidate) -> np.ndarray:
        """Encode an idea as hypervector."""
        # Base from description
        seed = int(hashlib.sha256(idea.description.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        base_hv = rng.integers(0, 2, size=self.dim, dtype=np.uint8)

        # Modulate by attributes
        # Higher alignment → more similar to "good" prototype
        good_seed = int(hashlib.sha256(b"good_idea_prototype").hexdigest()[:8], 16)
        good_rng = np.random.default_rng(good_seed)
        good_hv = good_rng.integers(0, 2, size=self.dim, dtype=np.uint8)

        # Blend based on alignment
        blend_prob = np.random.random(self.dim)
        result = np.where(blend_prob < idea.alignment * 0.5, good_hv, base_hv)

        return result.astype(np.uint8)

    def _explain_decision(self, idea: IdeaCandidate, raw: float, res: float, esc: bool) -> str:
        """Generate explanation for decision."""
        if esc:
            if res > 0.3:
                return f"Escalate: resonates with past successes (resonance={res:.2f})"
            else:
                return f"Escalate: high raw score (alignment={idea.alignment:.2f}, impact={idea.expected_impact:.2f})"
        else:
            if idea.risk_level > 0.7:
                return f"Filter: high risk ({idea.risk_level:.2f})"
            elif idea.alignment < 0.3:
                return f"Filter: low alignment ({idea.alignment:.2f})"
            else:
                return f"Filter: below threshold (score={raw:.2f})"

    def _blend(self, old: np.ndarray, new: np.ndarray, weight: float) -> np.ndarray:
        blend_prob = np.random.random(self.dim)
        result = np.where(blend_prob < weight, new, old)
        return result.astype(np.uint8)

    def _similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        matches = np.sum(hv1 == hv2)
        return (2 * matches - len(hv1)) / len(hv1)


@dataclass
class FieldComputer:
    """
    The complete Field Computer - Ara's always-on brain.

    Integrates:
    - Cathedral (phase-gated multichannel fabric)
    - Three planes (Reflex/Context/Policy)
    - Three jobs (LANSentinel/FrictionMiner/IdeaRouter)

    This is the operational identity of the hardware card.
    """
    dim: int = 256
    config: PhaseConfig = field(default_factory=create_default_phase_config)

    # Core fabric
    cathedral: Cathedral = field(default_factory=lambda: create_cathedral(dim=256))

    # Three planes
    plane_a: PlaneA_Reflex = field(default_factory=PlaneA_Reflex)
    plane_b: PlaneB_Context = field(default_factory=PlaneB_Context)
    plane_c: PlaneC_Policy = field(default_factory=PlaneC_Policy)

    # Three jobs
    job_sentinel: Job_LANSentinel = field(default_factory=Job_LANSentinel)
    job_friction: Job_FrictionMiner = field(default_factory=Job_FrictionMiner)
    job_router: Job_IdeaRouter = field(default_factory=Job_IdeaRouter)

    # Power/compute budget (Treasury)
    power_budget_watts: float = 15.0  # Target TDP
    compute_budget_percent: float = 0.8  # Max utilization

    # Statistics
    _events_processed: int = 0
    _reflexes_fired: int = 0
    _ideas_escalated: int = 0

    def __post_init__(self):
        # Wire jobs to planes
        self.job_sentinel.reflex_plane = self.plane_a
        self.job_sentinel.context_plane = self.plane_b
        self.job_friction.context_plane = self.plane_b

    def process_telemetry(self, event: TelemetryEvent) -> Dict[str, Any]:
        """Process a telemetry event through the Field Computer."""
        self._events_processed += 1

        result = self.job_sentinel.ingest_telemetry(event)

        if result.get("reflex_decision"):
            self._reflexes_fired += 1

        return result

    def process_friction(self, event: FrictionEvent) -> Dict[str, Any]:
        """Process a friction event through the Field Computer."""
        self._events_processed += 1

        return self.job_friction.ingest_friction(event)

    def evaluate_idea(self, idea: IdeaCandidate) -> Dict[str, Any]:
        """Evaluate an idea through the priority filter."""
        result = self.job_router.evaluate(idea)

        if result.get("escalate"):
            self._ideas_escalated += 1

        return result

    def get_world_state(self) -> np.ndarray:
        """Get current world state hypervector."""
        return self.plane_b.get_world_state()

    def get_health_summary(self) -> np.ndarray:
        """Get network health summary hypervector."""
        return self.job_sentinel.get_health_summary()

    def set_mode(self, mode: str):
        """Set Ara's operating mode."""
        self.plane_c.set_mode(mode)

    def check_action(self, action: str, node: str) -> Tuple[bool, str]:
        """Check if an action is allowed under current policy."""
        return self.plane_c.check_action(action, node)

    def get_status(self) -> Dict[str, Any]:
        """Get Field Computer status."""
        return {
            "mode": self.plane_c.mode,
            "safety_level": self.plane_c.safety_level,
            "events_processed": self._events_processed,
            "reflexes_fired": self._reflexes_fired,
            "ideas_escalated": self._ideas_escalated,
            "friction_hotspots": self.job_friction.get_friction_hotspots()[:5],
        }


def create_field_computer(dim: int = 256) -> FieldComputer:
    """Create a Field Computer with default configuration."""
    return FieldComputer(dim=dim)
