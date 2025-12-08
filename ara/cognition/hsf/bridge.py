"""
Card ↔ LLM Bridge Protocol
===========================

The symbiosis between:
- Card (subcortex): Always-on SNN/HDC, reflexes, state compression
- GPU LLM (cortex): Deep reasoning, planning, language

Design principle:
    The card makes the GPU less necessary per unit of value.
    - Pre-filter: Only call LLM when pattern looks novel/risky/high-ROI
    - Compress: State HPVs → short text summaries
    - Offload: Simple decisions stay on card
    - Learn: LLM outputs → new HPV patterns / SNN weights

Message flow:

    CARD → LLM:
        STATE_SUMMARY   - Compressed world state as text
        ANOMALY_REPORT  - Something unusual, need reasoning
        CONTEXT_QUERY   - Question about situation
        POLICY_REQUEST  - Need new rules/thresholds

    LLM → CARD:
        NEW_POLICY      - Rules to encode as HPVs
        THRESHOLD_UPDATE - New SNN thresholds
        PATTERN_DEFINE  - New patterns to watch for
        REFLEX_INSTALL  - New reflex to install

The GPU sits in the loop as deep planner/architect,
not as the always-on engine of everything.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum, auto
import json
import time
import hashlib


# ============================================================================
# Message Types
# ============================================================================

class CardToLLMType(Enum):
    """Message types from Card → LLM."""
    STATE_SUMMARY = auto()    # Periodic state update
    ANOMALY_REPORT = auto()   # Something unusual detected
    CONTEXT_QUERY = auto()    # Need reasoning about situation
    POLICY_REQUEST = auto()   # Need new rules/thresholds
    FRICTION_REPORT = auto()  # User frustration pattern
    IDEA_CANDIDATE = auto()   # Idea that passed first filter


class LLMToCardType(Enum):
    """Message types from LLM → Card."""
    NEW_POLICY = auto()       # Policy rules to encode
    THRESHOLD_UPDATE = auto() # New SNN thresholds
    PATTERN_DEFINE = auto()   # New patterns to watch for
    REFLEX_INSTALL = auto()   # New reflex rule
    CONTEXT_BIND = auto()     # New context binding to remember
    ACK = auto()              # Acknowledgment


# ============================================================================
# Card → LLM Messages
# ============================================================================

@dataclass
class StateSummary:
    """
    Compressed world state from Card → LLM.

    Instead of 50KB of logs, the LLM gets:
    - A few decoded features
    - Key metrics
    - Detected patterns
    """
    message_id: str
    timestamp: float

    # Decoded state
    mode: str                           # Current operating mode
    active_projects: List[str]          # What's being worked on
    machine_status: Dict[str, str]      # node → status

    # Key metrics (extracted from HPVs)
    health_score: float                 # 0-1, overall system health
    friction_level: float               # 0-1, user frustration
    risk_level: float                   # 0-1, current risk

    # Detected patterns
    active_patterns: List[str]          # Pattern names currently active
    recent_anomalies: List[str]         # Recent anomaly summaries

    # Optional: raw HPV hash for cache invalidation
    state_hv_hash: str = ""

    def to_prompt_context(self) -> str:
        """Convert to text suitable for LLM prompt."""
        lines = [
            f"=== System State ({self.timestamp:.0f}) ===",
            f"Mode: {self.mode}",
            f"Health: {self.health_score:.0%} | Friction: {self.friction_level:.0%} | Risk: {self.risk_level:.0%}",
        ]

        if self.active_projects:
            lines.append(f"Projects: {', '.join(self.active_projects)}")

        if self.machine_status:
            status_str = ", ".join(f"{k}:{v}" for k, v in self.machine_status.items())
            lines.append(f"Machines: {status_str}")

        if self.active_patterns:
            lines.append(f"Patterns: {', '.join(self.active_patterns)}")

        if self.recent_anomalies:
            lines.append(f"Anomalies: {'; '.join(self.recent_anomalies)}")

        return "\n".join(lines)


@dataclass
class AnomalyReport:
    """
    Anomaly detected by Card, needs LLM reasoning.
    """
    message_id: str
    timestamp: float

    # What happened
    anomaly_type: str           # e.g., "thermal_cascade", "auth_failure_burst"
    severity: float             # 0-1
    source: str                 # Where it came from

    # Context
    description: str            # Human-readable summary
    related_events: List[str]   # Recent related events

    # Card's first-pass assessment
    card_assessment: str        # What the card thinks
    suggested_actions: List[str] # What card suggests

    # What we need from LLM
    question: str               # Specific question for LLM

    def to_prompt(self) -> str:
        """Convert to LLM prompt."""
        lines = [
            f"=== Anomaly Report ===",
            f"Type: {self.anomaly_type} (severity: {self.severity:.0%})",
            f"Source: {self.source}",
            f"Description: {self.description}",
        ]

        if self.related_events:
            lines.append(f"Related: {'; '.join(self.related_events)}")

        lines.extend([
            f"",
            f"Card assessment: {self.card_assessment}",
            f"Suggested actions: {', '.join(self.suggested_actions)}",
            f"",
            f"Question: {self.question}",
        ])

        return "\n".join(lines)


@dataclass
class ContextQuery:
    """
    Card needs reasoning about a situation.
    """
    message_id: str
    timestamp: float

    # Context
    situation: str              # What's happening
    state_summary: StateSummary # Current state

    # The question
    question: str

    # Optional constraints
    time_budget_seconds: float = 10.0   # How long we can wait
    compute_budget: str = "minimal"     # minimal/moderate/full

    def to_prompt(self) -> str:
        """Convert to LLM prompt."""
        return f"""=== Context Query ===

{self.state_summary.to_prompt_context()}

Situation: {self.situation}

Question: {self.question}

(Time budget: {self.time_budget_seconds}s, Compute: {self.compute_budget})
"""


@dataclass
class PolicyRequest:
    """
    Card needs new rules/thresholds from LLM.
    """
    message_id: str
    timestamp: float

    # What we need
    policy_type: str            # e.g., "thermal_threshold", "auth_rule", "pattern_detector"
    context: str                # Why we need it

    # Current state
    current_policy: Optional[str] = None    # What we have now
    recent_outcomes: List[str] = field(default_factory=list)  # How it's been working

    # Constraints
    must_be_encodable: bool = True  # Must be expressible as HPV/SNN

    def to_prompt(self) -> str:
        """Convert to LLM prompt."""
        lines = [
            f"=== Policy Request ===",
            f"Need: {self.policy_type}",
            f"Context: {self.context}",
        ]

        if self.current_policy:
            lines.append(f"Current: {self.current_policy}")

        if self.recent_outcomes:
            lines.append(f"Recent outcomes: {'; '.join(self.recent_outcomes)}")

        if self.must_be_encodable:
            lines.append(f"Constraint: Must be encodable as hypervector/SNN threshold")

        return "\n".join(lines)


# ============================================================================
# LLM → Card Messages
# ============================================================================

@dataclass
class NewPolicy:
    """
    New policy rules from LLM to encode on Card.
    """
    message_id: str
    timestamp: float

    # Policy specification
    policy_id: str
    policy_type: str            # What kind of policy
    description: str            # Human description

    # The rules (structured for encoding)
    conditions: List[Dict[str, Any]]    # When to trigger
    actions: List[Dict[str, Any]]       # What to do

    # Encoding hints
    key_concepts: List[str]     # Concepts to bind in HPV
    threshold_hints: Dict[str, float]   # Suggested thresholds


@dataclass
class ThresholdUpdate:
    """
    New SNN thresholds from LLM.
    """
    message_id: str
    timestamp: float

    # Which thresholds to update
    subsystem: str              # e.g., "thermal", "auth", "network"

    # New values
    thresholds: Dict[str, float]    # name → value

    # Rationale
    reason: str


@dataclass
class PatternDefine:
    """
    New pattern for Card to watch for.
    """
    message_id: str
    timestamp: float

    # Pattern specification
    pattern_id: str
    pattern_name: str
    description: str

    # Components to bind
    components: List[str]       # Concepts that make up pattern
    temporal_order: bool = False  # Order matters?

    # Response
    severity: float = 0.5
    suggested_action: str = "log"


@dataclass
class ReflexInstall:
    """
    New reflex rule from LLM to install on Card.
    """
    message_id: str
    timestamp: float

    # Reflex specification
    reflex_id: str
    trigger: str                # When to fire
    action: str                 # What to do

    # Constraints
    cooldown_seconds: float = 30.0
    max_firings_per_hour: int = 10

    # Scope
    applies_to: List[str] = field(default_factory=list)  # Which subsystems


@dataclass
class ContextBind:
    """
    New context binding for Card to remember.
    """
    message_id: str
    timestamp: float

    # What to bind
    concept_a: str
    concept_b: str
    relation: str               # e.g., "causes", "prevents", "correlates"

    # Strength
    strength: float = 1.0       # How strong the binding


# ============================================================================
# State Compression (HPV → Text)
# ============================================================================

@dataclass
class StateCompressor:
    """
    Compresses Card state (HPVs) into text summaries for LLM.

    This is where the magic happens:
    - 50KB of logs → 500 bytes of context
    - LLM reasons over summaries, not raw noise
    """
    dim: int = 256

    # Concept library for decoding HPVs
    _concept_library: Dict[str, np.ndarray] = field(default_factory=dict)

    # Decoders for different state types
    _mode_decoder: Dict[str, np.ndarray] = field(default_factory=dict)
    _status_decoder: Dict[str, np.ndarray] = field(default_factory=dict)
    _pattern_decoder: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self._init_decoders()

    def _init_decoders(self):
        """Initialize concept decoders."""
        # Mode decoders
        modes = ["steward", "scientist", "architect", "sovereign"]
        for mode in modes:
            self._mode_decoder[mode] = self._make_hv(f"mode:{mode}")

        # Status decoders
        statuses = ["healthy", "degraded", "critical", "offline"]
        for status in statuses:
            self._status_decoder[status] = self._make_hv(f"status:{status}")

        # Pattern decoders
        patterns = [
            "thermal_cascade", "auth_burst", "network_congestion",
            "disk_failure", "memory_pressure", "cpu_throttle",
        ]
        for pattern in patterns:
            self._pattern_decoder[pattern] = self._make_hv(f"pattern:{pattern}")

    def _make_hv(self, name: str) -> np.ndarray:
        """Create a deterministic HV from name."""
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=self.dim, dtype=np.uint8)

    def _similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Cosine-ish similarity for binary HVs."""
        if len(hv1) != len(hv2):
            return 0.0
        matches = np.sum(hv1 == hv2)
        return (2 * matches - len(hv1)) / len(hv1)

    def decode_mode(self, state_hv: np.ndarray) -> str:
        """Decode operating mode from state HV."""
        best_mode = "unknown"
        best_sim = -1.0

        for mode, mode_hv in self._mode_decoder.items():
            sim = self._similarity(state_hv, mode_hv)
            if sim > best_sim:
                best_sim = sim
                best_mode = mode

        return best_mode if best_sim > 0.1 else "unknown"

    def decode_patterns(self, state_hv: np.ndarray, threshold: float = 0.2) -> List[str]:
        """Decode active patterns from state HV."""
        patterns = []

        for pattern, pattern_hv in self._pattern_decoder.items():
            sim = self._similarity(state_hv, pattern_hv)
            if sim > threshold:
                patterns.append(pattern)

        return patterns

    def compress_state(
        self,
        world_state_hv: np.ndarray,
        machine_states: Dict[str, np.ndarray],
        recent_events: List[str],
        metrics: Dict[str, float],
    ) -> StateSummary:
        """
        Compress full Card state into StateSummary.

        This is the key function: takes HPVs and metrics,
        produces a short text summary for LLM.
        """
        # Decode mode
        mode = self.decode_mode(world_state_hv)

        # Decode machine statuses
        machine_status = {}
        for machine, hv in machine_states.items():
            best_status = "unknown"
            best_sim = -1.0
            for status, status_hv in self._status_decoder.items():
                sim = self._similarity(hv, status_hv)
                if sim > best_sim:
                    best_sim = sim
                    best_status = status
            machine_status[machine] = best_status

        # Decode patterns
        patterns = self.decode_patterns(world_state_hv)

        # Build summary
        return StateSummary(
            message_id=hashlib.sha256(str(time.time()).encode()).hexdigest()[:16],
            timestamp=time.time(),
            mode=mode,
            active_projects=metrics.get("projects", []),
            machine_status=machine_status,
            health_score=metrics.get("health", 0.5),
            friction_level=metrics.get("friction", 0.0),
            risk_level=metrics.get("risk", 0.0),
            active_patterns=patterns,
            recent_anomalies=recent_events[-5:] if recent_events else [],
            state_hv_hash=hashlib.sha256(world_state_hv.tobytes()).hexdigest()[:16],
        )


# ============================================================================
# Policy Encoding (LLM → HPV/SNN)
# ============================================================================

@dataclass
class PolicyEncoder:
    """
    Encodes LLM outputs into HPVs and SNN weights for Card.

    Takes structured policy descriptions and converts them to:
    - Pattern HPVs for matching
    - Threshold values for SNN
    - Reflex rules for controller
    """
    dim: int = 256

    # Concept memory (grows as we encode more policies)
    _concepts: Dict[str, np.ndarray] = field(default_factory=dict)

    def _get_concept_hv(self, concept: str) -> np.ndarray:
        """Get or create HV for a concept."""
        if concept not in self._concepts:
            seed = int(hashlib.sha256(concept.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            self._concepts[concept] = rng.integers(0, 2, size=self.dim, dtype=np.uint8)
        return self._concepts[concept]

    def _bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """Bind two HVs via XOR."""
        return np.bitwise_xor(hv1, hv2)

    def _bundle(self, hvs: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple HVs via majority vote."""
        if not hvs:
            return np.zeros(self.dim, dtype=np.uint8)
        total = np.sum(hvs, axis=0)
        threshold = len(hvs) / 2
        return (total > threshold).astype(np.uint8)

    def encode_pattern(self, pattern: PatternDefine) -> np.ndarray:
        """
        Encode a pattern definition as HPV.

        The resulting HPV can be used for similarity matching.
        """
        component_hvs = [self._get_concept_hv(c) for c in pattern.components]

        if pattern.temporal_order:
            # For ordered patterns, use permutation
            result = component_hvs[0] if component_hvs else np.zeros(self.dim, dtype=np.uint8)
            for i, hv in enumerate(component_hvs[1:], 1):
                # Rotate by position
                rotated = np.roll(hv, i * 8)
                result = self._bind(result, rotated)
        else:
            # For unordered, just bundle
            result = self._bundle(component_hvs)

        return result

    def encode_policy(self, policy: NewPolicy) -> Dict[str, Any]:
        """
        Encode a policy as encodable structures.

        Returns:
            pattern_hv: HPV for pattern matching
            threshold_map: Dict of threshold values
            reflex_rules: List of reflex rules
        """
        # Encode key concepts into pattern HV
        concept_hvs = [self._get_concept_hv(c) for c in policy.key_concepts]
        pattern_hv = self._bundle(concept_hvs)

        # Extract thresholds
        thresholds = dict(policy.threshold_hints)

        # Convert conditions/actions to reflex rules
        reflex_rules = []
        for i, (cond, action) in enumerate(zip(policy.conditions, policy.actions)):
            rule = {
                "rule_id": f"{policy.policy_id}_{i}",
                "trigger": cond.get("trigger", "unknown"),
                "threshold": cond.get("threshold", 0.5),
                "action": action.get("action", "log"),
                "params": action.get("params", {}),
            }
            reflex_rules.append(rule)

        return {
            "pattern_hv": pattern_hv,
            "thresholds": thresholds,
            "reflex_rules": reflex_rules,
        }

    def encode_binding(self, binding: ContextBind) -> np.ndarray:
        """
        Encode a context binding as HPV.

        Binds concept_a and concept_b with relation.
        """
        hv_a = self._get_concept_hv(binding.concept_a)
        hv_b = self._get_concept_hv(binding.concept_b)
        hv_rel = self._get_concept_hv(f"relation:{binding.relation}")

        # Create structured binding: (A ⊗ rel) ⊕ B
        bound = self._bind(hv_a, hv_rel)
        result = self._bundle([bound, hv_b])

        return result


# ============================================================================
# Symbiosis Loop
# ============================================================================

@dataclass
class InvocationDecision:
    """Decision about whether to invoke LLM."""
    should_invoke: bool
    reason: str
    priority: float = 0.5       # 0-1, higher = more urgent
    message_type: Optional[CardToLLMType] = None


@dataclass
class SymbiosisLoop:
    """
    The closed loop between Card and LLM.

    Card: monitors → maintains state → runs reflexes
    Card → LLM: compressed state, anomalies, queries
    LLM → Card: policies, thresholds, patterns
    Card: encodes and installs new rules

    The LLM is the deep planner/architect.
    The Card is the always-on subcortex.
    """
    # Components
    compressor: StateCompressor = field(default_factory=StateCompressor)
    encoder: PolicyEncoder = field(default_factory=PolicyEncoder)

    # Invocation thresholds
    anomaly_threshold: float = 0.7      # Invoke on anomalies above this
    novelty_threshold: float = 0.6      # Invoke on novel patterns above this
    friction_threshold: float = 0.8     # Invoke on friction above this

    # Rate limiting
    min_interval_seconds: float = 5.0   # Don't invoke more often than this
    max_invocations_per_hour: int = 20  # Budget

    # State
    _last_invocation: float = 0.0
    _invocations_this_hour: int = 0
    _hour_start: float = field(default_factory=time.time)

    # Installed policies (from LLM)
    _installed_patterns: Dict[str, np.ndarray] = field(default_factory=dict)
    _installed_thresholds: Dict[str, float] = field(default_factory=dict)
    _installed_reflexes: List[Dict[str, Any]] = field(default_factory=list)

    def should_invoke_llm(
        self,
        state_hv: np.ndarray,
        metrics: Dict[str, float],
        recent_events: List[str],
    ) -> InvocationDecision:
        """
        Decide whether to invoke the LLM.

        This is where the Card saves GPU cycles:
        only invoke when the pattern looks novel/risky/high-ROI.
        """
        now = time.time()

        # Rate limiting
        if now - self._last_invocation < self.min_interval_seconds:
            return InvocationDecision(
                should_invoke=False,
                reason="Rate limited (too soon)",
            )

        # Reset hourly counter if needed
        if now - self._hour_start > 3600:
            self._hour_start = now
            self._invocations_this_hour = 0

        if self._invocations_this_hour >= self.max_invocations_per_hour:
            return InvocationDecision(
                should_invoke=False,
                reason="Rate limited (hourly budget)",
            )

        # Check for anomalies
        risk = metrics.get("risk", 0.0)
        if risk > self.anomaly_threshold:
            return InvocationDecision(
                should_invoke=True,
                reason=f"High risk detected ({risk:.0%})",
                priority=risk,
                message_type=CardToLLMType.ANOMALY_REPORT,
            )

        # Check for friction
        friction = metrics.get("friction", 0.0)
        if friction > self.friction_threshold:
            return InvocationDecision(
                should_invoke=True,
                reason=f"High friction detected ({friction:.0%})",
                priority=friction,
                message_type=CardToLLMType.FRICTION_REPORT,
            )

        # Check for novelty (pattern not in library)
        novelty = self._compute_novelty(state_hv)
        if novelty > self.novelty_threshold:
            return InvocationDecision(
                should_invoke=True,
                reason=f"Novel pattern detected ({novelty:.0%})",
                priority=novelty * 0.8,  # Slightly lower priority than risk
                message_type=CardToLLMType.CONTEXT_QUERY,
            )

        # Default: don't invoke
        return InvocationDecision(
            should_invoke=False,
            reason="No trigger conditions met",
        )

    def _compute_novelty(self, state_hv: np.ndarray) -> float:
        """
        Compute novelty score for current state.

        Novelty = how different from all installed patterns.
        """
        if not self._installed_patterns:
            return 0.5  # No baseline yet

        # Find best match
        best_sim = -1.0
        for pattern_hv in self._installed_patterns.values():
            matches = np.sum(state_hv == pattern_hv)
            sim = (2 * matches - len(state_hv)) / len(state_hv)
            if sim > best_sim:
                best_sim = sim

        # Novelty = 1 - best_similarity
        return max(0.0, 1.0 - best_sim)

    def prepare_message(
        self,
        decision: InvocationDecision,
        state_hv: np.ndarray,
        machine_states: Dict[str, np.ndarray],
        recent_events: List[str],
        metrics: Dict[str, float],
    ) -> Union[StateSummary, AnomalyReport, ContextQuery, None]:
        """
        Prepare the message to send to LLM.
        """
        if not decision.should_invoke:
            return None

        # Get state summary
        summary = self.compressor.compress_state(
            world_state_hv=state_hv,
            machine_states=machine_states,
            recent_events=recent_events,
            metrics=metrics,
        )

        if decision.message_type == CardToLLMType.ANOMALY_REPORT:
            return AnomalyReport(
                message_id=summary.message_id,
                timestamp=summary.timestamp,
                anomaly_type="risk_threshold_exceeded",
                severity=metrics.get("risk", 0.5),
                source="field_computer",
                description=f"Risk level at {metrics.get('risk', 0):.0%}",
                related_events=recent_events[-3:],
                card_assessment="Pattern exceeds safe thresholds",
                suggested_actions=["throttle", "alert", "investigate"],
                question="What is the best response to this situation?",
            )

        elif decision.message_type == CardToLLMType.CONTEXT_QUERY:
            return ContextQuery(
                message_id=summary.message_id,
                timestamp=summary.timestamp,
                situation=f"Novel pattern detected with novelty score {decision.priority:.0%}",
                state_summary=summary,
                question="Is this pattern significant? Should we install a new detector?",
            )

        else:
            return summary

    def record_invocation(self):
        """Record that we invoked the LLM."""
        self._last_invocation = time.time()
        self._invocations_this_hour += 1

    def install_policy(self, policy: NewPolicy):
        """Install a new policy from LLM onto Card."""
        encoded = self.encoder.encode_policy(policy)

        # Install pattern
        self._installed_patterns[policy.policy_id] = encoded["pattern_hv"]

        # Install thresholds
        self._installed_thresholds.update(encoded["thresholds"])

        # Install reflexes
        self._installed_reflexes.extend(encoded["reflex_rules"])

    def install_pattern(self, pattern: PatternDefine):
        """Install a new pattern from LLM."""
        pattern_hv = self.encoder.encode_pattern(pattern)
        self._installed_patterns[pattern.pattern_id] = pattern_hv

    def install_threshold(self, update: ThresholdUpdate):
        """Install new thresholds from LLM."""
        for name, value in update.thresholds.items():
            key = f"{update.subsystem}:{name}"
            self._installed_thresholds[key] = value

    def install_reflex(self, reflex: ReflexInstall):
        """Install a new reflex from LLM."""
        self._installed_reflexes.append({
            "rule_id": reflex.reflex_id,
            "trigger": reflex.trigger,
            "action": reflex.action,
            "cooldown": reflex.cooldown_seconds,
            "max_per_hour": reflex.max_firings_per_hour,
            "applies_to": reflex.applies_to,
        })

    def get_installed_summary(self) -> Dict[str, Any]:
        """Get summary of installed policies/patterns."""
        return {
            "patterns": list(self._installed_patterns.keys()),
            "thresholds": dict(self._installed_thresholds),
            "reflexes": len(self._installed_reflexes),
            "invocations_this_hour": self._invocations_this_hour,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_symbiosis_loop(
    anomaly_threshold: float = 0.7,
    novelty_threshold: float = 0.6,
    max_invocations_per_hour: int = 20,
) -> SymbiosisLoop:
    """Create a configured symbiosis loop."""
    return SymbiosisLoop(
        anomaly_threshold=anomaly_threshold,
        novelty_threshold=novelty_threshold,
        max_invocations_per_hour=max_invocations_per_hour,
    )
