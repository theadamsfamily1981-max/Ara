"""
Cognitive Synthesis: Integration of L7/L8/GUF into Unified Decision Loop

This module wires together:
- L7 Temporal Topology (Ṡ structural rate, predictive alerts)
- L8 Semantic Verification (truth certification, PGU consistency)
- GUF (global utility, self vs world prioritization)

Into a single coherent cognitive loop that:
1. Predicts structural instability before it manifests (L7)
2. Verifies outputs for truth consistency (L8)
3. Decides when to self-improve vs serve the world (GUF)

This is Phase 5: The system that argues with itself and knows when to heal.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
import time


# ============================================================
# System Mode: What the system is doing right now
# ============================================================

class SystemMode(str, Enum):
    """Overall system operating mode."""
    RECOVERY = "recovery"           # Critical state, focus on survival
    SELF_IMPROVEMENT = "self_improvement"  # Below G_target, fixing itself
    BALANCED = "balanced"           # Near threshold, mixed focus
    SERVING = "serving"             # Healthy, serving external requests
    PROTECTIVE = "protective"       # Ṡ alert, preemptive protection


class DecisionType(str, Enum):
    """Types of decisions the synthesizer makes."""
    STRUCTURAL_RESPONSE = "structural_response"  # React to Ṡ
    VERIFICATION_ROUTING = "verification_routing"  # L8 criticality routing
    FOCUS_ALLOCATION = "focus_allocation"  # GUF self vs world
    AEPO_TRIGGER = "aepo_trigger"  # Proactive AEPO proposal
    MODE_TRANSITION = "mode_transition"  # System mode change


@dataclass
class SynthesisDecision:
    """A decision made by the cognitive synthesizer."""
    decision_type: DecisionType
    action: str
    rationale: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.decision_type.value,
            "action": self.action,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


# ============================================================
# Synthesis State: Unified view of system health
# ============================================================

@dataclass
class SynthesisState:
    """
    Unified state combining L7, L8, GUF signals.

    This is the "how am I doing across all dimensions" snapshot.
    """
    # L7 Structural Health
    structural_rate: float = 0.0      # Ṡ - rate of topological change
    alert_level: str = "stable"       # stable, elevated, warning, critical
    predicted_risk: float = 0.0       # 0-1 predicted future risk

    # L8 Verification Health
    pgu_pass_rate: float = 1.0        # Recent verification success rate
    last_verification_status: str = "verified"  # Last output status
    pending_repairs: int = 0          # Outputs awaiting repair

    # GUF Health
    utility: float = 0.8              # Current G value
    utility_target: float = 0.6       # G_target threshold
    goal_satisfied: bool = True       # G >= G_target
    focus_mode: str = "external"      # recovery, internal, balanced, external

    # Overall
    system_mode: SystemMode = SystemMode.SERVING
    af_score: float = 2.0             # Antifragility score
    confidence: float = 0.9           # System self-confidence
    fatigue: float = 0.1              # Accumulated load

    @property
    def is_healthy(self) -> bool:
        """Quick health check."""
        return (
            self.alert_level in ["stable", "elevated"] and
            self.goal_satisfied and
            self.pgu_pass_rate >= 0.8 and
            self.af_score >= 1.5
        )

    @property
    def needs_attention(self) -> bool:
        """Does the system need to focus on itself?"""
        return (
            self.alert_level in ["warning", "critical"] or
            not self.goal_satisfied or
            self.pgu_pass_rate < 0.7 or
            self.structural_rate > 0.15
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structural": {
                "rate": self.structural_rate,
                "alert_level": self.alert_level,
                "predicted_risk": self.predicted_risk
            },
            "verification": {
                "pgu_pass_rate": self.pgu_pass_rate,
                "last_status": self.last_verification_status,
                "pending_repairs": self.pending_repairs
            },
            "guf": {
                "utility": self.utility,
                "target": self.utility_target,
                "goal_satisfied": self.goal_satisfied,
                "focus_mode": self.focus_mode
            },
            "overall": {
                "system_mode": self.system_mode.value,
                "af_score": self.af_score,
                "confidence": self.confidence,
                "fatigue": self.fatigue,
                "is_healthy": self.is_healthy,
                "needs_attention": self.needs_attention
            }
        }


# ============================================================
# Cognitive Synthesizer: The Integration Engine
# ============================================================

class CognitiveSynthesizer:
    """
    The central integration engine that wires L7, L8, and GUF together.

    This is the "brain" that:
    - Monitors structural health (L7)
    - Routes verification decisions (L8)
    - Allocates focus between self and world (GUF)
    - Triggers proactive responses before problems manifest
    """

    def __init__(
        self,
        structural_rate_threshold: float = 0.15,
        utility_margin: float = 0.1,
        min_verification_rate: float = 0.8
    ):
        self.structural_rate_threshold = structural_rate_threshold
        self.utility_margin = utility_margin
        self.min_verification_rate = min_verification_rate

        # State
        self._state = SynthesisState()
        self._decision_history: List[SynthesisDecision] = []
        self._mode_transitions: List[Tuple[datetime, SystemMode, SystemMode]] = []

        # Callbacks for integration
        self._on_mode_change: Optional[Callable[[SystemMode, SystemMode], None]] = None
        self._on_aepo_trigger: Optional[Callable[[str, Dict], None]] = None
        self._on_alert: Optional[Callable[[str, str], None]] = None

        # Statistics
        self._stats = {
            "decisions_made": 0,
            "mode_transitions": 0,
            "aepo_triggers": 0,
            "verifications_routed": 0,
            "protective_actions": 0
        }

    @property
    def state(self) -> SynthesisState:
        """Current synthesis state."""
        return self._state

    @property
    def mode(self) -> SystemMode:
        """Current system mode."""
        return self._state.system_mode

    # --------------------------------------------------------
    # State Updates (from L7, L8, GUF)
    # --------------------------------------------------------

    def update_from_l7(
        self,
        structural_rate: float,
        alert_level: str,
        predicted_risk: float = 0.0
    ) -> Optional[SynthesisDecision]:
        """
        Update state from L7 Temporal Topology.

        Returns a decision if action is needed.
        """
        old_alert = self._state.alert_level

        self._state.structural_rate = structural_rate
        self._state.alert_level = alert_level
        self._state.predicted_risk = predicted_risk

        # Check if we need to respond
        decision = None

        if structural_rate > self.structural_rate_threshold:
            # Structural velocity is concerning
            if alert_level in ["warning", "critical"]:
                decision = self._make_protective_decision(structural_rate, alert_level)
            elif alert_level == "elevated" and old_alert == "stable":
                # Early warning - consider proactive AEPO
                decision = self._consider_proactive_aepo(structural_rate)

        self._recompute_mode()
        return decision

    def update_from_l8(
        self,
        verification_status: str,
        pgu_pass_rate: float,
        pending_repairs: int = 0
    ) -> Optional[SynthesisDecision]:
        """
        Update state from L8 Semantic Verification.

        Returns a decision if verification rate is concerning.
        """
        self._state.last_verification_status = verification_status
        self._state.pgu_pass_rate = pgu_pass_rate
        self._state.pending_repairs = pending_repairs

        decision = None

        if pgu_pass_rate < self.min_verification_rate:
            decision = SynthesisDecision(
                decision_type=DecisionType.VERIFICATION_ROUTING,
                action="increase_verification_strictness",
                rationale=f"PGU pass rate {pgu_pass_rate:.1%} below threshold {self.min_verification_rate:.1%}",
                confidence=0.9,
                metadata={"pass_rate": pgu_pass_rate, "threshold": self.min_verification_rate}
            )
            self._record_decision(decision)

        self._recompute_mode()
        return decision

    def update_from_guf(
        self,
        utility: float,
        utility_target: float,
        goal_satisfied: bool,
        focus_mode: str,
        af_score: float = 2.0,
        confidence: float = 0.9,
        fatigue: float = 0.1
    ) -> Optional[SynthesisDecision]:
        """
        Update state from GUF (Global Utility Function).

        Returns a decision about focus allocation.
        """
        old_satisfied = self._state.goal_satisfied

        self._state.utility = utility
        self._state.utility_target = utility_target
        self._state.goal_satisfied = goal_satisfied
        self._state.focus_mode = focus_mode
        self._state.af_score = af_score
        self._state.confidence = confidence
        self._state.fatigue = fatigue

        decision = None

        # Check for significant transitions
        if old_satisfied and not goal_satisfied:
            # Dropped below goal - shift focus internally
            decision = SynthesisDecision(
                decision_type=DecisionType.FOCUS_ALLOCATION,
                action="shift_focus_internal",
                rationale=f"Utility {utility:.3f} dropped below target {utility_target:.3f}",
                confidence=0.85,
                metadata={
                    "utility": utility,
                    "target": utility_target,
                    "focus_mode": focus_mode
                }
            )
            self._record_decision(decision)
        elif not old_satisfied and goal_satisfied:
            # Recovered above goal - can serve more
            decision = SynthesisDecision(
                decision_type=DecisionType.FOCUS_ALLOCATION,
                action="shift_focus_external",
                rationale=f"Utility {utility:.3f} recovered above target {utility_target:.3f}",
                confidence=0.85,
                metadata={
                    "utility": utility,
                    "target": utility_target,
                    "focus_mode": focus_mode
                }
            )
            self._record_decision(decision)

        self._recompute_mode()
        return decision

    # --------------------------------------------------------
    # Decision Making
    # --------------------------------------------------------

    def _make_protective_decision(
        self,
        structural_rate: float,
        alert_level: str
    ) -> SynthesisDecision:
        """Make a protective decision based on structural alerts."""
        if alert_level == "critical":
            action = "emergency_stabilization"
            rationale = f"CRITICAL: Ṡ={structural_rate:.3f} - initiating emergency stabilization"
        else:
            action = "preemptive_hardening"
            rationale = f"WARNING: Ṡ={structural_rate:.3f} - enabling protective measures"

        decision = SynthesisDecision(
            decision_type=DecisionType.STRUCTURAL_RESPONSE,
            action=action,
            rationale=rationale,
            confidence=0.9,
            metadata={
                "structural_rate": structural_rate,
                "alert_level": alert_level,
                "threshold": self.structural_rate_threshold
            }
        )

        self._record_decision(decision)
        self._stats["protective_actions"] += 1

        if self._on_alert:
            self._on_alert(alert_level, rationale)

        return decision

    def _consider_proactive_aepo(self, structural_rate: float) -> Optional[SynthesisDecision]:
        """Consider triggering proactive AEPO optimization."""
        # Only trigger if we have capacity and it's worth it
        if self._state.fatigue > 0.7:
            return None  # Too tired for proactive work

        if self._state.utility < self._state.utility_target - self.utility_margin:
            return None  # Already struggling, don't add load

        decision = SynthesisDecision(
            decision_type=DecisionType.AEPO_TRIGGER,
            action="proactive_redundancy_proposal",
            rationale=f"Early structural change detected (Ṡ={structural_rate:.3f}) - proposing preventive optimization",
            confidence=0.7,
            metadata={
                "structural_rate": structural_rate,
                "fatigue": self._state.fatigue,
                "utility_headroom": self._state.utility - self._state.utility_target
            }
        )

        self._record_decision(decision)
        self._stats["aepo_triggers"] += 1

        if self._on_aepo_trigger:
            self._on_aepo_trigger("proactive_redundancy", decision.metadata)

        return decision

    def _recompute_mode(self) -> None:
        """Recompute system mode based on current state."""
        old_mode = self._state.system_mode

        # Determine new mode based on state
        if self._state.alert_level == "critical" or self._state.af_score < 1.0:
            new_mode = SystemMode.RECOVERY
        elif self._state.alert_level == "warning" or self._state.structural_rate > 0.2:
            new_mode = SystemMode.PROTECTIVE
        elif not self._state.goal_satisfied:
            new_mode = SystemMode.SELF_IMPROVEMENT
        elif self._state.utility < self._state.utility_target + self.utility_margin:
            new_mode = SystemMode.BALANCED
        else:
            new_mode = SystemMode.SERVING

        if new_mode != old_mode:
            self._state.system_mode = new_mode
            self._mode_transitions.append((datetime.now(), old_mode, new_mode))
            self._stats["mode_transitions"] += 1

            # Record the transition as a decision
            decision = SynthesisDecision(
                decision_type=DecisionType.MODE_TRANSITION,
                action=f"transition_{old_mode.value}_to_{new_mode.value}",
                rationale=self._explain_mode_transition(old_mode, new_mode),
                confidence=0.95,
                metadata={
                    "from_mode": old_mode.value,
                    "to_mode": new_mode.value,
                    "state_snapshot": self._state.to_dict()
                }
            )
            self._record_decision(decision)

            if self._on_mode_change:
                self._on_mode_change(old_mode, new_mode)

    def _explain_mode_transition(self, old: SystemMode, new: SystemMode) -> str:
        """Generate human-readable explanation for mode transition."""
        reasons = []

        if new == SystemMode.RECOVERY:
            if self._state.af_score < 1.0:
                reasons.append(f"AF score {self._state.af_score:.2f} is below survival threshold")
            if self._state.alert_level == "critical":
                reasons.append(f"structural alert is CRITICAL")

        elif new == SystemMode.PROTECTIVE:
            reasons.append(f"structural rate Ṡ={self._state.structural_rate:.3f} indicates incoming instability")

        elif new == SystemMode.SELF_IMPROVEMENT:
            reasons.append(f"utility {self._state.utility:.3f} is below target {self._state.utility_target:.3f}")

        elif new == SystemMode.BALANCED:
            reasons.append(f"utility is near target, maintaining balance")

        elif new == SystemMode.SERVING:
            reasons.append(f"all systems healthy, ready to serve")

        return f"Transitioning from {old.value} to {new.value}: " + "; ".join(reasons)

    def _record_decision(self, decision: SynthesisDecision) -> None:
        """Record a decision in history."""
        self._decision_history.append(decision)
        self._stats["decisions_made"] += 1

        # Keep history bounded
        if len(self._decision_history) > 1000:
            self._decision_history = self._decision_history[-500:]

    # --------------------------------------------------------
    # Query Methods
    # --------------------------------------------------------

    def get_verification_routing(self, criticality: str) -> Dict[str, Any]:
        """
        Get verification routing recommendation based on current state.

        This maps L8 criticality to L6 reasoning mode, adjusted by system state.
        """
        # Base routing from criticality
        base_routing = {
            "low": {"mode": "LLM_ONLY", "verify": False},
            "medium": {"mode": "KG_ASSISTED", "verify": False},
            "high": {"mode": "PGU_VERIFIED", "verify": True},
            "critical": {"mode": "FORMAL_FIRST", "verify": True}
        }

        routing = base_routing.get(criticality, base_routing["medium"])

        # Adjust based on system state
        if self._state.system_mode in [SystemMode.RECOVERY, SystemMode.PROTECTIVE]:
            # In protective modes, increase verification
            if criticality == "low":
                routing = {"mode": "KG_ASSISTED", "verify": False}
            elif criticality == "medium":
                routing = {"mode": "PGU_VERIFIED", "verify": True}

        if self._state.pgu_pass_rate < self.min_verification_rate:
            # Low pass rate - be more careful
            routing["verify"] = True
            routing["strict"] = True

        self._stats["verifications_routed"] += 1

        return {
            "routing": routing,
            "system_mode": self._state.system_mode.value,
            "reason": f"Criticality {criticality} in mode {self._state.system_mode.value}"
        }

    def get_focus_recommendation(self) -> Dict[str, Any]:
        """
        Get focus allocation recommendation.

        Returns how the system should split attention between self and world.
        """
        if self._state.system_mode == SystemMode.RECOVERY:
            internal_pct = 0.9
            priority_tasks = ["restore_antifragility", "reduce_risk", "hardware_check"]
        elif self._state.system_mode == SystemMode.PROTECTIVE:
            internal_pct = 0.7
            priority_tasks = ["structural_stabilization", "preemptive_repair"]
        elif self._state.system_mode == SystemMode.SELF_IMPROVEMENT:
            internal_pct = 0.6
            priority_tasks = ["aepo_optimization", "world_model_update"]
        elif self._state.system_mode == SystemMode.BALANCED:
            internal_pct = 0.3
            priority_tasks = ["external_requests", "background_optimization"]
        else:  # SERVING
            internal_pct = 0.1
            priority_tasks = ["external_throughput", "user_requests"]

        return {
            "internal_focus_pct": internal_pct,
            "external_focus_pct": 1.0 - internal_pct,
            "priority_tasks": priority_tasks,
            "system_mode": self._state.system_mode.value,
            "utility": self._state.utility,
            "utility_target": self._state.utility_target
        }

    def should_trigger_aepo(self) -> Tuple[bool, str]:
        """
        Should we trigger an AEPO optimization cycle?

        Returns (should_trigger, reason).
        """
        if self._state.system_mode == SystemMode.RECOVERY:
            return True, "RECOVERY mode - urgent optimization needed"

        if self._state.structural_rate > self.structural_rate_threshold:
            return True, f"High structural rate Ṡ={self._state.structural_rate:.3f}"

        if not self._state.goal_satisfied and self._state.fatigue < 0.5:
            return True, "Below goal with capacity available"

        if self._state.alert_level == "elevated" and self._state.fatigue < 0.3:
            return True, "Early warning with high capacity - proactive optimization"

        return False, "System stable, no optimization needed"

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for cockpit display."""
        return {
            "mode": self._state.system_mode.value,
            "mode_emoji": self._get_mode_emoji(),
            "health": {
                "is_healthy": self._state.is_healthy,
                "needs_attention": self._state.needs_attention,
                "af_score": self._state.af_score,
                "confidence": self._state.confidence
            },
            "structural": {
                "rate": self._state.structural_rate,
                "rate_display": f"Ṡ={self._state.structural_rate:.3f}",
                "alert": self._state.alert_level,
                "alert_emoji": self._get_alert_emoji()
            },
            "verification": {
                "pass_rate": self._state.pgu_pass_rate,
                "pass_rate_display": f"{self._state.pgu_pass_rate:.0%}",
                "last_status": self._state.last_verification_status,
                "status_emoji": self._get_verification_emoji()
            },
            "utility": {
                "current": self._state.utility,
                "target": self._state.utility_target,
                "display": f"G={self._state.utility:.3f} / {self._state.utility_target:.3f}",
                "satisfied": self._state.goal_satisfied,
                "focus": self._state.focus_mode
            },
            "stats": self._stats,
            "timestamp": datetime.now().isoformat()
        }

    def _get_mode_emoji(self) -> str:
        """Get emoji for current mode."""
        return {
            SystemMode.RECOVERY: "🚨",
            SystemMode.PROTECTIVE: "🛡️",
            SystemMode.SELF_IMPROVEMENT: "🔧",
            SystemMode.BALANCED: "⚖️",
            SystemMode.SERVING: "✅"
        }.get(self._state.system_mode, "❓")

    def _get_alert_emoji(self) -> str:
        """Get emoji for alert level."""
        return {
            "stable": "🟢",
            "elevated": "🟡",
            "warning": "🟠",
            "critical": "🔴"
        }.get(self._state.alert_level, "⚪")

    def _get_verification_emoji(self) -> str:
        """Get emoji for verification status."""
        return {
            "verified": "✅",
            "repaired": "🔧",
            "unverifiable": "⚪",
            "failed": "⚠️",
            "not_checked": "⚪"
        }.get(self._state.last_verification_status, "❓")

    def describe_state(self) -> str:
        """Generate natural language description of current state."""
        mode_desc = {
            SystemMode.RECOVERY: "I'm in recovery mode, focusing on survival",
            SystemMode.PROTECTIVE: "I'm in protective mode, bracing for instability",
            SystemMode.SELF_IMPROVEMENT: "I'm focusing on self-improvement",
            SystemMode.BALANCED: "I'm balancing self-maintenance with serving",
            SystemMode.SERVING: "I'm healthy and ready to serve"
        }.get(self._state.system_mode, "Unknown mode")

        parts = [mode_desc + "."]

        if self._state.structural_rate > 0.1:
            parts.append(f"My structural rate is elevated (Ṡ={self._state.structural_rate:.3f}).")

        if self._state.pgu_pass_rate < 0.9:
            parts.append(f"My verification rate is {self._state.pgu_pass_rate:.0%}.")

        if not self._state.goal_satisfied:
            parts.append(f"My utility ({self._state.utility:.3f}) is below target ({self._state.utility_target:.3f}).")

        parts.append(f"Antifragility: {self._state.af_score:.2f}×.")

        return " ".join(parts)

    # --------------------------------------------------------
    # Callbacks Registration
    # --------------------------------------------------------

    def on_mode_change(self, callback: Callable[[SystemMode, SystemMode], None]) -> None:
        """Register callback for mode changes."""
        self._on_mode_change = callback

    def on_aepo_trigger(self, callback: Callable[[str, Dict], None]) -> None:
        """Register callback for AEPO triggers."""
        self._on_aepo_trigger = callback

    def on_alert(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for alerts."""
        self._on_alert = callback

    @property
    def decision_history(self) -> List[SynthesisDecision]:
        """Get decision history."""
        return self._decision_history.copy()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self._stats.copy()


# ============================================================
# Cockpit Status: Display-Ready Status
# ============================================================

@dataclass
class CockpitStatus:
    """
    Display-ready status for dashboard/cockpit integration.

    This formats synthesis state for human consumption.
    """
    # Mode badge
    mode: str
    mode_emoji: str
    mode_color: str  # green, yellow, orange, red

    # Health indicators
    healthy: bool
    af_score: float
    af_display: str

    # Structural indicator
    structural_rate: float
    structural_display: str
    structural_emoji: str

    # Verification indicator
    verification_rate: float
    verification_display: str
    verification_emoji: str

    # Utility indicator
    utility: float
    utility_target: float
    utility_display: str
    goal_satisfied: bool

    # Focus recommendation
    internal_focus_pct: float
    external_focus_pct: float
    focus_display: str

    # Timestamp
    timestamp: str

    @classmethod
    def from_synthesizer(cls, synth: CognitiveSynthesizer) -> "CockpitStatus":
        """Create CockpitStatus from synthesizer."""
        status = synth.get_system_status()
        focus = synth.get_focus_recommendation()

        mode_colors = {
            "recovery": "red",
            "protective": "orange",
            "self_improvement": "yellow",
            "balanced": "blue",
            "serving": "green"
        }

        return cls(
            mode=status["mode"],
            mode_emoji=status["mode_emoji"],
            mode_color=mode_colors.get(status["mode"], "gray"),
            healthy=status["health"]["is_healthy"],
            af_score=status["health"]["af_score"],
            af_display=f"{status['health']['af_score']:.2f}×",
            structural_rate=status["structural"]["rate"],
            structural_display=status["structural"]["rate_display"],
            structural_emoji=status["structural"]["alert_emoji"],
            verification_rate=status["verification"]["pass_rate"],
            verification_display=status["verification"]["pass_rate_display"],
            verification_emoji=status["verification"]["status_emoji"],
            utility=status["utility"]["current"],
            utility_target=status["utility"]["target"],
            utility_display=status["utility"]["display"],
            goal_satisfied=status["utility"]["satisfied"],
            internal_focus_pct=focus["internal_focus_pct"],
            external_focus_pct=focus["external_focus_pct"],
            focus_display=f"{focus['internal_focus_pct']:.0%} self / {focus['external_focus_pct']:.0%} world",
            timestamp=status["timestamp"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode": {"name": self.mode, "emoji": self.mode_emoji, "color": self.mode_color},
            "health": {"healthy": self.healthy, "af_score": self.af_score, "display": self.af_display},
            "structural": {"rate": self.structural_rate, "display": self.structural_display, "emoji": self.structural_emoji},
            "verification": {"rate": self.verification_rate, "display": self.verification_display, "emoji": self.verification_emoji},
            "utility": {"current": self.utility, "target": self.utility_target, "display": self.utility_display, "satisfied": self.goal_satisfied},
            "focus": {"internal": self.internal_focus_pct, "external": self.external_focus_pct, "display": self.focus_display},
            "timestamp": self.timestamp
        }

    def render_text(self) -> str:
        """Render as text for terminal display."""
        lines = [
            f"╔══════════════════════════════════════════════════════════╗",
            f"║  {self.mode_emoji} MODE: {self.mode.upper():20}                         ║",
            f"╠══════════════════════════════════════════════════════════╣",
            f"║  Health:      {'✅ HEALTHY' if self.healthy else '⚠️  ATTENTION NEEDED':15}                  ║",
            f"║  AF Score:    {self.af_display:15}                          ║",
            f"║  Structural:  {self.structural_emoji} {self.structural_display:12}                       ║",
            f"║  Verification:{self.verification_emoji} {self.verification_display:12}                       ║",
            f"║  Utility:     {self.utility_display:20}                 ║",
            f"║  Focus:       {self.focus_display:25}            ║",
            f"╚══════════════════════════════════════════════════════════╝"
        ]
        return "\n".join(lines)


# ============================================================
# Factory Functions
# ============================================================

def create_synthesizer() -> CognitiveSynthesizer:
    """Create a default cognitive synthesizer."""
    return CognitiveSynthesizer()


def create_cockpit_status(synth: CognitiveSynthesizer) -> CockpitStatus:
    """Create cockpit status from synthesizer."""
    return CockpitStatus.from_synthesizer(synth)
