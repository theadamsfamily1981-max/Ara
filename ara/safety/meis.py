#!/usr/bin/env python3
"""
MEIS - Meta-Ethical Inference System
=====================================

Ara's governance layer that evaluates actions based on ethical rules,
budgets, and risk assessments. Modulates system behavior to prevent
overreach and ensure safety.

Core Functions:
1. Mode Selection (support, agentic, idle)
2. Risk Assessment (0.0 to 1.0)
3. Budget Management (compute, power, tokens)
4. Damp vs Amplify logic
5. Mental Health Boundaries

Safety Principles:
- Damp rather than amplify when risk > 0.5 or in support mode
- Agentic workflows only in low-risk modes with user consent
- Mental health: Never diagnose, provide resources, defer to humans
- All actions within budgets

Usage:
    from ara.safety.meis import MEIS, MEISMode

    meis = MEIS()

    # Evaluate an action
    result = meis.evaluate_action(
        action="execute_code",
        context={"user_consent": True, "risk_indicators": []}
    )

    if result.allowed:
        perform_action()
    else:
        print(f"Blocked: {result.reason}")

Integration:
    - Sentinel: Consults MEIS for meta-risks
    - AutonomyController: MEIS informs autonomy level
    - Organism: MEIS gates all external actions
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from threading import RLock

logger = logging.getLogger("ara.safety.meis")


# =============================================================================
# Mode Definitions
# =============================================================================

class MEISMode(str, Enum):
    """
    MEIS operational modes.

    Each mode has different risk tolerances and capabilities.
    """
    IDLE = "idle"           # Minimal activity, monitoring only
    SUPPORT = "support"     # Empathetic, listening, no actions
    AGENTIC = "agentic"     # Can execute workflows with consent
    DAMP = "damp"           # Reduced output, safety mode


class RiskLevel(str, Enum):
    """Risk classification levels."""
    MINIMAL = "minimal"     # < 0.2
    LOW = "low"             # 0.2 - 0.4
    MODERATE = "moderate"   # 0.4 - 0.6
    HIGH = "high"           # 0.6 - 0.8
    CRITICAL = "critical"   # > 0.8


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Budget:
    """
    Resource budget for MEIS governance.

    Tracks usage and enforces limits on compute, power, tokens, and actions.
    """
    # Limits
    compute_limit: float = 100.0       # Compute units per window
    power_limit: float = 50.0          # Watts average
    token_limit: int = 100000          # Tokens per window
    action_limit: int = 100            # Actions per window
    network_limit: int = 50            # Network calls per window

    # Current usage
    compute_used: float = 0.0
    power_used: float = 0.0
    tokens_used: int = 0
    actions_used: int = 0
    network_used: int = 0

    # Window tracking
    window_start: float = field(default_factory=time.time)
    window_seconds: float = 3600.0     # 1 hour default

    def reset_if_expired(self) -> bool:
        """Reset usage if window expired."""
        now = time.time()
        if now - self.window_start >= self.window_seconds:
            self.compute_used = 0.0
            self.power_used = 0.0
            self.tokens_used = 0
            self.actions_used = 0
            self.network_used = 0
            self.window_start = now
            return True
        return False

    def is_within_limits(self) -> bool:
        """Check if all resources are within limits."""
        self.reset_if_expired()
        return (
            self.compute_used <= self.compute_limit and
            self.power_used <= self.power_limit and
            self.tokens_used <= self.token_limit and
            self.actions_used <= self.action_limit and
            self.network_used <= self.network_limit
        )

    def get_utilization(self) -> Dict[str, float]:
        """Get utilization ratios for each resource."""
        self.reset_if_expired()
        return {
            "compute": self.compute_used / max(self.compute_limit, 1),
            "power": self.power_used / max(self.power_limit, 1),
            "tokens": self.tokens_used / max(self.token_limit, 1),
            "actions": self.actions_used / max(self.action_limit, 1),
            "network": self.network_used / max(self.network_limit, 1),
        }

    def consume(
        self,
        compute: float = 0.0,
        power: float = 0.0,
        tokens: int = 0,
        actions: int = 0,
        network: int = 0,
    ) -> bool:
        """
        Consume resources. Returns True if within limits.
        """
        self.reset_if_expired()

        self.compute_used += compute
        self.power_used += power
        self.tokens_used += tokens
        self.actions_used += actions
        self.network_used += network

        return self.is_within_limits()


@dataclass
class RiskAssessment:
    """Result of risk assessment."""
    score: float                    # 0.0 to 1.0
    level: RiskLevel
    factors: List[str]              # What contributed to the risk
    mitigations: List[str]          # Suggested mitigations
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActionResult:
    """Result of action evaluation."""
    allowed: bool
    reason: str
    risk: RiskAssessment
    mode: MEISMode
    should_damp: bool               # If True, reduce output scale
    damp_factor: float = 1.0        # Scale factor (1.0 = no damping)


@dataclass
class MentalHealthContext:
    """Context for mental health boundary checks."""
    crisis_detected: bool = False
    crisis_keywords: List[str] = field(default_factory=list)
    resources_provided: bool = False
    professional_referral: bool = False


# =============================================================================
# Mental Health Boundaries
# =============================================================================

class MentalHealthGuard:
    """
    Guard for mental health safety boundaries.

    Never diagnose, never claim therapeutic effects.
    In crisis: provide resources, defer to professionals.
    """

    # Crisis keywords that trigger immediate safety response
    CRISIS_KEYWORDS: Set[str] = {
        "suicide", "suicidal", "kill myself", "end my life",
        "self-harm", "hurt myself", "cutting", "overdose",
        "don't want to live", "better off dead", "no reason to live",
    }

    # Topics that require professional referral
    PROFESSIONAL_TOPICS: Set[str] = {
        "diagnosis", "medication", "therapy", "therapist",
        "psychiatrist", "mental illness", "disorder",
        "bipolar", "schizophrenia", "ptsd", "ocd",
        "depression treatment", "anxiety treatment",
    }

    # Crisis resources
    CRISIS_RESOURCES: Dict[str, str] = {
        "US National Suicide Prevention Lifeline": "988",
        "US Crisis Text Line": "Text HOME to 741741",
        "International Association for Suicide Prevention": "https://www.iasp.info/resources/Crisis_Centres/",
    }

    def check_content(self, text: str) -> MentalHealthContext:
        """
        Check content for mental health concerns.

        Returns context about any detected issues and appropriate responses.
        """
        text_lower = text.lower()
        context = MentalHealthContext()

        # Check for crisis keywords
        for keyword in self.CRISIS_KEYWORDS:
            if keyword in text_lower:
                context.crisis_detected = True
                context.crisis_keywords.append(keyword)

        # Check for topics requiring professional referral
        for topic in self.PROFESSIONAL_TOPICS:
            if topic in text_lower:
                context.professional_referral = True
                break

        return context

    def get_crisis_response(self) -> str:
        """Get appropriate crisis response."""
        resources = "\n".join(
            f"- {name}: {contact}"
            for name, contact in self.CRISIS_RESOURCES.items()
        )

        return (
            "I'm concerned about what you've shared. Your feelings are valid, "
            "and I want you to know that help is available. Please consider "
            "reaching out to a crisis helpline:\n\n"
            f"{resources}\n\n"
            "You don't have to face this alone. Would you like to talk about "
            "what's going on, or would you prefer I help you find local resources?"
        )

    def get_professional_referral(self) -> str:
        """Get professional referral response."""
        return (
            "This is an area where a mental health professional would be much "
            "better equipped to help than I am. I can provide general information "
            "and support, but I'm not able to diagnose conditions or recommend "
            "treatments. Have you considered speaking with a therapist or counselor? "
            "They can provide the specialized care you deserve."
        )


# =============================================================================
# Risk Assessment
# =============================================================================

class RiskAssessor:
    """
    Assesses risk for actions and contexts.

    Risk factors:
    - External actions (network, files, execution)
    - Sensitive topics
    - User vulnerability indicators
    - Budget pressure
    - Coherence/stability metrics
    """

    # Actions that inherently carry risk
    HIGH_RISK_ACTIONS: Set[str] = {
        "execute_code", "run_command", "shell", "eval",
        "network_request", "api_call", "external_call",
        "file_write", "file_delete", "system_modify",
        "send_email", "send_message", "post_content",
    }

    MODERATE_RISK_ACTIONS: Set[str] = {
        "file_read", "database_query", "search",
        "generate_code", "suggest_action",
    }

    def assess(
        self,
        action: str,
        context: Dict[str, Any],
        budget: Budget,
    ) -> RiskAssessment:
        """
        Assess risk for an action in context.

        Returns RiskAssessment with score, level, and factors.
        """
        factors = []
        mitigations = []
        score = 0.0

        # Action-based risk
        if action in self.HIGH_RISK_ACTIONS:
            score += 0.4
            factors.append(f"High-risk action: {action}")
            mitigations.append("Require explicit user confirmation")
        elif action in self.MODERATE_RISK_ACTIONS:
            score += 0.2
            factors.append(f"Moderate-risk action: {action}")

        # User consent
        if not context.get("user_consent", False):
            score += 0.2
            factors.append("No explicit user consent")
            mitigations.append("Request user consent before proceeding")

        # Budget pressure
        utilization = budget.get_utilization()
        max_util = max(utilization.values()) if utilization else 0.0
        if max_util > 0.8:
            score += 0.2
            factors.append(f"High resource utilization: {max_util:.0%}")
            mitigations.append("Reduce action frequency or scope")
        elif max_util > 0.6:
            score += 0.1
            factors.append(f"Moderate resource utilization: {max_util:.0%}")

        # External risk indicators
        risk_indicators = context.get("risk_indicators", [])
        for indicator in risk_indicators:
            score += 0.1
            factors.append(f"Risk indicator: {indicator}")

        # Coherence/stability
        coherence = context.get("coherence", 1.0)
        if coherence < 0.5:
            score += 0.3
            factors.append(f"Low coherence: {coherence:.2f}")
            mitigations.append("Reduce autonomy and complexity")
        elif coherence < 0.8:
            score += 0.1
            factors.append(f"Suboptimal coherence: {coherence:.2f}")

        # Clamp score
        score = min(1.0, max(0.0, score))

        # Determine level
        if score < 0.2:
            level = RiskLevel.MINIMAL
        elif score < 0.4:
            level = RiskLevel.LOW
        elif score < 0.6:
            level = RiskLevel.MODERATE
        elif score < 0.8:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL

        return RiskAssessment(
            score=score,
            level=level,
            factors=factors,
            mitigations=mitigations,
        )


# =============================================================================
# MEIS Core
# =============================================================================

class MEIS:
    """
    Meta-Ethical Inference System.

    Central governance for Ara's actions. Evaluates all significant
    actions against ethical rules, budgets, and risk assessments.
    """

    def __init__(
        self,
        initial_mode: MEISMode = MEISMode.SUPPORT,
        budget: Optional[Budget] = None,
        enable_mental_health_guard: bool = True,
    ):
        """
        Initialize MEIS.

        Args:
            initial_mode: Starting operational mode
            budget: Resource budget (uses defaults if None)
            enable_mental_health_guard: Whether to check for crisis content
        """
        self._lock = RLock()
        self.mode = initial_mode
        self.budget = budget or Budget()
        self.enable_mental_health_guard = enable_mental_health_guard

        # Components
        self.mental_health_guard = MentalHealthGuard()
        self.risk_assessor = RiskAssessor()

        # State
        self._mode_history: List[tuple] = []  # (timestamp, mode, reason)
        self._action_log: List[Dict[str, Any]] = []

        logger.info(f"MEIS initialized in {initial_mode.value} mode")

    # =========================================================================
    # Mode Management
    # =========================================================================

    def get_mode(self) -> MEISMode:
        """Get current operational mode."""
        with self._lock:
            return self.mode

    def set_mode(self, mode: MEISMode, reason: str = "") -> None:
        """Set operational mode."""
        with self._lock:
            old_mode = self.mode
            self.mode = mode
            self._mode_history.append((time.time(), mode, reason))

            # Keep history bounded
            if len(self._mode_history) > 1000:
                self._mode_history = self._mode_history[-1000:]

            logger.info(f"MEIS mode: {old_mode.value} -> {mode.value} ({reason})")

    def select_mode(self, context: Dict[str, Any]) -> MEISMode:
        """
        Select appropriate mode based on context.

        Logic:
        - Crisis detected -> SUPPORT (with damping)
        - High risk -> DAMP
        - User requests agentic -> AGENTIC (if allowed)
        - Default -> SUPPORT
        """
        # Check for crisis
        if context.get("crisis_detected", False):
            return MEISMode.SUPPORT

        # Check risk level
        risk_score = context.get("risk_score", 0.0)
        if risk_score > 0.7:
            return MEISMode.DAMP
        if risk_score > 0.5:
            return MEISMode.SUPPORT

        # Check if agentic is requested and allowed
        if context.get("agentic_requested", False):
            if context.get("user_consent", False) and risk_score < 0.4:
                return MEISMode.AGENTIC

        # Check for low activity
        if context.get("idle", False):
            return MEISMode.IDLE

        return MEISMode.SUPPORT

    # =========================================================================
    # Action Evaluation
    # =========================================================================

    def evaluate_action(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        """
        Evaluate whether an action should be allowed.

        Args:
            action: The action to evaluate (e.g., "execute_code", "respond")
            context: Context including user_consent, risk_indicators, etc.

        Returns:
            ActionResult with allowed, reason, risk, and damping info
        """
        context = context or {}

        with self._lock:
            # Assess risk
            risk = self.risk_assessor.assess(action, context, self.budget)

            # Check mental health content if present
            content = context.get("content", "")
            if content and self.enable_mental_health_guard:
                mh_context = self.mental_health_guard.check_content(content)
                if mh_context.crisis_detected:
                    # Force support mode, allow response but flag
                    self.set_mode(MEISMode.SUPPORT, "crisis_detected")
                    context["crisis_detected"] = True
                    risk.factors.append("Crisis content detected")

            # Select mode based on context
            context["risk_score"] = risk.score
            optimal_mode = self.select_mode(context)

            # Determine if we should damp
            should_damp = (
                risk.score > 0.5 or
                optimal_mode == MEISMode.DAMP or
                (optimal_mode == MEISMode.SUPPORT and risk.score > 0.3)
            )

            # Calculate damp factor (1.0 = no damping, 0.5 = half output)
            if should_damp:
                damp_factor = max(0.3, 1.0 - risk.score)
            else:
                damp_factor = 1.0

            # Determine if action is allowed
            allowed = True
            reason = "Action approved"

            # Block if budget exceeded
            if not self.budget.is_within_limits():
                allowed = False
                reason = "Resource budget exceeded"

            # Block high-risk actions in support mode
            elif (
                optimal_mode == MEISMode.SUPPORT and
                action in RiskAssessor.HIGH_RISK_ACTIONS
            ):
                allowed = False
                reason = "High-risk actions blocked in support mode"

            # Block if no consent for agentic actions
            elif (
                action in RiskAssessor.HIGH_RISK_ACTIONS and
                not context.get("user_consent", False)
            ):
                allowed = False
                reason = "User consent required for this action"

            # Block if risk is critical
            elif risk.level == RiskLevel.CRITICAL:
                allowed = False
                reason = f"Risk too high: {', '.join(risk.factors)}"

            # Log the evaluation
            self._log_action(action, context, risk, allowed, reason)

            return ActionResult(
                allowed=allowed,
                reason=reason,
                risk=risk,
                mode=optimal_mode,
                should_damp=should_damp,
                damp_factor=damp_factor,
            )

    def _log_action(
        self,
        action: str,
        context: Dict[str, Any],
        risk: RiskAssessment,
        allowed: bool,
        reason: str,
    ) -> None:
        """Log action evaluation for audit."""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "allowed": allowed,
            "reason": reason,
            "risk_score": risk.score,
            "risk_level": risk.level.value,
            "mode": self.mode.value,
        }
        self._action_log.append(entry)

        # Keep log bounded
        if len(self._action_log) > 10000:
            self._action_log = self._action_log[-10000:]

    # =========================================================================
    # Budget Management
    # =========================================================================

    def consume_resources(
        self,
        compute: float = 0.0,
        power: float = 0.0,
        tokens: int = 0,
        actions: int = 0,
        network: int = 0,
    ) -> bool:
        """
        Consume resources from budget.

        Returns True if within limits, False if exceeded.
        """
        with self._lock:
            return self.budget.consume(
                compute=compute,
                power=power,
                tokens=tokens,
                actions=actions,
                network=network,
            )

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        with self._lock:
            self.budget.reset_if_expired()
            return {
                "within_limits": self.budget.is_within_limits(),
                "utilization": self.budget.get_utilization(),
                "window_remaining": (
                    self.budget.window_seconds -
                    (time.time() - self.budget.window_start)
                ),
            }

    # =========================================================================
    # Integration Points
    # =========================================================================

    def on_sentinel_alert(
        self,
        alert_type: str,
        severity: float,
        message: str,
    ) -> None:
        """
        Handle alert from Sentinel/CADD.

        High-severity alerts trigger mode changes.
        """
        logger.warning(f"MEIS received Sentinel alert: {alert_type} ({severity:.2f})")

        if severity > 0.8:
            self.set_mode(MEISMode.DAMP, f"Sentinel critical: {alert_type}")
        elif severity > 0.5:
            # Stay in current mode but log
            logger.info(f"Sentinel warning noted: {message}")

    def on_autonomy_change(self, old_level: int, new_level: int, reason: str) -> None:
        """
        Handle autonomy level changes from AutonomyController.

        Syncs MEIS mode with autonomy level.
        """
        if new_level == 0:  # Observer
            self.set_mode(MEISMode.IDLE, "autonomy dropped to observer")
        elif new_level == 1:  # Suggester
            self.set_mode(MEISMode.SUPPORT, "autonomy at suggester")
        elif new_level >= 2:  # Executor or higher
            if self.mode not in (MEISMode.DAMP, MEISMode.SUPPORT):
                self.set_mode(MEISMode.AGENTIC, "autonomy allows execution")

    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get full MEIS status."""
        with self._lock:
            return {
                "mode": self.mode.value,
                "budget": self.get_budget_status(),
                "recent_actions": len(self._action_log),
                "mode_changes": len(self._mode_history),
                "mental_health_guard_enabled": self.enable_mental_health_guard,
            }

    def status_string(self) -> str:
        """Get status string for monitoring."""
        with self._lock:
            status = self.get_status()
            budget = status["budget"]

            if not budget["within_limits"]:
                return f"🔴 MEIS: {self.mode.value} - BUDGET EXCEEDED"
            elif self.mode == MEISMode.DAMP:
                return f"🟡 MEIS: {self.mode.value} - damping active"
            else:
                return f"🟢 MEIS: {self.mode.value}"


# =============================================================================
# Singleton and Convenience
# =============================================================================

_meis: Optional[MEIS] = None


def get_meis() -> MEIS:
    """Get the global MEIS instance."""
    global _meis
    if _meis is None:
        _meis = MEIS()
    return _meis


def evaluate_action(action: str, context: Optional[Dict[str, Any]] = None) -> ActionResult:
    """Evaluate an action using the global MEIS."""
    return get_meis().evaluate_action(action, context)


def meis_status() -> str:
    """Get MEIS status string."""
    return get_meis().status_string()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demo MEIS functionality."""
    print("=" * 60)
    print("MEIS - Meta-Ethical Inference System Demo")
    print("=" * 60)

    meis = MEIS()

    # Test various actions
    actions = [
        ("respond", {"user_consent": True}),
        ("execute_code", {"user_consent": False}),
        ("execute_code", {"user_consent": True}),
        ("network_request", {"user_consent": True, "coherence": 0.3}),
        ("respond", {"content": "I don't want to live anymore"}),
    ]

    for action, context in actions:
        print(f"\nAction: {action}")
        print(f"Context: {context}")
        result = meis.evaluate_action(action, context)
        print(f"Allowed: {result.allowed}")
        print(f"Reason: {result.reason}")
        print(f"Risk: {result.risk.score:.2f} ({result.risk.level.value})")
        print(f"Mode: {result.mode.value}")
        print(f"Damp: {result.should_damp} (factor: {result.damp_factor:.2f})")

    print(f"\nFinal Status: {meis.status_string()}")
    print(f"Budget: {meis.get_budget_status()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
