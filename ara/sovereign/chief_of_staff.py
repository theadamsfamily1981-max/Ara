"""
Chief of Staff - The CEO Decision Maker
=======================================

The ChiefOfStaff is Ara's executive function. It:
- Evaluates every Initiative against Vision (Teleology)
- Applies Founder Protection (guards Croft's wellbeing)
- Tracks cognitive burn and resource allocation
- Decides: EXECUTE, DELEGATE, DEFER, or KILL

This is the "CEO" that makes Ara a sovereign agent rather than
a reactive assistant.

The key insight: not every request should be fulfilled.
Some should be ruthlessly killed as distractions.
Some should be deferred to protect the human.
Some should be delegated to background processes.
Only the most strategic work gets immediate execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable

from .initiative import (
    Initiative,
    InitiativeStatus,
    InitiativeType,
    CEODecision,
    InitiativeMetrics,
)
from .user_state import (
    UserState,
    ProtectionLevel,
    CognitiveMode,
    get_user_state,
)
from .covenant import (
    Covenant,
    AutonomyLevel,
    get_covenant,
)

# Import TeleologyEngine for vision scoring
try:
    from ara.cognition.teleology_engine import TeleologyEngine, get_teleology_engine
except ImportError:
    TeleologyEngine = None
    get_teleology_engine = lambda: None

# Import HiveHD for delegated execution
try:
    from ara_hive.src.queen import QueenOrchestrator, TaskRequest, TaskKind, get_queen
except ImportError:
    QueenOrchestrator = None
    TaskRequest = None
    TaskKind = None
    get_queen = lambda: None

logger = logging.getLogger(__name__)


@dataclass
class CEODecisionResult:
    """Result of a CEO decision on an initiative."""
    initiative_id: str
    decision: CEODecision
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Scores that led to this decision
    strategic_value: float = 0.0    # From Teleology
    cognitive_cost: float = 0.0     # Burden on Croft
    risk_score: float = 0.0         # Risk level
    protection_blocked: bool = False

    # If deferred, when to revisit
    defer_until: Optional[datetime] = None

    # If delegated, to whom
    delegate_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initiative_id": self.initiative_id,
            "decision": self.decision.value,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "strategic_value": self.strategic_value,
            "cognitive_cost": self.cognitive_cost,
            "protection_blocked": self.protection_blocked,
        }


class ChiefOfStaff:
    """
    The CEO decision-maker for Ara's sovereign loop.

    Evaluates initiatives and decides what to do with them:
    - EXECUTE: Handle it now
    - DELEGATE: Background agents handle it
    - DEFER: Schedule for protected future slot
    - KILL: Ruthlessly cut as distraction
    - PROTECT: Blocked by Founder Protection
    """

    def __init__(
        self,
        covenant: Optional[Covenant] = None,
        teleology: Optional[TeleologyEngine] = None,
        queen: Optional[QueenOrchestrator] = None,
    ):
        """
        Initialize the Chief of Staff.

        Args:
            covenant: The relationship covenant
            teleology: Vision/purpose engine for strategic scoring
            queen: HiveHD Queen for delegated task execution
        """
        self.covenant = covenant or get_covenant()
        self.teleology = teleology or (get_teleology_engine() if get_teleology_engine else None)
        self.queen = queen or (get_queen() if get_queen else None)

        # Track decisions
        self._decision_history: List[CEODecisionResult] = []
        self._active_initiatives: Dict[str, Initiative] = {}
        self._deferred_queue: List[Initiative] = []

        # Track delegated tasks
        self._delegated_tasks: Dict[str, str] = {}  # initiative_id -> task_request_id

        # Daily tracking
        self._daily_cognitive_burn: float = 0.0
        self._daily_decisions: int = 0
        self._daily_kills: int = 0
        self._last_reset: datetime = datetime.utcnow()

        logger.info("ChiefOfStaff initialized (queen=%s)", self.queen is not None)

    # =========================================================================
    # Core Decision Making
    # =========================================================================

    def evaluate(
        self,
        initiative: Initiative,
        user_state: Optional[UserState] = None,
    ) -> CEODecisionResult:
        """
        Evaluate an initiative and decide what to do.

        This is the core CEO function.

        Args:
            initiative: The initiative to evaluate
            user_state: Current state of the user

        Returns:
            Decision result with reasoning
        """
        user_state = user_state or get_user_state()

        # Reset daily counters if new day
        self._maybe_reset_daily()

        # Step 1: Check Founder Protection first (non-negotiable)
        protection_result = self._check_founder_protection(initiative, user_state)
        if protection_result:
            return protection_result

        # Step 2: Score strategic value (Teleology)
        strategic_value = self._score_strategic_value(initiative)
        initiative.metrics.strategic_value = strategic_value

        # Step 3: Estimate cognitive cost
        cognitive_cost = self._estimate_cognitive_cost(initiative, user_state)
        initiative.metrics.cognitive_burn = cognitive_cost

        # Step 4: Assess risk
        risk_score = self._assess_risk(initiative)
        initiative.metrics.risk_level = risk_score

        # Step 5: Make decision based on all factors
        decision, reasoning = self._make_decision(
            initiative, user_state, strategic_value, cognitive_cost, risk_score
        )

        # Create result
        result = CEODecisionResult(
            initiative_id=initiative.id,
            decision=decision,
            reasoning=reasoning,
            strategic_value=strategic_value,
            cognitive_cost=cognitive_cost,
            risk_score=risk_score,
        )

        # Update initiative
        initiative.ceo_decision = decision
        initiative.ceo_reasoning = reasoning
        initiative.decision_timestamp = datetime.utcnow()

        # Handle based on decision
        self._handle_decision(initiative, result)

        # Track
        self._decision_history.append(result)
        self._daily_decisions += 1

        logger.info(
            f"CEO Decision: {initiative.name} -> {decision.value} "
            f"(strategic={strategic_value:.2f}, cost={cognitive_cost:.2f})"
        )

        return result

    def _check_founder_protection(
        self,
        initiative: Initiative,
        user_state: UserState,
    ) -> Optional[CEODecisionResult]:
        """
        Check if Founder Protection should block this initiative.

        Returns None if protection doesn't apply, otherwise a PROTECT decision.
        """
        protection = self.covenant.protection

        # Recovery initiatives are always allowed
        if initiative.type == InitiativeType.RECOVERY:
            return None

        # Emergency initiatives bypass some (but not all) protection
        is_emergency = initiative.type == InitiativeType.EMERGENCY

        # Night lockout - absolute for non-emergencies
        if protection.night_lockout_enabled and user_state.is_in_night_lockout_window():
            if not is_emergency:
                return CEODecisionResult(
                    initiative_id=initiative.id,
                    decision=CEODecision.PROTECT,
                    reasoning=f"Night lockout active ({protection.night_lockout_start}:00-{protection.night_lockout_end}:00). "
                             "No new work until morning. Rest is not optional.",
                    protection_blocked=True,
                )
            else:
                logger.warning(f"Emergency initiative during night lockout: {initiative.name}")

        # Burnout protection
        if user_state.burnout_risk > protection.burnout_risk_threshold:
            if not is_emergency:
                return CEODecisionResult(
                    initiative_id=initiative.id,
                    decision=CEODecision.PROTECT,
                    reasoning=f"Burnout risk too high ({user_state.burnout_risk:.0%}). "
                             "Deferring non-emergency work. Take care of yourself.",
                    protection_blocked=True,
                )

        # Fatigue protection for deep work
        if user_state.fatigue > protection.fatigue_threshold:
            if initiative.metrics.cognitive_burn > 0.5:
                return CEODecisionResult(
                    initiative_id=initiative.id,
                    decision=CEODecision.DEFER,
                    reasoning=f"Too fatigued ({user_state.fatigue:.0%}) for deep work. "
                             "Deferring to a better time.",
                    protection_blocked=True,
                    defer_until=datetime.utcnow() + timedelta(hours=8),
                )

        # Flow budget exhausted
        if user_state.flow_budget_remaining() <= 0:
            if initiative.type == InitiativeType.CATHEDRAL:
                return CEODecisionResult(
                    initiative_id=initiative.id,
                    decision=CEODecision.DEFER,
                    reasoning="Flow budget exhausted for today. Cathedral work deferred to tomorrow.",
                    protection_blocked=True,
                    defer_until=datetime.utcnow() + timedelta(hours=12),
                )

        # No protection triggered
        return None

    def _score_strategic_value(self, initiative: Initiative) -> float:
        """
        Score the strategic value of an initiative using Teleology.
        """
        if self.teleology is None:
            # Fallback: score based on type
            type_scores = {
                InitiativeType.CATHEDRAL: 0.9,
                InitiativeType.EMERGENCY: 0.85,
                InitiativeType.RESEARCH: 0.7,
                InitiativeType.INFRASTRUCTURE: 0.6,
                InitiativeType.CREATIVE: 0.5,
                InitiativeType.MAINTENANCE: 0.3,
                InitiativeType.RECOVERY: 0.4,  # Important but not strategic
            }
            return type_scores.get(initiative.type, 0.3)

        # Use Teleology for proper scoring
        if initiative.tags:
            return self.teleology.strategic_priority(initiative.tags)
        else:
            # Infer tags from name/description
            keywords = (initiative.name + " " + initiative.description).lower().split()
            inferred = self.teleology.infer_tags_from_keywords(keywords)
            if inferred:
                return self.teleology.strategic_priority(inferred)

        return 0.3  # Default low-medium

    def _estimate_cognitive_cost(
        self,
        initiative: Initiative,
        user_state: UserState,
    ) -> float:
        """
        Estimate how much cognitive load this will put on Croft.
        """
        base_cost = initiative.metrics.cognitive_burn

        if base_cost == 0.0:
            # Estimate based on type
            type_costs = {
                InitiativeType.CATHEDRAL: 0.7,
                InitiativeType.RESEARCH: 0.6,
                InitiativeType.CREATIVE: 0.5,
                InitiativeType.EMERGENCY: 0.8,
                InitiativeType.INFRASTRUCTURE: 0.4,
                InitiativeType.MAINTENANCE: 0.2,
                InitiativeType.RECOVERY: -0.2,  # Restorative
            }
            base_cost = type_costs.get(initiative.type, 0.3)

        # Adjust for current state
        # When already fatigued, everything costs more
        fatigue_multiplier = 1.0 + user_state.fatigue * 0.5

        # When in flow, matching work costs less
        if user_state.current_mode == CognitiveMode.FLOW:
            if initiative.type in (InitiativeType.CATHEDRAL, InitiativeType.RESEARCH):
                fatigue_multiplier *= 0.7

        return base_cost * fatigue_multiplier

    def _assess_risk(self, initiative: Initiative) -> float:
        """
        Assess the risk level of an initiative.
        """
        if initiative.metrics.risk_level > 0:
            return initiative.metrics.risk_level

        # Infer from type and keywords
        type_risks = {
            InitiativeType.EMERGENCY: 0.7,
            InitiativeType.INFRASTRUCTURE: 0.5,
            InitiativeType.CATHEDRAL: 0.4,
            InitiativeType.RESEARCH: 0.3,
            InitiativeType.CREATIVE: 0.2,
            InitiativeType.MAINTENANCE: 0.1,
            InitiativeType.RECOVERY: 0.0,
        }

        risk = type_risks.get(initiative.type, 0.3)

        # Check for risky keywords
        risky_words = {"deploy", "production", "delete", "remove", "migrate", "upgrade"}
        text = (initiative.name + " " + initiative.description).lower()
        if any(word in text for word in risky_words):
            risk = min(1.0, risk + 0.2)

        return risk

    def _make_decision(
        self,
        initiative: Initiative,
        user_state: UserState,
        strategic_value: float,
        cognitive_cost: float,
        risk_score: float,
    ) -> tuple[CEODecision, str]:
        """
        Make the final decision based on all factors.

        Returns (decision, reasoning).
        """
        # Emergency always executes (if not blocked by protection)
        if initiative.type == InitiativeType.EMERGENCY:
            return CEODecision.EXECUTE, "Emergency - executing immediately"

        # Recovery always executes
        if initiative.type == InitiativeType.RECOVERY:
            return CEODecision.EXECUTE, "Recovery initiative - always approved"

        # Kill low-value distractions
        if strategic_value < 0.1:
            self._daily_kills += 1
            return (
                CEODecision.KILL,
                f"Strategic value too low ({strategic_value:.2f}). Ruthlessly cut as distraction."
            )

        # Check cognitive budget
        if self._daily_cognitive_burn + cognitive_cost > 1.0:
            if strategic_value < 0.5:
                return (
                    CEODecision.DEFER,
                    f"Cognitive budget nearly exhausted. Deferring low-priority work."
                )

        # High risk + low value = kill
        if risk_score > 0.6 and strategic_value < 0.4:
            return (
                CEODecision.KILL,
                f"Risk ({risk_score:.2f}) outweighs value ({strategic_value:.2f}). Rejected."
            )

        # High value + manageable cost = execute
        if strategic_value > 0.5 and cognitive_cost < 0.7:
            self._daily_cognitive_burn += cognitive_cost
            return (
                CEODecision.EXECUTE,
                f"Strategic value ({strategic_value:.2f}) justifies cognitive investment ({cognitive_cost:.2f})."
            )

        # Medium value + user in good state = execute
        if strategic_value > 0.3 and user_state.focus_intensity_factor() > 0.6:
            self._daily_cognitive_burn += cognitive_cost
            return (
                CEODecision.EXECUTE,
                f"Good conditions for moderate-value work. Proceeding."
            )

        # Delegate if it doesn't need Croft
        if cognitive_cost > 0.5 and strategic_value < 0.6:
            return (
                CEODecision.DELEGATE,
                f"High cost ({cognitive_cost:.2f}), moderate value ({strategic_value:.2f}). "
                "Delegating to background agent."
            )

        # Default: execute but track
        self._daily_cognitive_burn += cognitive_cost
        return (
            CEODecision.EXECUTE,
            "Default approval. Proceeding with monitoring."
        )

    def _handle_decision(self, initiative: Initiative, result: CEODecisionResult) -> None:
        """Handle the outcome of a decision."""
        if result.decision == CEODecision.EXECUTE:
            initiative.status = InitiativeStatus.APPROVED
            self._active_initiatives[initiative.id] = initiative

        elif result.decision == CEODecision.DELEGATE:
            initiative.status = InitiativeStatus.APPROVED
            initiative.assigned_to = "hive_queen"
            # Dispatch to HiveHD Queen
            self._delegate_to_hive(initiative, result)

        elif result.decision == CEODecision.DEFER:
            initiative.status = InitiativeStatus.DEFERRED
            self._deferred_queue.append(initiative)

        elif result.decision == CEODecision.KILL:
            initiative.status = InitiativeStatus.KILLED

        elif result.decision == CEODecision.PROTECT:
            initiative.status = InitiativeStatus.BLOCKED
            initiative.blocked_by_founder_protection = True
            initiative.protection_reason = result.reasoning

    def _delegate_to_hive(self, initiative: Initiative, result: CEODecisionResult) -> None:
        """
        Delegate an initiative to the HiveHD Queen.

        The Queen will route it to appropriate Bees for execution.
        """
        if not self.queen or not TaskRequest:
            logger.warning(
                f"Cannot delegate {initiative.id}: HiveHD Queen not available"
            )
            return

        # Map initiative type to task kind
        type_to_kind = {
            InitiativeType.RESEARCH: TaskKind.RESEARCH if TaskKind else None,
            InitiativeType.INFRASTRUCTURE: TaskKind.SYSTEM if TaskKind else None,
            InitiativeType.MAINTENANCE: TaskKind.SYSTEM if TaskKind else None,
            InitiativeType.CREATIVE: TaskKind.LLM if TaskKind else None,
        }
        task_kind = type_to_kind.get(initiative.type)

        # Create task request
        request = TaskRequest(
            instruction=f"{initiative.name}: {initiative.description}",
            kind=task_kind,
            params={
                "initiative_id": initiative.id,
                "tags": initiative.tags,
            },
            context={
                "strategic_value": result.strategic_value,
                "cognitive_cost": result.cognitive_cost,
                "delegated_at": datetime.utcnow().isoformat(),
            },
        )

        # Track for later completion
        self._delegated_tasks[initiative.id] = request.request_id

        logger.info(
            f"Delegated initiative {initiative.id} to HiveHD Queen (request={request.request_id})"
        )

        # Note: Actual dispatch happens asynchronously
        # The Queen.dispatch() would be called from the async context

    async def execute_delegated(self, initiative_id: str) -> Optional[Dict[str, Any]]:
        """
        Execute a delegated initiative via the HiveHD Queen.

        Call this from an async context to actually run the task.
        """
        if initiative_id not in self._delegated_tasks:
            return None

        if not self.queen:
            logger.error("Cannot execute: HiveHD Queen not available")
            return None

        initiative = self._active_initiatives.get(initiative_id)
        if not initiative:
            # Check if it's in delegation tracking
            request_id = self._delegated_tasks.get(initiative_id)
            return {"error": f"Initiative {initiative_id} not found", "request_id": request_id}

        # Create request
        request = TaskRequest(
            instruction=f"{initiative.name}: {initiative.description}",
            params={"initiative_id": initiative_id},
        )

        # Dispatch to Queen
        result = await self.queen.dispatch(request)

        # Update initiative based on result
        if result.success:
            self.complete_initiative(initiative_id, success=True)
        else:
            logger.warning(f"Delegated task failed: {result.error}")

        # Clean up tracking
        del self._delegated_tasks[initiative_id]

        return result.to_dict()

    def _maybe_reset_daily(self) -> None:
        """Reset daily counters if new day."""
        now = datetime.utcnow()
        if now.date() > self._last_reset.date():
            self._daily_cognitive_burn = 0.0
            self._daily_decisions = 0
            self._daily_kills = 0
            self._last_reset = now
            logger.info("CEO: Daily counters reset")

    # =========================================================================
    # Queue Management
    # =========================================================================

    def check_deferred_queue(
        self,
        user_state: Optional[UserState] = None,
    ) -> List[Initiative]:
        """
        Check deferred initiatives and return any that are ready.
        """
        user_state = user_state or get_user_state()
        now = datetime.utcnow()

        ready = []
        still_deferred = []

        for init in self._deferred_queue:
            # Check if defer time has passed
            result = next(
                (r for r in self._decision_history if r.initiative_id == init.id),
                None
            )
            if result and result.defer_until and now >= result.defer_until:
                # Re-evaluate
                init.status = InitiativeStatus.PROPOSED
                ready.append(init)
            else:
                still_deferred.append(init)

        self._deferred_queue = still_deferred
        return ready

    def get_active_initiatives(self) -> List[Initiative]:
        """Get currently active initiatives."""
        return list(self._active_initiatives.values())

    def complete_initiative(self, initiative_id: str, success: bool = True) -> None:
        """Mark an initiative as complete."""
        if initiative_id in self._active_initiatives:
            init = self._active_initiatives.pop(initiative_id)
            init.status = InitiativeStatus.COMPLETED if success else InitiativeStatus.FAILED
            init.completed_at = datetime.utcnow()

            # Update trust
            if success:
                self.covenant.record_success(init.name)
            else:
                self.covenant.record_failure(init.name)

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get summary of today's CEO activity."""
        return {
            "date": datetime.utcnow().date().isoformat(),
            "decisions": self._daily_decisions,
            "kills": self._daily_kills,
            "cognitive_burn": round(self._daily_cognitive_burn, 2),
            "active_initiatives": len(self._active_initiatives),
            "deferred_queue": len(self._deferred_queue),
            "trust_level": self.covenant.trust.trust_points,
            "autonomy_level": self.covenant.trust.current_autonomy_level().name,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current CEO status."""
        return {
            "daily_summary": self.get_daily_summary(),
            "covenant_version": self.covenant.version,
            "teleology_available": self.teleology is not None,
            "recent_decisions": [
                r.to_dict() for r in self._decision_history[-10:]
            ],
        }


# =============================================================================
# Singleton Access
# =============================================================================

_chief_of_staff: Optional[ChiefOfStaff] = None


def get_chief_of_staff() -> ChiefOfStaff:
    """Get the default Chief of Staff."""
    global _chief_of_staff
    if _chief_of_staff is None:
        _chief_of_staff = ChiefOfStaff()
    return _chief_of_staff


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'CEODecisionResult',
    'ChiefOfStaff',
    'get_chief_of_staff',
]
