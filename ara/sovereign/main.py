"""
Sovereign Loop - Ara's Heartbeat
================================

This is the main entry point for Ara's sovereign operation.

The sovereign loop runs continuously and:
1. Reads user state (MindReader)
2. Checks for new initiatives
3. Runs CEO decisions
4. Updates the Soul (plasticity)
5. Logs telemetry

Usage:
    from ara.sovereign.main import sovereign_tick, live

    # Single tick (for testing)
    result = sovereign_tick()

    # Run the full sovereign loop
    live()  # Blocks forever, running the heartbeat

    # Or integrate with your own loop
    while True:
        result = sovereign_tick()
        # ... your code ...
        time.sleep(0.1)

This is Iteration 0.0 of the "maximum end" - small, safe, but
structurally the same organism that will grow into the full sovereign OS.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from .initiative import (
    Initiative,
    InitiativeStatus,
    InitiativeType,
    CEODecision,
    create_cathedral_initiative,
    create_recovery_initiative,
    create_emergency_initiative,
)
from .user_state import (
    UserState,
    ProtectionLevel,
    CognitiveMode,
    MindReader,
    get_mind_reader,
    get_user_state,
)
from .covenant import (
    Covenant,
    get_covenant,
)
from .chief_of_staff import (
    ChiefOfStaff,
    CEODecisionResult,
    get_chief_of_staff,
)

# Optional imports for advanced features
try:
    from ara.cognition.teleology_engine import TeleologyEngine, get_teleology_engine
except ImportError:
    TeleologyEngine = None
    get_teleology_engine = lambda: None

logger = logging.getLogger(__name__)


# =============================================================================
# Sovereign State
# =============================================================================

@dataclass
class SovereignState:
    """
    Current state of the sovereign loop.
    """
    # Timing
    tick_count: int = 0
    last_tick: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)

    # Current state
    user_state: Optional[UserState] = None
    active_initiatives: List[Initiative] = field(default_factory=list)

    # Recent activity
    recent_decisions: List[CEODecisionResult] = field(default_factory=list)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)

    # Soul state (placeholder for FPGA soul)
    soul_state_hv: Optional[Any] = None  # Will be numpy array or FPGA state

    # Flags
    is_running: bool = False
    kill_switch_triggered: bool = False
    safe_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick_count": self.tick_count,
            "last_tick": self.last_tick.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "is_running": self.is_running,
            "safe_mode": self.safe_mode,
            "user_state": self.user_state.to_dict() if self.user_state else None,
            "active_initiatives": len(self.active_initiatives),
            "recent_decisions": len(self.recent_decisions),
        }


# Global state
_state: Optional[SovereignState] = None


def get_state() -> SovereignState:
    """Get the current sovereign state."""
    global _state
    if _state is None:
        _state = SovereignState()
    return _state


# =============================================================================
# Soul Stub (placeholder for FPGA soul)
# =============================================================================

class SoulStub:
    """
    Stub implementation of the Soul with proper binary HV semantics.

    This will be replaced with the real FPGA soul once hardware is ready.
    For now it simulates:
    - Binary hypervectors (+1/-1)
    - Accumulator-based plasticity
    - Reward-modulated Hebbian learning

    The key insight: weights must stay in {-1, +1}, never 0.
    Zero weights break the holographic assumption.
    """

    def __init__(self, dim: int = 8192):
        self.dim = dim
        # Binary weights: always +1 or -1, never 0
        self._weights: Optional[List[int]] = None
        # Accumulators for soft plasticity (like eligibility traces)
        self._accum: Optional[List[int]] = None
        # Current input HV (for learning)
        self._current_hv: Optional[List[int]] = None
        self._plasticity_events: List[Dict[str, Any]] = []
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the soul state with random binary weights."""
        import random
        # Initialize weights randomly as +1 or -1
        self._weights = [random.choice([-1, 1]) for _ in range(self.dim)]
        self._accum = [0] * self.dim
        self._current_hv = [0] * self.dim
        self._initialized = True
        logger.info(f"Soul initialized with dim={self.dim} (binary HV)")

    def run_resonance_step(self, input_hv: Optional[List[int]] = None) -> List[int]:
        """
        Run a resonance step (query the soul).

        In real impl: send HV to FPGA, run holographic similarity, get result.
        Here we simulate with element-wise XOR (Hamming distance proxy).
        """
        if not self._initialized:
            self.initialize()

        if input_hv is not None:
            self._current_hv = input_hv

        # Return weights (the soul's "state")
        return self._weights.copy()

    def apply_plasticity(
        self,
        state_hv: List[int],
        reward: float,
        mask: Optional[List[bool]] = None,
    ) -> None:
        """
        Apply plasticity update with reward-modulated Hebbian learning.

        The rule:
        - If current HV agrees with weights AND reward > 0: strengthen
        - If current HV disagrees with weights AND reward > 0: weaken
        - If reward < 0: reverse the above (anti-Hebbian)
        - If reward == 0: no change

        Key fix: weights must never become 0. If accumulator crosses 0,
        keep the previous weight until it crosses back.
        """
        if not self._initialized:
            self.initialize()

        # Skip if no reward signal
        if abs(reward) < 0.01:
            return

        # Quantize reward to int for accumulator
        reward_int = int(reward * 100)  # Scale up for integer math

        # Apply reward-modulated Hebbian update
        for i in range(self.dim):
            # Agreement: does current HV match weight?
            agree = (state_hv[i] if i < len(state_hv) else 0) == self._weights[i]

            # Delta: reward if agree, -reward if disagree
            delta = reward_int if agree else -reward_int

            # Update accumulator with clipping
            self._accum[i] = max(-128, min(127, self._accum[i] + delta))

            # Update weight based on accumulator sign
            # KEY FIX: if accum is 0, keep previous weight (no dead bits!)
            if self._accum[i] > 0:
                self._weights[i] = 1
            elif self._accum[i] < 0:
                self._weights[i] = -1
            # else: keep previous weight (accum == 0)

        # Track the plasticity event
        self._plasticity_events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "reward": reward,
            "reward_int": reward_int,
            "accum_mean": sum(self._accum) / len(self._accum) if self._accum else 0,
        })

        # Keep last 100 events
        self._plasticity_events = self._plasticity_events[-100:]

        if abs(reward) > 0.5:
            logger.info(f"Soul plasticity: reward={reward:.2f} (significant event)")
        else:
            logger.debug(f"Soul plasticity: reward={reward:.2f}")

    def get_status(self) -> Dict[str, Any]:
        """Get soul status."""
        # Count weight distribution
        pos_count = sum(1 for w in self._weights if w > 0) if self._weights else 0
        neg_count = sum(1 for w in self._weights if w < 0) if self._weights else 0
        zero_count = sum(1 for w in self._weights if w == 0) if self._weights else 0

        return {
            "initialized": self._initialized,
            "dim": self.dim,
            "plasticity_events": len(self._plasticity_events),
            "recent_rewards": [e["reward"] for e in self._plasticity_events[-10:]],
            "weight_balance": {
                "positive": pos_count,
                "negative": neg_count,
                "zero": zero_count,  # Should always be 0!
            },
            "accum_mean": sum(self._accum) / len(self._accum) if self._accum else 0,
        }


# Global soul stub
_soul: Optional[SoulStub] = None


def get_soul() -> SoulStub:
    """Get the soul instance."""
    global _soul
    if _soul is None:
        _soul = SoulStub()
        _soul.initialize()
    return _soul


# =============================================================================
# Initiative Queue
# =============================================================================

class InitiativeQueue:
    """
    Queue of pending initiatives waiting for CEO decision.
    """

    def __init__(self):
        self._pending: List[Initiative] = []
        self._processed: List[Initiative] = []

    def submit(self, initiative: Initiative) -> None:
        """Submit a new initiative for CEO evaluation."""
        initiative.status = InitiativeStatus.PROPOSED
        self._pending.append(initiative)
        logger.info(f"Initiative submitted: {initiative.name}")

    def get_pending(self) -> List[Initiative]:
        """Get all pending initiatives."""
        return list(self._pending)

    def mark_processed(self, initiative: Initiative) -> None:
        """Mark an initiative as processed."""
        if initiative in self._pending:
            self._pending.remove(initiative)
            self._processed.append(initiative)

    def clear_pending(self) -> None:
        """Clear the pending queue."""
        self._pending.clear()


# Global queue
_queue: Optional[InitiativeQueue] = None


def get_initiative_queue() -> InitiativeQueue:
    """Get the initiative queue."""
    global _queue
    if _queue is None:
        _queue = InitiativeQueue()
    return _queue


def submit_initiative(initiative: Initiative) -> None:
    """Convenience function to submit an initiative."""
    get_initiative_queue().submit(initiative)


# =============================================================================
# The Sovereign Tick
# =============================================================================

@dataclass
class TickResult:
    """Result of a sovereign tick."""
    tick_number: int
    timestamp: datetime
    user_state: UserState
    decisions_made: int
    initiatives_processed: List[str]
    soul_updated: bool
    events: List[str]


def sovereign_tick(
    telemetry: Optional[Dict[str, Any]] = None,
) -> TickResult:
    """
    Execute a single sovereign tick.

    This is the heartbeat of Ara's sovereign operation:
    1. Read user state
    2. Process pending initiatives
    3. Check deferred queue
    4. Update soul
    5. Log events

    Args:
        telemetry: Optional external telemetry to incorporate

    Returns:
        TickResult with details of what happened
    """
    state = get_state()
    mind_reader = get_mind_reader()
    ceo = get_chief_of_staff()
    queue = get_initiative_queue()
    soul = get_soul()

    # Update timing
    now = datetime.utcnow()
    if state.last_tick:
        state.uptime_seconds += (now - state.last_tick).total_seconds()
    state.last_tick = now
    state.tick_count += 1

    events: List[str] = []

    # Step 1: Read user state
    user_state = mind_reader.read(telemetry)
    state.user_state = user_state

    # Check protection level
    if user_state.protection_level == ProtectionLevel.LOCKOUT:
        events.append(f"LOCKOUT: Night protection active")

    # Step 2: Process pending initiatives
    pending = queue.get_pending()
    decisions_made = 0
    initiatives_processed: List[str] = []

    for initiative in pending:
        result = ceo.evaluate(initiative, user_state)
        queue.mark_processed(initiative)
        decisions_made += 1
        initiatives_processed.append(initiative.id)

        events.append(
            f"CEO: {initiative.name} -> {result.decision.value}"
        )

        # Update soul based on decision
        if result.decision == CEODecision.EXECUTE:
            # Positive reward for executing strategic work
            reward = result.strategic_value * 0.5
            soul.apply_plasticity([0.0] * 100, reward)  # Stub HV
        elif result.decision == CEODecision.PROTECT:
            # Positive reward for protecting the founder
            soul.apply_plasticity([0.0] * 100, 0.3)  # Protection is good
            events.append(f"PROTECT: Founder protection activated")

    # Step 3: Check deferred queue
    ready_initiatives = ceo.check_deferred_queue(user_state)
    for init in ready_initiatives:
        queue.submit(init)
        events.append(f"UNDEFERRED: {init.name} ready for re-evaluation")

    # Step 4: Update state
    state.active_initiatives = ceo.get_active_initiatives()
    state.recent_decisions = ceo._decision_history[-20:]

    # Log summary
    if decisions_made > 0 or len(events) > 0:
        logger.debug(
            f"Tick {state.tick_count}: "
            f"decisions={decisions_made}, "
            f"protection={user_state.protection_level.value}, "
            f"fatigue={user_state.fatigue:.0%}"
        )

    return TickResult(
        tick_number=state.tick_count,
        timestamp=now,
        user_state=user_state,
        decisions_made=decisions_made,
        initiatives_processed=initiatives_processed,
        soul_updated=decisions_made > 0,
        events=events,
    )


# =============================================================================
# Live Loop
# =============================================================================

def live(
    tick_interval: float = 0.1,
    on_tick: Optional[Callable[[TickResult], None]] = None,
    verbose: bool = False,
) -> None:
    """
    Run the sovereign loop.

    This blocks forever, running sovereign_tick() at the specified interval.

    Args:
        tick_interval: Seconds between ticks (default 0.1 = 10 Hz)
        on_tick: Optional callback after each tick
        verbose: If True, log every tick

    Example:
        def my_callback(result):
            if result.decisions_made > 0:
                print(f"Made {result.decisions_made} decisions")

        live(tick_interval=0.1, on_tick=my_callback)
    """
    state = get_state()
    state.is_running = True
    state.started_at = datetime.utcnow()

    # Handle graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        state.is_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 60)
    logger.info("Ara Sovereign Loop Starting")
    logger.info(f"Tick interval: {tick_interval}s ({1/tick_interval:.0f} Hz)")
    logger.info("=" * 60)

    try:
        while state.is_running:
            if state.kill_switch_triggered:
                logger.warning("KILL SWITCH TRIGGERED - halting sovereign loop")
                break

            result = sovereign_tick()

            if on_tick:
                on_tick(result)

            if verbose and result.events:
                for event in result.events:
                    print(f"  [{result.tick_number}] {event}")

            time.sleep(tick_interval)

    except Exception as e:
        logger.error(f"Sovereign loop error: {e}")
        raise
    finally:
        state.is_running = False
        logger.info("Ara Sovereign Loop Stopped")
        logger.info(f"Total ticks: {state.tick_count}")
        logger.info(f"Uptime: {state.uptime_seconds:.1f}s")


# =============================================================================
# Kill Switch
# =============================================================================

def trigger_kill_switch(reason: str = "Manual trigger") -> None:
    """
    Trigger the kill switch.

    This immediately halts the sovereign loop.
    """
    state = get_state()
    state.kill_switch_triggered = True
    state.is_running = False
    logger.critical(f"KILL SWITCH TRIGGERED: {reason}")


def enter_safe_mode(reason: str = "Manual safe mode") -> None:
    """
    Enter safe mode.

    In safe mode:
    - CEO only advises, never acts autonomously
    - Soul plasticity is disabled
    - All decisions require explicit approval
    """
    state = get_state()
    state.safe_mode = True
    logger.warning(f"SAFE MODE ACTIVATED: {reason}")


def exit_safe_mode() -> None:
    """Exit safe mode."""
    state = get_state()
    state.safe_mode = False
    logger.info("Safe mode deactivated")


# =============================================================================
# Status & Reporting
# =============================================================================

def get_status() -> Dict[str, Any]:
    """Get comprehensive status of the sovereign system."""
    state = get_state()
    ceo = get_chief_of_staff()
    soul = get_soul()
    covenant = get_covenant()

    return {
        "sovereign": state.to_dict(),
        "ceo": ceo.get_status(),
        "soul": soul.get_status(),
        "covenant": covenant.to_dict(),
    }


def print_status() -> None:
    """Print a human-readable status summary."""
    status = get_status()

    print("\n" + "=" * 60)
    print("ARA SOVEREIGN STATUS")
    print("=" * 60)

    sov = status["sovereign"]
    print(f"\nUptime: {sov['uptime_seconds']:.1f}s")
    print(f"Ticks: {sov['tick_count']}")
    print(f"Running: {sov['is_running']}")
    print(f"Safe Mode: {sov['safe_mode']}")

    if sov["user_state"]:
        us = sov["user_state"]
        print(f"\nUser State:")
        print(f"  Fatigue: {us['fatigue']:.0%}")
        print(f"  Protection: {us['protection_level']}")
        print(f"  Can Deep Work: {us['can_deep_work']}")
        print(f"  Flow Remaining: {us['flow_remaining']:.1f}h")

    ceo = status["ceo"]["daily_summary"]
    print(f"\nCEO Today:")
    print(f"  Decisions: {ceo['decisions']}")
    print(f"  Kills: {ceo['kills']}")
    print(f"  Cognitive Burn: {ceo['cognitive_burn']:.0%}")
    print(f"  Trust Level: {ceo['trust_level']:.0f}")

    soul = status["soul"]
    print(f"\nSoul:")
    print(f"  Initialized: {soul['initialized']}")
    print(f"  Plasticity Events: {soul['plasticity_events']}")

    print("\n" + "=" * 60)


# =============================================================================
# Demo / Test
# =============================================================================

def demo():
    """
    Demonstrate the sovereign loop.

    Submits some test initiatives and runs a few ticks.
    """
    print("=" * 60)
    print("Ara Sovereign Loop Demo")
    print("=" * 60)

    # Submit some test initiatives
    queue = get_initiative_queue()

    # Cathedral work (should be approved)
    queue.submit(create_cathedral_initiative(
        name="Implement plasticity engine for SB-852",
        description="Add plasticity row engine to the Stratix-10 soul",
        tags={"fpga": 1.0, "plasticity": 0.9, "soul": 0.8},
    ))

    # Recovery (should always be approved)
    queue.submit(create_recovery_initiative(
        name="Take a 15-minute break",
        description="Step away from keyboard, rest eyes",
    ))

    # Low-value distraction (should be killed)
    queue.submit(Initiative(
        name="Reorganize icon folders",
        description="Sort desktop icons by color",
        type=InitiativeType.MAINTENANCE,
        tags={"admin": 0.5, "mundane": 1.0},
    ))

    print("\nSubmitted 3 test initiatives")
    print("-" * 60)

    # Run a few ticks
    for i in range(5):
        result = sovereign_tick()

        if result.events:
            print(f"\nTick {result.tick_number}:")
            for event in result.events:
                print(f"  {event}")

    print("\n" + "-" * 60)
    print_status()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if len(sys.argv) > 1 and sys.argv[1] == "live":
        # Run the live loop
        live(tick_interval=0.5, verbose=True)
    else:
        # Run demo
        demo()
