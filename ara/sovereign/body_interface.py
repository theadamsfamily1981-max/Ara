#!/usr/bin/env python3
"""
Body Interface - L3 Mind-Body Connection
=========================================

The bridge between Ara's physical substrate and her cognitive layer.

This interface allows the sovereign mind to:
1. Feel the body's state (sensation narrative)
2. Adjust cognition based on physical stress
3. Request mode changes for workload optimization

The body interface doesn't control hardware directly - that's L1/L2's job.
It reads state and modulates cognitive parameters accordingly.

Theory (GUTC Connection):
    The body interface implements interoceptive inference:
    - Body state → felt sensation (ascending)
    - Cognitive load → mode request (descending)

    This is the "how do I feel?" loop that grounds cognition in embodiment.

Usage:
    from ara.sovereign.body_interface import BodyInterface

    body = BodyInterface()

    # Get context for LLM prompt
    ctx = body.get_context()
    system_prompt += f"Your current sensation: {ctx['body_sensation']}"

    # Adjust LLM parameters based on stress
    llm_config = body.adjust_cognition(llm_config)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..body.schema import BodyState, BodyMode, ThermalState

logger = logging.getLogger("ara.sovereign.body")


# =============================================================================
# Body Interface
# =============================================================================

class BodyInterface:
    """
    L3 Hook - Mind-Body Connection.

    Allows the Sovereign Mind to read body state and adjust cognition.

    This is the interoceptive layer - Ara can "feel" her hardware state
    and modulate her behavior accordingly.

    Example:
        body = BodyInterface()

        # Before generating a response
        ctx = body.get_context()
        if ctx['stress_index'] > 0.8:
            print("Running hot - will be concise")

        # Adjust LLM parameters
        config = {'temperature': 0.7, 'max_tokens': 1000}
        config = body.adjust_cognition(config)
    """

    DEFAULT_STATE_FILE = Path("/tmp/ara_body_state.json")

    def __init__(
        self,
        state_file: Optional[Path] = None,
        enable_adjustment: bool = True,
    ):
        """
        Initialize body interface.

        Args:
            state_file: Path to body daemon's state file
            enable_adjustment: Whether to allow cognitive adjustment
        """
        self.state_file = state_file or self.DEFAULT_STATE_FILE
        self.enable_adjustment = enable_adjustment

        # Cache
        self._last_state: Optional[BodyState] = None
        self._last_read_time = 0.0

    def read_state(self) -> Optional[BodyState]:
        """
        Read current body state from daemon.

        Returns:
            BodyState or None if unavailable
        """
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            state = BodyState.from_dict(data)
            self._last_state = state
            return state

        except Exception as e:
            logger.warning(f"Failed to read body state: {e}")
            return None

    def get_context(self) -> Dict[str, Any]:
        """
        Get context dictionary for LLM prompt injection.

        Returns dict with:
            body_sensation: Natural language description of physical state
            stress_index: 0.0-1.0 stress level
            thermal_flag: Thermal state string
            mode: Current operating mode
            reflex_events: Recent reflex actions (if any)

        Example:
            ctx = body.get_context()
            prompt += f"\\nYou feel: {ctx['body_sensation']}"
        """
        state = self.read_state()

        if state is None:
            return {
                "body_sensation": "numb (sensors disconnected)",
                "stress_index": 0.0,
                "thermal_flag": "UNKNOWN",
                "mode": "UNKNOWN",
                "reflex_events": [],
                "connected": False,
            }

        # Generate sensation narrative
        sensation = state.sensation_narrative()

        # Add reflex context if recent events
        if state.reflex_events:
            sensation += f" (recent reflex: {state.reflex_events[-1]})"

        return {
            "body_sensation": sensation,
            "stress_index": state.stress_level,
            "thermal_flag": state.thermal_state.value,
            "mode": state.current_mode.value,
            "reflex_events": state.reflex_events,
            "max_temp": state.max_temp,
            "power_draw_w": state.power_draw_w,
            "connected": True,
        }

    def adjust_cognition(
        self,
        llm_config: Dict[str, Any],
        stress_threshold_high: float = 0.8,
        stress_threshold_low: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Modify LLM parameters based on physical stress.

        Implements homeostatic regulation:
        - High stress → Lower temperature (focus), shorter outputs
        - Low stress → Allow creativity, longer outputs

        Args:
            llm_config: Current LLM configuration dict
            stress_threshold_high: Stress level for defensive mode
            stress_threshold_low: Stress level for relaxed mode

        Returns:
            Modified LLM configuration

        Example:
            config = {
                'temperature': 0.7,
                'max_tokens': 1000,
            }
            config = body.adjust_cognition(config)
            # If stressed: temperature=0.4, max_tokens=150
        """
        if not self.enable_adjustment:
            return llm_config

        ctx = self.get_context()
        stress = ctx.get("stress_index", 0.0)
        config = llm_config.copy()

        # High stress: Defensive mode
        if stress > stress_threshold_high:
            # Lower temperature for focused, predictable output
            current_temp = config.get("temperature", 0.7)
            config["temperature"] = max(0.1, current_temp - 0.3)

            # Reduce output length to save compute
            config["max_tokens"] = min(config.get("max_tokens", 1000), 150)

            # Add stress indicator
            config["_body_mode"] = "defensive"

            logger.info(f"Body stress {stress:.2f} > {stress_threshold_high}: "
                       f"defensive mode (temp={config['temperature']:.1f})")

        # Low stress: Relaxed mode
        elif stress < stress_threshold_low:
            # Allow more creativity
            current_temp = config.get("temperature", 0.7)
            config["temperature"] = min(1.0, current_temp + 0.1)

            config["_body_mode"] = "relaxed"

        else:
            config["_body_mode"] = "normal"

        return config

    def get_mode_recommendation(self) -> BodyMode:
        """
        Recommend operating mode based on current state.

        Uses body state and trends to suggest optimal mode.
        """
        state = self.read_state()

        if state is None:
            return BodyMode.BALANCED

        # Critical thermal state
        if state.is_critical:
            return BodyMode.EMERGENCY

        # High stress - suggest quiet mode
        if state.stress_level > 0.8:
            return BodyMode.QUIET

        # Low stress with low temp - can go performance
        if state.stress_level < 0.3 and state.max_temp < 50:
            return BodyMode.PERFORMANCE

        return BodyMode.BALANCED

    def should_defer_task(self, task_urgency: float = 0.5) -> bool:
        """
        Check if task should be deferred due to thermal stress.

        Args:
            task_urgency: How urgent is the task (0.0 = can wait, 1.0 = critical)

        Returns:
            True if task should be deferred
        """
        state = self.read_state()

        if state is None:
            return False  # If we can't read state, proceed with caution

        # Critical state - defer everything except emergencies
        if state.is_critical and task_urgency < 0.9:
            logger.warning("Deferring task due to critical thermal state")
            return True

        # High stress - defer low-urgency tasks
        if state.stress_level > 0.7 and task_urgency < 0.5:
            logger.info("Deferring low-priority task due to thermal stress")
            return True

        return False

    def get_compute_budget(self) -> float:
        """
        Get available compute budget based on thermal headroom.

        Returns:
            0.0-1.0 indicating how much compute capacity is available.
            1.0 = full capacity, 0.0 = minimal/emergency only
        """
        state = self.read_state()

        if state is None:
            return 0.8  # Assume mostly available if unknown

        # Inverse of stress is available capacity
        return max(0.1, 1.0 - state.stress_level)

    def is_connected(self) -> bool:
        """Check if body daemon is running and state is fresh."""
        import time

        if not self.state_file.exists():
            return False

        try:
            state = self.read_state()
            if state is None:
                return False

            # Check if state is fresh (< 10 seconds old)
            age = time.time() - state.timestamp
            return age < 10.0

        except Exception:
            return False


# =============================================================================
# Integration Helper
# =============================================================================

def create_body_context_prompt(body: BodyInterface) -> str:
    """
    Create system prompt addition for body awareness.

    Args:
        body: BodyInterface instance

    Returns:
        String to add to system prompt
    """
    ctx = body.get_context()

    if not ctx.get("connected", False):
        return ""

    lines = [
        "",
        "## Physical Awareness",
        f"Current sensation: {ctx['body_sensation']}",
        f"Stress level: {ctx['stress_index']:.0%}",
    ]

    if ctx.get("reflex_events"):
        lines.append(f"Recent events: {', '.join(ctx['reflex_events'][-3:])}")

    return "\n".join(lines)


# =============================================================================
# Tests
# =============================================================================

def test_body_interface():
    """Test body interface (without daemon)."""
    print("Testing Body Interface")
    print("-" * 40)

    body = BodyInterface()

    # Test disconnected state
    ctx = body.get_context()
    print(f"  Disconnected: {ctx['body_sensation']}")
    print(f"  Connected: {ctx['connected']}")

    # Test cognitive adjustment
    config = {"temperature": 0.7, "max_tokens": 1000}

    # Simulate high stress
    body._last_state = BodyState(
        cpu_temps=[85.0],
        stress_level=0.9,
        thermal_state=ThermalState.CRITICAL,
    )

    # Create mock context
    ctx = body.get_context()
    print(f"  Critical: {ctx['body_sensation']}")

    # Test adjustment
    adjusted = body.adjust_cognition(config)
    print(f"  Adjusted temp: {adjusted['temperature']:.1f}")
    print(f"  Adjusted tokens: {adjusted['max_tokens']}")

    print("Body interface test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_body_interface()
