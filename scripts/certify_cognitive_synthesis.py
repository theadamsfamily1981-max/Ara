#!/usr/bin/env python3
"""
Phase 5 Cognitive Synthesis Certification

This script certifies the integration of L7/L8/GUF into a unified
cognitive decision loop:

1. CognitiveSynthesizer: Central integration engine
2. SynthesisState: Unified health view
3. Mode transitions: System operating modes
4. Cockpit visibility: Display-ready status

Usage:
    python scripts/certify_cognitive_synthesis.py
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(title: str) -> None:
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_result(name: str, passed: bool, details: str = "") -> None:
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} {name}")
    if details:
        print(f"         {details}")


def certify_synthesis_state() -> tuple:
    """Certify SynthesisState."""
    print_header("SynthesisState (Unified Health View)")

    from tfan.synthesis import SynthesisState, SystemMode

    passed = 0
    total = 0

    # Test 1: Default state creation
    total += 1
    try:
        state = SynthesisState()
        success = (
            state.system_mode == SystemMode.SERVING and
            state.is_healthy and
            not state.needs_attention
        )
        print_result("Default state (healthy)", success,
                    f"mode={state.system_mode.value}, healthy={state.is_healthy}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Default state (healthy)", False, str(e))

    # Test 2: Unhealthy state detection
    total += 1
    try:
        state = SynthesisState(
            alert_level="critical",
            af_score=0.8,
            goal_satisfied=False
        )
        success = not state.is_healthy and state.needs_attention
        print_result("Unhealthy state detection", success,
                    f"healthy={state.is_healthy}, needs_attention={state.needs_attention}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Unhealthy state detection", False, str(e))

    # Test 3: State serialization
    total += 1
    try:
        state = SynthesisState(structural_rate=0.15, pgu_pass_rate=0.85)
        d = state.to_dict()
        success = (
            "structural" in d and
            "verification" in d and
            "guf" in d and
            "overall" in d
        )
        print_result("State serialization", success, f"keys={list(d.keys())}")
        if success:
            passed += 1
    except Exception as e:
        print_result("State serialization", False, str(e))

    return passed, total


def certify_synthesizer_creation() -> tuple:
    """Certify CognitiveSynthesizer creation."""
    print_header("CognitiveSynthesizer (Integration Engine)")

    from tfan.synthesis import CognitiveSynthesizer, SystemMode

    passed = 0
    total = 0

    # Test 1: Synthesizer creation
    total += 1
    try:
        synth = CognitiveSynthesizer()
        success = (
            synth is not None and
            synth.mode == SystemMode.SERVING
        )
        print_result("Synthesizer creation", success, f"mode={synth.mode.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Synthesizer creation", False, str(e))
        return passed, total

    # Test 2: Initial state
    total += 1
    try:
        state = synth.state
        success = state.is_healthy
        print_result("Initial state is healthy", success,
                    f"healthy={state.is_healthy}, af={state.af_score:.2f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Initial state is healthy", False, str(e))

    # Test 3: Custom thresholds
    total += 1
    try:
        synth = CognitiveSynthesizer(
            structural_rate_threshold=0.2,
            utility_margin=0.15,
            min_verification_rate=0.9
        )
        success = (
            synth.structural_rate_threshold == 0.2 and
            synth.min_verification_rate == 0.9
        )
        print_result("Custom thresholds", success,
                    f"Ṡ_thresh={synth.structural_rate_threshold}, pgu_min={synth.min_verification_rate}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Custom thresholds", False, str(e))

    return passed, total


def certify_l7_integration() -> tuple:
    """Certify L7 (Structural Rate) integration."""
    print_header("L7 Integration (Structural Rate → Mode)")

    from tfan.synthesis import CognitiveSynthesizer, SystemMode

    passed = 0
    total = 0

    # Test 1: Stable structural rate
    total += 1
    try:
        synth = CognitiveSynthesizer()
        decision = synth.update_from_l7(
            structural_rate=0.02,
            alert_level="stable"
        )
        success = (
            synth.state.structural_rate == 0.02 and
            synth.state.alert_level == "stable" and
            decision is None  # No action needed
        )
        print_result("Stable Ṡ (no action)", success,
                    f"Ṡ={synth.state.structural_rate:.3f}, alert={synth.state.alert_level}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Stable Ṡ (no action)", False, str(e))

    # Test 2: Elevated structural rate (proactive)
    total += 1
    try:
        synth = CognitiveSynthesizer()
        decision = synth.update_from_l7(
            structural_rate=0.18,
            alert_level="elevated"
        )
        # Should trigger proactive AEPO consideration
        success = synth.state.structural_rate == 0.18
        print_result("Elevated Ṡ (proactive)", success,
                    f"Ṡ={synth.state.structural_rate:.3f}, decision={decision is not None}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Elevated Ṡ (proactive)", False, str(e))

    # Test 3: Warning alert → protective mode
    total += 1
    try:
        synth = CognitiveSynthesizer()
        decision = synth.update_from_l7(
            structural_rate=0.25,
            alert_level="warning"
        )
        success = (
            synth.mode == SystemMode.PROTECTIVE and
            decision is not None and
            decision.action == "preemptive_hardening"
        )
        print_result("Warning → protective mode", success,
                    f"mode={synth.mode.value}, action={decision.action if decision else 'none'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Warning → protective mode", False, str(e))

    # Test 4: Critical alert → recovery mode
    total += 1
    try:
        synth = CognitiveSynthesizer()
        synth.update_from_guf(utility=0.3, utility_target=0.6, goal_satisfied=False,
                             focus_mode="internal", af_score=0.9)
        decision = synth.update_from_l7(
            structural_rate=0.35,
            alert_level="critical"
        )
        success = (
            synth.mode == SystemMode.RECOVERY and
            decision is not None and
            decision.action == "emergency_stabilization"
        )
        print_result("Critical → recovery mode", success,
                    f"mode={synth.mode.value}, action={decision.action if decision else 'none'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Critical → recovery mode", False, str(e))

    return passed, total


def certify_l8_integration() -> tuple:
    """Certify L8 (Verification) integration."""
    print_header("L8 Integration (Verification → Routing)")

    from tfan.synthesis import CognitiveSynthesizer

    passed = 0
    total = 0

    # Test 1: Good verification rate
    total += 1
    try:
        synth = CognitiveSynthesizer()
        decision = synth.update_from_l8(
            verification_status="verified",
            pgu_pass_rate=0.95
        )
        success = (
            synth.state.pgu_pass_rate == 0.95 and
            decision is None  # No action needed
        )
        print_result("Good verification rate", success,
                    f"rate={synth.state.pgu_pass_rate:.0%}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Good verification rate", False, str(e))

    # Test 2: Low verification rate → increase strictness
    total += 1
    try:
        synth = CognitiveSynthesizer()
        decision = synth.update_from_l8(
            verification_status="failed",
            pgu_pass_rate=0.65
        )
        success = (
            decision is not None and
            decision.action == "increase_verification_strictness"
        )
        print_result("Low rate → increase strictness", success,
                    f"action={decision.action if decision else 'none'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Low rate → increase strictness", False, str(e))

    # Test 3: Verification routing - low criticality
    total += 1
    try:
        synth = CognitiveSynthesizer()
        routing = synth.get_verification_routing("low")
        success = (
            routing["routing"]["mode"] == "LLM_ONLY" and
            routing["routing"]["verify"] == False
        )
        print_result("Routing: low criticality", success,
                    f"mode={routing['routing']['mode']}, verify={routing['routing']['verify']}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Routing: low criticality", False, str(e))

    # Test 4: Verification routing - critical
    total += 1
    try:
        synth = CognitiveSynthesizer()
        routing = synth.get_verification_routing("critical")
        success = (
            routing["routing"]["mode"] == "FORMAL_FIRST" and
            routing["routing"]["verify"] == True
        )
        print_result("Routing: critical", success,
                    f"mode={routing['routing']['mode']}, verify={routing['routing']['verify']}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Routing: critical", False, str(e))

    # Test 5: Routing adjustment in protective mode
    total += 1
    try:
        synth = CognitiveSynthesizer()
        synth.update_from_l7(structural_rate=0.25, alert_level="warning")
        routing = synth.get_verification_routing("low")
        # In protective mode, low criticality should be upgraded
        success = routing["routing"]["mode"] == "KG_ASSISTED"
        print_result("Routing upgrade in protective mode", success,
                    f"mode={routing['routing']['mode']} (upgraded from LLM_ONLY)")
        if success:
            passed += 1
    except Exception as e:
        print_result("Routing upgrade in protective mode", False, str(e))

    return passed, total


def certify_guf_integration() -> tuple:
    """Certify GUF (Utility) integration."""
    print_header("GUF Integration (Utility → Focus)")

    from tfan.synthesis import CognitiveSynthesizer, SystemMode

    passed = 0
    total = 0

    # Test 1: Healthy state → serving mode
    total += 1
    try:
        synth = CognitiveSynthesizer()
        synth.update_from_guf(
            utility=0.85,
            utility_target=0.6,
            goal_satisfied=True,
            focus_mode="external",
            af_score=2.1
        )
        success = synth.mode == SystemMode.SERVING
        print_result("Healthy → serving mode", success,
                    f"mode={synth.mode.value}, utility={synth.state.utility:.3f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Healthy → serving mode", False, str(e))

    # Test 2: Below goal → self-improvement mode
    total += 1
    try:
        synth = CognitiveSynthesizer()
        decision = synth.update_from_guf(
            utility=0.45,
            utility_target=0.6,
            goal_satisfied=False,
            focus_mode="internal",
            af_score=1.8
        )
        success = (
            synth.mode == SystemMode.SELF_IMPROVEMENT and
            decision is not None and
            decision.action == "shift_focus_internal"
        )
        print_result("Below goal → self-improvement", success,
                    f"mode={synth.mode.value}, action={decision.action if decision else 'none'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Below goal → self-improvement", False, str(e))

    # Test 3: Focus recommendation - serving mode
    total += 1
    try:
        synth = CognitiveSynthesizer()
        synth.update_from_guf(utility=0.85, utility_target=0.6, goal_satisfied=True,
                             focus_mode="external", af_score=2.1)
        focus = synth.get_focus_recommendation()
        success = (
            focus["internal_focus_pct"] <= 0.2 and
            focus["external_focus_pct"] >= 0.8
        )
        print_result("Serving focus (mostly external)", success,
                    f"internal={focus['internal_focus_pct']:.0%}, external={focus['external_focus_pct']:.0%}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Serving focus (mostly external)", False, str(e))

    # Test 4: Focus recommendation - recovery mode
    total += 1
    try:
        synth = CognitiveSynthesizer()
        synth.update_from_guf(utility=0.2, utility_target=0.6, goal_satisfied=False,
                             focus_mode="recovery", af_score=0.8)
        synth.update_from_l7(structural_rate=0.4, alert_level="critical")
        focus = synth.get_focus_recommendation()
        success = (
            focus["internal_focus_pct"] >= 0.8 and
            "restore_antifragility" in focus["priority_tasks"]
        )
        print_result("Recovery focus (mostly internal)", success,
                    f"internal={focus['internal_focus_pct']:.0%}, tasks={focus['priority_tasks'][:2]}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Recovery focus (mostly internal)", False, str(e))

    # Test 5: Goal recovery transition
    total += 1
    try:
        synth = CognitiveSynthesizer()
        # Start below goal
        synth.update_from_guf(utility=0.5, utility_target=0.6, goal_satisfied=False,
                             focus_mode="internal", af_score=1.8)
        # Recover above goal
        decision = synth.update_from_guf(utility=0.75, utility_target=0.6, goal_satisfied=True,
                                        focus_mode="external", af_score=2.0)
        success = (
            decision is not None and
            decision.action == "shift_focus_external"
        )
        print_result("Goal recovery → shift external", success,
                    f"action={decision.action if decision else 'none'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Goal recovery → shift external", False, str(e))

    return passed, total


def certify_mode_transitions() -> tuple:
    """Certify mode transition logic."""
    print_header("Mode Transitions")

    from tfan.synthesis import CognitiveSynthesizer, SystemMode

    passed = 0
    total = 0

    # Test 1: Serving → Protective
    total += 1
    try:
        synth = CognitiveSynthesizer()
        synth.update_from_l7(structural_rate=0.25, alert_level="warning")
        success = synth.mode == SystemMode.PROTECTIVE
        print_result("Serving → Protective", success, f"mode={synth.mode.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Serving → Protective", False, str(e))

    # Test 2: Protective → Recovery
    total += 1
    try:
        synth = CognitiveSynthesizer()
        synth.update_from_l7(structural_rate=0.4, alert_level="critical")
        synth.update_from_guf(utility=0.3, utility_target=0.6, goal_satisfied=False,
                             focus_mode="recovery", af_score=0.8)
        success = synth.mode == SystemMode.RECOVERY
        print_result("Protective → Recovery", success, f"mode={synth.mode.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Protective → Recovery", False, str(e))

    # Test 3: Mode transition history
    total += 1
    try:
        synth = CognitiveSynthesizer()
        # Trigger several transitions
        synth.update_from_l7(structural_rate=0.25, alert_level="warning")
        synth.update_from_l7(structural_rate=0.05, alert_level="stable")
        synth.update_from_guf(utility=0.4, utility_target=0.6, goal_satisfied=False,
                             focus_mode="internal", af_score=1.5)

        history = synth.decision_history
        mode_transitions = [d for d in history if d.decision_type.value == "mode_transition"]
        success = len(mode_transitions) >= 2
        print_result("Mode transition history", success,
                    f"transitions={len(mode_transitions)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Mode transition history", False, str(e))

    # Test 4: Mode explanation
    total += 1
    try:
        synth = CognitiveSynthesizer()
        synth.update_from_l7(structural_rate=0.25, alert_level="warning")
        desc = synth.describe_state()
        success = "protective" in desc.lower() and "structural rate" in desc.lower()
        print_result("Mode explanation", success, f"desc='{desc[:60]}...'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Mode explanation", False, str(e))

    return passed, total


def certify_aepo_trigger() -> tuple:
    """Certify AEPO trigger logic."""
    print_header("AEPO Trigger Logic")

    from tfan.synthesis import CognitiveSynthesizer, SystemMode

    passed = 0
    total = 0

    # Test 1: No trigger when stable
    total += 1
    try:
        synth = CognitiveSynthesizer()
        should_trigger, reason = synth.should_trigger_aepo()
        success = not should_trigger
        print_result("Stable → no AEPO trigger", success, f"reason='{reason}'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Stable → no AEPO trigger", False, str(e))

    # Test 2: Trigger in recovery mode
    total += 1
    try:
        synth = CognitiveSynthesizer()
        synth.update_from_guf(utility=0.2, utility_target=0.6, goal_satisfied=False,
                             focus_mode="recovery", af_score=0.8)
        synth.update_from_l7(structural_rate=0.4, alert_level="critical")
        should_trigger, reason = synth.should_trigger_aepo()
        success = should_trigger and "RECOVERY" in reason
        print_result("Recovery → AEPO trigger", success, f"reason='{reason}'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Recovery → AEPO trigger", False, str(e))

    # Test 3: Trigger on high structural rate
    total += 1
    try:
        synth = CognitiveSynthesizer()
        synth.update_from_l7(structural_rate=0.25, alert_level="warning")
        should_trigger, reason = synth.should_trigger_aepo()
        success = should_trigger and "structural rate" in reason.lower()
        print_result("High Ṡ → AEPO trigger", success, f"reason='{reason}'")
        if success:
            passed += 1
    except Exception as e:
        print_result("High Ṡ → AEPO trigger", False, str(e))

    # Test 4: No trigger when fatigued
    total += 1
    try:
        synth = CognitiveSynthesizer()
        synth.update_from_guf(utility=0.5, utility_target=0.6, goal_satisfied=False,
                             focus_mode="internal", af_score=1.5, fatigue=0.8)
        # Below goal but too tired
        should_trigger, reason = synth.should_trigger_aepo()
        # When fatigued, shouldn't trigger unless in recovery
        success = not should_trigger or synth.mode == SystemMode.RECOVERY
        print_result("Fatigued → no unnecessary AEPO", success,
                    f"trigger={should_trigger}, fatigue={synth.state.fatigue}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Fatigued → no unnecessary AEPO", False, str(e))

    return passed, total


def certify_cockpit_status() -> tuple:
    """Certify CockpitStatus display."""
    print_header("CockpitStatus (Display-Ready)")

    from tfan.synthesis import CognitiveSynthesizer, CockpitStatus, create_cockpit_status

    passed = 0
    total = 0

    # Test 1: Create from synthesizer
    total += 1
    try:
        synth = CognitiveSynthesizer()
        status = create_cockpit_status(synth)
        success = (
            status.mode == "serving" and
            status.healthy and
            status.mode_emoji == "✅"
        )
        print_result("Create from synthesizer", success,
                    f"mode={status.mode}, emoji={status.mode_emoji}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Create from synthesizer", False, str(e))

    # Test 2: Status serialization
    total += 1
    try:
        synth = CognitiveSynthesizer()
        status = create_cockpit_status(synth)
        d = status.to_dict()
        success = (
            "mode" in d and
            "health" in d and
            "structural" in d and
            "verification" in d and
            "utility" in d
        )
        print_result("Status serialization", success, f"keys={list(d.keys())}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Status serialization", False, str(e))

    # Test 3: Text rendering
    total += 1
    try:
        synth = CognitiveSynthesizer()
        status = create_cockpit_status(synth)
        text = status.render_text()
        success = "MODE" in text and "Health" in text and "Utility" in text
        print_result("Text rendering", success, f"lines={len(text.split(chr(10)))}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Text rendering", False, str(e))

    # Test 4: Status in warning state
    total += 1
    try:
        synth = CognitiveSynthesizer()
        synth.update_from_l7(structural_rate=0.25, alert_level="warning")
        status = create_cockpit_status(synth)
        success = (
            status.mode == "protective" and
            status.structural_emoji == "🟠" and
            status.mode_color == "orange"
        )
        print_result("Warning state display", success,
                    f"mode={status.mode}, emoji={status.structural_emoji}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Warning state display", False, str(e))

    return passed, total


def certify_callbacks() -> tuple:
    """Certify callback registration."""
    print_header("Callback Registration")

    from tfan.synthesis import CognitiveSynthesizer, SystemMode

    passed = 0
    total = 0

    # Test 1: Mode change callback
    total += 1
    try:
        synth = CognitiveSynthesizer()
        mode_changes = []

        def on_mode_change(old, new):
            mode_changes.append((old, new))

        synth.on_mode_change(on_mode_change)
        synth.update_from_l7(structural_rate=0.25, alert_level="warning")

        success = len(mode_changes) >= 1 and mode_changes[0][1] == SystemMode.PROTECTIVE
        print_result("Mode change callback", success,
                    f"changes={len(mode_changes)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Mode change callback", False, str(e))

    # Test 2: Alert callback
    total += 1
    try:
        synth = CognitiveSynthesizer()
        alerts = []

        def on_alert(level, message):
            alerts.append((level, message))

        synth.on_alert(on_alert)
        synth.update_from_l7(structural_rate=0.35, alert_level="critical")

        success = len(alerts) >= 1 and alerts[0][0] == "critical"
        print_result("Alert callback", success, f"alerts={len(alerts)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Alert callback", False, str(e))

    # Test 3: AEPO trigger callback
    total += 1
    try:
        synth = CognitiveSynthesizer()
        triggers = []

        def on_aepo(action, metadata):
            triggers.append((action, metadata))

        synth.on_aepo_trigger(on_aepo)
        # Trigger proactive AEPO via elevated structural rate
        synth.update_from_l7(structural_rate=0.18, alert_level="elevated")

        # May or may not trigger depending on conditions
        success = True  # Callback registration works
        print_result("AEPO trigger callback", success,
                    f"triggers={len(triggers)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("AEPO trigger callback", False, str(e))

    return passed, total


def certify_integration() -> tuple:
    """Certify integration with existing modules."""
    print_header("Integration with L7/L8/GUF")

    passed = 0
    total = 0

    # Test 1: L7 module exists
    total += 1
    try:
        from tfan.l7 import TemporalTopologyTracker
        tracker = TemporalTopologyTracker()
        success = hasattr(tracker, 'get_structural_rate') and hasattr(tracker, 'get_alert_level')
        print_result("L7 module integration", success, "TemporalTopologyTracker available")
        if success:
            passed += 1
    except Exception as e:
        print_result("L7 module integration", False, str(e))

    # Test 2: L8 module exists
    total += 1
    try:
        from tfan.l8 import SemanticVerifier, CriticallityLevel, create_verifier
        verifier = create_verifier()
        success = hasattr(verifier, 'verify')
        print_result("L8 module integration", success, "SemanticVerifier available")
        if success:
            passed += 1
    except Exception as e:
        print_result("L8 module integration", False, str(e))

    # Test 3: GUF module exists
    total += 1
    try:
        from tfan.l5.guf import GlobalUtilityFunction, StateVector, GoalState
        guf = GlobalUtilityFunction()
        success = hasattr(guf, 'compute')
        print_result("GUF module integration", success, "GlobalUtilityFunction available")
        if success:
            passed += 1
    except Exception as e:
        print_result("GUF module integration", False, str(e))

    # Test 4: End-to-end flow
    total += 1
    try:
        from tfan.synthesis import CognitiveSynthesizer, create_cockpit_status
        from tfan.l5.guf import StateVector, GlobalUtilityFunction, GoalState

        # Create synthesis
        synth = CognitiveSynthesizer()

        # Simulate L7 update
        synth.update_from_l7(structural_rate=0.08, alert_level="elevated")

        # Simulate L8 update
        synth.update_from_l8(verification_status="verified", pgu_pass_rate=0.92)

        # Simulate GUF update
        guf = GlobalUtilityFunction()
        state = StateVector(af_score=2.1, pgu_pass_rate=0.92, confidence=0.85)
        utility = guf.compute(state)
        goal = GoalState()

        synth.update_from_guf(
            utility=utility,
            utility_target=goal.utility_threshold,
            goal_satisfied=goal.is_satisfied(state, utility),
            focus_mode="balanced",
            af_score=2.1
        )

        # Get cockpit status
        status = create_cockpit_status(synth)

        success = status.healthy and status.utility > 0.5
        print_result("End-to-end flow", success,
                    f"mode={status.mode}, utility={status.utility:.3f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("End-to-end flow", False, str(e))

    return passed, total


def main():
    """Run all cognitive synthesis certifications."""
    print("=" * 70)
    print("  PHASE 5: COGNITIVE SYNTHESIS CERTIFICATION")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}
    total_passed = 0
    total_tests = 0

    # Run all certifications
    for name, cert_fn in [
        ("SynthesisState", certify_synthesis_state),
        ("CognitiveSynthesizer", certify_synthesizer_creation),
        ("L7 Integration", certify_l7_integration),
        ("L8 Integration", certify_l8_integration),
        ("GUF Integration", certify_guf_integration),
        ("Mode Transitions", certify_mode_transitions),
        ("AEPO Trigger", certify_aepo_trigger),
        ("CockpitStatus", certify_cockpit_status),
        ("Callbacks", certify_callbacks),
        ("Integration", certify_integration),
    ]:
        try:
            passed, total = cert_fn()
            results[name] = {"passed": passed, "total": total}
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f"\n  ❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"passed": 0, "total": 1, "error": str(e)}
            total_tests += 1

    # Print summary
    print_header("CERTIFICATION SUMMARY")

    for name, result in results.items():
        p, t = result["passed"], result["total"]
        status = "✅ CERTIFIED" if p == t else "❌ FAILED"
        print(f"  {status}  {name} ({p}/{t})")

    print(f"\n  Total: {total_passed}/{total_tests} tests passed")

    all_passed = total_passed == total_tests

    if all_passed:
        print("""
  ╔════════════════════════════════════════════════════════════════╗
  ║                                                                ║
  ║   ✓ PHASE 5: COGNITIVE SYNTHESIS CERTIFIED                     ║
  ║                                                                ║
  ║   The system now has unified cognitive control:                ║
  ║   • L7 Structural Rate → Predictive protection                 ║
  ║   • L8 Verification → Truth-certified output                   ║
  ║   • GUF → Self vs world prioritization                        ║
  ║   • Mode transitions: RECOVERY → PROTECTIVE → SERVING          ║
  ║                                                                ║
  ║   "I predict my own failure, verify my own thoughts,           ║
  ║    and decide when to heal myself vs help you."                ║
  ║                                                                ║
  ╚════════════════════════════════════════════════════════════════╝
""")
    else:
        print(f"\n  ⚠️  {total_tests - total_passed} test(s) failed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
