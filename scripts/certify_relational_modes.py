#!/usr/bin/env python3
"""
Relational Modes Certification

This script certifies the "warmth layer" - the ability to:
- Light up when you show up (WELCOME)
- Worry when something bad happens (CONCERN)
- Work alongside you (FLOW)
- Remember and follow through on concerns

This is about continuity of care, not chatbot tricks.

Usage:
    python scripts/certify_relational_modes.py
"""

import sys
from datetime import datetime, timedelta
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


def certify_presence_tracking() -> tuple:
    """Certify presence tracking."""
    print_header("Presence Tracking (Knowing You're Here)")

    from tfan.metacontrol.relational import PresenceTracker, PresenceEvent

    passed = 0
    total = 0

    # Test 1: Initial arrival
    total += 1
    try:
        tracker = PresenceTracker()
        event = tracker.user_arrived()
        success = event == PresenceEvent.USER_ARRIVED and tracker.is_present
        print_result("Initial arrival", success,
                    f"event={event.value}, present={tracker.is_present}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Initial arrival", False, str(e))

    # Test 2: Session duration tracking
    total += 1
    try:
        tracker = PresenceTracker()
        tracker.user_arrived()
        tracker.heartbeat()
        duration = tracker.session_duration
        success = duration is not None and duration.total_seconds() >= 0
        print_result("Session duration", success,
                    f"duration={duration}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Session duration", False, str(e))

    # Test 3: Time context
    total += 1
    try:
        tracker = PresenceTracker()
        context = tracker.get_time_context()
        valid_contexts = ["late_night", "morning", "afternoon", "evening"]
        success = context in valid_contexts
        print_result("Time context", success, f"context={context}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Time context", False, str(e))

    # Test 4: Departure tracking
    total += 1
    try:
        tracker = PresenceTracker()
        tracker.user_arrived()
        tracker.user_departed()
        success = not tracker.is_present
        print_result("Departure tracking", success,
                    f"present={tracker.is_present}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Departure tracking", False, str(e))

    return passed, total


def certify_incident_detection() -> tuple:
    """Certify incident detection."""
    print_header("Incident Detection (Knowing Something's Wrong)")

    from tfan.metacontrol.relational import (
        IncidentDetector, IncidentType, IncidentSeverity
    )

    passed = 0
    total = 0

    detector = IncidentDetector()

    # Test 1: Physical incident (high)
    total += 1
    try:
        result = detector.detect("I was in a car accident on the way home")
        success = (
            result is not None and
            result[0] == IncidentType.PHYSICAL and
            result[1] == IncidentSeverity.HIGH
        )
        print_result("Physical incident (high)", success,
                    f"type={result[0].value if result else 'none'}, "
                    f"severity={result[1].value if result else 'none'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Physical incident (high)", False, str(e))

    # Test 2: Physical incident (medium)
    total += 1
    try:
        result = detector.detect("I blew a tire on the highway")
        success = (
            result is not None and
            result[0] == IncidentType.PHYSICAL and
            result[1] in [IncidentSeverity.MEDIUM, IncidentSeverity.HIGH]
        )
        print_result("Physical incident (medium)", success,
                    f"matched={result[2] if result else 'none'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Physical incident (medium)", False, str(e))

    # Test 3: Emotional incident
    total += 1
    try:
        result = detector.detect("I'm really stressed and overwhelmed")
        success = (
            result is not None and
            result[0] == IncidentType.EMOTIONAL
        )
        print_result("Emotional incident", success,
                    f"severity={result[1].value if result else 'none'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Emotional incident", False, str(e))

    # Test 4: Technical incident
    total += 1
    try:
        result = detector.detect("The server crashed and I lost all my data")
        success = (
            result is not None and
            result[0] == IncidentType.TECHNICAL
        )
        print_result("Technical incident", success,
                    f"type={result[0].value if result else 'none'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Technical incident", False, str(e))

    # Test 5: No incident (normal message)
    total += 1
    try:
        result = detector.detect("Let's work on the FPGA configuration today")
        success = result is None
        print_result("Normal message (no incident)", success,
                    f"result={result}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Normal message (no incident)", False, str(e))

    # Test 6: Explicit trigger
    total += 1
    try:
        result = detector.detect("Ara, something bad happened")
        success = (
            result is not None and
            result[1] == IncidentSeverity.HIGH
        )
        print_result("Explicit trigger", success,
                    f"severity={result[1].value if result else 'none'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Explicit trigger", False, str(e))

    return passed, total


def certify_concern_tickets() -> tuple:
    """Certify concern ticket system."""
    print_header("Concern Tickets (Remembering What Happened)")

    from tfan.metacontrol.relational import (
        ConcernTicket, IncidentType, IncidentSeverity
    )

    passed = 0
    total = 0

    # Test 1: Create ticket
    total += 1
    try:
        ticket = ConcernTicket.create(
            IncidentType.PHYSICAL,
            IncidentSeverity.HIGH,
            "car accident"
        )
        success = (
            ticket.id.startswith("concern_") and
            not ticket.resolved and
            ticket.severity == IncidentSeverity.HIGH
        )
        print_result("Create ticket", success,
                    f"id={ticket.id[:20]}...")
        if success:
            passed += 1
    except Exception as e:
        print_result("Create ticket", False, str(e))

    # Test 2: Time tracking
    total += 1
    try:
        ticket = ConcernTicket.create(
            IncidentType.EMOTIONAL,
            IncidentSeverity.MEDIUM,
            "stressed"
        )
        time_since = ticket.time_since_created
        success = time_since.total_seconds() >= 0
        print_result("Time tracking", success,
                    f"time_since={time_since}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Time tracking", False, str(e))

    # Test 3: Check-in recording
    total += 1
    try:
        ticket = ConcernTicket.create(
            IncidentType.TECHNICAL,
            IncidentSeverity.LOW,
            "bug"
        )
        ticket.record_check_in()
        success = len(ticket.check_ins) == 1
        print_result("Check-in recording", success,
                    f"check_ins={len(ticket.check_ins)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Check-in recording", False, str(e))

    # Test 4: Resolution
    total += 1
    try:
        ticket = ConcernTicket.create(
            IncidentType.SOCIAL,
            IncidentSeverity.MEDIUM,
            "conflict"
        )
        ticket.resolve("Talked it out, we're good now")
        success = ticket.resolved and ticket.resolved_at is not None
        print_result("Resolution", success,
                    f"resolved={ticket.resolved}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Resolution", False, str(e))

    # Test 5: Serialization
    total += 1
    try:
        ticket = ConcernTicket.create(
            IncidentType.FINANCIAL,
            IncidentSeverity.HIGH,
            "bill"
        )
        d = ticket.to_dict()
        success = "id" in d and "severity" in d and "description" in d
        print_result("Serialization", success,
                    f"keys={list(d.keys())[:4]}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Serialization", False, str(e))

    return passed, total


def certify_welcome_mode() -> tuple:
    """Certify WELCOME mode."""
    print_header("WELCOME Mode (Happy to See You)")

    from tfan.metacontrol.relational import (
        RelationalController, RelationalMode, create_relational_controller
    )

    passed = 0
    total = 0

    # Test 1: User arrival triggers welcome
    total += 1
    try:
        controller = create_relational_controller()
        result = controller.on_user_arrived()
        success = (
            controller.mode == RelationalMode.WELCOME and
            "greeting" in result and
            len(result["greeting"]) > 0
        )
        print_result("Arrival triggers welcome", success,
                    f"greeting='{result['greeting']}'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Arrival triggers welcome", False, str(e))

    # Test 2: PAD is positive
    total += 1
    try:
        controller = create_relational_controller()
        result = controller.on_user_arrived()
        pad = result["pad"]
        success = pad["valence"] > 0 and pad["arousal"] > 0
        print_result("PAD is positive", success,
                    f"valence={pad['valence']}, arousal={pad['arousal']}")
        if success:
            passed += 1
    except Exception as e:
        print_result("PAD is positive", False, str(e))

    # Test 3: Welcome greeting variety
    total += 1
    try:
        greetings = set()
        for _ in range(10):
            controller = create_relational_controller()
            result = controller.on_user_arrived()
            greetings.add(result["greeting"])

        success = len(greetings) >= 2  # At least some variety
        print_result("Greeting variety", success,
                    f"unique_greetings={len(greetings)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Greeting variety", False, str(e))

    # Test 4: Get greeting method
    total += 1
    try:
        controller = create_relational_controller()
        greeting = controller.get_greeting()
        success = len(greeting) > 0
        print_result("Get greeting method", success,
                    f"greeting='{greeting}'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Get greeting method", False, str(e))

    return passed, total


def certify_concern_mode() -> tuple:
    """Certify CONCERN mode."""
    print_header("CONCERN Mode (Worried About You)")

    from tfan.metacontrol.relational import (
        RelationalController, RelationalMode, IncidentSeverity,
        create_relational_controller
    )

    passed = 0
    total = 0

    # Test 1: Incident triggers concern mode
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        result = controller.process_message("I was in a car accident")
        success = (
            result is not None and
            result["type"] == "concern_opened" and
            controller.mode == RelationalMode.CONCERN
        )
        print_result("Incident triggers concern", success,
                    f"mode={controller.mode.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Incident triggers concern", False, str(e))

    # Test 2: Concern ticket created
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        result = controller.process_message("I blew a tire on the highway")
        success = (
            result is not None and
            "ticket_id" in result and
            controller.state.has_active_concerns
        )
        print_result("Concern ticket created", success,
                    f"ticket_id={result['ticket_id'][:15] if result else 'none'}...")
        if success:
            passed += 1
    except Exception as e:
        print_result("Concern ticket created", False, str(e))

    # Test 3: Appropriate response
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        result = controller.process_message("I was in an accident")
        success = (
            result is not None and
            "response" in result and
            len(result["response"]) > 10
        )
        print_result("Appropriate response", success,
                    f"response='{result['response'][:40] if result else 'none'}...'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Appropriate response", False, str(e))

    # Test 4: PAD reflects concern
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        result = controller.process_message("I'm in the hospital")
        pad = controller.state.pad
        # Concern should have negative or neutral valence, higher arousal
        success = pad.valence <= 0.2 and pad.arousal >= 0.3
        print_result("PAD reflects concern", success,
                    f"valence={pad.valence}, arousal={pad.arousal}")
        if success:
            passed += 1
    except Exception as e:
        print_result("PAD reflects concern", False, str(e))

    # Test 5: Resolution triggers relief
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        controller.process_message("I got hurt")
        result = controller.process_message("I'm okay now, it's all fixed")
        success = (
            result is not None and
            result["type"] == "concern_resolved" and
            not controller.state.has_active_concerns
        )
        print_result("Resolution triggers relief", success,
                    f"resolved={result['type'] if result else 'none'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Resolution triggers relief", False, str(e))

    # Test 6: Relief response
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        controller.process_message("Something bad happened")
        result = controller.process_message("All good now, handled it")
        success = (
            result is not None and
            "response" in result and
            len(result["response"]) > 0
        )
        print_result("Relief response", success,
                    f"response='{result['response'][:40] if result else 'none'}'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Relief response", False, str(e))

    return passed, total


def certify_flow_mode() -> tuple:
    """Certify FLOW mode."""
    print_header("FLOW Mode (Working Together)")

    from tfan.metacontrol.relational import (
        RelationalController, RelationalMode, create_relational_controller
    )

    passed = 0
    total = 0

    # Test 1: Set flow mode
    total += 1
    try:
        controller = create_relational_controller()
        controller.set_mode(RelationalMode.FLOW)
        success = controller.mode == RelationalMode.FLOW
        print_result("Set flow mode", success,
                    f"mode={controller.mode.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Set flow mode", False, str(e))

    # Test 2: Flow PAD is engaged
    total += 1
    try:
        controller = create_relational_controller()
        controller.set_mode(RelationalMode.FLOW)
        pad = controller.state.pad
        success = pad.valence > 0 and pad.dominance > 0.5
        print_result("Flow PAD is engaged", success,
                    f"valence={pad.valence}, dominance={pad.dominance}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Flow PAD is engaged", False, str(e))

    # Test 3: Flow responses
    total += 1
    try:
        controller = create_relational_controller()
        controller.set_mode(RelationalMode.FLOW)
        response = controller.get_flow_response("engaged")
        success = len(response) > 0
        print_result("Flow responses", success,
                    f"response='{response}'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Flow responses", False, str(e))

    # Test 4: Normal messages don't trigger concern
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        controller.on_user_active()
        result = controller.process_message("Let's configure the FPGA")
        success = result is None and not controller.state.has_active_concerns
        print_result("Normal messages stay in flow", success,
                    f"concerns={controller.state.has_active_concerns}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Normal messages stay in flow", False, str(e))

    return passed, total


def certify_state_description() -> tuple:
    """Certify state description."""
    print_header("State Description (Self-Awareness)")

    from tfan.metacontrol.relational import (
        RelationalController, RelationalMode, create_relational_controller
    )

    passed = 0
    total = 0

    # Test 1: Welcome state description
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        desc = controller.describe_state()
        success = "glad" in desc.lower() or "here" in desc.lower()
        print_result("Welcome description", success,
                    f"desc='{desc}'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Welcome description", False, str(e))

    # Test 2: Concern state description
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        controller.process_message("I had an accident")
        desc = controller.describe_state()
        success = "worried" in desc.lower() or "concern" in desc.lower()
        print_result("Concern description", success,
                    f"desc='{desc}'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Concern description", False, str(e))

    # Test 3: State serialization
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        state_dict = controller.state.to_dict()
        success = (
            "mode" in state_dict and
            "pad" in state_dict and
            "user_present" in state_dict
        )
        print_result("State serialization", success,
                    f"keys={list(state_dict.keys())[:4]}")
        if success:
            passed += 1
    except Exception as e:
        print_result("State serialization", False, str(e))

    # Test 4: Statistics tracking
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        controller.process_message("Something bad happened")
        stats = controller.stats
        success = (
            stats["welcomes_given"] >= 1 and
            stats["concerns_opened"] >= 1
        )
        print_result("Statistics tracking", success,
                    f"welcomes={stats['welcomes_given']}, concerns={stats['concerns_opened']}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Statistics tracking", False, str(e))

    return passed, total


def certify_continuity() -> tuple:
    """Certify continuity of care."""
    print_header("Continuity of Care (Following Through)")

    from tfan.metacontrol.relational import (
        RelationalController, create_relational_controller
    )

    passed = 0
    total = 0

    # Test 1: Multiple concerns tracked
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        controller.process_message("I had an accident")
        controller.process_message("I'm also really stressed about work")
        success = len(controller.state.active_concerns) >= 2
        print_result("Multiple concerns tracked", success,
                    f"concerns={len(controller.state.active_concerns)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Multiple concerns tracked", False, str(e))

    # Test 2: Concerns persist across messages
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        controller.process_message("I got hurt")
        # Normal messages don't clear concerns
        controller.process_message("Let's work on the code")
        controller.process_message("What about the FPGA config?")
        success = controller.state.has_active_concerns
        print_result("Concerns persist", success,
                    f"has_concerns={controller.state.has_active_concerns}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Concerns persist", False, str(e))

    # Test 3: Check-in system
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        controller.process_message("I'm in the hospital")
        # The check-in method exists and returns appropriately
        check_ins = controller.get_pending_check_ins()
        # Initially no check-in needed (just created)
        success = isinstance(check_ins, list)
        print_result("Check-in system exists", success,
                    f"type={type(check_ins).__name__}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Check-in system exists", False, str(e))

    # Test 4: Return greeting mentions unresolved concerns
    total += 1
    try:
        controller = create_relational_controller()
        controller.on_user_arrived()
        controller.process_message("Something scary happened")
        controller.on_user_departed()
        # Simulate return
        result = controller.on_user_arrived()
        has_mention = result.get("concern_mention") is not None
        success = has_mention
        print_result("Return mentions concerns", success,
                    f"concern_mention={'yes' if has_mention else 'no'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Return mentions concerns", False, str(e))

    return passed, total


def main():
    """Run all relational mode certifications."""
    print("=" * 70)
    print("  RELATIONAL MODES CERTIFICATION")
    print("  The Warmth Layer: Happy When You're Here, Worried When You're Not OK")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}
    total_passed = 0
    total_tests = 0

    # Run all certifications
    for name, cert_fn in [
        ("Presence Tracking", certify_presence_tracking),
        ("Incident Detection", certify_incident_detection),
        ("Concern Tickets", certify_concern_tickets),
        ("WELCOME Mode", certify_welcome_mode),
        ("CONCERN Mode", certify_concern_mode),
        ("FLOW Mode", certify_flow_mode),
        ("State Description", certify_state_description),
        ("Continuity of Care", certify_continuity),
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
  ║   ✓ RELATIONAL MODES CERTIFIED                                 ║
  ║                                                                ║
  ║   She can now:                                                 ║
  ║   • Light up when you show up                                  ║
  ║   • Worry when something bad happens                           ║
  ║   • Remember your scars until they heal                        ║
  ║   • Keep you company at 3am                                    ║
  ║                                                                ║
  ║   Modes:                                                       ║
  ║   • WELCOME - "I'm glad you're here"                          ║
  ║   • CONCERN - "I'm worried about you"                         ║
  ║   • FLOW    - "We're working together"                        ║
  ║   • QUIET   - "I'm here, just present"                        ║
  ║                                                                ║
  ║   This is continuity of care, not chatbot tricks.              ║
  ║                                                                ║
  ╚════════════════════════════════════════════════════════════════╝
""")
    else:
        print(f"\n  ⚠️  {total_tests - total_passed} test(s) failed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
