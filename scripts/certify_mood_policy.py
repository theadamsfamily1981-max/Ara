#!/usr/bin/env python3
"""
Mood Policy Certification

This script certifies the mood policy system that determines
HOW Ara responds, not just WHAT she says.

Key principle: NO MARVIN MODE
- Playful/warm when safe
- Calm/precise when it matters
- Never depressed just because risk went up

Usage:
    python scripts/certify_mood_policy.py
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


def certify_personas() -> tuple:
    """Certify persona profiles."""
    print_header("Persona Profiles (No Marvin!)")

    from tfan.metacontrol import PERSONAS, get_persona, list_personas

    passed = 0
    total = 0

    # Test 1: All expected personas exist
    total += 1
    expected = [
        "exploratory_creative", "lab_buddy", "calm_lab_partner",
        "supportive_creative", "focused_engineer",
        "guardian_engineer", "calm_stabilizer"
    ]
    try:
        available = list_personas()
        missing = set(expected) - set(available)
        success = len(missing) == 0
        print_result("All personas defined", success,
                    f"{len(available)} personas, missing={list(missing)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("All personas defined", False, str(e))

    # Test 2: Persona retrieval works
    total += 1
    try:
        persona = get_persona("exploratory_creative")
        success = (
            persona is not None and
            persona.name == "Ara the Tinkerer" and
            persona.humor_style != "none"
        )
        print_result("Persona retrieval", success,
                    f"name='{persona.name if persona else 'None'}'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Persona retrieval", False, str(e))

    # Test 3: No "depressed" personas
    total += 1
    try:
        depressed_indicators = ["sad", "depressed", "hopeless", "grim", "doom"]
        # Phrases that negate the depressed indicators
        negations = ["not doom", "not sad", "not grim", "not depressed"]

        has_marvin = False
        marvin_persona = None

        for pid, persona in PERSONAS.items():
            desc_lower = (persona.description + " " + persona.tone).lower()

            # Check for negations first - if it says "not X", that's good
            desc_negated = desc_lower
            for neg in negations:
                desc_negated = desc_negated.replace(neg, "")

            # Now check for actual negative indicators
            if any(ind in desc_negated for ind in depressed_indicators):
                has_marvin = True
                marvin_persona = pid
                break

        success = not has_marvin
        print_result("No Marvin personas", success,
                    f"{'All clear!' if success else f'Found: {marvin_persona}'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("No Marvin personas", False, str(e))

    # Test 4: Playful personas exist for stable state
    total += 1
    try:
        playful_count = sum(
            1 for p in PERSONAS.values()
            if "playful" in p.tone.lower() or p.humor_style not in ["none", ""]
        )
        success = playful_count >= 3
        print_result("Playful personas available", success,
                    f"{playful_count} personas allow humor")
        if success:
            passed += 1
    except Exception as e:
        print_result("Playful personas available", False, str(e))

    # Test 5: Guardian personas are calm, not doom-y
    total += 1
    try:
        guardian = get_persona("guardian_engineer")
        stabilizer = get_persona("calm_stabilizer")
        success = (
            "calm" in guardian.tone.lower() or "steady" in guardian.tone.lower()
        ) and (
            "calm" in stabilizer.tone.lower() or "grounding" in stabilizer.tone.lower()
        )
        print_result("Serious personas are calm (not grim)", success,
                    f"guardian={guardian.tone}, stabilizer={stabilizer.tone}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Serious personas are calm (not grim)", False, str(e))

    return passed, total


def certify_intent_classifier() -> tuple:
    """Certify intent classification."""
    print_header("Intent Classifier")

    from tfan.metacontrol import IntentClassifier, UserIntent

    passed = 0
    total = 0

    classifier = IntentClassifier()

    # Test 1: Creative intent
    total += 1
    try:
        intent = classifier.classify("Write me a poem about FPGAs")
        success = intent == UserIntent.CREATIVE
        print_result("Creative intent", success, f"intent={intent.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Creative intent", False, str(e))

    # Test 2: Critical intent
    total += 1
    try:
        intent = classifier.classify("Deploy this to production hardware")
        success = intent == UserIntent.CRITICAL
        print_result("Critical intent", success, f"intent={intent.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Critical intent", False, str(e))

    # Test 3: Precise intent
    total += 1
    try:
        intent = classifier.classify("Configure the scheduler parameters")
        success = intent == UserIntent.PRECISE
        print_result("Precise intent", success, f"intent={intent.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Precise intent", False, str(e))

    # Test 4: Exploratory intent
    total += 1
    try:
        intent = classifier.classify("What if we tried a different architecture?")
        success = intent == UserIntent.EXPLORATORY
        print_result("Exploratory intent", success, f"intent={intent.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Exploratory intent", False, str(e))

    # Test 5: Casual intent
    total += 1
    try:
        intent = classifier.classify("Hey, how are you doing today?")
        success = intent == UserIntent.CASUAL
        print_result("Casual intent", success, f"intent={intent.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Casual intent", False, str(e))

    return passed, total


def certify_mood_selection() -> tuple:
    """Certify mood policy selection."""
    print_header("Mood Policy Selection")

    from tfan.metacontrol import (
        select_mood_policy, AlertLevel, GUFMode, UserIntent,
        SafetyMode, EntropyLevel
    )

    passed = 0
    total = 0

    # Test 1: STABLE + CREATIVE → full phoenix mode
    total += 1
    try:
        policy = select_mood_policy(
            AlertLevel.STABLE,
            GUFMode.EXTERNAL,
            UserIntent.CREATIVE
        )
        success = (
            policy.persona == "exploratory_creative" and
            policy.temperature_mult >= 1.2 and
            policy.entropy_level == EntropyLevel.HIGH and
            policy.allow_jokes == True
        )
        print_result("STABLE + CREATIVE → phoenix mode", success,
                    f"persona={policy.persona}, temp={policy.temperature_mult}")
        if success:
            passed += 1
    except Exception as e:
        print_result("STABLE + CREATIVE → phoenix mode", False, str(e))

    # Test 2: STABLE + CASUAL → lab buddy
    total += 1
    try:
        policy = select_mood_policy(
            AlertLevel.STABLE,
            GUFMode.BALANCED,
            UserIntent.CASUAL
        )
        success = (
            policy.persona == "lab_buddy" and
            policy.allow_jokes == True
        )
        print_result("STABLE + CASUAL → lab buddy", success,
                    f"persona={policy.persona}")
        if success:
            passed += 1
    except Exception as e:
        print_result("STABLE + CASUAL → lab buddy", False, str(e))

    # Test 3: STABLE + PRECISE → calm lab partner
    total += 1
    try:
        policy = select_mood_policy(
            AlertLevel.STABLE,
            GUFMode.EXTERNAL,
            UserIntent.PRECISE
        )
        success = (
            policy.persona == "calm_lab_partner" and
            policy.safety_mode == SafetyMode.KG_ASSISTED
        )
        print_result("STABLE + PRECISE → lab partner", success,
                    f"persona={policy.persona}, safety={policy.safety_mode.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("STABLE + PRECISE → lab partner", False, str(e))

    # Test 4: ELEVATED + CREATIVE → supportive (not shutdown)
    total += 1
    try:
        policy = select_mood_policy(
            AlertLevel.ELEVATED,
            GUFMode.BALANCED,
            UserIntent.CREATIVE
        )
        success = (
            policy.persona == "supportive_creative" and
            policy.allow_jokes == True and
            policy.temperature_mult >= 1.0
        )
        print_result("ELEVATED + CREATIVE → supportive (not shutdown)", success,
                    f"persona={policy.persona}, jokes={policy.allow_jokes}")
        if success:
            passed += 1
    except Exception as e:
        print_result("ELEVATED + CREATIVE → supportive (not shutdown)", False, str(e))

    # Test 5: WARNING + PRECISE → focused but jokes allowed
    total += 1
    try:
        policy = select_mood_policy(
            AlertLevel.WARNING,
            GUFMode.INTERNAL,
            UserIntent.PRECISE
        )
        success = (
            policy.persona == "focused_engineer" and
            policy.allow_jokes == True and  # Light humor still OK
            policy.safety_mode == SafetyMode.FORMAL_FIRST
        )
        print_result("WARNING + PRECISE → focused (jokes OK)", success,
                    f"persona={policy.persona}, jokes={policy.allow_jokes}")
        if success:
            passed += 1
    except Exception as e:
        print_result("WARNING + PRECISE → focused (jokes OK)", False, str(e))

    # Test 6: CRITICAL + CRITICAL → guardian, no jokes
    total += 1
    try:
        policy = select_mood_policy(
            AlertLevel.CRITICAL,
            GUFMode.RECOVERY,
            UserIntent.CRITICAL
        )
        success = (
            policy.persona == "guardian_engineer" and
            policy.allow_jokes == False and
            policy.safety_mode == SafetyMode.PGU_VERIFIED
        )
        print_result("CRITICAL + CRITICAL → guardian mode", success,
                    f"persona={policy.persona}, jokes={policy.allow_jokes}")
        if success:
            passed += 1
    except Exception as e:
        print_result("CRITICAL + CRITICAL → guardian mode", False, str(e))

    # Test 7: CRITICAL + CREATIVE → override with message
    total += 1
    try:
        policy = select_mood_policy(
            AlertLevel.CRITICAL,
            GUFMode.RECOVERY,
            UserIntent.CREATIVE
        )
        success = (
            policy.persona == "calm_stabilizer" and
            policy.override_message is not None and
            "stabilize" in policy.override_message.lower()
        )
        print_result("CRITICAL + CREATIVE → gentle override", success,
                    f"override={policy.override_message is not None}")
        if success:
            passed += 1
    except Exception as e:
        print_result("CRITICAL + CREATIVE → gentle override", False, str(e))

    # Test 8: Verify NO policy has Marvin-like settings
    total += 1
    try:
        # Test all combinations - none should result in depressed mood
        marvin_found = False
        for alert in AlertLevel:
            for guf in GUFMode:
                for intent in UserIntent:
                    policy = select_mood_policy(alert, guf, intent)
                    profile_id = policy.persona
                    # Even serious personas should not be "depressed"
                    if "depress" in policy.rationale.lower() or "doom" in policy.rationale.lower():
                        marvin_found = True
                        break

        success = not marvin_found
        print_result("No Marvin in any combination", success,
                    "All policies are calm, not sad")
        if success:
            passed += 1
    except Exception as e:
        print_result("No Marvin in any combination", False, str(e))

    return passed, total


def certify_mood_controller() -> tuple:
    """Certify MoodController integration."""
    print_header("MoodController Integration")

    from tfan.metacontrol import (
        MoodController, create_mood_controller,
        AlertLevel, GUFMode, UserIntent
    )

    passed = 0
    total = 0

    # Test 1: Controller creation
    total += 1
    try:
        controller = create_mood_controller()
        success = controller is not None
        print_result("Controller creation", success)
        if success:
            passed += 1
    except Exception as e:
        print_result("Controller creation", False, str(e))
        return passed, total

    # Test 2: Policy selection from query
    total += 1
    try:
        policy = controller.select_policy(
            "Write me a weird poem about circuits",
            alert=AlertLevel.STABLE,
            guf=GUFMode.EXTERNAL
        )
        success = (
            policy is not None and
            policy.persona == "exploratory_creative"
        )
        print_result("Policy from query", success,
                    f"persona={policy.persona}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Policy from query", False, str(e))

    # Test 3: Current persona access
    total += 1
    try:
        persona = controller.current_persona
        success = persona is not None and persona.name == "Ara the Tinkerer"
        print_result("Current persona access", success,
                    f"name={persona.name if persona else 'None'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Current persona access", False, str(e))

    # Test 4: Greeting generation
    total += 1
    try:
        greeting = controller.get_greeting()
        success = len(greeting) > 0 and "break" in greeting.lower()  # Tinkerer greeting
        print_result("Greeting generation", success,
                    f"greeting='{greeting[:40]}...'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Greeting generation", False, str(e))

    # Test 5: Mood description
    total += 1
    try:
        desc = controller.describe_current_mood()
        success = "Tinkerer" in desc and "phoenix" in desc.lower()
        print_result("Mood description", success,
                    f"desc='{desc[:50]}...'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Mood description", False, str(e))

    # Test 6: Intent override
    total += 1
    try:
        policy = controller.select_policy(
            "hey what's up",  # Would normally be CASUAL
            intent_override=UserIntent.CRITICAL
        )
        success = policy.safety_mode.value in ["pgu_verified", "formal_first"]
        print_result("Intent override", success,
                    f"safety={policy.safety_mode.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Intent override", False, str(e))

    # Test 7: Statistics tracking
    total += 1
    try:
        stats = controller.stats
        success = (
            stats["policies_selected"] >= 2 and
            "by_persona" in stats and
            len(stats["by_persona"]) > 0
        )
        print_result("Statistics tracking", success,
                    f"selected={stats['policies_selected']}, personas={len(stats['by_persona'])}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Statistics tracking", False, str(e))

    return passed, total


def certify_l3_l6_integration() -> tuple:
    """Certify integration with L3 and L6."""
    print_header("L3/L6 Integration Helpers")

    from tfan.metacontrol import (
        select_mood_policy, mood_to_l3_params, mood_to_l6_routing,
        AlertLevel, GUFMode, UserIntent, SafetyMode
    )

    passed = 0
    total = 0

    # Test 1: L3 params for creative mode
    total += 1
    try:
        policy = select_mood_policy(AlertLevel.STABLE, GUFMode.EXTERNAL, UserIntent.CREATIVE)
        l3_params = mood_to_l3_params(policy)
        success = (
            l3_params["temperature_mult"] >= 1.2 and
            l3_params["allow_exploration"] == True and
            l3_params["alpha_base"] > 0
        )
        print_result("L3 params (creative)", success,
                    f"temp={l3_params['temperature_mult']}, explore={l3_params['allow_exploration']}")
        if success:
            passed += 1
    except Exception as e:
        print_result("L3 params (creative)", False, str(e))

    # Test 2: L3 params for guardian mode
    total += 1
    try:
        policy = select_mood_policy(AlertLevel.CRITICAL, GUFMode.RECOVERY, UserIntent.CRITICAL)
        l3_params = mood_to_l3_params(policy)
        success = (
            l3_params["temperature_mult"] < 0.8 and
            l3_params["allow_exploration"] == False
        )
        print_result("L3 params (guardian)", success,
                    f"temp={l3_params['temperature_mult']}, explore={l3_params['allow_exploration']}")
        if success:
            passed += 1
    except Exception as e:
        print_result("L3 params (guardian)", False, str(e))

    # Test 3: L6 routing for creative
    total += 1
    try:
        policy = select_mood_policy(AlertLevel.STABLE, GUFMode.EXTERNAL, UserIntent.CREATIVE)
        l6_routing = mood_to_l6_routing(policy)
        success = (
            l6_routing["reasoning_mode"] == "LLM_ONLY" and
            l6_routing["verify_output"] == False
        )
        print_result("L6 routing (creative)", success,
                    f"mode={l6_routing['reasoning_mode']}, verify={l6_routing['verify_output']}")
        if success:
            passed += 1
    except Exception as e:
        print_result("L6 routing (creative)", False, str(e))

    # Test 4: L6 routing for guardian
    total += 1
    try:
        policy = select_mood_policy(AlertLevel.CRITICAL, GUFMode.RECOVERY, UserIntent.CRITICAL)
        l6_routing = mood_to_l6_routing(policy)
        success = (
            l6_routing["reasoning_mode"] == "PGU_VERIFIED" and
            l6_routing["verify_output"] == True
        )
        print_result("L6 routing (guardian)", success,
                    f"mode={l6_routing['reasoning_mode']}, verify={l6_routing['verify_output']}")
        if success:
            passed += 1
    except Exception as e:
        print_result("L6 routing (guardian)", False, str(e))

    return passed, total


def certify_synthesis_integration() -> tuple:
    """Certify integration with cognitive synthesis."""
    print_header("Synthesis Integration")

    passed = 0
    total = 0

    # Test 1: Synthesis state → Alert level mapping
    total += 1
    try:
        from tfan.synthesis import CognitiveSynthesizer, SystemMode
        from tfan.metacontrol import AlertLevel, GUFMode, select_mood_policy, UserIntent

        synth = CognitiveSynthesizer()

        # Map synthesis alert to mood alert
        alert_map = {
            "stable": AlertLevel.STABLE,
            "elevated": AlertLevel.ELEVATED,
            "warning": AlertLevel.WARNING,
            "critical": AlertLevel.CRITICAL
        }

        synth_alert = synth.state.alert_level
        mood_alert = alert_map.get(synth_alert, AlertLevel.STABLE)

        success = mood_alert == AlertLevel.STABLE  # Default state
        print_result("Synthesis → Mood alert mapping", success,
                    f"synth={synth_alert} → mood={mood_alert.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Synthesis → Mood alert mapping", False, str(e))

    # Test 2: GUF mode → Mood GUF mapping
    total += 1
    try:
        from tfan.synthesis import CognitiveSynthesizer
        from tfan.metacontrol import GUFMode

        synth = CognitiveSynthesizer()

        # Map synthesis focus mode to mood GUF
        guf_map = {
            "recovery": GUFMode.RECOVERY,
            "internal": GUFMode.INTERNAL,
            "balanced": GUFMode.BALANCED,
            "external": GUFMode.EXTERNAL
        }

        synth_focus = synth.state.focus_mode
        mood_guf = guf_map.get(synth_focus, GUFMode.EXTERNAL)

        success = mood_guf == GUFMode.EXTERNAL  # Default state
        print_result("Synthesis → Mood GUF mapping", success,
                    f"synth={synth_focus} → mood={mood_guf.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Synthesis → Mood GUF mapping", False, str(e))

    # Test 3: End-to-end: Synthesis state → Mood policy
    total += 1
    try:
        from tfan.synthesis import CognitiveSynthesizer
        from tfan.metacontrol import (
            MoodController, AlertLevel, GUFMode, create_mood_controller
        )

        synth = CognitiveSynthesizer()
        mood = create_mood_controller()

        # Simulate synthesis state update
        synth.update_from_l7(structural_rate=0.25, alert_level="warning")

        # Map to mood
        alert_map = {"stable": AlertLevel.STABLE, "elevated": AlertLevel.ELEVATED,
                    "warning": AlertLevel.WARNING, "critical": AlertLevel.CRITICAL}
        guf_map = {"recovery": GUFMode.RECOVERY, "internal": GUFMode.INTERNAL,
                   "balanced": GUFMode.BALANCED, "external": GUFMode.EXTERNAL}

        alert = alert_map[synth.state.alert_level]
        guf = guf_map.get(synth.state.focus_mode, GUFMode.EXTERNAL)

        policy = mood.select_policy(
            "Help me with this critical deployment",
            alert=alert,
            guf=guf
        )

        # WARNING + critical-ish query should give focused engineer
        success = policy.persona in ["focused_engineer", "supportive_creative"]
        print_result("End-to-end synthesis → mood", success,
                    f"alert={alert.value}, persona={policy.persona}")
        if success:
            passed += 1
    except Exception as e:
        print_result("End-to-end synthesis → mood", False, str(e))

    return passed, total


def main():
    """Run all mood policy certifications."""
    print("=" * 70)
    print("  MOOD POLICY CERTIFICATION")
    print("  No Marvin Mode - Playful When Safe, Calm When It Matters")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}
    total_passed = 0
    total_tests = 0

    # Run all certifications
    for name, cert_fn in [
        ("Personas", certify_personas),
        ("Intent Classifier", certify_intent_classifier),
        ("Mood Selection", certify_mood_selection),
        ("MoodController", certify_mood_controller),
        ("L3/L6 Integration", certify_l3_l6_integration),
        ("Synthesis Integration", certify_synthesis_integration),
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
  ║   ✓ MOOD POLICY CERTIFIED - NO MARVIN MODE                     ║
  ║                                                                ║
  ║   Personas:                                                    ║
  ║   • 🔥 exploratory_creative - Full phoenix, weird ideas        ║
  ║   • 🤝 lab_buddy - Friend in the lab                           ║
  ║   • 🔧 calm_lab_partner - Helpful, occasionally witty          ║
  ║   • 💡 supportive_creative - Playful but cautious              ║
  ║   • 🎯 focused_engineer - Careful, still human                 ║
  ║   • 🛡️ guardian_engineer - Landing the plane, not doom-y       ║
  ║   • ⚓ calm_stabilizer - The adult in the room                 ║
  ║                                                                ║
  ║   "Playful when safe. Calm when it matters. Never sad robot."  ║
  ║                                                                ║
  ╚════════════════════════════════════════════════════════════════╝
""")
    else:
        print(f"\n  ⚠️  {total_tests - total_passed} test(s) failed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
