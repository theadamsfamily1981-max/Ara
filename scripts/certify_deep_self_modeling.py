#!/usr/bin/env python3
"""
Phase 7: Deep Self-Modeling Certification

This script certifies the GUF (Global Utility Function) and Self-Improvement
Scheduler that enable the system to:

1. Learn its own values (what signals predict good/bad futures)
2. Decide when to focus on self-improvement vs external tasks
3. Track self-improvement episodes and their outcomes
4. Explain its scheduling decisions

Key components:
- GUFWeights: Learnable weights for utility computation
- GoalState: Defines "good enough" to serve the world
- GlobalUtilityFunction: U(state) = Σ w_i * s_i
- SelfImprovementScheduler: Allocates resources between self and world

Usage:
    python scripts/certify_deep_self_modeling.py
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


def certify_state_vector() -> tuple:
    """Certify StateVector representation."""
    print_header("StateVector (System State Representation)")

    from tfan.l5.guf import StateVector

    passed = 0
    total = 0

    # Test 1: Default creation
    total += 1
    try:
        state = StateVector()
        success = state.af_score == 1.0 and state.risk == 0.0
        print_result("Default StateVector", success,
                    f"af={state.af_score}, risk={state.risk:.2f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Default StateVector", False, str(e))

    # Test 2: To vector conversion
    total += 1
    try:
        state = StateVector(
            af_score=2.21,
            clv_instability=0.3,
            clv_resource=0.2,
            clv_structural=0.1
        )
        vec = state.to_vector()
        success = len(vec) == 13 and vec[0] == 2.21
        print_result("Vector conversion", success, f"len={len(vec)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Vector conversion", False, str(e))

    # Test 3: Risk computation
    total += 1
    try:
        state = StateVector(
            clv_instability=0.6,
            clv_resource=0.4,
            clv_structural=0.3
        )
        # risk = 0.5*0.6 + 0.3*0.4 + 0.2*0.3 = 0.3 + 0.12 + 0.06 = 0.48
        success = abs(state.risk - 0.48) < 0.01
        print_result("Risk computation", success, f"risk={state.risk:.2f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Risk computation", False, str(e))

    return passed, total


def certify_guf_weights() -> tuple:
    """Certify GUFWeights (learnable values)."""
    print_header("GUFWeights (Learnable Values)")

    from tfan.l5.guf import GUFWeights

    passed = 0
    total = 0

    # Test 1: Default weights
    total += 1
    try:
        weights = GUFWeights()
        vec = weights.to_vector()
        success = len(vec) == 13 and weights.w_af == 0.30
        print_result("Default weights", success, f"w_af={weights.w_af}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Default weights", False, str(e))

    # Test 2: Bounded update
    total += 1
    try:
        weights = GUFWeights()
        # Try to push w_af beyond bounds
        weights.update("w_af", 1.0)  # Should hit upper bound 0.5
        success = weights.w_af == 0.5
        print_result("Bounded update", success, f"w_af clamped to {weights.w_af}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Bounded update", False, str(e))

    # Test 3: Perturbation for exploration
    total += 1
    try:
        weights = GUFWeights()
        perturbed = weights.perturb(sigma=0.05)
        # Should be different but within bounds
        success = perturbed.w_af != weights.w_af and 0.1 <= perturbed.w_af <= 0.5
        print_result("Weight perturbation", success,
                    f"original={weights.w_af:.3f}, perturbed={perturbed.w_af:.3f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Weight perturbation", False, str(e))

    return passed, total


def certify_goal_state() -> tuple:
    """Certify GoalState (defines "good enough")."""
    print_header("GoalState (Goal Definition)")

    from tfan.l5.guf import GoalState, StateVector

    passed = 0
    total = 0

    # Test 1: Default goal
    total += 1
    try:
        goal = GoalState()
        success = goal.min_af_score == 1.8 and goal.max_risk == 0.4
        print_result("Default goal", success,
                    f"min_af={goal.min_af_score}, max_risk={goal.max_risk}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Default goal", False, str(e))

    # Test 2: Goal satisfaction check (satisfied)
    total += 1
    try:
        goal = GoalState()
        state = StateVector(
            af_score=2.21,
            clv_instability=0.1,
            clv_resource=0.1,
            clv_structural=0.1,
            confidence=0.9,
            fatigue=0.2,
            pgu_pass_rate=0.95
        )
        satisfied = goal.is_satisfied(state, 0.7)  # utility 0.7 > threshold 0.6
        success = satisfied == True
        print_result("Goal satisfied (good state)", success)
        if success:
            passed += 1
    except Exception as e:
        print_result("Goal satisfied (good state)", False, str(e))

    # Test 3: Goal satisfaction check (not satisfied)
    total += 1
    try:
        goal = GoalState()
        state = StateVector(
            af_score=1.2,  # Below min 1.8
            clv_instability=0.6,
            clv_resource=0.4,
            clv_structural=0.3,
            confidence=0.4,  # Below min 0.6
            fatigue=0.8,  # Above max 0.7
            pgu_pass_rate=0.85  # Below min 0.9
        )
        satisfied = goal.is_satisfied(state, 0.4)
        violations = goal.get_violations(state, 0.4)
        success = satisfied == False and len(violations) >= 3
        print_result("Goal not satisfied (bad state)", success,
                    f"violations={len(violations)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Goal not satisfied (bad state)", False, str(e))

    # Test 4: Distance to goal
    total += 1
    try:
        goal = GoalState()
        state = StateVector(
            af_score=1.5,  # Below 1.8
            confidence=0.5  # Below 0.6
        )
        distance = goal.distance_to_goal(state, 0.5)
        success = distance > 0
        print_result("Distance to goal", success, f"distance={distance:.2f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Distance to goal", False, str(e))

    return passed, total


def certify_guf() -> tuple:
    """Certify GlobalUtilityFunction."""
    print_header("GlobalUtilityFunction (GUF)")

    from tfan.l5.guf import GlobalUtilityFunction, StateVector, GUFWeights

    passed = 0
    total = 0

    # Test 1: GUF creation
    total += 1
    try:
        guf = GlobalUtilityFunction()
        success = guf.weights is not None and guf.goal is not None
        print_result("GUF creation", success)
        if success:
            passed += 1
    except Exception as e:
        print_result("GUF creation", False, str(e))

    # Test 2: Utility computation (good state)
    total += 1
    try:
        guf = GlobalUtilityFunction()
        good_state = StateVector(
            af_score=2.5,
            throughput=0.9,
            clv_instability=0.1,
            confidence=0.9,
            pgu_pass_rate=1.0
        )
        utility = guf.compute(good_state)
        success = utility > 0.6
        print_result("Utility (good state)", success, f"utility={utility:.3f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Utility (good state)", False, str(e))

    # Test 3: Utility computation (bad state)
    total += 1
    try:
        bad_state = StateVector(
            af_score=0.8,
            throughput=0.3,
            clv_instability=0.8,
            clv_resource=0.7,
            confidence=0.3,
            fatigue=0.9,
            pgu_pass_rate=0.6
        )
        utility = guf.compute(bad_state)
        good_utility = guf.compute(good_state)
        success = utility < good_utility  # Bad state should have lower utility than good
        print_result("Utility (bad state)", success, f"utility={utility:.3f} < {good_utility:.3f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Utility (bad state)", False, str(e))

    # Test 4: Utility with breakdown
    total += 1
    try:
        breakdown = guf.compute_with_breakdown(good_state)
        success = (
            "utility" in breakdown and
            "contributions" in breakdown and
            "goal_satisfied" in breakdown
        )
        print_result("Utility breakdown", success,
                    f"utility={breakdown['utility']:.3f}, goal_ok={breakdown['goal_satisfied']}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Utility breakdown", False, str(e))

    # Test 5: Focus recommendation
    total += 1
    try:
        from tfan.l5.guf import FocusMode
        focus_good = guf.recommend_focus(good_state)
        focus_bad = guf.recommend_focus(bad_state)
        success = focus_good in [FocusMode.EXTERNAL, FocusMode.BALANCED] and \
                  focus_bad in [FocusMode.INTERNAL, FocusMode.RECOVERY]
        print_result("Focus recommendation", success,
                    f"good→{focus_good.value}, bad→{focus_bad.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Focus recommendation", False, str(e))

    # Test 6: State explanation
    total += 1
    try:
        explanation = guf.explain_state(good_state)
        success = "utility" in explanation.lower() and len(explanation) > 50
        print_result("State explanation", success, f"'{explanation[:60]}...'")
        if success:
            passed += 1
    except Exception as e:
        print_result("State explanation", False, str(e))

    return passed, total


def certify_learning() -> tuple:
    """Certify GUF learning from outcomes."""
    print_header("GUF Learning")

    from tfan.l5.guf import GlobalUtilityFunction, StateVector

    passed = 0
    total = 0

    # Test 1: Record outcome
    total += 1
    try:
        guf = GlobalUtilityFunction()
        state_before = StateVector(af_score=1.5, clv_instability=0.4)
        state_after = StateVector(af_score=2.0, clv_instability=0.2)

        record = guf.record_outcome(state_before, state_after, "aepo_optimization", 0.5)
        success = record["delta_utility"] > 0
        print_result("Record outcome", success,
                    f"Δu={record['delta_utility']:.3f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Record outcome", False, str(e))

    # Test 2: Multiple outcomes for learning
    total += 1
    try:
        # Generate 15 samples
        for i in range(14):
            s1 = StateVector(af_score=1.0 + i * 0.1, clv_instability=0.5 - i * 0.02)
            s2 = StateVector(af_score=1.2 + i * 0.1, clv_instability=0.4 - i * 0.02)
            guf.record_outcome(s1, s2, f"opt_{i}", 0.3)

        success = len(guf._history) >= 15
        print_result("Accumulate history", success, f"samples={len(guf._history)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Accumulate history", False, str(e))

    # Test 3: Learn from history
    total += 1
    try:
        original_w_af = guf.weights.w_af
        result = guf.learn_from_history(learning_rate=0.05, min_samples=10)
        success = result["status"] == "updated"
        print_result("Learn from history", success,
                    f"w_af: {original_w_af:.3f} → {guf.weights.w_af:.3f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Learn from history", False, str(e))

    return passed, total


def certify_scheduler() -> tuple:
    """Certify SelfImprovementScheduler."""
    print_header("SelfImprovementScheduler")

    from tfan.l5.guf import (
        SelfImprovementScheduler, GlobalUtilityFunction,
        StateVector, FocusMode
    )

    passed = 0
    total = 0

    # Test 1: Scheduler creation
    total += 1
    try:
        scheduler = SelfImprovementScheduler()
        success = scheduler.guf is not None
        print_result("Scheduler creation", success)
        if success:
            passed += 1
    except Exception as e:
        print_result("Scheduler creation", False, str(e))

    # Test 2: Decision for good state (external focus)
    total += 1
    try:
        good_state = StateVector(
            af_score=2.5,
            clv_instability=0.1,
            confidence=0.9,
            fatigue=0.2,
            pgu_pass_rate=0.98
        )
        decision = scheduler.decide(good_state)
        success = decision.external_allocation > decision.internal_allocation
        print_result("Good state → external focus", success,
                    f"internal={decision.internal_allocation:.0%}, external={decision.external_allocation:.0%}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Good state → external focus", False, str(e))

    # Test 3: Decision for bad state (internal focus)
    total += 1
    try:
        bad_state = StateVector(
            af_score=1.2,
            clv_instability=0.6,
            confidence=0.4,
            fatigue=0.8,
            pgu_pass_rate=0.7
        )
        decision = scheduler.decide(bad_state)
        success = decision.internal_allocation > decision.external_allocation
        print_result("Bad state → internal focus", success,
                    f"internal={decision.internal_allocation:.0%}, external={decision.external_allocation:.0%}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Bad state → internal focus", False, str(e))

    # Test 4: Recovery mode for critical state
    total += 1
    try:
        critical_state = StateVector(
            af_score=0.7,  # Below 1.0 threshold
            clv_instability=0.9
        )
        decision = scheduler.decide(critical_state)
        success = decision.focus_mode == FocusMode.RECOVERY
        print_result("Critical state → recovery", success,
                    f"focus={decision.focus_mode.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Critical state → recovery", False, str(e))

    # Test 5: Priority tasks
    total += 1
    try:
        decision = scheduler.decide(bad_state)
        success = len(decision.priority_tasks) > 0
        print_result("Priority tasks generated", success,
                    f"tasks={decision.priority_tasks[:2]}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Priority tasks generated", False, str(e))

    # Test 6: Decision explanation
    total += 1
    try:
        decision = scheduler.decide(good_state)
        explanation = scheduler.explain_decision(decision)
        success = "Focus mode" in explanation and len(explanation) > 50
        print_result("Decision explanation", success, f"'{explanation[:60]}...'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Decision explanation", False, str(e))

    # Test 7: Self-improvement episode tracking
    total += 1
    try:
        ep_id = scheduler.start_improvement_episode(bad_state, "aepo_optimization")
        # Simulate improvement
        improved_state = StateVector(
            af_score=1.8,
            clv_instability=0.3,
            confidence=0.7,
            fatigue=0.5,
            pgu_pass_rate=0.9
        )
        result = scheduler.end_improvement_episode(ep_id, improved_state, success=True)
        success = "utility_delta" in result
        print_result("Improvement episode tracking", success,
                    f"Δu={result.get('utility_delta', 0):.3f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Improvement episode tracking", False, str(e))

    return passed, total


def certify_integration() -> tuple:
    """Certify integration with existing components."""
    print_header("Integration with Phase 6 Components")

    passed = 0
    total = 0

    # Test 1: GUF + SelfState integration
    total += 1
    try:
        from tfan.cognition import SelfState, create_self_state
        from tfan.l5.guf import StateVector, GlobalUtilityFunction

        self_state = create_self_state()
        self_state.update_pad(valence=0.2, arousal=0.3, dominance=0.5)
        self_state.update_clv(instability=0.2, resource=0.1, structural=0.1)
        self_state.confidence = 0.85

        # Convert to StateVector
        state = StateVector(
            af_score=2.0,
            clv_instability=self_state.clv.instability,
            clv_resource=self_state.clv.resource,
            clv_structural=self_state.clv.structural,
            confidence=self_state.confidence,
            fatigue=self_state.fatigue,
            mood_valence=self_state.pad.valence
        )

        guf = GlobalUtilityFunction()
        utility = guf.compute(state)
        success = 0 < utility < 1
        print_result("GUF + SelfState", success, f"utility={utility:.3f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("GUF + SelfState", False, str(e))

    # Test 2: GUF + GoalVector alignment
    total += 1
    try:
        from tfan.cognition import GoalVector as CogGoal
        from tfan.l5.guf import GoalState, GUFWeights

        cog_goal = CogGoal(stability_weight=0.4, latency_weight=0.3)
        guf_weights = GUFWeights(w_af=0.4, w_latency=0.3)

        # Verify alignment
        success = abs(cog_goal.stability_weight - guf_weights.w_af) < 0.01
        print_result("GoalVector ↔ GUFWeights alignment", success,
                    f"stability={cog_goal.stability_weight}, w_af={guf_weights.w_af}")
        if success:
            passed += 1
    except Exception as e:
        print_result("GoalVector ↔ GUFWeights alignment", False, str(e))

    # Test 3: Scheduler + Deliberation depth
    total += 1
    try:
        from tfan.l5.guf import SelfImprovementScheduler, StateVector, FocusMode

        scheduler = SelfImprovementScheduler()

        # High risk state should trigger more internal focus
        high_risk = StateVector(clv_instability=0.7, af_score=1.3)
        decision = scheduler.decide(high_risk)

        # Map focus to deliberation depth
        depth_map = {
            FocusMode.RECOVERY: 3,
            FocusMode.INTERNAL: 2,
            FocusMode.BALANCED: 1,
            FocusMode.EXTERNAL: 1
        }
        depth = depth_map[decision.focus_mode]
        success = decision.focus_mode in [FocusMode.INTERNAL, FocusMode.RECOVERY] and depth >= 2
        print_result("Scheduler → Deliberation depth", success,
                    f"focus={decision.focus_mode.value}, depth={depth}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Scheduler → Deliberation depth", False, str(e))

    return passed, total


def certify_presets() -> tuple:
    """Certify GUF presets."""
    print_header("GUF Presets")

    from tfan.l5.guf import create_guf, StateVector

    passed = 0
    total = 0

    # Test 1: Balanced preset
    total += 1
    try:
        guf = create_guf("balanced")
        success = guf.weights.w_af == 0.30
        print_result("Balanced preset", success, f"w_af={guf.weights.w_af}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Balanced preset", False, str(e))

    # Test 2: Safety-first preset
    total += 1
    try:
        guf = create_guf("safety_first")
        success = guf.weights.w_af == 0.35 and guf.goal.min_af_score == 2.0
        print_result("Safety-first preset", success,
                    f"w_af={guf.weights.w_af}, min_af={guf.goal.min_af_score}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Safety-first preset", False, str(e))

    # Test 3: Performance preset
    total += 1
    try:
        guf = create_guf("performance")
        success = guf.weights.w_throughput == 0.25 and guf.goal.max_risk == 0.5
        print_result("Performance preset", success,
                    f"w_throughput={guf.weights.w_throughput}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Performance preset", False, str(e))

    # Test 4: Preset behavior difference
    total += 1
    try:
        guf_safe = create_guf("safety_first")
        guf_perf = create_guf("performance")

        # Same state, different utilities
        state = StateVector(af_score=1.6, throughput=0.9, clv_instability=0.35)
        u_safe = guf_safe.compute(state)
        u_perf = guf_perf.compute(state)

        # Performance preset should give higher utility for high throughput
        success = u_perf != u_safe  # Different utilities for same state
        print_result("Preset behavior difference", success,
                    f"safety={u_safe:.3f}, perf={u_perf:.3f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Preset behavior difference", False, str(e))

    return passed, total


def main():
    """Run all Phase 7 certifications."""
    print("=" * 70)
    print("  PHASE 7: DEEP SELF-MODELING CERTIFICATION")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}
    total_passed = 0
    total_tests = 0

    # Run all certifications
    for name, cert_fn in [
        ("StateVector", certify_state_vector),
        ("GUFWeights", certify_guf_weights),
        ("GoalState", certify_goal_state),
        ("GlobalUtilityFunction", certify_guf),
        ("GUF Learning", certify_learning),
        ("SelfImprovementScheduler", certify_scheduler),
        ("Integration", certify_integration),
        ("Presets", certify_presets),
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
  ║   ✓ PHASE 7 DEEP SELF-MODELING CERTIFIED                       ║
  ║                                                                ║
  ║   The system demonstrates autonomous self-management:          ║
  ║   • GUF: Learned utility function U(state) = Σ w_i * s_i       ║
  ║   • GoalState: Defines "good enough" to serve the world        ║
  ║   • Learning: Weights adapt from experience                    ║
  ║   • Scheduler: Chooses self vs external focus                  ║
  ║   • Episodes: Tracks self-improvement outcomes                 ║
  ║                                                                ║
  ║   "It knows when to work on itself vs serve the world"         ║
  ║                                                                ║
  ╚════════════════════════════════════════════════════════════════╝
""")
    else:
        print(f"\n  ⚠️  {total_tests - total_passed} test(s) failed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
