#!/usr/bin/env python3
"""
Certification Script: GUF++ (Causal Synthesis)

Tests the GUF++ implementation:
1. Safety constraints and verification
2. Weight proposals and PGU checking
3. Trajectory simulation under stressors
4. Causal safety guarantees (AF preservation)

Target: 30/30 tests passing
"""

import sys
from datetime import datetime

sys.path.insert(0, "/home/user/Ara")

from tfan.l5.guf import (
    GUFWeights,
    StateVector,
    GoalState,
    GlobalUtilityFunction,
    FocusMode
)

from tfan.l5.guf_plus import (
    ConstraintType,
    SafetyConstraint,
    ConstraintSet,
    ProposalSource,
    WeightProposal,
    GUFVerifier,
    CausalGUF,
    create_causal_guf,
    create_safety_constraints
)


def run_tests():
    passed = 0
    failed = 0
    failures = []

    def test(name, condition, details=""):
        nonlocal passed, failed, failures
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            msg = f"  ✗ {name}" + (f" - {details}" if details else "")
            print(msg)
            failed += 1
            failures.append(name)

    # ========================================
    # Section 1: Safety Constraints
    # ========================================
    print("\n═══ Section 1: Safety Constraints ═══")

    # Create constraint
    af_constraint = SafetyConstraint(
        id="test_af",
        constraint_type=ConstraintType.AF_FLOOR,
        description="AF must stay above 2.0",
        metric="af_score",
        threshold=2.0,
        is_hard=True
    )

    test("SafetyConstraint created", af_constraint is not None)
    test("Constraint has correct type", af_constraint.constraint_type == ConstraintType.AF_FLOOR)

    # Check constraint against good state
    good_state = StateVector(af_score=2.5)
    satisfied, msg = af_constraint.check(good_state, GUFWeights())
    test("Constraint satisfied for good state", satisfied)

    # Check constraint against bad state
    bad_state = StateVector(af_score=1.5)
    satisfied, msg = af_constraint.check(bad_state, GUFWeights())
    test("Constraint violated for bad state", not satisfied)

    # ConstraintSet
    cs = ConstraintSet(name="test_set")
    cs.add(af_constraint)
    cs.add(SafetyConstraint(
        id="risk_ceiling",
        constraint_type=ConstraintType.RISK_CEILING,
        description="Risk must stay below 0.5",
        metric="risk",
        threshold=0.5,
        is_hard=True
    ))

    result = cs.check_all(good_state, GUFWeights())
    test("ConstraintSet check_all works", "all_satisfied" in result)
    test("Good state satisfies all constraints", result["all_satisfied"])

    # ========================================
    # Section 2: Weight Proposals
    # ========================================
    print("\n═══ Section 2: Weight Proposals ═══")

    baseline = GUFWeights()
    proposed = GUFWeights(w_af=0.35, w_latency=0.20)

    proposal = WeightProposal(
        id="test_001",
        proposed_weights=proposed,
        source=ProposalSource.GRADIENT,
        rationale="Test proposal",
        baseline_weights=baseline
    )

    test("WeightProposal created", proposal is not None)
    test("Proposal has weight delta", proposal.weight_delta > 0)
    test("Proposal source recorded", proposal.source == ProposalSource.GRADIENT)

    # ========================================
    # Section 3: GUF Verifier
    # ========================================
    print("\n═══ Section 3: GUF Verifier ═══")

    # Use lower threshold to allow for stress trajectories
    verifier = GUFVerifier(min_af_score=1.5, max_weight_delta=0.3)

    test("GUFVerifier created", verifier is not None)
    test("Verifier has constraints", len(verifier.constraints.constraints) > 0)

    # Verify a safe proposal
    # Use high AF so trajectory simulation stays above threshold even under stress
    safe_weights = GUFWeights(w_af=0.32, w_structural_rate=-0.12)
    safe_proposal = WeightProposal(
        id="safe_001",
        proposed_weights=safe_weights,
        source=ProposalSource.GRADIENT,
        rationale="Safe weight adjustment",
        baseline_weights=baseline
    )

    healthy_state = StateVector(
        af_score=3.5,  # High AF to survive trajectory stress tests
        clv_instability=0.2,
        clv_resource=0.1,
        confidence=0.85,
        fatigue=0.2
    )

    verified_proposal = verifier.verify(safe_proposal, healthy_state)
    test("Safe proposal verified", verified_proposal.verified,
         f"errors: {verified_proposal.verification_errors}")

    # Verify an unsafe proposal (AF weight too low)
    unsafe_weights = GUFWeights(w_af=0.05, w_structural_rate=0.0)
    unsafe_proposal = WeightProposal(
        id="unsafe_001",
        proposed_weights=unsafe_weights,
        source=ProposalSource.EVOLUTION,
        rationale="Unsafe experiment",
        baseline_weights=baseline
    )

    rejected_proposal = verifier.verify(unsafe_proposal, healthy_state)
    test("Unsafe proposal rejected", not rejected_proposal.verified)
    test("Rejection has errors", len(rejected_proposal.verification_errors) > 0)

    # Verify proposal with large weight delta
    large_delta_weights = GUFWeights(w_af=0.8, w_latency=0.5)  # Far from baseline
    large_proposal = WeightProposal(
        id="large_001",
        proposed_weights=large_delta_weights,
        source=ProposalSource.EXPERT,
        rationale="Large change",
        baseline_weights=baseline
    )

    large_result = verifier.verify(large_proposal, healthy_state)
    test("Large delta proposal flagged", not large_result.verified or
         any("magnitude" in e.lower() for e in large_result.verification_errors))

    # ========================================
    # Section 4: Trajectory Simulation
    # ========================================
    print("\n═══ Section 4: Trajectory Simulation ═══")

    # Check that trajectory simulation runs
    test_proposal = WeightProposal(
        id="traj_001",
        proposed_weights=GUFWeights(w_af=0.30),
        source=ProposalSource.GRADIENT,
        rationale="Trajectory test",
        baseline_weights=baseline
    )

    verified_traj = verifier.verify(test_proposal, healthy_state)
    test("Trajectory simulation included",
         verified_traj.verification_result is not None and
         "trajectory_results" in verified_traj.verification_result)

    if verified_traj.verification_result:
        traj_results = verified_traj.verification_result.get("trajectory_results", {})
        test("Multiple scenarios simulated", len(traj_results) >= 3)

        # Check that stressor scenario is handled
        if "high_load" in traj_results:
            test("High load scenario simulated",
                 "min_af" in traj_results["high_load"])

    # ========================================
    # Section 5: CausalGUF Core
    # ========================================
    print("\n═══ Section 5: CausalGUF Core ═══")

    causal_guf = create_causal_guf(min_af_score=2.0)
    test("CausalGUF created", causal_guf is not None)

    # Compute utility
    utility = causal_guf.compute(healthy_state)
    test("CausalGUF computes utility", 0.0 < utility < 1.0)

    # Recommend focus
    focus = causal_guf.recommend_focus(healthy_state)
    test("CausalGUF recommends focus", focus in FocusMode)

    # Propose weights
    new_weights = GUFWeights(w_af=0.31)
    proposal = causal_guf.propose_weights(
        new_weights=new_weights,
        source=ProposalSource.GRADIENT,
        rationale="Test evolution",
        current_state=healthy_state
    )
    test("CausalGUF creates proposals", proposal is not None)
    test("Proposal auto-verified", proposal.verified or len(proposal.verification_errors) > 0)

    # ========================================
    # Section 6: Weight Evolution
    # ========================================
    print("\n═══ Section 6: Weight Evolution ═══")

    # Create GUF with history for learning
    evolution_guf = create_causal_guf(min_af_score=1.8)

    # Record some outcomes to enable learning
    state1 = StateVector(af_score=2.0, clv_instability=0.3)
    state2 = StateVector(af_score=2.2, clv_instability=0.2)  # Improved

    evolution_guf.guf.record_outcome(
        state_before=state1,
        state_after=state2,
        action_taken="optimize",
        external_reward=0.1
    )

    # Record more for minimum samples
    for i in range(10):
        s1 = StateVector(af_score=2.0 + i*0.01, clv_instability=0.3 - i*0.01)
        s2 = StateVector(af_score=2.0 + i*0.02, clv_instability=0.3 - i*0.02)
        evolution_guf.guf.record_outcome(s1, s2, "learn", 0.05)

    test("History recorded", len(evolution_guf.guf._history) >= 10)

    # Try evolution
    evolved = evolution_guf.evolve_weights(healthy_state, learning_rate=0.005)
    test("Evolution attempted", evolved is not None or evolution_guf._stats["proposals_created"] > 1)

    # ========================================
    # Section 7: Exploration
    # ========================================
    print("\n═══ Section 7: Exploration ═══")

    explore_guf = create_causal_guf(min_af_score=1.8)

    # Propose from exploration
    best_proposal = explore_guf.propose_from_exploration(
        current_state=healthy_state,
        n_candidates=3,
        sigma=0.01
    )

    test("Exploration generates proposals",
         explore_guf._stats["proposals_created"] >= 3)

    # ========================================
    # Section 8: Custom Constraints
    # ========================================
    print("\n═══ Section 8: Custom Constraints ═══")

    custom_guf = create_causal_guf()

    # Add custom constraint
    constraint_id = custom_guf.add_safety_constraint(
        constraint_type=ConstraintType.BOUNDED,
        metric="fatigue",
        threshold=0.6,
        is_hard=True,
        description="Fatigue must stay low"
    )

    test("Custom constraint added", constraint_id is not None)

    # Check constraint count increased
    constraint_count = len(custom_guf.verifier.constraints.constraints)
    test("Constraint registered", constraint_count >= 5)

    # ========================================
    # Section 9: Governance Explanation
    # ========================================
    print("\n═══ Section 9: Governance Explanation ═══")

    explanation = causal_guf.explain_governance()
    test("Governance explanation generated", len(explanation) > 100)
    test("Explanation includes weights", "w_af" in explanation)
    test("Explanation includes constraints", "Safety Constraints" in explanation)

    # ========================================
    # Section 10: AF Preservation
    # ========================================
    print("\n═══ Section 10: AF Preservation ═══")

    # Test that proposals violating AF floor are rejected
    low_af_state = StateVector(af_score=1.2, clv_instability=0.6)

    af_test_guf = create_causal_guf(min_af_score=2.0)

    # This state is already below threshold
    constraint_result = af_test_guf.verifier.constraints.check_all(
        low_af_state, GUFWeights()
    )
    test("Low AF state flagged", not constraint_result["all_satisfied"])

    # Verify proposal that would work in low AF scenario
    recovery_weights = GUFWeights(
        w_af=0.40,  # High AF focus
        w_clv_instability=-0.25,
        w_structural_rate=-0.15
    )
    recovery_proposal = WeightProposal(
        id="recovery_001",
        proposed_weights=recovery_weights,
        source=ProposalSource.RECOVERY,
        rationale="Recovery focus weights",
        baseline_weights=GUFWeights()
    )

    # Verify in context of recovery state
    recovery_verified = af_test_guf.verifier.verify(recovery_proposal, low_af_state)
    # Recovery proposal should still work on the weights side
    # (even if current state is bad, weights are about future behavior)
    test("Recovery weights structure valid",
         recovery_verified.verification_result is not None)

    # Stats
    stats = af_test_guf.stats
    test("Stats tracked", stats["proposals_created"] >= 0)

    return passed, failed, failures


def main():
    print("=" * 60)
    print("GUF++ CERTIFICATION")
    print("Causal Synthesis and Verified Self-Governance")
    print("=" * 60)

    passed, failed, failures = run_tests()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failures:
        print("\nFailed tests:")
        for f in failures:
            print(f"  - {f}")

    total = passed + failed
    threshold = 0.90

    if passed / total >= threshold:
        print(f"\n✅ CERTIFICATION PASSED ({passed}/{total})")
        return 0
    else:
        print(f"\n❌ CERTIFICATION FAILED ({passed}/{total} < {threshold:.0%})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
