#!/usr/bin/env python3
"""
Certification Script: L9 Autonomy (Staged Hardware Self-Modification)

Tests the L9 autonomy implementation:
1. Autonomy stages (ADVISOR, SANDBOX, PARTIAL)
2. Permission checks at each stage
3. Stage progression based on track record
4. Safety gates and veto capabilities
5. Integration with Autosynth

Target: 30/30 tests passing
"""

import sys
from datetime import datetime, timedelta

sys.path.insert(0, "/home/user/Ara")

from tfan.hardware.l9_autonomy import (
    AutonomyStage,
    KernelCriticality,
    AutonomyPolicy,
    AutonomyState,
    AutonomyController,
    L9AutosynthIntegration,
    create_autonomy_controller,
    create_integration
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
    # Section 1: Basic Types
    # ========================================
    print("\n═══ Section 1: Basic Types ═══")

    test("AutonomyStage enum exists", AutonomyStage.ADVISOR is not None)
    test("Three stages defined",
         len(AutonomyStage) == 3)
    test("ADVISOR is first stage",
         AutonomyStage.ADVISOR.value == "advisor")

    test("KernelCriticality enum exists", KernelCriticality.NON_CRITICAL is not None)
    test("Four criticality levels",
         len(KernelCriticality) == 4)
    test("SAFETY is most critical",
         KernelCriticality.SAFETY.value == "safety")

    # ========================================
    # Section 2: Autonomy Policy
    # ========================================
    print("\n═══ Section 2: Autonomy Policy ═══")

    policy = AutonomyPolicy()
    test("AutonomyPolicy created", policy is not None)
    test("Default stage is ADVISOR",
         policy.current_stage == AutonomyStage.ADVISOR)
    test("Stage B requirements set",
         policy.stage_b_min_proposals > 0)
    test("Stage C requirements set",
         policy.stage_c_min_deployments > 0)
    test("Sandbox limits set",
         policy.sandbox_max_kernels > 0)
    test("Fallback required by default",
         policy.require_fallback_path)

    # ========================================
    # Section 3: Autonomy State
    # ========================================
    print("\n═══ Section 3: Autonomy State ═══")

    state = AutonomyState(
        stage=AutonomyStage.ADVISOR,
        stage_since=datetime.now() - timedelta(days=5),
        proposals_created=20,
        proposals_verified=18,
        proposals_rejected=2
    )

    test("AutonomyState created", state is not None)
    test("Success rate computed",
         abs(state.success_rate - 0.9) < 0.01)
    test("Days at stage computed",
         state.days_at_stage >= 4)  # May be 4 or 5 depending on timing
    test("State serializes", "stage" in state.to_dict())

    # ========================================
    # Section 4: Autonomy Controller - Stage A
    # ========================================
    print("\n═══ Section 4: Controller - Stage A ═══")

    controller = create_autonomy_controller(start_stage=AutonomyStage.ADVISOR)
    test("Controller created", controller is not None)
    test("Starts at ADVISOR", controller.stage == AutonomyStage.ADVISOR)

    # Can always propose
    test("Can propose at ADVISOR", controller.can_propose())

    # Cannot auto-deploy at ADVISOR
    can_deploy, reason = controller.can_auto_deploy(
        KernelCriticality.NON_CRITICAL,
        {"LUT": 5000}
    )
    test("Cannot auto-deploy at ADVISOR", not can_deploy)
    test("Reason mentions Stage A", "Stage A" in reason or "ADVISOR" in reason)

    # Record proposals
    for i in range(5):
        controller.record_proposal(verified=True)
    test("Proposals recorded", controller.state.proposals_verified == 5)

    # ========================================
    # Section 5: Stage Progression A → B
    # ========================================
    print("\n═══ Section 5: Stage Progression A → B ═══")

    # Create controller with relaxed policy for testing
    relaxed_policy = AutonomyPolicy(
        stage_b_min_proposals=5,
        stage_b_min_days=0,  # No day requirement for test
        stage_b_success_rate=0.8
    )
    prog_controller = AutonomyController(
        policy=relaxed_policy,
        start_stage=AutonomyStage.ADVISOR
    )

    # Record enough proposals
    for i in range(6):
        prog_controller.record_proposal(verified=True)

    test("Progressed to SANDBOX",
         prog_controller.stage == AutonomyStage.SANDBOX)

    # ========================================
    # Section 6: Controller - Stage B (Sandbox)
    # ========================================
    print("\n═══ Section 6: Controller - Stage B ═══")

    sandbox_controller = AutonomyController(
        policy=relaxed_policy,
        start_stage=AutonomyStage.SANDBOX
    )

    # Can auto-deploy non-critical at SANDBOX
    can_deploy, reason = sandbox_controller.can_auto_deploy(
        KernelCriticality.NON_CRITICAL,
        {"LUT": 5000}
    )
    test("Can auto-deploy NON_CRITICAL at SANDBOX", can_deploy)

    # Cannot auto-deploy critical at SANDBOX
    can_deploy, reason = sandbox_controller.can_auto_deploy(
        KernelCriticality.CRITICAL,
        {"LUT": 5000}
    )
    test("Cannot auto-deploy CRITICAL at SANDBOX", not can_deploy)

    # Resource limits enforced
    can_deploy, reason = sandbox_controller.can_auto_deploy(
        KernelCriticality.NON_CRITICAL,
        {"LUT": 50000}  # Exceeds sandbox limit
    )
    test("Resource limits enforced at SANDBOX", not can_deploy)

    # Kernel tracking
    sandbox_controller.record_kernel_active()
    test("Kernel active tracked",
         sandbox_controller.state.current_sandbox_kernels == 1)
    sandbox_controller.record_kernel_retired()
    test("Kernel retired tracked",
         sandbox_controller.state.current_sandbox_kernels == 0)

    # ========================================
    # Section 7: Stage Progression B → C
    # ========================================
    print("\n═══ Section 7: Stage Progression B → C ═══")

    partial_policy = AutonomyPolicy(
        stage_c_min_deployments=3,
        stage_c_min_days=0,
        stage_c_incident_free_days=0
    )
    partial_controller = AutonomyController(
        policy=partial_policy,
        start_stage=AutonomyStage.SANDBOX
    )

    # Record successful deployments
    for i in range(4):
        partial_controller.record_deployment(success=True, sandbox=True)

    test("Progressed to PARTIAL",
         partial_controller.stage == AutonomyStage.PARTIAL)

    # ========================================
    # Section 8: Controller - Stage C (Partial)
    # ========================================
    print("\n═══ Section 8: Controller - Stage C ═══")

    full_controller = AutonomyController(
        policy=partial_policy,
        start_stage=AutonomyStage.PARTIAL
    )

    # Can auto-deploy standard at PARTIAL
    can_deploy, reason = full_controller.can_auto_deploy(
        KernelCriticality.STANDARD,
        {"LUT": 20000}
    )
    test("Can auto-deploy STANDARD at PARTIAL", can_deploy)

    # Still cannot auto-deploy SAFETY kernels
    can_deploy, reason = full_controller.can_auto_deploy(
        KernelCriticality.SAFETY,
        {"LUT": 5000}
    )
    test("Cannot auto-deploy SAFETY at PARTIAL", not can_deploy)

    # ========================================
    # Section 9: Safety Features
    # ========================================
    print("\n═══ Section 9: Safety Features ═══")

    safety_controller = AutonomyController(start_stage=AutonomyStage.SANDBOX)

    # Veto callback
    veto_called = False
    def veto_callback(action_id):
        nonlocal veto_called
        veto_called = True
        return True  # Veto the action

    safety_controller.set_veto_callback(veto_callback)
    vetoed = safety_controller.request_veto_window("test_action")
    test("Veto callback called", veto_called)
    test("Veto returned True", vetoed)

    # Emergency stop
    safety_controller.emergency_stop()
    test("Emergency stop reverts to ADVISOR",
         safety_controller.stage == AutonomyStage.ADVISOR)

    # Force stage
    safety_controller.force_stage(AutonomyStage.SANDBOX, "Test override")
    test("Force stage works",
         safety_controller.stage == AutonomyStage.SANDBOX)

    # ========================================
    # Section 10: Reporting
    # ========================================
    print("\n═══ Section 10: Reporting ═══")

    report_controller = create_autonomy_controller()

    # Record some activity
    for i in range(3):
        report_controller.record_proposal(verified=True)

    status = report_controller.get_status()
    test("Status contains stage", "stage" in status)
    test("Status contains state", "state" in status)
    test("Status contains progression", "progression" in status)

    explanation = report_controller.explain_autonomy()
    test("Explanation generated", len(explanation) > 100)
    test("Explanation mentions stage", "ADVISOR" in explanation or "Stage" in explanation)

    # ========================================
    # Section 11: Integration
    # ========================================
    print("\n═══ Section 11: Integration ═══")

    integration = create_integration()
    test("Integration created", integration is not None)

    # Record proposal
    integration.on_proposal_verified(
        "prop_001",
        verified=True,
        criticality=KernelCriticality.NON_CRITICAL
    )

    # Check deployment recommendation
    rec = integration.get_deployment_recommendation("prop_001", {"LUT": 5000})
    test("Recommendation generated", "recommendation" in rec)
    test("Stage in recommendation", rec["stage"] == "advisor")

    # At ADVISOR, should recommend human approval
    test("Recommends human approval at ADVISOR",
         rec["recommendation"] == "human_approval")

    return passed, failed, failures


def main():
    print("=" * 60)
    print("L9 AUTONOMY CERTIFICATION")
    print("Staged Hardware Self-Modification")
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
