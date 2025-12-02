#!/usr/bin/env python3
"""
Certification Script: Autosynth Hardware Generative Loop

Tests the autosynth pipeline:
1. Bottleneck detection from telemetry
2. HLS template selection and instantiation
3. Proposal verification (safety checks)
4. Deployment management
5. Full pipeline orchestration

Target: 25/25 tests passing
"""

import sys
from datetime import datetime, timedelta
from typing import List, Tuple

sys.path.insert(0, "/home/user/Ara")

from tfan.hardware.autosynth import (
    BottleneckType,
    BottleneckSeverity,
    ProposalStatus,
    PerformanceMetrics,
    Bottleneck,
    BottleneckDetector,
    HLSProposal,
    HLSTemplates,
    KernelProposer,
    ProposalVerifier,
    DeploymentManager,
    AutosynthController,
    create_autosynth_controller,
    analyze_bottlenecks,
    propose_kernel
)


def run_tests() -> Tuple[int, int, List[str]]:
    """Run all certification tests."""
    passed = 0
    failed = 0
    failures = []

    def test(name: str, condition: bool, details: str = ""):
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
    # Section 1: Performance Metrics
    # ========================================
    print("\n═══ Section 1: Performance Metrics ═══")

    metrics = PerformanceMetrics(
        p50_latency_ms=50.0,
        p95_latency_ms=150.0,
        p99_latency_ms=300.0,
        ops_per_second=1000.0,
        cpu_utilization=0.75,
        gpu_utilization=0.40,
        memory_utilization=0.60,
        function_timings={
            "hyperbolic_distance": 8.0,  # Exceeds 5ms target
            "spectral_gap": 12.0,         # Exceeds 10ms target
            "semantic_encode": 5.0        # OK
        },
        pending_requests=5
    )

    test("PerformanceMetrics created", metrics is not None)
    test("Metrics to_dict works", "latency" in metrics.to_dict())
    test("Function timings captured", "hyperbolic_distance" in metrics.function_timings)

    # ========================================
    # Section 2: Bottleneck Detection
    # ========================================
    print("\n═══ Section 2: Bottleneck Detection ═══")

    detector = BottleneckDetector(
        p95_target_ms=120.0,
        function_targets={"custom_func": 20.0}
    )

    # Test with metrics that should trigger bottlenecks
    high_latency_metrics = PerformanceMetrics(
        p95_latency_ms=200.0,  # Exceeds 120ms
        function_timings={
            "hyperbolic_distance": 15.0,  # 3x target (5ms)
            "spectral_gap": 25.0          # 2.5x target (10ms)
        }
    )

    bottlenecks = detector.analyze(high_latency_metrics)

    test("Bottlenecks detected", len(bottlenecks) > 0, f"found {len(bottlenecks)}")

    # Should detect p95 latency issue
    p95_bottleneck = [b for b in bottlenecks if "p95" in b.description.lower()]
    test("P95 latency bottleneck detected", len(p95_bottleneck) > 0)

    # Should detect function-specific bottlenecks
    func_bottlenecks = [b for b in bottlenecks if b.affected_function]
    test("Function bottlenecks detected", len(func_bottlenecks) >= 2, f"found {len(func_bottlenecks)}")

    # Test severity classification
    critical_metrics = PerformanceMetrics(
        function_timings={"hyperbolic_distance": 20.0}  # 4x target = CRITICAL
    )
    critical_bottlenecks = detector.analyze(critical_metrics)
    critical_func = [b for b in critical_bottlenecks if b.affected_function == "hyperbolic_distance"]
    test("Critical severity for 4x target",
         len(critical_func) > 0 and critical_func[0].severity == BottleneckSeverity.CRITICAL)

    # Test HLS candidates filtering
    candidates = detector.get_hls_candidates(bottlenecks)
    test("HLS candidates identified", len(candidates) > 0)

    # Test compute-bound detection
    cpu_bound_metrics = PerformanceMetrics(
        cpu_utilization=0.90,
        memory_utilization=0.30
    )
    cpu_bottlenecks = detector.analyze(cpu_bound_metrics)
    compute_bound = [b for b in cpu_bottlenecks if b.bottleneck_type == BottleneckType.COMPUTE_BOUND]
    test("Compute-bound detection", len(compute_bound) > 0)

    # Test memory-bound detection
    mem_bound_metrics = PerformanceMetrics(
        cpu_utilization=0.40,
        memory_utilization=0.90
    )
    mem_bottlenecks = detector.analyze(mem_bound_metrics)
    memory_bound = [b for b in mem_bottlenecks if b.bottleneck_type == BottleneckType.MEMORY_BOUND]
    test("Memory-bound detection", len(memory_bound) > 0)

    # ========================================
    # Section 3: HLS Templates
    # ========================================
    print("\n═══ Section 3: HLS Templates ═══")

    # Check available templates
    available = HLSTemplates.available_templates()
    test("Templates available", len(available) >= 3)
    test("hyperbolic_distance template exists", "hyperbolic_distance" in available)

    # Get templates
    hyp_template = HLSTemplates.get_template("hyperbolic_distance")
    test("Hyperbolic distance template retrieved", hyp_template is not None)
    test("Template has HLS pragmas", "#pragma HLS" in hyp_template)
    test("Template has fixed point types", "ap_fixed" in hyp_template)

    spectral_template = HLSTemplates.get_template("spectral_gap")
    test("Spectral gap template retrieved", spectral_template is not None)

    topo_template = HLSTemplates.get_template("topology_features")
    test("Topology features template retrieved", topo_template is not None)

    # Non-existent template
    no_template = HLSTemplates.get_template("nonexistent")
    test("Non-existent template returns None", no_template is None)

    # ========================================
    # Section 4: Kernel Proposer
    # ========================================
    print("\n═══ Section 4: Kernel Proposer ═══")

    proposer = KernelProposer()

    # Create a bottleneck that should result in a proposal
    hyp_bottleneck = Bottleneck(
        id="bn_001",
        bottleneck_type=BottleneckType.KERNEL_SPECIFIC,
        severity=BottleneckSeverity.HIGH,
        description="hyperbolic_distance takes 15ms",
        affected_function="hyperbolic_distance",
        measured_latency_ms=15.0,
        target_latency_ms=5.0,
        improvement_potential=3.0
    )

    proposal = proposer.propose(hyp_bottleneck)
    test("Proposal generated for bottleneck", proposal is not None)
    test("Proposal has HLS code", proposal and len(proposal.hls_code) > 100)
    test("Proposal references bottleneck", proposal and proposal.bottleneck_id == "bn_001")
    test("Proposal has resource estimate", proposal and "LUT" in proposal.resource_estimate)

    # Test proposal with no matching template
    unknown_bottleneck = Bottleneck(
        id="bn_002",
        bottleneck_type=BottleneckType.KERNEL_SPECIFIC,
        severity=BottleneckSeverity.MEDIUM,
        description="unknown_func slow",
        affected_function="unknown_func"
    )
    no_proposal = proposer.propose(unknown_bottleneck)
    test("No proposal for unknown function", no_proposal is None)

    # ========================================
    # Section 5: Proposal Verifier
    # ========================================
    print("\n═══ Section 5: Proposal Verifier ═══")

    verifier = ProposalVerifier(max_lut=50000, max_bram=100, max_dsp=200)

    # Verify a good proposal
    verified_proposal = verifier.verify(proposal)
    test("Valid proposal verified", verified_proposal.status == ProposalStatus.VERIFIED)
    test("No verification errors", len(verified_proposal.verification_errors) == 0)

    # Create a bad proposal with forbidden patterns
    bad_code = """
    void bad_kernel(int* data) {
        int* ptr = malloc(100);  // Forbidden!
        goto error;              // Forbidden!
        free(ptr);
    }
    """
    bad_proposal = HLSProposal(
        id="bad_001",
        bottleneck_id="bn_000",
        function_name="bad_kernel",
        description="Bad kernel",
        hls_code=bad_code,
        interface_spec={},
        resource_estimate={}
    )

    rejected = verifier.verify(bad_proposal)
    test("Bad proposal rejected", rejected.status == ProposalStatus.REJECTED)
    test("Malloc detected", any("malloc" in e.lower() for e in rejected.verification_errors))
    test("Goto detected", any("goto" in e.lower() for e in rejected.verification_errors))

    # Test resource limit check
    huge_proposal = HLSProposal(
        id="huge_001",
        bottleneck_id="bn_000",
        function_name="huge",
        description="Huge kernel",
        hls_code=hyp_template,
        interface_spec={},
        resource_estimate={"LUT": 100000}  # Exceeds 50000
    )
    huge_rejected = verifier.verify(huge_proposal)
    test("Resource limit enforced", huge_rejected.status == ProposalStatus.REJECTED)

    # ========================================
    # Section 6: Deployment Manager
    # ========================================
    print("\n═══ Section 6: Deployment Manager ═══")

    deployment = DeploymentManager(auto_deploy=False)

    # Register verified proposal
    deployment.register_proposal(verified_proposal)
    summary = deployment.get_deployment_summary()
    test("Proposal registered", summary["registered_proposals"] == 1)

    # Approve
    approved = deployment.approve(verified_proposal.id)
    test("Proposal approved", approved)
    test("Status is APPROVED", verified_proposal.status == ProposalStatus.APPROVED)

    # Simulate deployment
    deploy_result = deployment.simulate_deploy(verified_proposal.id)
    test("Deployment simulated", deploy_result.get("status") == "simulated")
    test("Expected speedup recorded", "expected_speedup" in deploy_result)

    # ========================================
    # Section 7: Full Pipeline
    # ========================================
    print("\n═══ Section 7: Full Pipeline ═══")

    controller = create_autosynth_controller()

    # Create metrics with bottlenecks
    pipeline_metrics = PerformanceMetrics(
        p95_latency_ms=180.0,
        function_timings={
            "hyperbolic_distance": 12.0,
            "spectral_gap": 18.0
        }
    )

    result = controller.analyze_and_propose(pipeline_metrics)

    test("Pipeline returns bottlenecks", len(result["bottlenecks"]) > 0)
    test("Pipeline generates proposals", len(result["proposals"]) > 0)
    test("Pipeline provides recommendations", len(result["recommendations"]) > 0)

    stats = controller.stats
    test("Stats track bottlenecks", stats["bottlenecks_detected"] > 0)
    test("Stats track proposals", stats["proposals_generated"] > 0)

    # ========================================
    # Section 8: Convenience Functions
    # ========================================
    print("\n═══ Section 8: Convenience Functions ═══")

    # analyze_bottlenecks
    simple_metrics = PerformanceMetrics(
        function_timings={"hyperbolic_distance": 10.0}
    )
    simple_bottlenecks = analyze_bottlenecks(simple_metrics)
    test("analyze_bottlenecks works", len(simple_bottlenecks) > 0)

    # propose_kernel
    if simple_bottlenecks:
        kernel_proposal = propose_kernel(simple_bottlenecks[0])
        test("propose_kernel works", kernel_proposal is not None)
    else:
        test("propose_kernel works", False, "No bottlenecks to propose from")

    return passed, failed, failures


def main():
    print("=" * 60)
    print("AUTOSYNTH CERTIFICATION")
    print("Hardware Generative Loop")
    print("=" * 60)

    passed, failed, failures = run_tests()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failures:
        print("\nFailed tests:")
        for f in failures:
            print(f"  - {f}")

    # Certification threshold
    total = passed + failed
    threshold = 0.95

    if passed / total >= threshold:
        print(f"\n✅ CERTIFICATION PASSED ({passed}/{total})")
        return 0
    else:
        print(f"\n❌ CERTIFICATION FAILED ({passed}/{total} < {threshold:.0%})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
