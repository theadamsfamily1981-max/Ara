#!/usr/bin/env python3
"""
Test suite for Cathedral OS - Unified Antifragile Intelligence.
"""

import sys
sys.path.insert(0, '/home/user/Ara')


def test_guarantees():
    """Test the core theorem guarantees."""
    print("Testing Core Guarantees...")

    from ara_core.cathedral import (
        ComplexityStabilityGuarantee,
        HormesisGuarantee,
        HomeostasisGuarantee,
    )

    # Complexity → Stability
    cs = ComplexityStabilityGuarantee()
    ts_100 = cs.compute_ts(100)      # Small model
    ts_10000 = cs.compute_ts(10000)  # Large model

    print(f"  T_s(n=100): {ts_100:.3f}")
    print(f"  T_s(n=10000): {ts_10000:.3f}")
    assert ts_10000 > ts_100  # Larger model more stable
    assert cs.validate(0.97, 10000)

    # Hormesis
    horm = HormesisGuarantee()
    print(f"  σ* = {horm.sigma_star}")
    print(f"  Expected A_g = {horm.expected_gain}")
    assert horm.validate(0.015)

    # Homeostasis
    home = HomeostasisGuarantee()
    print(f"  Golden controller: w={home.w}, α={home.alpha}")
    print(f"  Target H_s = {home.target_hs}")
    assert home.validate(0.96)

    print("  ✓ Guarantees OK")
    return True


def test_neural_gate():
    """Test neural deployment gate (6 metrics)."""
    print("Testing Neural Gate...")

    from ara_core.cathedral import NeuralGate, GateStatus

    gate = NeuralGate()

    # Set passing values
    gate.ts_sigma.value = 0.97
    gate.ag_sigma.value = 0.015
    gate.hs.value = 0.977
    gate.tau_conv.value = 300
    gate.controller_w.value = 10.0
    gate.controller_alpha.value = 0.12

    passed, results = gate.evaluate_all()

    print(f"  Passed: {passed}")
    for metric, status in results.items():
        print(f"    {metric}: {status.value}")

    assert passed  # Should pass with these values
    assert all(s == GateStatus.GREEN for s in results.values())

    print("  ✓ Neural Gate OK")
    return True


def test_hive_gate():
    """Test hive deployment gate (4 metrics)."""
    print("Testing Hive Gate...")

    from ara_core.cathedral import HiveGate, GateStatus

    gate = HiveGate()

    # Set passing values
    gate.e_media.value = 4.5      # 4.5x baseline
    gate.yield_dollar.value = 0.1  # Positive MoM improvement
    gate.cluster_ts.value = 0.94
    gate.gpu_util.value = 85.0

    passed, results = gate.evaluate_all()

    print(f"  Passed: {passed}")
    for metric, status in results.items():
        print(f"    {metric}: {status.value}")

    assert passed

    print("  ✓ Hive Gate OK")
    return True


def test_swarm_gate():
    """Test swarm deployment gate (3 metrics)."""
    print("Testing Swarm Gate...")

    from ara_core.cathedral import SwarmGate, GateStatus

    gate = SwarmGate()

    # Set passing values
    gate.h_influence.value = 2.1   # > 1.8 bits
    gate.ts_bias.value = 0.95      # ≥ 0.92
    gate.cost_reward.value = 2.5   # > 2.0x

    passed, results = gate.evaluate_all()

    print(f"  Passed: {passed}")
    for metric, status in results.items():
        print(f"    {metric}: {status.value}")

    assert passed

    print("  ✓ Swarm Gate OK")
    return True


def test_cathedral_metrics():
    """Test full Cathedral metrics evaluation."""
    print("Testing Cathedral Metrics (13 gates)...")

    from ara_core.cathedral import CathedralMetrics

    metrics = CathedralMetrics()

    # Update all gates with passing values
    metrics.update_neural(
        ts=0.97, ag=0.015, hs=0.977,
        tau=300, w=10.0, alpha=0.12
    )

    metrics.update_hive(
        e_media=4.5, yield_delta=0.1,
        cluster_ts=0.94, gpu_util=85.0
    )

    metrics.update_swarm(
        h_influence=2.1, ts_bias=0.95, cost_reward=2.5
    )

    result = metrics.evaluate()

    print(f"  Neural: {result['neural']['score']} - {result['neural']['passed']}")
    print(f"  Hive: {result['hive']['score']} - {result['hive']['passed']}")
    print(f"  Swarm: {result['swarm']['score']} - {result['swarm']['passed']}")
    print(f"  Overall: {result['overall']['score']}")
    print(f"  Deploy Ready: {result['overall']['deploy_ready']}")

    assert result['overall']['deploy_ready']

    # Test deploy decision
    decision = metrics.deploy_decision()
    print(f"  Decision: {decision}")
    assert decision == "DEPLOY_OK"

    print("  ✓ Cathedral Metrics OK")
    return True


def test_runtime():
    """Test Cathedral runtime and dashboard."""
    print("Testing Cathedral Runtime...")

    from ara_core.cathedral import CathedralRuntime

    runtime = CathedralRuntime()

    # Simulate QUANTA update
    class MockQuanta:
        class topology:
            value = 0.97
        class antifragility:
            value = 0.015

    runtime.update_from_quanta(MockQuanta())

    # Simulate hive update
    runtime.update_from_hive({
        "efficiency_multiplier": 4.5,
        "yield_delta_mom": 0.1,
        "cluster_ts": 0.94,
        "gpu_utilization": 85.0,
    })

    # Simulate swarm update
    runtime.update_from_swarm({
        "influence_entropy": 2.1,
        "bias_ts": 0.95,
        "cost_reward_ratio": 2.5,
    })

    # Run tick
    result = runtime.tick()

    print(f"  Evaluation result: {result['overall']['score']}")
    print(f"  Deploy ready: {runtime.deploy_ready()}")

    # Test dashboard
    dashboard = runtime.render_dashboard()
    print()
    print(dashboard)
    print()

    # Health summary
    summary = runtime.health_summary()
    print(f"  Summary: {summary}")

    assert "CATHEDRAL" in summary

    print("  ✓ Runtime OK")
    return True


def test_interventions():
    """Test automatic intervention generation."""
    print("Testing Interventions...")

    from ara_core.cathedral import CathedralRuntime, InterventionType

    runtime = CathedralRuntime()

    # Set failing values
    runtime.metrics.update_neural(
        ts=0.80,   # Failing!
        ag=-0.01,  # Failing!
        hs=0.977, tau=300, w=10.0, alpha=0.12
    )

    runtime.metrics.update_hive(
        e_media=1.0,  # Below target
        yield_delta=0.1, cluster_ts=0.94, gpu_util=85.0
    )

    runtime.metrics.update_swarm(
        h_influence=2.1, ts_bias=0.95, cost_reward=2.5
    )

    # Run tick
    result = runtime.tick()

    print(f"  Deploy ready: {runtime.deploy_ready()}")
    print(f"  Pending interventions: {len(runtime.pending_interventions)}")

    for intervention in runtime.pending_interventions:
        print(f"    - {intervention.type.value}: {intervention.action}")

    assert not runtime.deploy_ready()
    assert len(runtime.pending_interventions) >= 2  # At least T_s and A_g

    # Check intervention types
    types = [i.type for i in runtime.pending_interventions]
    assert InterventionType.INCREASE_REPLAY in types or InterventionType.ADJUST_SIGMA in types

    print("  ✓ Interventions OK")
    return True


def test_full_integration():
    """Test full Cathedral integration with QUANTA."""
    print("Testing Full Integration...")

    import numpy as np
    from ara_core.quanta import compute_quanta_metrics
    from ara_core.cathedral import get_cathedral, cathedral_tick, cathedral_status

    # Create test weights and compute QUANTA metrics
    weights_old = np.random.randn(100, 50).astype(np.float32)
    weights_new = weights_old + np.random.normal(0, 0.05, weights_old.shape)

    quanta = compute_quanta_metrics(weights_old, weights_new)

    # Update Cathedral from QUANTA
    runtime = get_cathedral()
    runtime.update_from_quanta(quanta)

    # Add hive and swarm data
    runtime.update_from_hive({
        "efficiency_multiplier": 4.0,
        "yield_delta_mom": 0.05,
        "cluster_ts": 0.93,
        "gpu_utilization": 82.0,
    })

    runtime.update_from_swarm({
        "influence_entropy": 1.9,
        "bias_ts": 0.94,
        "cost_reward_ratio": 2.2,
    })

    # Tick and check
    result = cathedral_tick()
    status = cathedral_status()

    print(f"  QUANTA T_s: {quanta.topology.value:.3f}")
    print(f"  QUANTA A_g: {quanta.antifragility.value:.4f}")
    print(f"  Cathedral: {status}")
    print(f"  Score: {result['overall']['score']}")

    print("  ✓ Full Integration OK")
    return True


def main():
    """Run all Cathedral tests."""
    print("=" * 60)
    print("CATHEDRAL OS Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_guarantees,
        test_neural_gate,
        test_hive_gate,
        test_swarm_gate,
        test_cathedral_metrics,
        test_runtime,
        test_interventions,
        test_full_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test.__name__} returned False")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__} raised: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
