#!/usr/bin/env python3
"""
Test suite for QUANTA v2.0 Memory Consolidation.
"""

import sys
sys.path.insert(0, '/home/user/Ara')

import numpy as np


def test_topology_metric():
    """Test T_s (topology stability) computation."""
    print("Testing T_s (Topology Stability)...")

    from ara_core.quanta import TopologyMetric

    # Create test weights
    weights = np.random.randn(100, 50).astype(np.float32)
    weights_perturbed = weights + np.random.normal(0, 0.1, weights.shape)

    # Compute T_s
    ts = TopologyMetric()
    ts.compute(weights, weights_perturbed)

    print(f"  T_s value: {ts.value:.3f}")
    print(f"  Bottleneck distance: {ts.bottleneck_distance:.3f}")
    print(f"  Status: {ts.status.value}")
    print(f"  Target: {ts.TARGET}")

    assert 0 <= ts.value <= 1
    assert ts.bottleneck_distance >= 0

    print("  ✓ Topology OK")
    return True


def test_antifragility_metric():
    """Test A_g (antifragility gain) computation."""
    print("Testing A_g (Antifragility)...")

    from ara_core.quanta import AntifragilityMetric

    # Create test weights
    weights = np.random.randn(100, 50).astype(np.float32)

    # Compute A_g at optimal σ*
    ag = AntifragilityMetric()
    ag.compute(weights, sigma=0.10)

    print(f"  A_g value: {ag.value:.4f}")
    print(f"  σ used: {ag.sigma_used}")
    print(f"  T_s (no stress): {ag.ts_no_stress:.3f}")
    print(f"  T_s (with stress): {ag.ts_with_stress:.3f}")
    print(f"  Status: {ag.status.value}")

    # Should be close to target at optimal σ*
    assert ag.value >= -0.05  # Not too fragile

    print("  ✓ Antifragility OK")
    return True


def test_nib_metric():
    """Test NIB (identity preservation) computation."""
    print("Testing NIB (Identity)...")

    from ara_core.quanta import NIBMetric

    # Create test weights - small change
    weights_old = np.random.randn(100, 50).astype(np.float32)
    weights_new = weights_old + np.random.normal(0, 0.05, weights_old.shape)

    # Compute NIB
    nib = NIBMetric()
    nib.compute(weights_old, weights_new)

    print(f"  ΔD value: {nib.value:.3f}")
    print(f"  Weight change ratio: {nib.weight_change_ratio:.3f}")
    print(f"  Status: {nib.status.value}")
    print(f"  Target: <{nib.TARGET}")

    assert nib.value >= 0
    assert nib.weight_change_ratio >= 0

    print("  ✓ NIB OK")
    return True


def test_gft_metric():
    """Test GFT η (damping) computation."""
    print("Testing GFT (Damping)...")

    from ara_core.quanta import GFTMetric

    # Create test layer weights
    layer_weights = [
        np.random.randn(100, 50).astype(np.float32),
        np.random.randn(50, 30).astype(np.float32),
        np.random.randn(30, 10).astype(np.float32),
    ]

    # Compute GFT
    gft = GFTMetric()
    gft.compute(layer_weights)

    print(f"  η (mean): {gft.value:.2f}")
    print(f"  Per-layer: {[f'{e:.2f}' for e in gft.eta_per_layer]}")
    print(f"  Critical %: {gft.critical_percentage*100:.0f}%")
    print(f"  Status: {gft.status.value}")

    assert len(gft.eta_per_layer) == 3
    assert gft.critical_percentage >= 0

    print("  ✓ GFT OK")
    return True


def test_capacity_metric():
    """Test C (capacity) computation."""
    print("Testing Capacity...")

    from ara_core.quanta import CapacityMetric

    # Create test layer weights
    layer_weights = [
        np.random.randn(100, 50).astype(np.float32),
        np.random.randn(50, 30).astype(np.float32),
    ]

    # Compute capacity
    cap = CapacityMetric()
    cap.compute(layer_weights, sigma=0.10)

    print(f"  Bits/layer: {cap.value:.1f}")
    print(f"  Per-layer: {[f'{b:.1f}' for b in cap.bits_per_layer]}")
    print(f"  Retention: {cap.capacity_retention*100:.0f}%")
    print(f"  Status: {cap.status.value}")

    assert cap.value > 0
    assert cap.capacity_retention > 0

    print("  ✓ Capacity OK")
    return True


def test_quanta_metrics_all():
    """Test all QUANTA metrics together."""
    print("Testing QUANTA Metrics (all 5)...")

    from ara_core.quanta import QUANTAMetrics, MetricStatus

    # Create test weights
    weights_old = np.random.randn(100, 50).astype(np.float32)
    weights_new = weights_old + np.random.normal(0, 0.05, weights_old.shape)
    layer_weights = [weights_new[:50], weights_new[50:]]

    # Compute all metrics
    metrics = QUANTAMetrics()
    metrics.compute_all(weights_old, weights_new, layer_weights, sigma=0.10)

    print(f"  T_s: {metrics.topology.value:.3f} [{metrics.topology.status.value}]")
    print(f"  A_g: {metrics.antifragility.value:.4f} [{metrics.antifragility.status.value}]")
    print(f"  NIB: {metrics.nib.value:.3f} [{metrics.nib.status.value}]")
    print(f"  GFT: {metrics.gft.value:.2f} [{metrics.gft.status.value}]")
    print(f"  Cap: {metrics.capacity.value:.1f} [{metrics.capacity.status.value}]")
    print(f"  Overall: {metrics.overall_status.value}")
    print(f"  All green: {metrics.all_green}")

    # Check we have values for all metrics
    assert metrics.topology.value > 0
    assert metrics.capacity.value > 0

    # Check summary
    summary = metrics.summary()
    assert "QUANTA" in summary
    print()
    print(summary)

    print("  ✓ All metrics OK")
    return True


def test_consolidation():
    """Test QUANTA consolidation."""
    print("Testing Consolidation...")

    from ara_core.quanta import (
        QUANTAConsolidator, ConsolidationPhase, compute_quanta_metrics
    )

    # Create test weights
    weights = np.random.randn(100, 50).astype(np.float32)

    # Create consolidator
    consolidator = QUANTAConsolidator()

    # Run each phase
    for phase in [ConsolidationPhase.MICRO, ConsolidationPhase.REPLAY, ConsolidationPhase.STRUCTURAL]:
        weights_new = consolidator.consolidate(weights, phase)
        print(f"  {phase.value}: shape={weights_new.shape}")
        weights = weights_new

    # Check we have events
    assert len(consolidator.schedule.events) == 3

    # Check health summary
    health = consolidator.get_health_summary()
    print(f"  Events: {health['events_total']}")
    print(f"  Objective: {health['objective']:.3f}")

    print("  ✓ Consolidation OK")
    return True


def test_cockpit():
    """Test memory health cockpit."""
    print("Testing Cockpit Dashboard...")

    from ara_core.quanta import (
        MemoryHealthCockpit, compute_quanta_metrics
    )

    # Create cockpit
    cockpit = MemoryHealthCockpit()

    # Generate some metrics
    for i in range(5):
        weights_old = np.random.randn(50, 30).astype(np.float32)
        weights_new = weights_old + np.random.normal(0, 0.05 + i*0.02, weights_old.shape)

        metrics = compute_quanta_metrics(weights_old, weights_new)
        cockpit.update(metrics)

    # Render dashboard
    dashboard = cockpit.render()
    print()
    print(dashboard)
    print()

    # Check JSON export
    json_export = cockpit.export_json()
    assert "widgets" in json_export
    assert "alerts" in json_export

    # Check D-Bus export
    dbus_export = cockpit.export_dbus()
    assert "org.ara.memory.topology" in dbus_export

    print(f"  History: {len(cockpit.metrics_history)} samples")
    print(f"  Alerts: {len(cockpit.alerts)}")

    print("  ✓ Cockpit OK")
    return True


def test_full_pipeline():
    """Test complete QUANTA pipeline."""
    print("Testing Full Pipeline (metrics → consolidation → cockpit)...")

    from ara_core.quanta import (
        QUANTAConsolidator, ConsolidationPhase,
        get_cockpit, update_cockpit, render_cockpit
    )

    # Create consolidator
    consolidator = QUANTAConsolidator()

    # Initial weights
    weights = np.random.randn(100, 50).astype(np.float32)

    # Run consolidation cycle and update cockpit
    for _ in range(3):
        weights = consolidator.consolidate(weights, ConsolidationPhase.REPLAY)

        if consolidator.current_metrics:
            update_cockpit(consolidator.current_metrics)

    # Get final dashboard
    dashboard = render_cockpit()
    health = consolidator.get_health_summary()

    print(f"  Final status: {health['status']}")
    print(f"  Objective: {health['objective']:.3f}")

    # Check recommendations
    if health['recommendation']['should_act']:
        print(f"  Actions needed: {len(health['recommendation']['actions'])}")
        for action in health['recommendation']['actions'][:2]:
            print(f"    - {action['action']}: {action['reason'][:40]}...")

    print("  ✓ Full Pipeline OK")
    return True


def main():
    """Run all QUANTA tests."""
    print("=" * 60)
    print("QUANTA v2.0 Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_topology_metric,
        test_antifragility_metric,
        test_nib_metric,
        test_gft_metric,
        test_capacity_metric,
        test_quanta_metrics_all,
        test_consolidation,
        test_cockpit,
        test_full_pipeline,
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
