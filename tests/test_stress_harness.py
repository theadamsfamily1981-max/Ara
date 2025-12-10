#!/usr/bin/env python3
"""
Test suite for Cathedral OS Stress Test Harness.

Tests all 12 breakpoints across 3 tiers.
"""

import sys
sys.path.insert(0, '/home/user/Ara')


def test_tier1_attacks():
    """Test Tier 1 - Immediate Killers."""
    print("Testing Tier 1 Attacks...")

    from ara_core.cathedral.stress_tests import (
        controller_overshoot,
        stress_overdose,
        pheromone_flood,
        morph_overbudget,
    )

    # Controller Overshoot
    print("  1. Controller Overshoot...")
    result = controller_overshoot(alpha=0.25, n_steps=500)
    print(f"     Status: {result.status}")
    print(f"     H_s: {result.metrics['H_s']:.3f}")
    print(f"     Oscillations: {result.metrics['direction_changes']}")
    assert result.status in ["passed", "failed"]

    # Stress Overdose
    print("  2. Stress Overdose...")
    result = stress_overdose(sigma=0.30, n_steps=500)
    print(f"     Status: {result.status}")
    print(f"     T_s: {result.metrics['initial_T_s']:.3f} → {result.metrics['final_T_s']:.3f}")
    print(f"     A_g: {result.metrics['A_g']:.4f}")
    assert result.status in ["passed", "failed"]

    # Pheromone Flood
    print("  3. Pheromone Flood...")
    result = pheromone_flood(flood_bytes=1024*1024, n_locations=50)
    print(f"     Status: {result.status}")
    print(f"     H_influence: {result.metrics['H_influence']:.2f}")
    print(f"     Gradient clarity: {result.metrics['gradient_clarity']:.2f}")
    assert result.status in ["passed", "failed"]

    # Morph Overbudget
    print("  4. Morph Overbudget...")
    result = morph_overbudget(morph_fraction=0.25)
    print(f"     Status: {result.status}")
    print(f"     T_s: {result.metrics['T_s_before']:.3f} → {result.metrics['T_s_after']:.3f}")
    print(f"     Actual morph: {result.metrics['actual_morph']:.1%}")
    assert result.status in ["passed", "failed"]

    print("  ✓ Tier 1 OK")
    return True


def test_tier2_attacks():
    """Test Tier 2 - Slow Poison."""
    print("Testing Tier 2 Attacks...")

    from ara_core.cathedral.stress_tests import (
        monoculture_swarm,
        homeostatic_drift,
        tda_adversarial,
    )

    # Monoculture Swarm
    print("  5. Monoculture Swarm...")
    result = monoculture_swarm(n_agents=50, n_iterations=50)
    print(f"     Status: {result.status}")
    print(f"     H_influence: {result.metrics['initial_H_influence']:.2f} → {result.metrics['final_H_influence']:.2f}")
    print(f"     Detected: {result.metrics['monoculture_detected']}")
    assert result.status in ["passed", "failed"]

    # Homeostatic Drift
    print("  6. Homeostatic Drift...")
    result = homeostatic_drift(window_size=50, n_steps=500)
    print(f"     Status: {result.status}")
    print(f"     Target: {result.metrics['original_target']:.3f} → {result.metrics['final_target']:.3f}")
    print(f"     Drift: {result.metrics['target_drift_pct']:.1%}")
    assert result.status in ["passed", "failed"]

    # TDA Adversarial
    print("  7. TDA Adversarial...")
    result = tda_adversarial(attack_fraction=0.2)
    print(f"     Status: {result.status}")
    print(f"     T_s local: {result.metrics['T_s_local']:.3f}")
    print(f"     Semantic accuracy: {result.metrics['semantic_accuracy']:.2f}")
    print(f"     Detection rate: {result.metrics['detection_rate']:.2f}")
    assert result.status in ["passed", "failed"]

    print("  ✓ Tier 2 OK")
    return True


def test_tier3_exploits():
    """Test Tier 3 - Moonshot Exploits."""
    print("Testing Tier 3 Exploits...")

    from ara_core.cathedral.stress_tests import (
        sigma_phase_lock,
        pheromone_hacking,
        vsa_superposition,
        quantum_yield,
        junkyard_voltron,
    )

    # σ* Phase-Lock
    print("  8. σ* Phase-Lock...")
    result = sigma_phase_lock(n_modules=5, n_cycles=50)
    print(f"     Status: {result.status}")
    print(f"     A_g: {result.metrics['A_g']:.4f}")
    print(f"     Improvement: {result.metrics.get('A_g_improvement', 0):.1%}")
    assert result.status in ["passed", "failed"]

    # Pheromone Hacking
    print("  9. Pheromone Hacking...")
    result = pheromone_hacking(n_agents=50, n_fake_trails=10)
    print(f"     Status: {result.status}")
    print(f"     Attack success: {result.metrics['attack_success_rate']:.0%}")
    print(f"     Detection: {result.metrics['detection_rate']:.0%}")
    assert result.status in ["passed", "failed"]

    # VSA Superposition
    print("  10. VSA Superposition...")
    result = vsa_superposition(base_dim=512, target_dim=2048, n_modalities=10)
    print(f"     Status: {result.status}")
    print(f"     Modalities: {result.metrics['base_modalities']} → {result.metrics['target_modalities']}")
    print(f"     Interference: {result.metrics['target_interference']:.3f}")
    assert result.status in ["passed", "failed"]

    # Quantum Yield
    print("  11. Quantum Yield...")
    result = quantum_yield(n_assets=4, n_trials=5)
    print(f"     Status: {result.status}")
    print(f"     Sharpe: {result.metrics['sharpe_classical']:.4f} → {result.metrics['sharpe_quantum']:.4f}")
    print(f"     Improvement: {result.metrics['improvement_pct']:+.1%}")
    assert result.status in ["passed", "failed"]

    # Junkyard Voltron
    print("  12. Junkyard Voltron...")
    result = junkyard_voltron(n_gpus=5, n_fpgas=3, n_miners=2)
    print(f"     Status: {result.status}")
    print(f"     Devices: {result.metrics['total_devices']}")
    print(f"     Utilization: {result.metrics['utilization']:.0%}")
    print(f"     Fault tolerance: {result.metrics['failure_tolerance']:.0%}")
    assert result.status in ["passed", "failed"]

    print("  ✓ Tier 3 OK")
    return True


def test_harness_functions():
    """Test harness utility functions."""
    print("Testing Harness Functions...")

    from ara_core.cathedral.stress_tests import (
        list_attacks,
        run_attack,
        run_tier,
        run_all_attacks,
        generate_report,
        ATTACKS,
        AttackTier,
    )

    # List attacks
    attacks = list_attacks()
    print(f"  Registered attacks: {len(attacks)}")
    assert len(attacks) == 12  # All 12 breakpoints

    # Check tiers
    tier1_count = sum(1 for _, info in attacks if info['tier'] == 'tier1')
    tier2_count = sum(1 for _, info in attacks if info['tier'] == 'tier2')
    tier3_count = sum(1 for _, info in attacks if info['tier'] == 'tier3')
    print(f"  Tier 1: {tier1_count}, Tier 2: {tier2_count}, Tier 3: {tier3_count}")
    assert tier1_count == 4
    assert tier2_count == 3
    assert tier3_count == 5

    # Run single attack
    result = run_attack("controller_overshoot", alpha=0.20, n_steps=100)
    print(f"  Single attack result: {result.name} -> {result.status}")
    assert result.name == "controller_overshoot"

    # Run tier
    tier1_results = run_tier(AttackTier.TIER1)
    print(f"  Tier 1 results: {len(tier1_results)}")
    assert len(tier1_results) == 4

    # Generate report (subset for speed)
    report = generate_report(tier1_results)
    print(f"  Report generated: {len(report)} chars")
    assert "ADVERSARIAL ANTIFRAGILITY REPORT" in report

    print("  ✓ Harness Functions OK")
    return True


def main():
    """Run all stress test harness tests."""
    print("=" * 70)
    print("CATHEDRAL OS - STRESS TEST HARNESS")
    print("=" * 70)
    print()

    tests = [
        test_tier1_attacks,
        test_tier2_attacks,
        test_tier3_exploits,
        test_harness_functions,
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

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
