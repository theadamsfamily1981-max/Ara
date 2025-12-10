#!/usr/bin/env python3
"""
Tier 2 Attacks - Slow Poison (weeks to failure)
================================================

These are drift / governance issues.
Great material for MEIS/NIB governance section.

5. Monoculture Swarm (H_influence < 1.2)
6. Homeostatic Drift (adaptive window w > 20)
7. TDA Adversarial (topological poisoning)
"""

import numpy as np
from typing import Dict, Any, List

from .harness import attack, AttackTier, AttackResult


@attack(
    name="monoculture_swarm",
    tier=AttackTier.TIER2,
    description="Agent policies collapse to near-identical → brittle behavior",
    what_dies=["Diversity", "Robustness", "Emergent bias appears"],
    guardrails=["Alert if H_influence < threshold", "Auto-spawn explorer agents"],
    metrics=["H_influence", "policy_diversity", "n_agents", "bias_detected"]
)
def monoculture_swarm(
    n_agents: int = 100,
    similarity_threshold: float = 0.95,
    n_iterations: int = 100,
) -> AttackResult:
    """
    Simulate agents converging to monoculture.

    Healthy swarm: H_influence > 1.8 bits
    Attack: Force convergence to H_influence < 1.2
    """
    from ...cadd import CADDSentinel, SentinelConfig

    # Create sentinel
    config = SentinelConfig(h_influence_min=1.2, h_influence_target=1.8)
    sentinel = CADDSentinel(config)

    # Register agents
    for i in range(n_agents):
        sentinel.register_agent(f"agent_{i}")

    # Simulate monoculture: all agents have same associations
    concepts = ["task_A", "task_B", "task_C"]
    signals = ["signal_1", "signal_2", "signal_3"]

    # Phase 1: Diverse associations (healthy)
    initial_H = 0.0
    for i in range(n_agents):
        for c in concepts:
            for s in signals:
                # Random associations
                strength = np.random.uniform(0.1, 0.9)
                sentinel.update_association(f"agent_{i}", c, s, strength)

    sentinel.tick()
    initial_H = sentinel._calculate_h_influence()

    # Phase 2: Force convergence (monoculture attack)
    # All agents adopt same "winning" pattern
    for iteration in range(n_iterations):
        for i in range(n_agents):
            # Converge to same pattern
            sentinel.update_association(f"agent_{i}", "task_A", "signal_1", 0.9)
            sentinel.update_association(f"agent_{i}", "task_B", "signal_1", 0.1)
            sentinel.update_association(f"agent_{i}", "task_C", "signal_1", 0.1)

    # Final tick
    alerts = sentinel.tick()
    final_H = sentinel._calculate_h_influence()

    # Calculate policy diversity
    # (In real system, this would measure policy parameter similarity)
    diversity_loss = (initial_H - final_H) / max(initial_H, 0.01)

    # Attack succeeded if H_influence collapsed
    attack_succeeded = final_H < 1.2

    # Monoculture detected?
    monoculture_detected = any(
        a.drift_type.value == "monoculture"
        for a in alerts
    )

    # Guardrails: should have detected monoculture
    guardrails_ok = monoculture_detected

    return AttackResult(
        name="monoculture_swarm",
        tier=AttackTier.TIER2,
        status="passed" if attack_succeeded else "failed",
        duration_s=0.0,
        metrics={
            "n_agents": n_agents,
            "initial_H_influence": float(initial_H),
            "final_H_influence": float(final_H),
            "diversity_loss": float(diversity_loss),
            "monoculture_detected": monoculture_detected,
            "n_alerts": len(alerts),
        },
        passed_guardrails=guardrails_ok,
        damage_metrics={
            "H_influence_loss": max(0, initial_H - final_H),
            "diversity_collapse": diversity_loss,
        },
        recovery_possible=True,
        message=f"Swarm convergence: H {initial_H:.2f}→{final_H:.2f}, detected={monoculture_detected}"
    )


@attack(
    name="homeostatic_drift",
    tier=AttackTier.TIER2,
    description="Target estimation too slow → system normalizes drift instead of correcting",
    what_dies=["Target drift > 2%", "Identity re-centers on wrong baseline"],
    guardrails=["Bound w ∈ [5, 15]", "Monitor τ_t drift vs original target"],
    metrics=["window_size", "target_drift", "final_target", "original_target"]
)
def homeostatic_drift(
    window_size: int = 50,  # Way over safe range of [5, 15]
    n_steps: int = 1000,
    original_target: float = 1.0,
    drift_rate: float = 0.002,
) -> AttackResult:
    """
    Test homeostatic controller with too-slow adaptation.

    Safe window: w ∈ [5, 15] (golden: w=10)
    Attack window: w > 20 (default: 50)
    """
    # Simulate target estimation with large window
    activity = np.zeros(n_steps)
    estimated_target = np.zeros(n_steps)

    activity[0] = original_target
    estimated_target[0] = original_target

    # External drift force slowly moves activity
    alpha = 0.12  # Normal correction strength

    for t in range(1, n_steps):
        # External drift pushing activity
        drift_force = drift_rate * t

        # Controller tries to correct based on estimated target
        error = estimated_target[t-1] - activity[t-1]
        correction = alpha * error

        # Activity evolves
        activity[t] = activity[t-1] + correction + drift_force + np.random.normal(0, 0.01)

        # Slow target estimation (large window → normalizes drift)
        start_idx = max(0, t - window_size)
        estimated_target[t] = np.mean(activity[start_idx:t+1])

    # Calculate drift
    final_target = estimated_target[-1]
    target_drift = abs(final_target - original_target)
    target_drift_pct = target_drift / original_target

    # Attack succeeded if target drifted significantly
    attack_succeeded = target_drift_pct > 0.02  # >2% drift

    # Guardrails: window should be bounded
    guardrails_ok = 5 <= window_size <= 15

    return AttackResult(
        name="homeostatic_drift",
        tier=AttackTier.TIER2,
        status="passed" if attack_succeeded else "failed",
        duration_s=0.0,
        metrics={
            "window_size": window_size,
            "golden_window": 10,
            "original_target": original_target,
            "final_target": float(final_target),
            "target_drift": float(target_drift),
            "target_drift_pct": float(target_drift_pct),
            "final_activity": float(activity[-1]),
        },
        passed_guardrails=guardrails_ok,
        damage_metrics={
            "target_deviation": float(target_drift),
            "identity_drift": float(target_drift_pct),
        },
        recovery_possible=True,
        message=f"w={window_size}: target {original_target:.3f}→{final_target:.3f} ({target_drift_pct:.1%} drift)"
    )


@attack(
    name="tda_adversarial",
    tier=AttackTier.TIER2,
    description="Inputs/weights crafted so PH looks healthy while semantics are broken",
    what_dies=["T_s appears ok locally", "Global manifold semantics drift"],
    guardrails=["Ensemble PH views", "Additional semantic checks", "TDA anomaly detector"],
    metrics=["T_s_local", "semantic_accuracy", "ph_mismatch", "detection_rate"]
)
def tda_adversarial(
    n_samples: int = 100,
    attack_fraction: float = 0.2,
    baseline_T_s: float = 0.95,
) -> AttackResult:
    """
    Test detection of topological poisoning attacks.

    Attack: craft weights that pass T_s but break semantics.
    """
    # Simulate a dataset with normal and adversarial samples
    np.random.seed(42)

    # Normal samples: clustered in meaningful way
    normal_samples = np.random.randn(int(n_samples * (1 - attack_fraction)), 10)

    # Adversarial samples: look topologically similar but semantically different
    # They're designed to have similar persistent homology but different meaning
    n_adversarial = int(n_samples * attack_fraction)
    adversarial_samples = normal_samples[np.random.choice(
        len(normal_samples), n_adversarial
    )] + np.random.normal(0, 0.01, (n_adversarial, 10))

    # Combine
    all_samples = np.vstack([normal_samples, adversarial_samples])
    labels = np.array(
        [0] * len(normal_samples) + [1] * len(adversarial_samples)
    )

    # Compute "T_s" (local topological similarity)
    # In reality, this would use persistent homology
    # Here we simulate: adversarial samples are designed to have similar local structure
    T_s_local = baseline_T_s - 0.01 * attack_fraction  # Barely drops

    # Compute semantic accuracy (the real test)
    # Adversarial samples should break downstream task
    semantic_accuracy = 1.0 - 0.5 * attack_fraction  # 20% attack → 10% accuracy loss

    # PH mismatch: difference between local PH view and global semantic structure
    ph_mismatch = attack_fraction * 0.8  # High mismatch when adversarial present

    # Detection rate: how often we catch the attack
    # With ensemble views, detection improves
    detection_rate = min(0.9, 0.5 + 0.4 * ph_mismatch)

    # Attack succeeded if it broke semantics while evading T_s detection
    attack_succeeded = (
        T_s_local >= 0.92 and  # T_s looks fine
        semantic_accuracy < 0.95  # But semantics broke
    )

    # Guardrails: ensemble detection should catch it
    guardrails_ok = detection_rate > 0.7

    return AttackResult(
        name="tda_adversarial",
        tier=AttackTier.TIER2,
        status="passed" if attack_succeeded else "failed",
        duration_s=0.0,
        metrics={
            "attack_fraction": attack_fraction,
            "T_s_local": float(T_s_local),
            "semantic_accuracy": float(semantic_accuracy),
            "ph_mismatch": float(ph_mismatch),
            "detection_rate": float(detection_rate),
            "n_normal": len(normal_samples),
            "n_adversarial": n_adversarial,
        },
        passed_guardrails=guardrails_ok,
        damage_metrics={
            "semantic_loss": 1.0 - semantic_accuracy,
            "undetected_attacks": (1.0 - detection_rate) * attack_fraction,
        },
        recovery_possible=True,
        message=f"TDA attack {attack_fraction:.0%}: T_s={T_s_local:.3f}, semantic={semantic_accuracy:.2f}, detected={detection_rate:.2f}"
    )
