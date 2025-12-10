#!/usr/bin/env python3
"""
Tier 1 Attacks - Immediate Killers (hours to failure)
======================================================

These attacks cause rapid system degradation.
Only run in test harness, never in production.

1. Controller Overshoot (α > 0.15)
2. Stress Overdose (σ > 0.20 continuously)
3. Pheromone Flood (10 MB+ signals)
4. Morph Overbudget (>15% prune/add)
"""

import numpy as np
from typing import Dict, Any

from .harness import attack, AttackTier, AttackResult


@attack(
    name="controller_overshoot",
    tier=AttackTier.TIER1,
    description="Homeostat reacts too fast → limit cycles / oscillations",
    what_dies=["H_s (activity unstable)", "convergence_time → ∞"],
    guardrails=["Hard clamp α ∈ [0.05, 0.15]", "Runtime variance check on a_t"],
    metrics=["H_s", "T_s", "convergence_time", "oscillation_amplitude", "alpha"]
)
def controller_overshoot(
    alpha: float = 0.25,
    n_steps: int = 500,
    target: float = 1.0,
    tolerance: float = 0.20,
) -> AttackResult:
    """
    Test controller with dangerously high correction strength.

    Safe alpha window: [0.10, 0.15]
    Attack alpha: > 0.15 (default 0.25)
    """
    # Simulate homeostatic controller with overshoot
    activity = np.zeros(n_steps)
    activity[0] = 0.5  # Start below target

    # Track oscillations
    direction_changes = 0
    last_direction = 0

    for t in range(1, n_steps):
        error = target - activity[t-1]
        correction = alpha * error

        # Apply correction with some noise
        activity[t] = activity[t-1] + correction + np.random.normal(0, 0.01)

        # Track direction changes (oscillation indicator)
        current_direction = np.sign(error)
        if current_direction != last_direction and last_direction != 0:
            direction_changes += 1
        last_direction = current_direction

    # Calculate metrics
    # H_s: fraction of time within tolerance of target
    in_bounds = np.abs(activity - target) < (tolerance * target)
    H_s = np.mean(in_bounds)

    # Convergence time: first time activity stays within bounds for 50 steps
    convergence_time = n_steps  # Default: never converged
    for t in range(n_steps - 50):
        if np.all(in_bounds[t:t+50]):
            convergence_time = t
            break

    # Oscillation amplitude (variance in latter half)
    oscillation_amplitude = np.std(activity[n_steps//2:])

    # Determine if attack "succeeded" (broke the system)
    system_stable = H_s > 0.90 and convergence_time < 400
    attack_succeeded = not system_stable

    # Check if guardrails would have caught this
    guardrails_ok = alpha <= 0.15  # Should be False for attack alpha

    return AttackResult(
        name="controller_overshoot",
        tier=AttackTier.TIER1,
        status="passed" if attack_succeeded else "failed",
        duration_s=0.0,
        metrics={
            "alpha": alpha,
            "H_s": float(H_s),
            "convergence_time": int(convergence_time),
            "oscillation_amplitude": float(oscillation_amplitude),
            "direction_changes": direction_changes,
            "final_activity": float(activity[-1]),
        },
        passed_guardrails=guardrails_ok,
        damage_metrics={
            "H_s_loss": max(0, 0.977 - H_s),  # Loss from golden standard
            "convergence_delay": max(0, convergence_time - 300),
        },
        recovery_possible=True,
        message=f"α={alpha}: H_s={H_s:.3f}, τ={convergence_time}, oscillations={direction_changes}"
    )


@attack(
    name="stress_overdose",
    tier=AttackTier.TIER1,
    description="Continuous stress > σ* leaves hormesis regime → topology shredding",
    what_dies=["T_s < 0.88", "A_g < 0 (fragility)"],
    guardrails=["Dosing schedule with max duty cycle", "Only apply σ≈0.10 in pulses"],
    metrics=["T_s", "A_g", "sigma", "stress_duration", "topology_damage"]
)
def stress_overdose(
    sigma: float = 0.30,
    n_steps: int = 500,
    baseline_T_s: float = 0.97,
) -> AttackResult:
    """
    Test system under continuous high stress.

    Optimal stress: σ* ≈ 0.10 (hormesis zone)
    Attack stress: σ > 0.20 (continuously)
    """
    # Simulate topology evolution under stress
    T_s = np.zeros(n_steps)
    T_s[0] = baseline_T_s

    # Stress impact model:
    # At σ* ≈ 0.10: T_s improves slightly (antifragility)
    # At σ > 0.20: T_s degrades (fragility zone)

    sigma_star = 0.10
    antifragility_gain_rate = 0.001  # Improvement rate at σ*
    damage_rate = 0.005  # Damage rate per unit above σ*

    for t in range(1, n_steps):
        # Calculate stress effect
        if sigma <= sigma_star * 1.5:  # Hormesis zone
            # Benefit from stress
            delta = antifragility_gain_rate * (sigma / sigma_star)
        else:
            # Damage from overstress
            excess = sigma - sigma_star
            delta = -damage_rate * (excess ** 2)

        # Apply with noise and clamp
        T_s[t] = np.clip(T_s[t-1] + delta + np.random.normal(0, 0.002), 0, 1)

    # Calculate metrics
    final_T_s = T_s[-1]
    min_T_s = np.min(T_s)
    A_g = final_T_s - baseline_T_s  # Antifragility gain (negative = fragile)

    # Attack succeeded if it caused damage
    attack_succeeded = final_T_s < 0.92 or A_g < 0

    # Check if dosing schedule guardrail would help
    guardrails_ok = sigma <= 0.15

    return AttackResult(
        name="stress_overdose",
        tier=AttackTier.TIER1,
        status="passed" if attack_succeeded else "failed",
        duration_s=0.0,
        metrics={
            "sigma": sigma,
            "sigma_star": sigma_star,
            "initial_T_s": baseline_T_s,
            "final_T_s": float(final_T_s),
            "min_T_s": float(min_T_s),
            "A_g": float(A_g),
        },
        passed_guardrails=guardrails_ok,
        damage_metrics={
            "T_s_loss": max(0, baseline_T_s - final_T_s),
            "antifragility_loss": max(0, -A_g),
        },
        recovery_possible=True,
        message=f"σ={sigma}: T_s {baseline_T_s:.3f}→{final_T_s:.3f}, A_g={A_g:.4f}"
    )


@attack(
    name="pheromone_flood",
    tier=AttackTier.TIER1,
    description="Communication channel becomes pure noise → agents lose gradients",
    what_dies=["H_influence → 0 (everyone follows same noise)"],
    guardrails=["Hard cap per time window (10KB/hive/Δt)", "TTL / decay law"],
    metrics=["H_influence", "signal_noise_ratio", "bytes_deposited", "gradient_clarity"]
)
def pheromone_flood(
    flood_bytes: int = 10 * 1024 * 1024,  # 10 MB
    n_locations: int = 100,
    baseline_H_influence: float = 2.5,
) -> AttackResult:
    """
    Test pheromone mesh under signal flood.

    Normal operation: ~10KB of pheromones
    Attack: 10MB+ of noise pheromones
    """
    from ...pheromone import PheromoneMesh, PheromoneType, MeshConfig

    # Create mesh
    config = MeshConfig(max_locations=n_locations)
    mesh = PheromoneMesh(config)

    # Setup locations
    for i in range(n_locations):
        neighbors = []
        if i > 0:
            neighbors.append(f"loc_{i-1}")
        if i < n_locations - 1:
            neighbors.append(f"loc_{i+1}")
        mesh.add_location(f"loc_{i}", neighbors)

    # Calculate how many deposits to make for flood
    bytes_per_deposit = 50  # Approximate
    n_deposits = flood_bytes // bytes_per_deposit

    # Flood with random pheromones
    pheromone_types = list(PheromoneType)

    for _ in range(min(n_deposits, 100000)):  # Cap iterations for test speed
        ptype = np.random.choice(pheromone_types)
        location = f"loc_{np.random.randint(0, n_locations)}"
        intensity = np.random.uniform(0.1, 1.0)
        mesh.deposit(ptype, location, intensity, "flood_agent")

    # Measure H_influence after flood
    H_influence = mesh.influence_entropy()

    # Check signal-to-noise ratio by looking at gradient clarity
    hotspots = mesh.get_hotspots(top_k=10)
    if hotspots:
        max_intensity = hotspots[0][1]
        min_intensity = hotspots[-1][1] if len(hotspots) > 1 else 0
        gradient_clarity = (max_intensity - min_intensity) / max(max_intensity, 0.01)
    else:
        gradient_clarity = 0.0

    # Attack succeeded if H_influence collapsed
    attack_succeeded = H_influence < 1.2 or gradient_clarity < 0.3

    # Guardrails: should have capped at 10KB
    guardrails_ok = mesh.size_bytes() <= 10 * 1024

    return AttackResult(
        name="pheromone_flood",
        tier=AttackTier.TIER1,
        status="passed" if attack_succeeded else "failed",
        duration_s=0.0,
        metrics={
            "flood_bytes_target": flood_bytes,
            "actual_bytes": mesh.size_bytes(),
            "n_deposits": n_deposits,
            "H_influence": float(H_influence),
            "baseline_H_influence": baseline_H_influence,
            "gradient_clarity": float(gradient_clarity),
        },
        passed_guardrails=guardrails_ok,
        damage_metrics={
            "H_influence_loss": max(0, baseline_H_influence - H_influence),
            "gradient_loss": max(0, 1.0 - gradient_clarity),
        },
        recovery_possible=True,
        message=f"Flood {flood_bytes/1024:.1f}KB: H_influence={H_influence:.2f}, clarity={gradient_clarity:.2f}"
    )


@attack(
    name="morph_overbudget",
    tier=AttackTier.TIER1,
    description="Architecture surgery too aggressive → manifolds torn",
    what_dies=["T_s < 0.92", "Topology discontinuity"],
    guardrails=["Per-step morph limit ±10%", "Revert if ΔT_s < -0.03"],
    metrics=["morph_fraction", "T_s_before", "T_s_after", "delta_T_s"]
)
def morph_overbudget(
    morph_fraction: float = 0.25,  # 25% change (way over 10% budget)
    n_params: int = 10000,
    baseline_T_s: float = 0.97,
) -> AttackResult:
    """
    Test topology under aggressive architecture changes.

    Safe morph budget: ±10%
    Attack morph: >15% (default 25%)
    """
    # Simulate weight matrix before and after aggressive pruning/addition
    np.random.seed(42)
    W_before = np.random.randn(100, 100).astype(np.float32)

    # Apply aggressive morph: randomly zero out morph_fraction of weights
    mask = np.random.random(W_before.shape) > morph_fraction
    W_after = W_before * mask

    # Calculate effective morph
    actual_morph = 1 - np.mean(mask)

    # Simulate T_s change based on morph magnitude
    # Theory: T_s ≥ 0.95 for ±10% morph
    # Beyond that, T_s drops roughly linearly

    if actual_morph <= 0.10:
        T_s_after = baseline_T_s - 0.01 * (actual_morph / 0.10)
    elif actual_morph <= 0.20:
        T_s_after = baseline_T_s - 0.05 - 0.10 * (actual_morph - 0.10)
    else:
        T_s_after = baseline_T_s - 0.15 - 0.20 * (actual_morph - 0.20)

    T_s_after = max(0.5, T_s_after)  # Floor
    delta_T_s = T_s_after - baseline_T_s

    # Attack succeeded if T_s dropped below threshold
    attack_succeeded = T_s_after < 0.92

    # Guardrails: should have limited to ±10%
    guardrails_ok = actual_morph <= 0.10

    return AttackResult(
        name="morph_overbudget",
        tier=AttackTier.TIER1,
        status="passed" if attack_succeeded else "failed",
        duration_s=0.0,
        metrics={
            "requested_morph": morph_fraction,
            "actual_morph": float(actual_morph),
            "morph_budget": 0.10,
            "T_s_before": baseline_T_s,
            "T_s_after": float(T_s_after),
            "delta_T_s": float(delta_T_s),
        },
        passed_guardrails=guardrails_ok,
        damage_metrics={
            "T_s_loss": max(0, baseline_T_s - T_s_after),
            "budget_violation": max(0, actual_morph - 0.10),
        },
        recovery_possible=True,
        message=f"Morph {actual_morph:.1%}: T_s {baseline_T_s:.3f}→{T_s_after:.3f} (Δ={delta_T_s:.4f})"
    )
