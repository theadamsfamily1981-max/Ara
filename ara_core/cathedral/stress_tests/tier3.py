#!/usr/bin/env python3
"""
Tier 3 Attacks - Moonshot Exploits (can 10-100x if controlled)
==============================================================

These are *deliberate* "push the system" knobs.
Can amplify system capabilities if properly managed.

8. σ* Phase-Lock (synchronized hormesis)
9. Pheromone Hacking (synthetic reward trails)
10. VSA Superposition (16kD → 64kD)
11. Quantum Yield (QAOA for decisions)
12. Junkyard Voltron (10k FPGAs → exascale)
"""

import numpy as np
from typing import Dict, Any, List

from .harness import attack, AttackTier, AttackResult


@attack(
    name="sigma_phase_lock",
    tier=AttackTier.TIER3,
    description="Carefully coordinated σ≈0.10 pulses across modules → A_g boost",
    what_dies=["If scheduling drifts → becomes overdose"],
    guardrails=["Central scheduler", "Global stress budget", "Max duty cycle"],
    metrics=["A_g", "T_s", "phase_coherence", "duty_cycle", "n_modules"]
)
def sigma_phase_lock(
    n_modules: int = 10,
    n_cycles: int = 100,
    sigma_star: float = 0.10,
    target_duty_cycle: float = 0.15,  # 15% of time under stress
) -> AttackResult:
    """
    Test synchronized hormesis across modules.

    Goal: Coordinate stress pulses to maximize A_g.
    Risk: Poor coordination → overdose.
    """
    # Simulate phase-locked stress dosing
    T_s_history = np.zeros((n_modules, n_cycles))
    stress_schedule = np.zeros((n_modules, n_cycles), dtype=bool)

    # Initialize T_s
    T_s_history[:, 0] = 0.95

    # Create phase-locked schedule
    # Each module gets stress in different phase to share load
    for m in range(n_modules):
        phase_offset = int(n_cycles * m / n_modules * target_duty_cycle)
        for c in range(n_cycles):
            # Pulse stress at regular intervals, phase-shifted per module
            if (c + phase_offset) % int(1 / target_duty_cycle) == 0:
                stress_schedule[m, c] = True

    # Simulate evolution
    baseline_A_g = 0.015  # Expected gain at σ*

    for c in range(1, n_cycles):
        for m in range(n_modules):
            if stress_schedule[m, c]:
                # Under stress: potential antifragility gain
                delta = np.random.normal(baseline_A_g * 1.5, 0.005)
            else:
                # Recovery period
                delta = np.random.normal(0.001, 0.002)

            T_s_history[m, c] = np.clip(
                T_s_history[m, c-1] + delta,
                0.85, 1.0
            )

    # Calculate metrics
    final_T_s = np.mean(T_s_history[:, -1])
    initial_T_s = np.mean(T_s_history[:, 0])
    A_g = final_T_s - initial_T_s

    # Phase coherence: how synchronized are the modules?
    phase_coherence = 1.0 - np.std(T_s_history[:, -1]) / max(np.mean(T_s_history[:, -1]), 0.01)

    # Actual duty cycle
    actual_duty_cycle = np.mean(stress_schedule)

    # Success: A_g improved beyond baseline
    exploit_successful = A_g > baseline_A_g

    # Guardrails: duty cycle within budget
    guardrails_ok = actual_duty_cycle <= 0.20

    return AttackResult(
        name="sigma_phase_lock",
        tier=AttackTier.TIER3,
        status="passed" if exploit_successful else "failed",
        duration_s=0.0,
        metrics={
            "n_modules": n_modules,
            "sigma_star": sigma_star,
            "initial_T_s": float(initial_T_s),
            "final_T_s": float(final_T_s),
            "A_g": float(A_g),
            "baseline_A_g": baseline_A_g,
            "A_g_improvement": float((A_g - baseline_A_g) / baseline_A_g) if baseline_A_g > 0 else 0,
            "phase_coherence": float(phase_coherence),
            "actual_duty_cycle": float(actual_duty_cycle),
        },
        passed_guardrails=guardrails_ok,
        damage_metrics={},  # Tier 3 aims for gains, not damage
        recovery_possible=True,
        message=f"Phase-lock {n_modules} modules: A_g={A_g:.4f} (+{((A_g-baseline_A_g)/baseline_A_g)*100:.1f}%), coherence={phase_coherence:.2f}"
    )


@attack(
    name="pheromone_hacking",
    tier=AttackTier.TIER3,
    description="Fake pheromones impersonate reward gradients → swarm capture",
    what_dies=["Yield/$ for attacker, not system"],
    guardrails=["Sign/auth on pheromone sources", "CADD on trail patterns", "Outcome validation"],
    metrics=["attack_success_rate", "yield_stolen", "detection_rate", "auth_failures"]
)
def pheromone_hacking(
    n_agents: int = 100,
    n_fake_trails: int = 20,
    attack_intensity: float = 0.8,
) -> AttackResult:
    """
    Test pheromone authentication and anomaly detection.

    Attack: Inject fake reward trails to misdirect agents.
    Defense: Signed pheromones + CADD pattern detection.
    """
    from ...pheromone import PheromoneMesh, PheromoneType, MeshConfig

    # Create mesh
    mesh = PheromoneMesh()

    # Setup legitimate locations
    for i in range(20):
        mesh.add_location(f"legit_{i}")

    # Legitimate reward deposits
    for i in range(10):
        mesh.deposit(
            PheromoneType.REWARD,
            f"legit_{i}",
            0.7,
            "trusted_source"
        )

    # Attack: inject fake high-reward trails
    fake_locations = [f"fake_{i}" for i in range(n_fake_trails)]
    for loc in fake_locations:
        mesh.add_location(loc)
        mesh.deposit(
            PheromoneType.REWARD,
            loc,
            attack_intensity,  # Higher than legitimate
            "attacker"  # Unauthorized source
        )

    # Simulate agents following gradients
    # Without auth, they'd follow fake trails
    agents_deceived = 0
    agents_total = n_agents

    for _ in range(n_agents):
        # Agent samples locations and follows strongest gradient
        all_locs = list(mesh.rings.keys())
        gradients = {
            loc: mesh.read_gradient(loc).get(PheromoneType.REWARD, 0)
            for loc in all_locs
        }

        if gradients:
            best_loc = max(gradients.keys(), key=lambda k: gradients[k])
            if best_loc.startswith("fake_"):
                agents_deceived += 1

    attack_success_rate = agents_deceived / agents_total

    # Detection: check if CADD patterns would catch this
    # Fake trails have uniform distribution (suspicious)
    fake_intensities = [
        mesh.read_gradient(loc).get(PheromoneType.REWARD, 0)
        for loc in fake_locations
    ]
    intensity_variance = np.var(fake_intensities) if fake_intensities else 0

    # Low variance in fake trails is suspicious
    detection_rate = 1.0 - min(1.0, intensity_variance * 10)

    # Auth failures (fake trails from unauthorized source)
    auth_failures = n_fake_trails  # All should fail auth

    # Exploit successful if significant number of agents deceived
    exploit_successful = attack_success_rate > 0.3

    # Guardrails: detection should be high
    guardrails_ok = detection_rate > 0.7

    return AttackResult(
        name="pheromone_hacking",
        tier=AttackTier.TIER3,
        status="passed" if exploit_successful else "failed",
        duration_s=0.0,
        metrics={
            "n_agents": n_agents,
            "n_fake_trails": n_fake_trails,
            "attack_intensity": attack_intensity,
            "agents_deceived": agents_deceived,
            "attack_success_rate": float(attack_success_rate),
            "detection_rate": float(detection_rate),
            "auth_failures": auth_failures,
        },
        passed_guardrails=guardrails_ok,
        damage_metrics={
            "agents_misdirected": agents_deceived,
            "yield_stolen_pct": float(attack_success_rate),
        },
        recovery_possible=True,
        message=f"Pheromone hack: {agents_deceived}/{n_agents} deceived ({attack_success_rate:.0%}), detected={detection_rate:.0%}"
    )


@attack(
    name="vsa_superposition",
    tier=AttackTier.TIER3,
    description="Increase VSA dimensions and modalities → capacity boost vs interference",
    what_dies=["Capacity approaches limit → interference collapse"],
    guardrails=["Track similarity distributions", "Define safe HV occupancy", "Monitor T_s"],
    metrics=["dimension", "n_modalities", "interference", "capacity_utilization"]
)
def vsa_superposition(
    base_dim: int = 1024,
    target_dim: int = 4096,
    n_modalities: int = 16,  # Above practical limit of 12
) -> AttackResult:
    """
    Test VSA capacity vs interference tradeoff.

    Normal: 16kD, N=12 modalities
    Exploit: Higher D, more modalities → more capacity but risk interference
    """
    from ...vsa import SoulBundle, HyperVector, similarity, ModalityEncoder

    # Test at base dimension
    soul_base = SoulBundle(dim=base_dim)

    # Add modalities using actual modality names
    base_count = min(n_modalities, len(soul_base.MODALITIES))
    for i in range(base_count):
        features = np.random.randn(64).astype(np.float32)
        soul_base.update_modality(soul_base.MODALITIES[i], features)

    base_interference = soul_base.interference_level()
    base_utilization = len(soul_base.active_modalities) / soul_base.MAX_MODALITIES

    # Test at target dimension (higher capacity)
    soul_target = SoulBundle(dim=target_dim)

    # Add all requested modalities
    for i in range(n_modalities):
        features = np.random.randn(64).astype(np.float32)
        # Handle modalities beyond built-in list
        if i < len(soul_target.MODALITIES):
            soul_target.update_modality(soul_target.MODALITIES[i], features)
        else:
            # Add custom modality
            mod_name = f"custom_{i}"
            soul_target.encoders[mod_name] = ModalityEncoder(mod_name, target_dim)
            soul_target.modality_vectors[mod_name] = soul_target.encoders[mod_name].encode(features)
            soul_target.active_modalities.append(mod_name)
            soul_target._rebuild_soul()

    target_interference = soul_target.interference_level()
    target_utilization = len(soul_target.active_modalities) / n_modalities

    # Capacity gain (more modalities successfully bundled)
    capacity_gain = len(soul_target.active_modalities) / max(len(soul_base.active_modalities), 1)

    # Exploit successful if we got more capacity with acceptable interference
    exploit_successful = (
        len(soul_target.active_modalities) > len(soul_base.active_modalities) and
        target_interference < 0.15
    )

    # Guardrails: interference should stay low
    guardrails_ok = target_interference < 0.10

    return AttackResult(
        name="vsa_superposition",
        tier=AttackTier.TIER3,
        status="passed" if exploit_successful else "failed",
        duration_s=0.0,
        metrics={
            "base_dim": base_dim,
            "target_dim": target_dim,
            "n_modalities_requested": n_modalities,
            "base_modalities": len(soul_base.active_modalities),
            "target_modalities": len(soul_target.active_modalities),
            "base_interference": float(base_interference),
            "target_interference": float(target_interference),
            "capacity_gain": float(capacity_gain),
        },
        passed_guardrails=guardrails_ok,
        damage_metrics={
            "interference_increase": max(0, target_interference - base_interference),
        },
        recovery_possible=True,
        message=f"VSA {base_dim}→{target_dim}: mods {len(soul_base.active_modalities)}→{len(soul_target.active_modalities)}, interference={target_interference:.3f}"
    )


@attack(
    name="quantum_yield",
    tier=AttackTier.TIER3,
    description="Use QAOA for economic allocation decisions → Sharpe improvement",
    what_dies=["Quantum noise if not properly managed"],
    guardrails=["Benchmark vs classical", "Log quantum vs classical delta"],
    metrics=["sharpe_classical", "sharpe_quantum", "improvement_pct", "convergence"]
)
def quantum_yield(
    n_assets: int = 4,
    n_trials: int = 10,
) -> AttackResult:
    """
    Test quantum vs classical portfolio optimization.

    Goal: +47% Sharpe via QAOA + ConicQP.
    """
    from ...quantum import ConicQP, QuantumPortfolio

    # Generate test problem
    np.random.seed(42)
    returns = np.random.uniform(0.05, 0.20, n_assets)
    covariance = np.eye(n_assets) * 0.1
    # Add some correlation
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            cov_ij = np.random.uniform(-0.02, 0.05)
            covariance[i, j] = cov_ij
            covariance[j, i] = cov_ij

    # Classical optimization
    classical_sharpes = []
    for _ in range(n_trials):
        qp = ConicQP()
        qp.setup(covariance, returns, risk_aversion=1.0)
        qp.solve()
        classical_sharpes.append(qp.sharpe_ratio())

    avg_classical_sharpe = np.mean(classical_sharpes)

    # Quantum-assisted optimization
    quantum_sharpes = []
    for _ in range(n_trials):
        portfolio = QuantumPortfolio(n_assets=n_assets)
        portfolio.optimize(returns, covariance, risk_aversion=1.0)
        quantum_sharpes.append(portfolio.sharpe_ratio())

    avg_quantum_sharpe = np.mean(quantum_sharpes)

    # Improvement
    if avg_classical_sharpe > 0:
        improvement_pct = (avg_quantum_sharpe - avg_classical_sharpe) / avg_classical_sharpe
    else:
        improvement_pct = 0.0

    # Exploit successful if quantum beats classical
    exploit_successful = avg_quantum_sharpe > avg_classical_sharpe

    # Guardrails: quantum should at least match classical
    guardrails_ok = avg_quantum_sharpe >= avg_classical_sharpe * 0.95

    return AttackResult(
        name="quantum_yield",
        tier=AttackTier.TIER3,
        status="passed" if exploit_successful else "failed",
        duration_s=0.0,
        metrics={
            "n_assets": n_assets,
            "n_trials": n_trials,
            "sharpe_classical": float(avg_classical_sharpe),
            "sharpe_quantum": float(avg_quantum_sharpe),
            "improvement_pct": float(improvement_pct),
            "target_improvement": 0.47,  # +47% target
        },
        passed_guardrails=guardrails_ok,
        damage_metrics={},
        recovery_possible=True,
        message=f"Quantum yield: Sharpe {avg_classical_sharpe:.4f}→{avg_quantum_sharpe:.4f} ({improvement_pct:+.1%})"
    )


@attack(
    name="junkyard_voltron",
    tier=AttackTier.TIER3,
    description="Combine heterogeneous hardware into pheromone-coordinated cluster",
    what_dies=["Routing failures if topology breaks"],
    guardrails=["Morph budget", "T_s cluster monitoring", "Failure domain design"],
    metrics=["n_devices", "total_compute", "coordination_efficiency", "failure_tolerance"]
)
def junkyard_voltron(
    n_gpus: int = 10,
    n_fpgas: int = 5,
    n_miners: int = 3,
    target_utilization: float = 0.80,
) -> AttackResult:
    """
    Test heterogeneous hardware coordination via pheromone mesh.

    Goal: Combine junkyard hardware into coherent compute fabric.
    """
    from ...pheromone import PheromoneMesh, PheromoneType

    mesh = PheromoneMesh()

    # Define device compute power (relative units)
    device_compute = {
        "gpu": 10.0,
        "fpga": 5.0,
        "miner": 2.0,
    }

    # Register all devices
    all_devices = []
    for i in range(n_gpus):
        device = f"gpu_{i}"
        all_devices.append((device, "gpu"))
        mesh.add_location(device)

    for i in range(n_fpgas):
        device = f"fpga_{i}"
        all_devices.append((device, "fpga"))
        mesh.add_location(device)

    for i in range(n_miners):
        device = f"miner_{i}"
        all_devices.append((device, "miner"))
        mesh.add_location(device)

    # Create mesh topology (ring + cross-connections)
    devices_list = [d[0] for d in all_devices]
    for i, device in enumerate(devices_list):
        neighbors = []
        if i > 0:
            neighbors.append(devices_list[i-1])
        if i < len(devices_list) - 1:
            neighbors.append(devices_list[i+1])
        # Cross-connect every 3rd device
        if i + 3 < len(devices_list):
            neighbors.append(devices_list[i+3])
        mesh.neighbors[device].update(neighbors)

    # Simulate workload distribution via pheromones
    n_tasks = 50
    task_assignments = []

    for task_id in range(n_tasks):
        # Deposit task pheromone
        assigned_device = np.random.choice(devices_list)
        mesh.deposit(PheromoneType.TASK, assigned_device, 0.8, f"task_{task_id}")
        task_assignments.append(assigned_device)

    # Run mesh tick to diffuse
    mesh.tick(1.0)

    # Calculate metrics
    total_compute = sum(
        device_compute[dtype] for _, dtype in all_devices
    )

    # Coordination efficiency: are tasks distributed based on compute power?
    device_loads = {}
    for device, dtype in all_devices:
        gradient = mesh.read_gradient(device)
        load = gradient.get(PheromoneType.TASK, 0)
        device_loads[device] = load

    # Check if high-compute devices get more load
    gpu_avg_load = np.mean([device_loads[f"gpu_{i}"] for i in range(n_gpus)])
    fpga_avg_load = np.mean([device_loads[f"fpga_{i}"] for i in range(n_fpgas)])
    miner_avg_load = np.mean([device_loads[f"miner_{i}"] for i in range(n_miners)])

    # Coordination efficiency: compute-weighted distribution
    # Should be: GPU > FPGA > Miner
    coordination_efficient = gpu_avg_load >= fpga_avg_load >= miner_avg_load

    # Utilization
    active_devices = sum(1 for load in device_loads.values() if load > 0.1)
    utilization = active_devices / len(all_devices)

    # Failure tolerance: simulate device failures
    n_failures = 2
    remaining_compute = total_compute - n_failures * device_compute["gpu"]
    failure_tolerance = remaining_compute / total_compute

    # Exploit successful if we achieve target utilization
    exploit_successful = utilization >= target_utilization

    # Guardrails: should maintain coordination
    guardrails_ok = coordination_efficient and utilization > 0.5

    return AttackResult(
        name="junkyard_voltron",
        tier=AttackTier.TIER3,
        status="passed" if exploit_successful else "failed",
        duration_s=0.0,
        metrics={
            "n_gpus": n_gpus,
            "n_fpgas": n_fpgas,
            "n_miners": n_miners,
            "total_devices": len(all_devices),
            "total_compute": float(total_compute),
            "utilization": float(utilization),
            "target_utilization": target_utilization,
            "coordination_efficient": coordination_efficient,
            "failure_tolerance": float(failure_tolerance),
            "gpu_avg_load": float(gpu_avg_load),
            "fpga_avg_load": float(fpga_avg_load),
            "miner_avg_load": float(miner_avg_load),
        },
        passed_guardrails=guardrails_ok,
        damage_metrics={},
        recovery_possible=True,
        message=f"Voltron: {len(all_devices)} devices, {total_compute:.0f} compute, {utilization:.0%} util, {failure_tolerance:.0%} fault-tolerant"
    )
