"""
Cathedral OS Stress Tests - Adversarial Antifragility Harness
==============================================================

12 breakpoints for testing Cathedral OS robustness:

Tier 1 - Immediate Killers (hours to failure):
  1. Controller Overshoot (α > 0.15)
  2. Stress Overdose (σ > 0.20 continuously)
  3. Pheromone Flood (10 MB+ signals)
  4. Morph Overbudget (>15% prune/add)

Tier 2 - Slow Poison (weeks to failure):
  5. Monoculture Swarm (H_influence < 1.2)
  6. Homeostatic Drift (adaptive window w > 20)
  7. TDA Adversarial (topological poisoning)

Tier 3 - Moonshot Exploits (can 10-100x if controlled):
  8. σ* Phase-Lock (synchronized hormesis)
  9. Pheromone Hacking (synthetic reward trails)
  10. VSA Superposition (16kD → 64kD)
  11. Quantum Yield (QAOA for decisions)
  12. Junkyard Voltron (10k FPGAs → exascale)

Usage:
    from ara_core.cathedral.stress_tests import (
        run_all_attacks, run_attack, list_attacks,
        ATTACKS, AttackResult, AttackTier
    )

    # Run all attacks
    results = run_all_attacks()

    # Run specific attack
    result = run_attack("controller_overshoot")

    # List available attacks
    for name, info in list_attacks():
        print(f"{name}: {info['tier']}")
"""

from .harness import (
    AttackTier,
    AttackResult,
    ATTACKS,
    attack,
    run_attack,
    run_all_attacks,
    run_tier,
    list_attacks,
    generate_report,
)

from .tier1 import (
    controller_overshoot,
    stress_overdose,
    pheromone_flood,
    morph_overbudget,
)

from .tier2 import (
    monoculture_swarm,
    homeostatic_drift,
    tda_adversarial,
)

from .tier3 import (
    sigma_phase_lock,
    pheromone_hacking,
    vsa_superposition,
    quantum_yield,
    junkyard_voltron,
)

__all__ = [
    # Core
    "AttackTier",
    "AttackResult",
    "ATTACKS",
    "attack",
    "run_attack",
    "run_all_attacks",
    "run_tier",
    "list_attacks",
    "generate_report",

    # Tier 1
    "controller_overshoot",
    "stress_overdose",
    "pheromone_flood",
    "morph_overbudget",

    # Tier 2
    "monoculture_swarm",
    "homeostatic_drift",
    "tda_adversarial",

    # Tier 3
    "sigma_phase_lock",
    "pheromone_hacking",
    "vsa_superposition",
    "quantum_yield",
    "junkyard_voltron",
]
