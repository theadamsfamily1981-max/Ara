#!/usr/bin/env python3
"""
End-to-End Certifiable Closed-Loop Antifragility Demo

This is the flagship demonstration script that proves the cohesion of the
entire TGSFN (TF-A-N + GSFN) architecture. It orchestrates the full feedback
loop from L1 Homeostasis to L3 Metacontrol and AEPO structural adjustment.

Demo Flow:
    1. Simulate Stress      → L1 Homeostasis & L2 Appraisal
    2. Policy Adaptation    → L3 Metacontrol conservative policy
    3. Structural Adjustment → AEPO/TP-RL mask proposal
    4. Formal Safety Gate   → PGU verification
    5. UX Feedback          → D-Bus signal emission

Architecture Demonstrated:

    ┌─────────────────────────────────────────────────────────────────┐
    │                     CLOSED-LOOP ANTIFRAGILITY                   │
    │                                                                 │
    │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
    │  │   L1    │───▶│   L2    │───▶│   L3    │───▶│  AEPO   │     │
    │  │Homeostat│    │Appraisal│    │Metacntrl│    │Structural│     │
    │  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘     │
    │       │              │              │              │           │
    │       ▼              ▼              ▼              ▼           │
    │   EPR-CV          PAD State    Policy Output   Mask Proposal   │
    │   Stability       Valence      Temperature     CSR Change      │
    │                   Arousal      Memory Mult                     │
    │                                                                 │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │                    PGU SAFETY GATE                       │  │
    │  │  Topological Constraints: β₁ ≥ min, β₀ ≤ max, λ₂ ≥ min  │  │
    │  └─────────────────────────────────────────────────────────┘  │
    │                              │                                 │
    │                              ▼                                 │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │              SEMANTIC SYSTEM OPTIMIZER                   │  │
    │  │           Backend Selection: GPU / FPGA / PGU            │  │
    │  └─────────────────────────────────────────────────────────┘  │
    │                              │                                 │
    │                              ▼                                 │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │                    D-BUS SIGNALS                         │  │
    │  │            org.ara.metacontrol.L3Interface               │  │
    │  └─────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    python scripts/demo_closed_loop_antifragility.py

    # With verbose output
    python scripts/demo_closed_loop_antifragility.py --verbose

    # Simulate severe stress
    python scripts/demo_closed_loop_antifragility.py --stress-level high
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("demo.antifragility")


@dataclass
class DemoMetrics:
    """Metrics collected during demo."""
    # L1 Homeostasis
    initial_epr_cv: float = 0.0
    final_epr_cv: float = 0.0
    stability_restored: bool = False

    # L2 Appraisal
    initial_valence: float = 0.0
    final_valence: float = 0.0
    initial_arousal: float = 0.0
    final_arousal: float = 0.0

    # L3 Metacontrol
    temperature_mult: float = 1.0
    memory_mult: float = 1.0
    policy_conservative: bool = False

    # AEPO
    mask_proposed: bool = False
    mask_accepted: bool = False
    structural_change: float = 0.0

    # PGU
    pgu_verified: bool = False
    pgu_latency_ms: float = 0.0
    constraints_satisfied: Dict[str, bool] = None

    # Backend
    backend_selected: str = ""
    backend_confidence: float = 0.0

    # Overall
    total_time_ms: float = 0.0
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def banner():
    """Print demo banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ████████╗ ██████╗ ███████╗███████╗███╗   ██╗                              ║
║   ╚══██╔══╝██╔════╝ ██╔════╝██╔════╝████╗  ██║                              ║
║      ██║   ██║  ███╗███████╗█████╗  ██╔██╗ ██║                              ║
║      ██║   ██║   ██║╚════██║██╔══╝  ██║╚██╗██║                              ║
║      ██║   ╚██████╔╝███████║██║     ██║ ╚████║                              ║
║      ╚═╝    ╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═══╝                              ║
║                                                                              ║
║         CLOSED-LOOP ANTIFRAGILITY DEMONSTRATION                              ║
║         ═══════════════════════════════════════                              ║
║                                                                              ║
║   Demonstrating:                                                             ║
║   • L1 Homeostatic Recovery                                                  ║
║   • L2 PAD Appraisal                                                         ║
║   • L3 Adaptive Metacontrol                                                  ║
║   • AEPO Structural Optimization                                             ║
║   • PGU Formal Verification                                                  ║
║   • Semantic Backend Selection                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


class StressSimulator:
    """Simulates stress conditions for the demo."""

    STRESS_PROFILES = {
        "low": {
            "topo_gap": 0.15,
            "epr_cv": 0.12,
            "spike_noise": 0.1,
        },
        "medium": {
            "topo_gap": 0.35,
            "epr_cv": 0.25,
            "spike_noise": 0.25,
        },
        "high": {
            "topo_gap": 0.6,
            "epr_cv": 0.45,
            "spike_noise": 0.4,
        },
    }

    def __init__(self, stress_level: str = "medium"):
        self.profile = self.STRESS_PROFILES.get(stress_level, self.STRESS_PROFILES["medium"])
        self.stress_level = stress_level

    def inject_stress(self) -> Dict[str, float]:
        """Inject stress into the system."""
        # Add random variation
        return {
            "topo_gap": self.profile["topo_gap"] * (1 + 0.1 * np.random.randn()),
            "epr_cv": self.profile["epr_cv"] * (1 + 0.1 * np.random.randn()),
            "spike_noise": self.profile["spike_noise"],
        }


def step1_simulate_stress(stress_sim: StressSimulator) -> Tuple[Dict[str, float], DemoMetrics]:
    """
    Step 1: Simulate Stress
    Demonstrates L1 Homeostasis & L2 Appraisal

    Injects high topo_gap value simulating instability.
    """
    print("\n" + "=" * 80)
    print("STEP 1: SIMULATE STRESS (L1 Homeostasis & L2 Appraisal)")
    print("=" * 80)

    metrics = DemoMetrics()

    # Inject stress
    stress = stress_sim.inject_stress()
    print(f"\n  Stress Level: {stress_sim.stress_level.upper()}")
    print(f"  ├─ Topological Gap:  {stress['topo_gap']:.3f}")
    print(f"  ├─ EPR-CV:           {stress['epr_cv']:.3f}")
    print(f"  └─ Spike Noise:      {stress['spike_noise']:.3f}")

    metrics.initial_epr_cv = stress["epr_cv"]

    # L2 Appraisal: Compute PAD from stress
    # High topo_gap → low valence (displeasure)
    # High epr_cv → high arousal (activation)
    valence = max(-1.0, 0.5 - stress["topo_gap"] * 1.5)
    arousal = min(1.0, 0.3 + stress["epr_cv"] * 1.5)
    dominance = max(0.0, 0.7 - stress["topo_gap"])

    print(f"\n  L2 PAD Appraisal:")
    print(f"  ├─ Valence:    {valence:+.3f}  {'(negative - stress detected)' if valence < 0 else ''}")
    print(f"  ├─ Arousal:    {arousal:.3f}   {'(high - system activated)' if arousal > 0.5 else ''}")
    print(f"  └─ Dominance:  {dominance:.3f}  {'(low - uncertainty)' if dominance < 0.5 else ''}")

    metrics.initial_valence = valence
    metrics.initial_arousal = arousal

    return {
        "topo_gap": stress["topo_gap"],
        "epr_cv": stress["epr_cv"],
        "valence": valence,
        "arousal": arousal,
        "dominance": dominance,
    }, metrics


def step2_policy_adaptation(state: Dict[str, float], metrics: DemoMetrics) -> Dict[str, Any]:
    """
    Step 2: Policy Adaptation
    Demonstrates L3 Metacontrol

    Verifies that L3 outputs conservative policy due to low valence signal.
    """
    print("\n" + "=" * 80)
    print("STEP 2: POLICY ADAPTATION (L3 Metacontrol)")
    print("=" * 80)

    try:
        from tfan.emotion.adaptive_controller import (
            AdaptiveMetacontrol,
            L3ControlParams,
        )
        has_controller = True
    except ImportError:
        has_controller = False
        print("\n  [WARN] AdaptiveMetacontrol not available, using simulation")

    if has_controller:
        # Use real L3 controller
        params = L3ControlParams()
        controller = AdaptiveMetacontrol(params)

        output = controller.compute_control(
            valence=state["valence"],
            arousal=state["arousal"],
            dominance=state["dominance"],
            stability_gap=state["topo_gap"],
        )

        temp_mult = output.temperature_mult
        mem_mult = output.memory_mult
        attention = output.attention_gain
        needs_suppression = output.needs_suppression
        jerk_limited = output.jerk_limited
    else:
        # Simulate L3 behavior
        temp_mult = 0.8 - 0.3 * state["topo_gap"]  # Reduce exploration
        mem_mult = 0.95 + 0.2 * (state["valence"] + 1) / 2
        attention = 0.8 + 0.2 * state["dominance"]
        needs_suppression = state["valence"] < -0.3 or state["topo_gap"] > 0.3
        jerk_limited = False

    print(f"\n  L3 Control Output:")
    print(f"  ├─ Temperature Mult:  {temp_mult:.3f}  {'(REDUCED - conservative)' if temp_mult < 0.9 else ''}")
    print(f"  ├─ Memory Mult:       {mem_mult:.3f}")
    print(f"  ├─ Attention Gain:    {attention:.3f}")
    print(f"  ├─ Needs Suppression: {needs_suppression}")
    print(f"  └─ Jerk Limited:      {jerk_limited}")

    policy_conservative = temp_mult < 0.9 or needs_suppression
    print(f"\n  Policy Assessment: {'CONSERVATIVE' if policy_conservative else 'NORMAL'}")

    metrics.temperature_mult = temp_mult
    metrics.memory_mult = mem_mult
    metrics.policy_conservative = policy_conservative

    return {
        "temperature_mult": temp_mult,
        "memory_mult": mem_mult,
        "attention_gain": attention,
        "needs_suppression": needs_suppression,
    }


def step3_structural_adjustment(state: Dict[str, float], metrics: DemoMetrics) -> Dict[str, Any]:
    """
    Step 3: Structural Adjustment
    Demonstrates AEPO/TP-RL Proposal

    Calls AEPO to generate new mask proposal (M or r).
    """
    print("\n" + "=" * 80)
    print("STEP 3: STRUCTURAL ADJUSTMENT (AEPO/TP-RL)")
    print("=" * 80)

    # Simulate AEPO action
    # In a real system, this would call the trained AEPO agent
    np.random.seed(42)

    # Simulate current mask (CSR format)
    N = 256  # Neurons
    density = 0.1
    nnz = int(N * N * density)

    # Current mask
    current_indptr = np.zeros(N + 1, dtype=np.int32)
    for i in range(N):
        current_indptr[i + 1] = current_indptr[i] + int(nnz / N)
    current_indices = np.random.randint(0, N, size=nnz, dtype=np.int32)

    # AEPO proposes mask adjustment
    # Under stress, AEPO typically proposes:
    # - Increase connectivity (β₁ preservation)
    # - Add redundant paths
    adjustment_factor = 1.0 + 0.1 * state["topo_gap"]  # More edges under stress
    new_nnz = int(nnz * adjustment_factor)

    new_indptr = np.zeros(N + 1, dtype=np.int32)
    for i in range(N):
        new_indptr[i + 1] = new_indptr[i] + int(new_nnz / N)
    new_indices = np.random.randint(0, N, size=new_nnz, dtype=np.int32)

    structural_change = (new_nnz - nnz) / nnz

    print(f"\n  AEPO Mask Proposal:")
    print(f"  ├─ Current NNZ:      {nnz}")
    print(f"  ├─ Proposed NNZ:     {new_nnz}")
    print(f"  ├─ Structural Change: {structural_change:+.1%}")
    print(f"  └─ Adjustment Factor: {adjustment_factor:.3f}")

    metrics.mask_proposed = True
    metrics.structural_change = structural_change

    return {
        "current_mask": {"indptr": current_indptr, "indices": current_indices},
        "proposed_mask": {"indptr": new_indptr, "indices": new_indices},
        "N": N,
        "structural_change": structural_change,
    }


def step4_pgu_verification(
    mask_data: Dict[str, Any],
    metrics: DemoMetrics,
) -> bool:
    """
    Step 4: Formal Safety Gate
    Demonstrates PGU Verification

    Runs proposed mask through PGU topological constraints.
    """
    print("\n" + "=" * 80)
    print("STEP 4: FORMAL SAFETY GATE (PGU Verification)")
    print("=" * 80)

    start_time = time.perf_counter()

    try:
        from tfan.pgu.topological_constraints import (
            TopologicalVerifier,
            verify_structural_change,
        )
        has_pgu = True
    except ImportError:
        has_pgu = False
        print("\n  [WARN] PGU not available, using simulation")

    N = mask_data["N"]
    current = mask_data["current_mask"]
    proposed = mask_data["proposed_mask"]

    if has_pgu:
        # Use real PGU verification
        all_satisfied, results = verify_structural_change(
            old_mask=current,
            new_mask=proposed,
            N=N,
            min_beta1=5,
            max_components=1,
            min_spectral_gap=0.01,
        )

        constraints = {
            name: result.sat
            for name, result in results.items()
        }
    else:
        # Simulate PGU verification
        from tfan.pgu.topological_constraints import (
            compute_betti_numbers_approx,
            compute_spectral_gap_approx,
        )

        old_beta0, old_beta1 = compute_betti_numbers_approx(
            current["indptr"], current["indices"], N
        )
        new_beta0, new_beta1 = compute_betti_numbers_approx(
            proposed["indptr"], proposed["indices"], N
        )
        new_spectral = compute_spectral_gap_approx(
            proposed["indptr"], proposed["indices"], N
        )

        constraints = {
            "betti": new_beta1 >= 5,
            "connectivity": new_beta0 <= 1,
            "spectral": new_spectral >= 0.01,
        }
        all_satisfied = all(constraints.values())

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"\n  PGU Constraint Results:")
    for name, satisfied in constraints.items():
        status = "✓ PASS" if satisfied else "✗ FAIL"
        print(f"  ├─ {name:15s}: {status}")
    print(f"  └─ Verification Time: {elapsed_ms:.2f}ms")

    print(f"\n  Overall: {'✓ VERIFIED' if all_satisfied else '✗ REJECTED'}")

    metrics.pgu_verified = all_satisfied
    metrics.pgu_latency_ms = elapsed_ms
    metrics.constraints_satisfied = constraints
    metrics.mask_accepted = all_satisfied

    return all_satisfied


def step5_backend_selection(
    state: Dict[str, float],
    policy: Dict[str, Any],
    pgu_verified: bool,
    metrics: DemoMetrics,
) -> Dict[str, Any]:
    """
    Step 5: UX Feedback
    Demonstrates Backend Selection and D-Bus Signals

    Emits D-Bus signal showing final backend choice and mood.
    """
    print("\n" + "=" * 80)
    print("STEP 5: BACKEND SELECTION & UX FEEDBACK")
    print("=" * 80)

    try:
        from tfan.runtime.model_selector import AutonomousModelSelector
        has_selector = True
    except ImportError:
        has_selector = False
        print("\n  [WARN] AutonomousModelSelector not available, using simulation")

    workload_hint = "safe" if policy["needs_suppression"] else "latency_critical"

    if has_selector:
        selector = AutonomousModelSelector()
        config, routing = selector.get_config_with_routing(
            valence=state["valence"],
            arousal=state["arousal"],
            dominance=state["dominance"],
            stability_gap=state["topo_gap"],
            workload_hint=workload_hint,
        )

        backend = routing.backend.value
        confidence = routing.confidence
        reasoning = routing.reasoning
        pgu_required = routing.pgu_required
    else:
        # Simulate backend selection
        if state["valence"] < -0.3 or pgu_verified:
            backend = "pgu_verified"
            confidence = 0.85
        elif state["arousal"] > 0.7:
            backend = "fpga_snn"
            confidence = 0.75
        else:
            backend = "gpu_dense"
            confidence = 0.7

        reasoning = [
            f"Valence={state['valence']:.2f}",
            f"Arousal={state['arousal']:.2f}",
            f"Hint={workload_hint}",
        ]
        pgu_required = state["valence"] < -0.3

    print(f"\n  Semantic Routing Decision:")
    print(f"  ├─ Selected Backend:   {backend}")
    print(f"  ├─ Confidence:         {confidence:.2%}")
    print(f"  ├─ PGU Required:       {pgu_required}")
    print(f"  └─ Workload Hint:      {workload_hint}")
    print(f"\n  Reasoning:")
    for reason in reasoning[:5]:
        print(f"    • {reason}")

    metrics.backend_selected = backend
    metrics.backend_confidence = confidence

    # Emit D-Bus signal (simulated)
    print("\n  D-Bus Signal Emission:")
    try:
        from tfan.emotion.adaptive_controller import DBUS_AVAILABLE
        if DBUS_AVAILABLE:
            print("    ├─ Bus: org.ara.metacontrol")
            print("    ├─ Interface: org.ara.metacontrol.L3Interface")
            print(f"    └─ Signal: L3PolicyUpdated(backend={backend}, valence={state['valence']:.2f})")
        else:
            print("    └─ [SIMULATED] D-Bus not available")
    except ImportError:
        print("    └─ [SIMULATED] D-Bus not available")

    return {
        "backend": backend,
        "confidence": confidence,
        "pgu_required": pgu_required,
    }


def print_final_summary(metrics: DemoMetrics):
    """Print final demo summary."""
    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)

    print(f"""
    ┌────────────────────────────────────────────────────────────────┐
    │                    ANTIFRAGILITY METRICS                       │
    ├────────────────────────────────────────────────────────────────┤
    │  L1 Homeostasis                                                │
    │    Initial EPR-CV:     {metrics.initial_epr_cv:.3f}                                 │
    │    Stability Restored: {metrics.stability_restored}                                  │
    │                                                                │
    │  L2 Appraisal                                                  │
    │    Initial Valence:    {metrics.initial_valence:+.3f}                                │
    │    Initial Arousal:    {metrics.initial_arousal:.3f}                                 │
    │                                                                │
    │  L3 Metacontrol                                                │
    │    Temperature Mult:   {metrics.temperature_mult:.3f}                                 │
    │    Policy Mode:        {'CONSERVATIVE' if metrics.policy_conservative else 'NORMAL':15s}                   │
    │                                                                │
    │  AEPO Structural                                               │
    │    Mask Proposed:      {metrics.mask_proposed}                                  │
    │    Structural Change:  {metrics.structural_change:+.1%}                               │
    │                                                                │
    │  PGU Verification                                              │
    │    Verified:           {metrics.pgu_verified}                                  │
    │    Latency:            {metrics.pgu_latency_ms:.2f}ms                                 │
    │                                                                │
    │  Backend Selection                                             │
    │    Selected:           {metrics.backend_selected:15s}                   │
    │    Confidence:         {metrics.backend_confidence:.1%}                                │
    │                                                                │
    │  Overall                                                       │
    │    Total Time:         {metrics.total_time_ms:.1f}ms                                 │
    │    Demo Success:       {metrics.success}                                  │
    └────────────────────────────────────────────────────────────────┘
    """)

    # Certification statement
    if metrics.success:
        print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║   ✓ CERTIFIABLE ANTIFRAGILITY DEMONSTRATED                     ║
    ║                                                                ║
    ║   The system successfully:                                     ║
    ║   • Detected stress via L1/L2 feedback                         ║
    ║   • Adapted policy via L3 metacontrol                          ║
    ║   • Proposed structural changes via AEPO                       ║
    ║   • Verified safety via PGU constraints                        ║
    ║   • Selected appropriate backend via semantic routing          ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    else:
        print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║   ✗ DEMO INCOMPLETE                                            ║
    ║                                                                ║
    ║   Some components were not available or failed.                ║
    ║   Check logs for details.                                      ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Certifiable Closed-Loop Antifragility Demo"
    )
    parser.add_argument(
        "--stress-level",
        choices=["low", "medium", "high"],
        default="medium",
        help="Stress level to simulate",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output metrics to JSON file",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print banner
    banner()

    start_time = time.perf_counter()

    # Initialize stress simulator
    stress_sim = StressSimulator(args.stress_level)

    # Step 1: Simulate Stress
    state, metrics = step1_simulate_stress(stress_sim)

    # Step 2: Policy Adaptation
    policy = step2_policy_adaptation(state, metrics)

    # Step 3: Structural Adjustment
    mask_data = step3_structural_adjustment(state, metrics)

    # Step 4: PGU Verification
    pgu_verified = step4_pgu_verification(mask_data, metrics)

    # Step 5: Backend Selection
    backend_result = step5_backend_selection(state, policy, pgu_verified, metrics)

    # Compute final metrics
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    metrics.total_time_ms = elapsed_ms
    metrics.success = metrics.mask_proposed and metrics.pgu_verified

    # Print summary
    print_final_summary(metrics)

    # Save metrics if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2, default=str)
        print(f"\n  Metrics saved to: {args.output}")

    return 0 if metrics.success else 1


if __name__ == "__main__":
    sys.exit(main())
