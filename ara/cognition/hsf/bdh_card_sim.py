#!/usr/bin/env python3
"""
BDH-Card Integration Simulation (Minimal Prototype)
====================================================

A runnable prototype of the Card ↔ BDH symbiosis:
- State hypervector: 128-d float
- BDH neurons: 256-d, sparse activations + Hebbian plasticity
- 3 scenarios:
  1) Normal telemetry  -> no anomaly
  2) High telemetry    -> anomaly -> BDH policy
  3) Same high telem   -> check adaptation after policy

The Card is the always-on state machine (hypervector accumulator).
The BDH is the plastic "cortex" (sparse spiking + Hebbian learning).
A tiny policy loop closes the gap.

Usage:
    python -m ara.cognition.hsf.bdh_card_sim
    python -m ara.cognition.hsf.bdh_card_sim --evolve 50
    python -m ara.cognition.hsf.bdh_card_sim --evolve 100 --plot

Knobs to play with:
    BDHConfig.hebb_lr    - Hebbian learning rate (try 0.02 → 0.1)
    BDHConfig.sparsity   - Fraction of neurons active (try 0.05 → 0.01)
    BDHConfig.steps      - Temporal steps per inference (try 3 → 10)
"""

import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any

# ----------------------------
# Configuration
# ----------------------------

DIM_STATE = 128
DIM_NEURONS = 256


# ----------------------------
# Hypervector utilities (numpy)
# ----------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def init_hv(dim: int = DIM_STATE) -> np.ndarray:
    """Initialize a random unit hypervector."""
    return normalize(np.random.randn(dim))


def bind(a: np.ndarray, b: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Simple binding via scaled addition."""
    return normalize(a + alpha * b)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two hypervectors."""
    return float(np.dot(a, b))


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def topk_indices(x: np.ndarray, k: int) -> np.ndarray:
    """Get indices of top-k values."""
    return np.argpartition(x, -k)[-k:]


# ----------------------------
# BDH "Dragon Hatchling" core
# ----------------------------

@dataclass
class BDHConfig:
    """Configuration for the BDH spiking layer."""
    dim_state: int = DIM_STATE
    dim_neurons: int = DIM_NEURONS
    sparsity: float = 0.05         # top-k fraction active
    hebb_lr: float = 0.02          # Hebbian learning rate
    steps: int = 3                 # temporal steps per inference


class BDHCore:
    """
    Simplified BDH-like spiking layer:
      - Linear projection: state -> neurons
      - ReLU + top-k sparsity (spikes)
      - Hebbian weight updates based on co-activation

    This simulates what would run on neuromorphic hardware.
    """

    def __init__(self, cfg: BDHConfig):
        self.cfg = cfg
        # dense weight for sim; hardware would be sparse
        self.weight = np.random.randn(cfg.dim_neurons, cfg.dim_state) * 0.1
        # per-neuron threshold (can be tuned by "policy")
        self.threshold = np.zeros(cfg.dim_neurons)
        # Track activation history for analysis
        self.activation_history: List[float] = []

    def forward(self, state_hv: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Run 'steps' temporal iterations, apply Hebbian plasticity.

        Returns:
          - new_state_hv: updated hypervector
          - debug: dict for logging
        """
        cfg = self.cfg
        state = state_hv.reshape(1, -1)  # [1, dim_state]
        hebb_lr = cfg.hebb_lr

        total_spikes = 0
        sparse_act = None

        for _ in range(cfg.steps):
            # 1. Project: state -> neuron potentials
            pot = state @ self.weight.T  # [1, dim_neurons]

            # 2. Apply per-neuron thresholds and ReLU
            pot = pot - self.threshold
            act = relu(pot)

            # 3. Sparsify: keep top-k activations
            k = max(1, int(cfg.sparsity * cfg.dim_neurons))
            topk_idx = topk_indices(act.flatten(), k)
            sparse_act = np.zeros_like(act)
            sparse_act[0, topk_idx] = act[0, topk_idx]

            total_spikes += int((sparse_act > 0).sum())

            # 4. Hebbian update: ΔW ~ hebb_lr * (act^T * state)
            #    (local plasticity; we clamp to avoid explosions)
            delta_w = hebb_lr * (sparse_act.T @ state)
            self.weight += delta_w
            self.weight = np.clip(self.weight, -1.0, 1.0)

            # 5. Fold back to state (readout) - simple linear + norm
            state = normalize((sparse_act @ self.weight).flatten()).reshape(1, -1)

        new_state = state.flatten()
        mean_act = float(sparse_act.mean()) if sparse_act is not None else 0.0
        max_act = float(sparse_act.max()) if sparse_act is not None else 0.0
        self.activation_history.append(mean_act)

        debug = {
            "mean_act": mean_act,
            "max_act": max_act,
            "total_spikes": total_spikes,
            "weight_norm": float(np.linalg.norm(self.weight)),
        }
        return new_state, debug

    def propose_policy(self, anomaly_score: float, threshold: float) -> Tuple[float, str]:
        """
        Very simple "policy head":
        - If anomaly is high => suggest slightly increasing threshold.
        - Also returns a text description.
        """
        # Scale delta by how far we are beyond threshold
        over = max(0.0, anomaly_score - threshold)
        delta = 0.05 * (over / max(threshold, 1e-6))
        desc = f"Adjust anomaly threshold by +{delta:.4f}"
        return delta, desc


# ----------------------------
# Card-side logic
# ----------------------------

@dataclass
class CardConfig:
    """Configuration for the neuromorphic card."""
    anomaly_threshold: float = 1.25   # lowered for normalized vectors
    bind_alpha: float = 1.0           # how strongly new telemetry binds


@dataclass
class TelemetryFrame:
    """A single telemetry frame."""
    friction: float = 0.0
    flap: float = 0.0
    heat: float = 0.0
    noise: float = 0.0


class NeuromorphicCard:
    """
    The always-on neuromorphic card.

    Maintains a state hypervector that accumulates telemetry.
    Detects anomalies and invokes BDH when needed.
    """

    def __init__(self, card_cfg: CardConfig, bdh_cfg: BDHConfig):
        self.card_cfg = card_cfg
        self.bdh_cfg = bdh_cfg
        self.state_hv = init_hv()
        self.baseline_hv = self.state_hv.copy()  # reference for drift
        self.bdh = BDHCore(bdh_cfg)

        # History for plotting
        self.history: List[Dict] = []

    def anomaly_score(self) -> float:
        """
        Compute anomaly score.

        Uses both norm deviation and drift from baseline.
        """
        norm_score = float(np.linalg.norm(self.state_hv))
        drift_score = 1.0 - similarity(self.state_hv, self.baseline_hv)
        # Combine: norm deviation + drift
        return norm_score + drift_score * 0.5

    def ingest_telemetry(self, frame: TelemetryFrame):
        """
        Ingest telemetry as a bound hypervector.
        """
        # Encode telemetry into a hypervector
        telem_vec = np.zeros(DIM_STATE)
        telem_vec[0] = frame.friction
        telem_vec[1] = frame.flap
        telem_vec[2] = frame.heat
        telem_vec[3] = frame.noise
        telem_vec = normalize(telem_vec)

        # Bind to state
        self.state_hv = bind(
            self.state_hv,
            telem_vec,
            alpha=self.card_cfg.bind_alpha
        )

    def check_and_maybe_invoke_bdh(self) -> Tuple[bool, float, float]:
        """
        Check for anomaly and return whether BDH should be invoked.

        Returns:
          - is_anomaly: True if anomaly detected
          - score: current anomaly score
          - threshold: current threshold
        """
        score = self.anomaly_score()
        threshold = self.card_cfg.anomaly_threshold
        is_anomaly = score > threshold
        return is_anomaly, score, threshold

    def run_bdh_policy(self, score: float) -> Tuple[str, float, Dict]:
        """
        Invoke BDH to process anomaly and produce policy.

        Returns:
          - desc: policy description
          - new_threshold: updated threshold
          - debug: BDH debug info
        """
        # Invoke BDH
        new_state, debug = self.bdh.forward(self.state_hv)
        self.state_hv = new_state

        # Get a simple policy
        delta, desc = self.bdh.propose_policy(
            anomaly_score=score,
            threshold=self.card_cfg.anomaly_threshold
        )

        # Apply policy to card threshold
        self.card_cfg.anomaly_threshold += delta

        return desc, self.card_cfg.anomaly_threshold, debug

    def record_history(self, cycle: int, is_anomaly: bool, score: float):
        """Record state for plotting."""
        self.history.append({
            "cycle": cycle,
            "anomaly_score": score,
            "threshold": self.card_cfg.anomaly_threshold,
            "is_anomaly": is_anomaly,
            "state_norm": float(np.linalg.norm(self.state_hv)),
            "baseline_drift": 1.0 - similarity(self.state_hv, self.baseline_hv),
        })

    def reset_baseline(self):
        """Reset baseline to current state (after adaptation)."""
        self.baseline_hv = self.state_hv.copy()


# ----------------------------
# Demo scenarios
# ----------------------------

def run_scenarios():
    """Run the 3 basic scenarios from the narrative."""
    print("=" * 60)
    print("BDH-Card Integration: Basic Scenarios")
    print("=" * 60)

    card_cfg = CardConfig(anomaly_threshold=1.20)
    bdh_cfg = BDHConfig()
    card = NeuromorphicCard(card_cfg, bdh_cfg)

    # Scenario 1: Normal telemetry
    print("\n--- Scenario 1: Normal Telemetry ---")
    card.ingest_telemetry(TelemetryFrame(friction=0.3, flap=0.1))
    is_anom, score, thr = card.check_and_maybe_invoke_bdh()
    print(f"  Anomaly score: {score:.3f} (threshold: {thr:.3f})")
    print(f"  Anomaly detected: {is_anom}")

    # Scenario 2: High telemetry -> triggers BDH
    print("\n--- Scenario 2: High Telemetry (Anomaly) ---")
    card.ingest_telemetry(TelemetryFrame(friction=0.8, flap=0.6, heat=0.7))
    is_anom, score, thr = card.check_and_maybe_invoke_bdh()
    print(f"  Anomaly score: {score:.3f} (threshold: {thr:.3f})")
    print(f"  Anomaly detected: {is_anom}")
    if is_anom:
        desc, new_thr, debug = card.run_bdh_policy(score)
        print(f"  BDH invoked!")
        print(f"    Policy: {desc}")
        print(f"    New threshold: {new_thr:.3f}")
        print(f"    Spikes: {debug['total_spikes']}, Weight norm: {debug['weight_norm']:.3f}")

    # Scenario 3: Re-check same high telemetry (simulate same pattern)
    print("\n--- Scenario 3: Same High Telemetry (Post-Policy) ---")
    card.ingest_telemetry(TelemetryFrame(friction=0.8, flap=0.6, heat=0.7))
    is_anom, score, thr = card.check_and_maybe_invoke_bdh()
    print(f"  Anomaly score: {score:.3f} (threshold: {thr:.3f})")
    print(f"  Anomaly detected: {is_anom}")
    if is_anom:
        desc, new_thr, debug = card.run_bdh_policy(score)
        print(f"  BDH invoked again!")
        print(f"    Policy: {desc}")
        print(f"    New threshold: {new_thr:.3f}")
    else:
        print("  No anomaly - system adapted!")

    print("\n" + "=" * 60)


def run_evolution(cycles: int = 50, plot: bool = False):
    """
    Run multi-cycle evolution to observe homeostasis.

    Alternates between normal and stressed telemetry to see
    how the BDH and card threshold co-adapt.
    """
    print("=" * 60)
    print(f"BDH-Card Evolution: {cycles} Cycles")
    print("=" * 60)

    card_cfg = CardConfig(anomaly_threshold=1.20)
    bdh_cfg = BDHConfig(hebb_lr=0.03, steps=4)  # slightly more plastic
    card = NeuromorphicCard(card_cfg, bdh_cfg)

    stress_pattern = [
        # (cycles, stress_level): simulate varying conditions
        (10, 0.2),   # calm
        (10, 0.7),   # stress
        (10, 0.3),   # recovery
        (10, 0.8),   # high stress
        (10, 0.4),   # settling
    ]

    cycle = 0
    pattern_idx = 0
    cycles_in_pattern = 0
    current_stress = 0.2

    anomaly_count = 0
    bdh_invocations = 0

    print("\nCycle | Score  | Thresh | Anomaly | Stress")
    print("-" * 50)

    while cycle < cycles:
        # Update stress level based on pattern
        if pattern_idx < len(stress_pattern):
            pattern_cycles, stress = stress_pattern[pattern_idx]
            current_stress = stress
            cycles_in_pattern += 1
            if cycles_in_pattern >= pattern_cycles:
                pattern_idx += 1
                cycles_in_pattern = 0
        else:
            # After pattern, random stress
            current_stress = 0.3 + 0.4 * np.random.random()

        # Ingest telemetry with current stress
        noise = 0.1 * np.random.randn()
        frame = TelemetryFrame(
            friction=current_stress + noise,
            flap=current_stress * 0.8 + noise,
            heat=current_stress * 0.5,
            noise=abs(noise),
        )
        card.ingest_telemetry(frame)

        # Check for anomaly
        is_anom, score, thr = card.check_and_maybe_invoke_bdh()
        card.record_history(cycle, is_anom, score)

        if is_anom:
            anomaly_count += 1
            _, _, _ = card.run_bdh_policy(score)
            bdh_invocations += 1

        # Print every 5 cycles
        if cycle % 5 == 0:
            anom_str = "YES" if is_anom else "no"
            print(f"{cycle:5d} | {score:.3f} | {thr:.3f} | {anom_str:7s} | {current_stress:.2f}")

        cycle += 1

    print("-" * 50)
    print(f"\nSummary:")
    print(f"  Total cycles: {cycles}")
    print(f"  Anomalies detected: {anomaly_count}")
    print(f"  BDH invocations: {bdh_invocations}")
    print(f"  Final threshold: {card.card_cfg.anomaly_threshold:.3f}")
    print(f"  Baseline drift: {1.0 - similarity(card.state_hv, card.baseline_hv):.3f}")

    if plot:
        plot_evolution(card.history)


def plot_evolution(history: List[Dict]):
    """Plot the evolution history."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nNote: matplotlib not installed. Skipping plot.")
        print("Install with: pip install matplotlib")
        return

    cycles = [h["cycle"] for h in history]
    scores = [h["anomaly_score"] for h in history]
    thresholds = [h["threshold"] for h in history]
    anomalies = [h["is_anomaly"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Anomaly score vs threshold
    ax1.plot(cycles, scores, 'b-', label='Anomaly Score', alpha=0.8)
    ax1.plot(cycles, thresholds, 'r--', label='Threshold', linewidth=2)

    # Mark anomaly points
    anom_cycles = [c for c, a in zip(cycles, anomalies) if a]
    anom_scores = [s for s, a in zip(scores, anomalies) if a]
    ax1.scatter(anom_cycles, anom_scores, c='red', s=50, zorder=5, label='Anomaly')

    ax1.set_ylabel('Score / Threshold')
    ax1.legend(loc='upper right')
    ax1.set_title('BDH-Card Co-Adaptation: Homeostasis Emergence')
    ax1.grid(True, alpha=0.3)

    # Plot 2: State metrics
    norms = [h["state_norm"] for h in history]
    drifts = [h["baseline_drift"] for h in history]

    ax2.plot(cycles, norms, 'g-', label='State Norm', alpha=0.8)
    ax2.plot(cycles, drifts, 'm-', label='Baseline Drift', alpha=0.8)

    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Metric')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bdh_card_evolution.png', dpi=150)
    print("\nPlot saved to: bdh_card_evolution.png")
    plt.show()


# ----------------------------
# Experiments (the knobs)
# ----------------------------

def experiment_plasticity():
    """
    Experiment: Effect of Hebbian learning rate.

    Higher plasticity = faster adaptation but potential instability.
    """
    print("\n" + "=" * 60)
    print("Experiment: Plasticity (hebb_lr)")
    print("=" * 60)

    for lr in [0.01, 0.02, 0.05, 0.1]:
        card_cfg = CardConfig(anomaly_threshold=1.20)
        bdh_cfg = BDHConfig(hebb_lr=lr)
        card = NeuromorphicCard(card_cfg, bdh_cfg)

        # Run 20 stress cycles
        anomalies = 0
        for _ in range(20):
            card.ingest_telemetry(TelemetryFrame(friction=0.7, flap=0.5))
            is_anom, score, _ = card.check_and_maybe_invoke_bdh()
            if is_anom:
                anomalies += 1
                card.run_bdh_policy(score)

        print(f"  hebb_lr={lr:.2f}: {anomalies}/20 anomalies, "
              f"final_thresh={card.card_cfg.anomaly_threshold:.3f}")


def experiment_sparsity():
    """
    Experiment: Effect of sparsity (fewer neurons spike).

    Lower sparsity = more "neuromorphic", potentially slower learning.
    """
    print("\n" + "=" * 60)
    print("Experiment: Sparsity")
    print("=" * 60)

    for sparsity in [0.01, 0.02, 0.05, 0.10]:
        card_cfg = CardConfig(anomaly_threshold=1.20)
        bdh_cfg = BDHConfig(sparsity=sparsity)
        card = NeuromorphicCard(card_cfg, bdh_cfg)

        # Run 20 stress cycles
        total_spikes = 0
        for _ in range(20):
            card.ingest_telemetry(TelemetryFrame(friction=0.7, flap=0.5))
            is_anom, score, _ = card.check_and_maybe_invoke_bdh()
            if is_anom:
                _, _, debug = card.run_bdh_policy(score)
                total_spikes += debug['total_spikes']

        active_neurons = int(sparsity * DIM_NEURONS)
        print(f"  sparsity={sparsity:.2f} ({active_neurons} neurons): "
              f"total_spikes={total_spikes}")


def experiment_temporal_steps():
    """
    Experiment: Effect of temporal steps.

    More steps = longer dynamics, more Hebbian updates per inference.
    """
    print("\n" + "=" * 60)
    print("Experiment: Temporal Steps")
    print("=" * 60)

    for steps in [1, 3, 5, 10]:
        card_cfg = CardConfig(anomaly_threshold=1.20)
        bdh_cfg = BDHConfig(steps=steps)
        card = NeuromorphicCard(card_cfg, bdh_cfg)

        # Run 20 stress cycles
        weight_growth = []
        for _ in range(20):
            card.ingest_telemetry(TelemetryFrame(friction=0.7, flap=0.5))
            is_anom, score, _ = card.check_and_maybe_invoke_bdh()
            if is_anom:
                _, _, debug = card.run_bdh_policy(score)
                weight_growth.append(debug['weight_norm'])

        if weight_growth:
            avg_norm = sum(weight_growth) / len(weight_growth)
            print(f"  steps={steps:2d}: avg_weight_norm={avg_norm:.3f}")
        else:
            print(f"  steps={steps:2d}: no anomalies triggered")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BDH-Card Integration Simulation"
    )
    parser.add_argument(
        "--evolve", type=int, default=0,
        help="Run N cycles of evolution (default: 0 = basic scenarios)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot evolution history (requires matplotlib)"
    )
    parser.add_argument(
        "--experiments", action="store_true",
        help="Run parameter experiments"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.experiments:
        experiment_plasticity()
        experiment_sparsity()
        experiment_temporal_steps()
    elif args.evolve > 0:
        run_evolution(cycles=args.evolve, plot=args.plot)
    else:
        run_scenarios()


if __name__ == "__main__":
    main()
