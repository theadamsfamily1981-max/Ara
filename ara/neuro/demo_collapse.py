#!/usr/bin/env python3
"""
Demo: Protocol Collapse - From IPC to On-Chip Learning
=======================================================

Demonstrates the evolution from NeuroSymbiosis v1 to v2:

v1 (2025): STATE_HPV_QUERY → GPU LLM → NEW_POLICY_HDC
    - Card detects anomaly
    - Sends query to GPU
    - LLM reasons and returns policy
    - Card patches weights
    - Round-trip: ~100ms-1s

v2 (Protocol Collapse): pre × post × ρ → Δw
    - Card detects anomaly
    - Local neuromodulator computes ρ
    - Hebbian update on-chip
    - No GPU, no IPC
    - Update time: ~1ms

This demo shows:
1. The old flow (simulated LLM round-trip)
2. The new flow (on-chip Hebbian)
3. Comparison: latency, energy proxy, adaptation speed

Run:
    python -m ara.neuro.demo_collapse
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

from ara.hdc.encoder import HDEncoder
from ara.hdc.state_stream import StateStream
from ara.neuro.hebbian import HebbianPolicyLearner, SimpleNeuromodulator
from ara.neuro.spike_encoder import SpikeEncoder
from ara.neuro.unified_head import UnifiedHead, UnifiedHeadConfig, HeadOutput, ReflexUnit


# ============================================================================
# Event Generation
# ============================================================================

def generate_anomaly_sequence(n_normal: int = 50, n_anomaly: int = 10,
                               seed: int = 42) -> List[Tuple[Dict, bool]]:
    """Generate a sequence of normal and anomaly events."""
    rng = np.random.default_rng(seed)
    events = []

    # Normal events
    for _ in range(n_normal):
        events.append(({
            "cpu": rng.uniform(0.1, 0.4),
            "memory": rng.uniform(0.2, 0.5),
            "network": rng.uniform(0.1, 0.3),
        }, False))

    # Anomaly events
    for _ in range(n_anomaly):
        events.append(({
            "cpu": rng.uniform(0.8, 0.99),
            "memory": rng.uniform(0.7, 0.95),
            "network": rng.uniform(0.6, 0.9),
        }, True))

    # Shuffle
    rng.shuffle(events)
    return events


# ============================================================================
# V1: Simulated LLM Round-Trip
# ============================================================================

class V1SimulatedLLM:
    """
    Simulates the v1 flow with LLM round-trip.

    STATE_HPV_QUERY → (simulated delay) → NEW_POLICY_HDC
    """

    def __init__(self, llm_latency_ms: float = 200.0,
                 tokens_per_call: int = 500):
        self.llm_latency_ms = llm_latency_ms
        self.tokens_per_call = tokens_per_call

        self.encoder = HDEncoder()
        self.stream = StateStream(encoder=self.encoder)

        # Fixed detection threshold (no learning)
        self.threshold = 0.6

        # Stats
        self.llm_calls = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0

    def process(self, metrics: Dict, is_anomaly: bool) -> Tuple[bool, Dict]:
        """Process one event through v1 flow."""
        # Encode and add to state
        self.stream.add_metrics(metrics)
        state_mag = self.stream.get_state_magnitude()

        # Detect based on fixed threshold
        detected = state_mag > self.threshold

        info = {
            "detected": detected,
            "magnitude": state_mag,
            "llm_called": False,
        }

        # If detected, simulate LLM call
        if detected:
            # Simulate latency
            time.sleep(self.llm_latency_ms / 1000.0 * 0.01)  # Scaled for demo

            self.llm_calls += 1
            self.total_tokens += self.tokens_per_call
            self.total_latency_ms += self.llm_latency_ms

            info["llm_called"] = True

            # LLM would return a policy, but in v1 we can't learn on-chip
            # so the threshold stays fixed

        return detected, info

    def get_stats(self) -> Dict:
        return {
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "threshold": self.threshold,
        }


# ============================================================================
# V2: On-Chip Hebbian Learning
# ============================================================================

class V2OnChipLearning:
    """
    V2 flow with on-chip Hebbian learning.

    pre × post × ρ → Δw (no GPU, no IPC)
    """

    def __init__(self, eta: float = 0.01):
        self.encoder = HDEncoder()
        self.spike_encoder = SpikeEncoder(dim=1024, method="rate")

        # Adaptive reflex unit
        self.reflex = ReflexUnit(hpv_dim=1024, n_classes=4, eta=eta)

        # Stats
        self.updates = 0
        self.total_dw = 0.0

    def process(self, metrics: Dict, is_anomaly: bool) -> Tuple[bool, Dict]:
        """Process one event through v2 flow."""
        # Encode to HPV
        hpv = self.encoder.encode_metrics(metrics)

        # Ground truth for learning
        ground_truth = HeadOutput.ANOMALY if is_anomaly else HeadOutput.NORMAL

        # Process through reflex unit (forward + learn)
        decision, info = self.reflex.process(hpv, ground_truth=ground_truth)

        detected = decision in (HeadOutput.ANOMALY, HeadOutput.CRITICAL, HeadOutput.WARNING)

        if info["dw_norm"] > 0:
            self.updates += 1
            self.total_dw += info["dw_norm"]

        info["detected"] = detected
        return detected, info

    def get_stats(self) -> Dict:
        return {
            "updates": self.updates,
            "total_dw": self.total_dw,
            **self.reflex.get_stats(),
        }


# ============================================================================
# Comparison Demo
# ============================================================================

def demo_comparison():
    """Compare v1 and v2 flows side by side."""
    print("=" * 60)
    print("Protocol Collapse: v1 vs v2 Comparison")
    print("=" * 60)

    # Generate events
    events = generate_anomaly_sequence(n_normal=100, n_anomaly=20)
    n_anomalies = sum(1 for _, a in events if a)
    print(f"Events: {len(events)} total, {n_anomalies} anomalies")

    # V1: Simulated LLM
    print("\n--- V1: LLM Round-Trip ---")
    v1 = V1SimulatedLLM(llm_latency_ms=200.0)

    v1_tp = v1_fp = v1_fn = v1_tn = 0
    v1_start = time.time()

    for metrics, is_anomaly in events:
        detected, _ = v1.process(metrics, is_anomaly)
        if is_anomaly:
            if detected:
                v1_tp += 1
            else:
                v1_fn += 1
        else:
            if detected:
                v1_fp += 1
            else:
                v1_tn += 1

    v1_time = (time.time() - v1_start) * 1000
    v1_stats = v1.get_stats()

    print(f"  LLM calls: {v1_stats['llm_calls']}")
    print(f"  Total tokens: {v1_stats['total_tokens']}")
    print(f"  Simulated latency: {v1_stats['total_latency_ms']:.0f}ms")
    v1_f1 = 2 * v1_tp / (2 * v1_tp + v1_fp + v1_fn) if (v1_tp + v1_fp + v1_fn) > 0 else 0
    print(f"  F1: {v1_f1:.3f} (TP={v1_tp}, FP={v1_fp}, FN={v1_fn})")

    # V2: On-chip learning
    print("\n--- V2: On-Chip Hebbian ---")
    v2 = V2OnChipLearning(eta=0.02)

    v2_tp = v2_fp = v2_fn = v2_tn = 0
    v2_start = time.time()

    for metrics, is_anomaly in events:
        detected, _ = v2.process(metrics, is_anomaly)
        if is_anomaly:
            if detected:
                v2_tp += 1
            else:
                v2_fn += 1
        else:
            if detected:
                v2_fp += 1
            else:
                v2_tn += 1

    v2_time = (time.time() - v2_start) * 1000
    v2_stats = v2.get_stats()

    print(f"  Weight updates: {v2_stats['updates']}")
    print(f"  Total Δw: {v2_stats['total_dw']:.3f}")
    print(f"  Actual runtime: {v2_time:.1f}ms")
    v2_f1 = 2 * v2_tp / (2 * v2_tp + v2_fp + v2_fn) if (v2_tp + v2_fp + v2_fn) > 0 else 0
    print(f"  F1: {v2_f1:.3f} (TP={v2_tp}, FP={v2_fp}, FN={v2_fn})")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    headers = ["Metric", "V1 (LLM)", "V2 (Hebbian)", "Improvement"]
    rows = [
        ["GPU calls", str(v1_stats["llm_calls"]), "0", "∞"],
        ["Tokens", str(v1_stats["total_tokens"]), "0", "∞"],
        ["Latency (ms)", f"{v1_stats['total_latency_ms']:.0f}", f"{v2_time:.1f}",
         f"{v1_stats['total_latency_ms']/max(v2_time, 0.1):.0f}x"],
        ["Adapts?", "No", "Yes", "-"],
        ["F1", f"{v1_f1:.3f}", f"{v2_f1:.3f}",
         f"{'+' if v2_f1 > v1_f1 else ''}{(v2_f1-v1_f1):.3f}"],
    ]

    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(4)]
    fmt = "  " + " | ".join(f"{{:{w}}}" for w in col_widths)

    print(fmt.format(*headers))
    print("  " + "-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))

    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
  V1: STATE_HPV_QUERY → GPU LLM → NEW_POLICY_HDC
      - Requires network round-trip
      - ~200ms latency per policy update
      - Tokens cost energy and money

  V2: pre × post × ρ → Δw
      - All on-chip, no GPU needed
      - ~1ms update time
      - Continuous adaptation
      - "Habits become hardware"
""")


def demo_learning_curve():
    """Show how v2 adapts over time."""
    print("\n" + "=" * 60)
    print("Learning Curve: V2 Adaptation Over Time")
    print("=" * 60)

    v2 = V2OnChipLearning(eta=0.03)

    # Track accuracy over time
    window_size = 20
    accuracies = []
    current_correct = 0

    events = generate_anomaly_sequence(n_normal=200, n_anomaly=50, seed=123)

    print("\nEvent | Accuracy (last 20) | Weight Norm")
    print("-" * 45)

    for i, (metrics, is_anomaly) in enumerate(events):
        detected, info = v2.process(metrics, is_anomaly)

        # Track accuracy
        correct = (detected == is_anomaly)
        current_correct += int(correct)

        if (i + 1) % window_size == 0:
            accuracy = current_correct / window_size
            accuracies.append(accuracy)
            weight_norm = v2.reflex.head._learner.get_stats()["weight_norm"]

            print(f"{i+1:5d} | {accuracy:.1%}               | {weight_norm:.3f}")
            current_correct = 0

    # Show improvement
    if len(accuracies) >= 2:
        improvement = accuracies[-1] - accuracies[0]
        print(f"\nAccuracy improved by {improvement:+.1%} through on-chip learning")


def demo_hebbian_rule():
    """Visualize the Hebbian update rule."""
    print("\n" + "=" * 60)
    print("Hebbian Update Rule Visualization")
    print("=" * 60)

    print("""
    The three-factor Hebbian rule:

        Δw = η · ρ · (post ⊗ pre)

    Where:
        pre  = state HPV as spikes (what we're observing)
        post = detector output (what we detected)
        ρ    = neuromodulator (reward/policy signal)

    Example update:
    """)

    # Create a small example
    learner = HebbianPolicyLearner(n_pre=8, n_post=4, eta=0.1)

    print("Initial weights (4x8):")
    print(np.round(learner.w, 2))

    # Simulate pre and post activations
    pre = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.float32)  # State pattern
    post = np.array([1, 0, 0, 1], dtype=np.float32)             # Detector response
    rho = 1.0                                                    # Positive reward

    print(f"\npre (state):  {pre}")
    print(f"post (detector): {post}")
    print(f"ρ (reward):   {rho}")

    dw = learner.step(pre, post, rho)

    print(f"\nΔw = {learner.cfg.eta} × {rho} × outer(post, pre):")
    print(np.round(dw, 3))

    print(f"\nUpdated weights:")
    print(np.round(learner.w, 2))

    print("""
    Notice: Only weights connecting active pre and post neurons changed.
    This is "fire together, wire together" - the basis of on-chip learning.
    """)


# ============================================================================
# Main
# ============================================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║              Protocol Collapse Demo                            ║
║                                                                ║
║  From: STATE_HPV_QUERY → GPU LLM → NEW_POLICY_HDC              ║
║  To:   pre × post × ρ → Δw (on-chip)                           ║
║                                                                ║
║  "The whole 2025 STATE/POLICY IPC becomes local               ║
║   three-factor learning: pre, post, neuromodulator"           ║
╚════════════════════════════════════════════════════════════════╝
""")

    demos = [
        ("V1 vs V2 Comparison", demo_comparison),
        ("Learning Curve", demo_learning_curve),
        ("Hebbian Rule", demo_hebbian_rule),
    ]

    print("Available demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print("  0. Run all\n")

    try:
        choice = input("Select demo (0-3) [0]: ").strip()
        choice = int(choice) if choice else 0
    except (ValueError, EOFError):
        choice = 0

    if choice == 0:
        for name, demo_fn in demos:
            demo_fn()
            print("\n" + "▓" * 60 + "\n")
    elif 1 <= choice <= len(demos):
        demos[choice - 1][1]()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
