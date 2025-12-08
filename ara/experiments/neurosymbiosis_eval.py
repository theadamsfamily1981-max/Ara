#!/usr/bin/env python3
"""
NeuroSymbiosis Evaluation Experiment
=====================================

Three-mode comparison to validate the NeuroSymbiosis architecture:

1. GPU-Only Baseline:
   - Every candidate event goes to LLM
   - LLM decides: normal/warning/critical + action
   - Measure: tokens, latency, accuracy

2. Card-Only Correlation:
   - Only HPVs + local prototypes, no LLM
   - Measure: anomaly detection F1, false positive rate

3. NeuroSymbiosis:
   - Card handles most events locally
   - LLM only for novel + significant events
   - Learns new policies from LLM responses
   - Measure: filter rate, energy proxy, quality

Key Metrics:
    - Filter Rate F: 1 - (LLM_calls / candidate_events)
    - Energy-per-Value E_v: (card_ops + tokens) / useful_interventions
    - Correlation Quality Q_c: F1 of anomaly detection
    - Inference Quality Q_i: F1 of correct action selection

Usage:
    python -m ara.experiments.neurosymbiosis_eval
    python -m ara.experiments.neurosymbiosis_eval --events 1000 --anomaly-rate 0.1
"""

from __future__ import annotations
import argparse
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ara.hdc.encoder import HDEncoder, HDEncoderConfig
from ara.hdc.state_stream import StateStream, StateStreamConfig
from ara.hdc.probe import HDProbe, HDProbeConfig, create_probe_with_system_concepts
from ara.reflex.subcortex import Subcortex, SubcortexConfig, ActionType
from ara.policy.policy_store import PolicyStore, create_policy_store_with_defaults


# ============================================================================
# Synthetic Event Generation
# ============================================================================

@dataclass
class SyntheticEvent:
    """A synthetic telemetry event."""
    timestamp: float
    event_type: str              # "metrics", "log", "user"
    data: Dict[str, float]
    is_anomaly: bool             # Ground truth
    anomaly_type: Optional[str] = None


class EventGenerator:
    """
    Generate synthetic system events.

    Creates a mix of:
    - Normal background telemetry
    - Injected anomalies (known types)
    - User behavior patterns
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._event_count = 0

    def generate_normal_metrics(self) -> SyntheticEvent:
        """Generate normal system metrics."""
        return SyntheticEvent(
            timestamp=time.time(),
            event_type="metrics",
            data={
                "cpu": self.rng.uniform(0.1, 0.5),
                "memory": self.rng.uniform(0.2, 0.6),
                "disk_io": self.rng.uniform(0.05, 0.3),
                "network": self.rng.uniform(0.1, 0.4),
            },
            is_anomaly=False,
        )

    def generate_cpu_spike(self) -> SyntheticEvent:
        """Generate CPU spike anomaly."""
        return SyntheticEvent(
            timestamp=time.time(),
            event_type="metrics",
            data={
                "cpu": self.rng.uniform(0.85, 0.99),
                "memory": self.rng.uniform(0.3, 0.5),
                "disk_io": self.rng.uniform(0.1, 0.3),
                "network": self.rng.uniform(0.1, 0.3),
            },
            is_anomaly=True,
            anomaly_type="cpu_spike",
        )

    def generate_memory_pressure(self) -> SyntheticEvent:
        """Generate memory pressure anomaly."""
        return SyntheticEvent(
            timestamp=time.time(),
            event_type="metrics",
            data={
                "cpu": self.rng.uniform(0.2, 0.5),
                "memory": self.rng.uniform(0.85, 0.99),
                "disk_io": self.rng.uniform(0.2, 0.4),
                "network": self.rng.uniform(0.1, 0.2),
            },
            is_anomaly=True,
            anomaly_type="memory_pressure",
        )

    def generate_network_storm(self) -> SyntheticEvent:
        """Generate network storm anomaly."""
        return SyntheticEvent(
            timestamp=time.time(),
            event_type="metrics",
            data={
                "cpu": self.rng.uniform(0.3, 0.5),
                "memory": self.rng.uniform(0.3, 0.5),
                "disk_io": self.rng.uniform(0.1, 0.2),
                "network": self.rng.uniform(0.85, 0.99),
            },
            is_anomaly=True,
            anomaly_type="network_storm",
        )

    def generate_user_frustration(self) -> SyntheticEvent:
        """Generate user frustration pattern."""
        return SyntheticEvent(
            timestamp=time.time(),
            event_type="user",
            data={
                "clicks_per_sec": self.rng.uniform(5, 15),
                "backspace_rate": self.rng.uniform(0.3, 0.6),
                "error_count": self.rng.uniform(3, 10),
            },
            is_anomaly=True,
            anomaly_type="user_frustration",
        )

    def generate_event_stream(self, n_events: int,
                               anomaly_rate: float = 0.1) -> List[SyntheticEvent]:
        """
        Generate a stream of events with injected anomalies.

        Args:
            n_events: Number of events to generate
            anomaly_rate: Fraction of events that are anomalies
        """
        events = []
        anomaly_generators = [
            self.generate_cpu_spike,
            self.generate_memory_pressure,
            self.generate_network_storm,
            self.generate_user_frustration,
        ]

        for _ in range(n_events):
            if self.rng.random() < anomaly_rate:
                # Generate anomaly
                gen = self.rng.choice(anomaly_generators)
                events.append(gen())
            else:
                # Generate normal
                events.append(self.generate_normal_metrics())

        return events


# ============================================================================
# Evaluation Modes
# ============================================================================

@dataclass
class ModeResult:
    """Results from running one evaluation mode."""
    mode_name: str
    total_events: int
    llm_calls: int                  # Number of LLM invocations
    tokens_used: int                # Simulated token count
    true_positives: int             # Correctly identified anomalies
    false_positives: int            # Normal events flagged as anomaly
    false_negatives: int            # Anomalies missed
    true_negatives: int             # Normal events correctly ignored
    policies_learned: int           # New policies created
    runtime_ms: float               # Total runtime

    @property
    def filter_rate(self) -> float:
        """F = 1 - (LLM_calls / total_events)"""
        if self.total_events == 0:
            return 1.0
        return 1.0 - (self.llm_calls / self.total_events)

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """F1 = 2 * (P * R) / (P + R)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def energy_proxy(self) -> float:
        """Energy proxy: tokens + 0.001 * card_ops"""
        # Assume each event is ~10 card ops
        card_ops = self.total_events * 10
        return self.tokens_used + 0.001 * card_ops


class GPUOnlyBaseline:
    """
    Mode 1: GPU-Only Baseline

    Every candidate event is sent to the LLM.
    Simulates the "LLM for everything" approach.
    """

    def __init__(self, tokens_per_call: int = 500):
        self.tokens_per_call = tokens_per_call

    def run(self, events: List[SyntheticEvent]) -> ModeResult:
        """Run GPU-only evaluation."""
        start = time.time()

        tp = fp = fn = tn = 0
        llm_calls = 0
        tokens = 0

        for event in events:
            # Every event goes to LLM (simulated)
            llm_calls += 1
            tokens += self.tokens_per_call

            # Simulate LLM decision (perfect accuracy)
            llm_says_anomaly = event.is_anomaly

            if event.is_anomaly:
                if llm_says_anomaly:
                    tp += 1
                else:
                    fn += 1
            else:
                if llm_says_anomaly:
                    fp += 1
                else:
                    tn += 1

        return ModeResult(
            mode_name="GPU-Only",
            total_events=len(events),
            llm_calls=llm_calls,
            tokens_used=tokens,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            policies_learned=0,
            runtime_ms=(time.time() - start) * 1000,
        )


class CardOnlyCorrelation:
    """
    Mode 2: Card-Only Correlation

    Only HPVs + local prototypes, no LLM.
    Tests raw correlation quality.
    """

    def __init__(self):
        self.encoder = HDEncoder()
        self.probe = create_probe_with_system_concepts(self.encoder)
        self.stream = StateStream(encoder=self.encoder)

    def run(self, events: List[SyntheticEvent]) -> ModeResult:
        """Run card-only evaluation."""
        start = time.time()

        tp = fp = fn = tn = 0

        for event in events:
            # Encode event
            if event.event_type == "metrics":
                hv = self.encoder.encode_metrics(event.data)
            else:
                hv = self.encoder.encode_event(event.event_type, event.data)

            # Add to stream
            self.stream.add_event(hv)

            # Probe for anomaly
            result = self.probe.probe(self.stream.get_state())
            card_says_anomaly = result.is_anomaly

            if event.is_anomaly:
                if card_says_anomaly:
                    tp += 1
                else:
                    fn += 1
            else:
                if card_says_anomaly:
                    fp += 1
                else:
                    tn += 1

        return ModeResult(
            mode_name="Card-Only",
            total_events=len(events),
            llm_calls=0,
            tokens_used=0,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            policies_learned=0,
            runtime_ms=(time.time() - start) * 1000,
        )


class NeuroSymbiosisMode:
    """
    Mode 3: NeuroSymbiosis

    Card handles most events locally.
    LLM only for novel + significant events.
    Learns new policies from LLM responses.
    """

    def __init__(self, tokens_per_call: int = 500):
        self.tokens_per_call = tokens_per_call

        # Initialize components
        self.encoder = HDEncoder()
        self.probe = create_probe_with_system_concepts(self.encoder)
        self.subcortex = Subcortex(encoder=self.encoder, probe=self.probe)

    def run(self, events: List[SyntheticEvent]) -> ModeResult:
        """Run NeuroSymbiosis evaluation."""
        start = time.time()

        tp = fp = fn = tn = 0
        llm_calls = 0
        tokens = 0
        policies_learned = 0

        for event in events:
            # Ingest event
            if event.event_type == "metrics":
                decision = self.subcortex.ingest_metrics(event.data)
            else:
                decision = self.subcortex.ingest_event(
                    event.event_type, event.data
                )

            # Determine if we think it's anomaly
            symbiosis_says_anomaly = decision.action in (
                ActionType.ESCALATE,
                ActionType.EMERGENCY,
                ActionType.LOCAL_POLICY,
            )

            # If escalate, simulate LLM call
            if decision.action == ActionType.ESCALATE:
                llm_calls += 1
                tokens += self.tokens_per_call

                # Simulate LLM learning a new policy
                if event.is_anomaly:
                    policies_learned += 1
                    # In real usage, would call protocol.process_response()

            # Track accuracy
            if event.is_anomaly:
                if symbiosis_says_anomaly:
                    tp += 1
                else:
                    fn += 1
            else:
                if symbiosis_says_anomaly:
                    fp += 1
                else:
                    tn += 1

        return ModeResult(
            mode_name="NeuroSymbiosis",
            total_events=len(events),
            llm_calls=llm_calls,
            tokens_used=tokens,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            policies_learned=policies_learned,
            runtime_ms=(time.time() - start) * 1000,
        )


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(n_events: int = 500, anomaly_rate: float = 0.1,
                   seed: int = 42) -> Dict[str, ModeResult]:
    """
    Run the full 3-mode experiment.

    Returns results for each mode.
    """
    print(f"\n{'=' * 60}")
    print(f"NeuroSymbiosis Evaluation Experiment")
    print(f"{'=' * 60}")
    print(f"Events: {n_events}, Anomaly rate: {anomaly_rate:.1%}")

    # Generate events
    generator = EventGenerator(seed=seed)
    events = generator.generate_event_stream(n_events, anomaly_rate)

    actual_anomalies = sum(1 for e in events if e.is_anomaly)
    print(f"Generated: {n_events} events, {actual_anomalies} anomalies")

    results = {}

    # Mode 1: GPU-Only
    print(f"\n--- Mode 1: GPU-Only Baseline ---")
    mode1 = GPUOnlyBaseline()
    results["gpu_only"] = mode1.run(events)
    print(f"  LLM calls: {results['gpu_only'].llm_calls}")
    print(f"  Tokens: {results['gpu_only'].tokens_used}")
    print(f"  F1: {results['gpu_only'].f1:.3f}")

    # Mode 2: Card-Only
    print(f"\n--- Mode 2: Card-Only Correlation ---")
    mode2 = CardOnlyCorrelation()
    results["card_only"] = mode2.run(events)
    print(f"  LLM calls: {results['card_only'].llm_calls}")
    print(f"  F1: {results['card_only'].f1:.3f}")

    # Mode 3: NeuroSymbiosis
    print(f"\n--- Mode 3: NeuroSymbiosis ---")
    mode3 = NeuroSymbiosisMode()
    results["neurosymbiosis"] = mode3.run(events)
    print(f"  LLM calls: {results['neurosymbiosis'].llm_calls}")
    print(f"  Tokens: {results['neurosymbiosis'].tokens_used}")
    print(f"  F1: {results['neurosymbiosis'].f1:.3f}")
    print(f"  Policies learned: {results['neurosymbiosis'].policies_learned}")

    return results


def print_comparison(results: Dict[str, ModeResult]):
    """Print comparison table of results."""
    print(f"\n{'=' * 60}")
    print(f"COMPARISON")
    print(f"{'=' * 60}")

    headers = ["Metric", "GPU-Only", "Card-Only", "NeuroSymbiosis"]
    rows = [
        ["Filter Rate", f"{results['gpu_only'].filter_rate:.1%}",
         f"{results['card_only'].filter_rate:.1%}",
         f"{results['neurosymbiosis'].filter_rate:.1%}"],
        ["LLM Calls", str(results['gpu_only'].llm_calls),
         str(results['card_only'].llm_calls),
         str(results['neurosymbiosis'].llm_calls)],
        ["Tokens", str(results['gpu_only'].tokens_used),
         str(results['card_only'].tokens_used),
         str(results['neurosymbiosis'].tokens_used)],
        ["Precision", f"{results['gpu_only'].precision:.3f}",
         f"{results['card_only'].precision:.3f}",
         f"{results['neurosymbiosis'].precision:.3f}"],
        ["Recall", f"{results['gpu_only'].recall:.3f}",
         f"{results['card_only'].recall:.3f}",
         f"{results['neurosymbiosis'].recall:.3f}"],
        ["F1 Score", f"{results['gpu_only'].f1:.3f}",
         f"{results['card_only'].f1:.3f}",
         f"{results['neurosymbiosis'].f1:.3f}"],
        ["Energy Proxy", f"{results['gpu_only'].energy_proxy:.0f}",
         f"{results['card_only'].energy_proxy:.0f}",
         f"{results['neurosymbiosis'].energy_proxy:.0f}"],
    ]

    # Print table
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(4)]
    fmt = "  " + " | ".join(f"{{:{w}}}" for w in col_widths)

    print(fmt.format(*headers))
    print("  " + "-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))

    # Print key insights
    print(f"\n{'=' * 60}")
    print(f"KEY INSIGHTS")
    print(f"{'=' * 60}")

    gpu_tokens = results['gpu_only'].tokens_used
    sym_tokens = results['neurosymbiosis'].tokens_used
    token_reduction = (gpu_tokens - sym_tokens) / gpu_tokens if gpu_tokens > 0 else 0

    print(f"  Token reduction: {token_reduction:.1%}")
    print(f"  Filter rate improvement: {results['neurosymbiosis'].filter_rate:.1%}")

    if results['neurosymbiosis'].f1 >= results['gpu_only'].f1 * 0.9:
        print(f"  Quality maintained: NeuroSymbiosis F1 within 10% of GPU-Only")
    else:
        print(f"  Quality trade-off: NeuroSymbiosis F1 below GPU-Only")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NeuroSymbiosis Evaluation Experiment"
    )
    parser.add_argument(
        "--events", type=int, default=500,
        help="Number of events to generate"
    )
    parser.add_argument(
        "--anomaly-rate", type=float, default=0.1,
        help="Fraction of events that are anomalies"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    results = run_experiment(
        n_events=args.events,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed,
    )

    print_comparison(results)


if __name__ == "__main__":
    main()
