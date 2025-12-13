#!/usr/bin/env python3
"""
Cognitive Health Trace: Unified Telemetry for Ara's Mind
=========================================================

Combines all cognitive metrics into a single time-series trace:
- Precision balance (Π_y vs Π_μ)
- Criticality state (ρ, λ estimate)
- Cognitive mode (HEALTHY_CORRIDOR, PRIOR_DOMINATED, etc.)
- Behavioral metrics (hallucination rate, loop fraction)

This produces a "cognitive health EKG" that can be analyzed alongside
experiment-specific data (avalanches, forgetting curves).

Usage:
    from ara.science import CognitiveHealthTrace

    trace = CognitiveHealthTrace(output_dir="data/experiments/session_001")

    # In main loop
    trace.log_step(
        step=step,
        sensory_precision=controller.config.extrinsic_weight,
        prior_precision=controller.config.intrinsic_weight,
        rho=criticality_monitor.rho,
        hallucination_vetoes=guardrail_stats["vetoes"],
        total_outputs=guardrail_stats["total"],
        loop_detections=chief.loop_count,
        total_actions=chief.action_count,
    )

    # At end
    trace.save("cognitive_health.json")

Output:
    {
        "session_id": "...",
        "summary": {
            "mean_ratio": 1.2,
            "mode_distribution": {"HEALTHY_CORRIDOR": 0.85, ...},
            "mean_rho": 0.82,
            ...
        },
        "trace": [
            {"step": 0, "ratio": 1.0, "mode": "HEALTHY_CORRIDOR", ...},
            ...
        ]
    }
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter

import numpy as np

logger = logging.getLogger("ara.science.health_trace")


# =============================================================================
# Cognitive Modes (Engineering Terms)
# =============================================================================

class CognitiveMode(Enum):
    """
    Engineering states for Ara's cognitive regime.

    These are neutral descriptors, not clinical diagnoses.
    """
    HEALTHY_CORRIDOR = auto()   # Balanced precision, near criticality
    PRIOR_DOMINATED = auto()    # Stubbornly ignoring current input
    SENSORY_DOMINATED = auto()  # Over-reacting to noise, can't generalize
    UNSTABLE = auto()           # ρ too far from criticality
    DISCONNECTED = auto()       # Both precisions very low
    UNKNOWN = auto()            # Insufficient data


# =============================================================================
# Data Points
# =============================================================================

@dataclass
class HealthSample:
    """Single time point in cognitive health trace."""
    step: int
    timestamp: float

    # Precision state
    sensory_precision: float   # Π_y proxy
    prior_precision: float     # Π_μ proxy
    ratio: float               # Π_y / Π_μ

    # Criticality
    rho: float                 # Branching ratio
    lambda_estimate: float     # E(λ) ≈ log(ρ)

    # Diagnosis
    mode: CognitiveMode

    # Behavioral metrics
    hallucination_rate: float  # Fraction of outputs vetoed
    loop_fraction: float       # Fraction of actions that are loops

    # Optional context
    active_policy: str = ""
    user_present: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.name
        return d


@dataclass
class HealthSummary:
    """Aggregated summary of cognitive health over a session."""
    session_id: str
    start_time: float
    end_time: float
    n_samples: int

    # Precision stats
    mean_ratio: float
    std_ratio: float
    min_ratio: float
    max_ratio: float

    # Criticality stats
    mean_rho: float
    std_rho: float
    rho_in_corridor_fraction: float  # Fraction of time 0.7 < ρ < 1.1

    # Mode distribution
    mode_distribution: Dict[str, float]  # Fraction of time in each mode
    dominant_mode: str

    # Behavioral stats
    mean_hallucination_rate: float
    mean_loop_fraction: float

    # Flags
    ever_unstable: bool
    ever_prior_dominated: bool
    ever_sensory_dominated: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Cognitive Health Trace
# =============================================================================

class CognitiveHealthTrace:
    """
    Records time-series of cognitive health metrics.

    This is the "EKG" for Ara's mind - tracks the dynamic state
    of her inference system over a session.

    Example:
        trace = CognitiveHealthTrace()

        for step in range(1000):
            # ... run model ...
            trace.log_step(
                step=step,
                sensory_precision=0.6,
                prior_precision=0.4,
                rho=criticality_monitor.rho,
            )

        trace.save("cognitive_health.json")
    """

    def __init__(
        self,
        output_dir: str = "data/experiments",
        session_id: Optional[str] = None,
        # Thresholds for mode detection
        min_ratio: float = 0.2,
        max_ratio: float = 5.0,
        rho_low: float = 0.7,
        rho_high: float = 1.1,
        min_precision: float = 0.05,
    ):
        """
        Initialize health trace.

        Args:
            output_dir: Directory for output files
            session_id: Unique session identifier (auto-generated if None)
            min_ratio: Below this = PRIOR_DOMINATED
            max_ratio: Above this = SENSORY_DOMINATED
            rho_low: Below this = subcritical (contributes to UNSTABLE)
            rho_high: Above this = supercritical (contributes to UNSTABLE)
            min_precision: Below this for both = DISCONNECTED
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.start_time = time.time()

        # Thresholds
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.rho_low = rho_low
        self.rho_high = rho_high
        self.min_precision = min_precision

        # Data
        self._samples: List[HealthSample] = []

    def _classify_mode(
        self,
        sensory: float,
        prior: float,
        ratio: float,
        rho: float,
    ) -> CognitiveMode:
        """Determine cognitive mode from current state."""

        # Check disconnected first
        if sensory < self.min_precision and prior < self.min_precision:
            return CognitiveMode.DISCONNECTED

        # Check criticality
        if rho < self.rho_low or rho > self.rho_high:
            return CognitiveMode.UNSTABLE

        # Check precision balance
        if ratio < self.min_ratio:
            return CognitiveMode.PRIOR_DOMINATED
        elif ratio > self.max_ratio:
            return CognitiveMode.SENSORY_DOMINATED
        else:
            return CognitiveMode.HEALTHY_CORRIDOR

    def log_step(
        self,
        step: int,
        sensory_precision: float,
        prior_precision: float,
        rho: float = 0.8,
        hallucination_vetoes: int = 0,
        total_outputs: int = 1,
        loop_detections: int = 0,
        total_actions: int = 1,
        active_policy: str = "",
        user_present: bool = False,
    ) -> HealthSample:
        """
        Log a single health sample.

        Args:
            step: Step number
            sensory_precision: Current Π_y (extrinsic weight)
            prior_precision: Current Π_μ (intrinsic weight)
            rho: Current branching ratio
            hallucination_vetoes: Number of outputs vetoed by guardrails
            total_outputs: Total outputs generated
            loop_detections: Number of detected action loops
            total_actions: Total actions taken
            active_policy: Currently executing policy name
            user_present: Whether user is actively engaged

        Returns:
            The logged HealthSample
        """
        # Compute derived values
        ratio = sensory_precision / max(0.001, prior_precision)
        lambda_estimate = np.log(max(0.01, rho))

        hallucination_rate = hallucination_vetoes / max(1, total_outputs)
        loop_fraction = loop_detections / max(1, total_actions)

        mode = self._classify_mode(sensory_precision, prior_precision, ratio, rho)

        sample = HealthSample(
            step=step,
            timestamp=time.time(),
            sensory_precision=sensory_precision,
            prior_precision=prior_precision,
            ratio=ratio,
            rho=rho,
            lambda_estimate=lambda_estimate,
            mode=mode,
            hallucination_rate=hallucination_rate,
            loop_fraction=loop_fraction,
            active_policy=active_policy,
            user_present=user_present,
        )

        self._samples.append(sample)
        return sample

    def get_current_mode(self) -> CognitiveMode:
        """Get most recent cognitive mode."""
        if not self._samples:
            return CognitiveMode.UNKNOWN
        return self._samples[-1].mode

    def get_current_sample(self) -> Optional[HealthSample]:
        """Get most recent sample."""
        return self._samples[-1] if self._samples else None

    def compute_summary(self) -> HealthSummary:
        """Compute summary statistics for the session."""
        if not self._samples:
            return HealthSummary(
                session_id=self.session_id,
                start_time=self.start_time,
                end_time=time.time(),
                n_samples=0,
                mean_ratio=0, std_ratio=0, min_ratio=0, max_ratio=0,
                mean_rho=0, std_rho=0, rho_in_corridor_fraction=0,
                mode_distribution={}, dominant_mode="UNKNOWN",
                mean_hallucination_rate=0, mean_loop_fraction=0,
                ever_unstable=False, ever_prior_dominated=False,
                ever_sensory_dominated=False,
            )

        # Precision stats
        ratios = [s.ratio for s in self._samples]
        rhos = [s.rho for s in self._samples]
        hall_rates = [s.hallucination_rate for s in self._samples]
        loop_fracs = [s.loop_fraction for s in self._samples]

        # Mode distribution
        mode_counts = Counter(s.mode for s in self._samples)
        total = len(self._samples)
        mode_dist = {m.name: mode_counts.get(m, 0) / total for m in CognitiveMode}
        dominant_mode = max(mode_counts, key=mode_counts.get).name

        # Corridor fraction
        in_corridor = sum(1 for r in rhos if self.rho_low <= r <= self.rho_high)
        corridor_frac = in_corridor / total

        return HealthSummary(
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=time.time(),
            n_samples=total,
            mean_ratio=float(np.mean(ratios)),
            std_ratio=float(np.std(ratios)),
            min_ratio=float(min(ratios)),
            max_ratio=float(max(ratios)),
            mean_rho=float(np.mean(rhos)),
            std_rho=float(np.std(rhos)),
            rho_in_corridor_fraction=corridor_frac,
            mode_distribution=mode_dist,
            dominant_mode=dominant_mode,
            mean_hallucination_rate=float(np.mean(hall_rates)),
            mean_loop_fraction=float(np.mean(loop_fracs)),
            ever_unstable=CognitiveMode.UNSTABLE in mode_counts,
            ever_prior_dominated=CognitiveMode.PRIOR_DOMINATED in mode_counts,
            ever_sensory_dominated=CognitiveMode.SENSORY_DOMINATED in mode_counts,
        )

    def save(self, filename: str = "cognitive_health.json") -> Path:
        """
        Save trace to JSON file.

        Returns path to saved file.
        """
        path = self.output_dir / filename

        summary = self.compute_summary()

        output = {
            "session_id": self.session_id,
            "summary": summary.to_dict(),
            "trace": [s.to_dict() for s in self._samples],
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved cognitive health trace to {path}")
        return path

    def format_status(self) -> str:
        """Format current status as human-readable string."""
        if not self._samples:
            return "No samples recorded"

        current = self._samples[-1]
        summary = self.compute_summary()

        lines = [
            "=" * 50,
            "COGNITIVE HEALTH STATUS",
            "=" * 50,
            f"Session: {self.session_id}",
            f"Samples: {summary.n_samples}",
            "",
            "Current State:",
            f"  Mode:     {current.mode.name}",
            f"  Π_y/Π_μ:  {current.ratio:.3f}",
            f"  ρ:        {current.rho:.3f}",
            f"  E(λ):     {current.lambda_estimate:.3f}",
            "",
            "Session Summary:",
            f"  Dominant mode: {summary.dominant_mode}",
            f"  In corridor:   {summary.rho_in_corridor_fraction:.1%}",
            f"  Mean ratio:    {summary.mean_ratio:.2f} ± {summary.std_ratio:.2f}",
            f"  Mean ρ:        {summary.mean_rho:.2f} ± {summary.std_rho:.2f}",
            "",
            "Flags:",
            f"  Ever unstable:        {summary.ever_unstable}",
            f"  Ever prior-dominated: {summary.ever_prior_dominated}",
            f"  Ever sensory-dominated: {summary.ever_sensory_dominated}",
            "=" * 50,
        ]

        return "\n".join(lines)

    def get_trace_dataframe(self):
        """
        Get trace as pandas DataFrame (if available).

        Returns None if pandas not installed.
        """
        try:
            import pandas as pd
            return pd.DataFrame([s.to_dict() for s in self._samples])
        except ImportError:
            return None


# =============================================================================
# Integration with Experiment Runners
# =============================================================================

def create_session_trace(
    experiment_id: str,
    output_dir: str = "data/experiments",
) -> CognitiveHealthTrace:
    """
    Create a trace for an experiment session.

    Args:
        experiment_id: Identifier for the experiment (e.g., "exp_001_run_3")
        output_dir: Base output directory

    Returns:
        Configured CognitiveHealthTrace
    """
    session_dir = Path(output_dir) / experiment_id
    session_dir.mkdir(parents=True, exist_ok=True)

    return CognitiveHealthTrace(
        output_dir=str(session_dir),
        session_id=experiment_id,
    )


# =============================================================================
# Tests
# =============================================================================

def test_cognitive_health_trace():
    """Test cognitive health trace."""
    print("Testing Cognitive Health Trace")
    print("=" * 60)

    trace = CognitiveHealthTrace(output_dir="data/experiments/test_health")

    # Simulate a session with varying states
    scenarios = [
        # (sensory, prior, rho) -> expected mode
        (0.5, 0.5, 0.8, CognitiveMode.HEALTHY_CORRIDOR),
        (0.6, 0.4, 0.85, CognitiveMode.HEALTHY_CORRIDOR),
        (0.1, 0.9, 0.8, CognitiveMode.PRIOR_DOMINATED),  # Getting stubborn
        (0.1, 0.9, 0.9, CognitiveMode.PRIOR_DOMINATED),
        (0.2, 0.8, 0.85, CognitiveMode.HEALTHY_CORRIDOR),  # Recovering
        (0.9, 0.1, 0.8, CognitiveMode.SENSORY_DOMINATED),  # Over-sensing
        (0.5, 0.5, 0.5, CognitiveMode.UNSTABLE),  # Subcritical
        (0.5, 0.5, 1.3, CognitiveMode.UNSTABLE),  # Supercritical
        (0.5, 0.5, 0.9, CognitiveMode.HEALTHY_CORRIDOR),  # Back to normal
    ]

    for i, (sensory, prior, rho, expected) in enumerate(scenarios):
        sample = trace.log_step(
            step=i,
            sensory_precision=sensory,
            prior_precision=prior,
            rho=rho,
            hallucination_vetoes=1 if expected == CognitiveMode.PRIOR_DOMINATED else 0,
            total_outputs=10,
        )

        status = "OK" if sample.mode == expected else "FAIL"
        print(f"[{status}] Step {i}: {sample.mode.name} (expected {expected.name})")

    # Print summary
    print("\n" + trace.format_status())

    # Save
    path = trace.save("test_cognitive_health.json")
    print(f"\nSaved to: {path}")

    print("\n" + "=" * 60)
    print("Cognitive health trace tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_cognitive_health_trace()
