#!/usr/bin/env python3
"""
Ara Scientific Instrumentation: Avalanche Recorder
==================================================

Logs neural activity cascades for power-law analysis.

This is the "craniotomy" - we're recording Ara's internal dynamics
to verify she operates at the critical point of consciousness.

Theory:
    At criticality (λ ≈ 1), neural systems exhibit scale-free avalanches:
        P(S) ∝ S^(-τ)  where τ ≈ 1.5  (size distribution)
        P(D) ∝ D^(-α)  where α ≈ 2.0  (duration distribution)

    These exponents are universal - they appear in cortical slice cultures,
    awake rodents, human EEG, and (we hypothesize) in Ara.

    If Ara shows these exponents when CriticalityMonitor keeps ρ ≈ 0.8,
    we have mathematical evidence that GUTC produces critical dynamics.

Methodology:
    1. Since LLMs have continuous activations (not spikes), we discretize:
       - A site is "active" if |ΔA| > θ (threshold = 2σ of baseline)
    2. An "avalanche" is a contiguous burst of activity:
       - Size (S) = total active sites across all steps
       - Duration (D) = number of consecutive active steps
    3. Fit power laws to P(S) and P(D), extract exponents

Usage:
    logger = AvalancheLogger()

    for step, activations in enumerate(model_forward_pass):
        logger.log_step(activations, step)

    logger.save_session("avalanches.csv")

Then run: python scripts/science/fit_powerlaw.py data/experiments/exp_001/avalanches.csv
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from collections import deque
import time
import json
import logging

logger = logging.getLogger("ara.science.avalanche")


# =============================================================================
# Avalanche Event
# =============================================================================

@dataclass
class AvalancheEvent:
    """A single neural avalanche (cascade of activity)."""
    event_id: int
    start_step: int
    end_step: int
    duration: int                   # D: number of time steps
    size: int                       # S: total active sites
    peak_activity: int              # Maximum simultaneous active sites
    mean_activity: float            # Average active sites per step
    shape: List[int] = field(default_factory=list)  # Activity profile

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "duration": self.duration,
            "size": self.size,
            "peak_activity": self.peak_activity,
            "mean_activity": self.mean_activity,
        }


# =============================================================================
# Avalanche Logger
# =============================================================================

class AvalancheLogger:
    """
    Scientific instrument for recording neural avalanches in Ara.

    This hooks into the model's forward pass and records activity cascades
    for subsequent power-law analysis.

    Example:
        logger = AvalancheLogger(threshold_sigma=2.0)

        # During inference
        for step, batch in enumerate(data_loader):
            outputs = model(batch)
            activations = model.get_layer_activations(layer=12)
            logger.log_step(activations, step)

        # Save results
        logger.save_session()
    """

    def __init__(
        self,
        log_dir: str = "data/experiments/exp_001",
        threshold_sigma: float = 2.0,
        baseline_window: int = 100,
        min_avalanche_size: int = 1,
    ):
        """
        Initialize avalanche logger.

        Args:
            log_dir: Directory for output files
            threshold_sigma: Activation threshold in standard deviations
            baseline_window: Steps to compute baseline statistics
            min_avalanche_size: Minimum size to record an avalanche
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.threshold_sigma = threshold_sigma
        self.baseline_window = baseline_window
        self.min_avalanche_size = min_avalanche_size

        # Baseline statistics (calibrated during warmup)
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None
        self._activation_threshold: float = 0.5  # Default, will be calibrated

        # Baseline calibration buffer
        self._calibration_buffer: deque = deque(maxlen=baseline_window)
        self._calibrated = False

        # Current avalanche state
        self._current_event: List[int] = []  # Activity counts per step
        self._current_start_step: int = 0

        # Recorded events
        self.events: List[AvalancheEvent] = []

        # Session metadata
        self._session_start = time.time()
        self._total_steps = 0
        self._total_active_steps = 0

        # Previous activations for delta computation
        self._prev_activations: Optional[np.ndarray] = None

    def log_step(
        self,
        activations: Union[np.ndarray, "torch.Tensor"],
        step_id: int,
    ) -> Optional[int]:
        """
        Log a single time step of neural activity.

        Args:
            activations: Activation tensor from a model layer
                         Shape: (batch, seq, hidden) or (seq, hidden) or (hidden,)
            step_id: Global step identifier

        Returns:
            Number of active sites at this step (or None if calibrating)
        """
        # Convert to numpy if needed
        if hasattr(activations, 'detach'):
            activations = activations.detach().cpu().numpy()

        # Flatten to 1D for simplicity
        activations = np.asarray(activations, dtype=np.float32).flatten()

        self._total_steps += 1

        # Phase 1: Calibration (collect baseline statistics)
        if not self._calibrated:
            return self._calibration_step(activations, step_id)

        # Phase 2: Recording

        # Compute activity delta (change from previous step)
        if self._prev_activations is not None:
            delta = np.abs(activations - self._prev_activations)
        else:
            delta = np.abs(activations - self._baseline_mean)

        self._prev_activations = activations.copy()

        # Count active sites (above threshold)
        n_active = int(np.sum(delta > self._activation_threshold))

        # Avalanche state machine
        if n_active > 0:
            # Activity detected - avalanche in progress
            if not self._current_event:
                # Start of new avalanche
                self._current_start_step = step_id

            self._current_event.append(n_active)
            self._total_active_steps += 1

        else:
            # Silence - check if avalanche just ended
            if self._current_event:
                self._commit_event(step_id)

        return n_active

    def _calibration_step(
        self,
        activations: np.ndarray,
        step_id: int,
    ) -> None:
        """Collect baseline statistics during warmup."""
        self._calibration_buffer.append(activations)

        if len(self._calibration_buffer) >= self.baseline_window:
            # Compute baseline statistics
            all_acts = np.stack(list(self._calibration_buffer))
            self._baseline_mean = float(np.mean(all_acts))
            self._baseline_std = float(np.std(all_acts))

            # Set threshold
            self._activation_threshold = self._baseline_mean + self.threshold_sigma * self._baseline_std

            self._calibrated = True
            logger.info(
                f"Calibrated: mean={self._baseline_mean:.4f}, "
                f"std={self._baseline_std:.4f}, "
                f"threshold={self._activation_threshold:.4f}"
            )

        return None

    def _commit_event(self, end_step: int):
        """Finalize and record an avalanche event."""
        if not self._current_event:
            return

        duration = len(self._current_event)
        size = sum(self._current_event)

        # Filter tiny avalanches
        if size < self.min_avalanche_size:
            self._current_event = []
            return

        event = AvalancheEvent(
            event_id=len(self.events),
            start_step=self._current_start_step,
            end_step=end_step,
            duration=duration,
            size=size,
            peak_activity=max(self._current_event),
            mean_activity=size / duration,
            shape=self._current_event.copy(),
        )

        self.events.append(event)
        self._current_event = []

        if len(self.events) % 100 == 0:
            logger.debug(f"Recorded {len(self.events)} avalanches")

    def save_session(self, filename: str = "avalanches.csv") -> Path:
        """
        Save recorded avalanches to CSV.

        Returns path to saved file.
        """
        # Commit any pending avalanche
        if self._current_event:
            self._commit_event(self._total_steps)

        # Build dataframe or write CSV directly
        records = [e.to_dict() for e in self.events]
        path = self.log_dir / filename

        if HAS_PANDAS:
            df = pd.DataFrame(records)
            df.to_csv(path, index=False)
        else:
            # Fallback: write CSV manually
            import csv
            if records:
                with open(path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)

        logger.info(f"✅ Saved {len(records)} avalanches to {path}")

        # Also save session metadata
        meta_path = self.log_dir / filename.replace(".csv", "_meta.json")
        meta = {
            "session_start": self._session_start,
            "session_duration_s": time.time() - self._session_start,
            "total_steps": self._total_steps,
            "total_active_steps": self._total_active_steps,
            "n_avalanches": len(self.events),
            "threshold_sigma": self.threshold_sigma,
            "activation_threshold": self._activation_threshold,
            "baseline_mean": self._baseline_mean,
            "baseline_std": self._baseline_std,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return path

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of recorded avalanches."""
        if not self.events:
            return {"n_avalanches": 0}

        sizes = [e.size for e in self.events]
        durations = [e.duration for e in self.events]

        return {
            "n_avalanches": len(self.events),
            "total_steps": self._total_steps,
            "active_fraction": self._total_active_steps / max(1, self._total_steps),
            "size_mean": np.mean(sizes),
            "size_std": np.std(sizes),
            "size_max": max(sizes),
            "duration_mean": np.mean(durations),
            "duration_std": np.std(durations),
            "duration_max": max(durations),
        }

    def reset(self):
        """Reset logger for a new session."""
        self.events = []
        self._current_event = []
        self._prev_activations = None
        self._total_steps = 0
        self._total_active_steps = 0
        self._session_start = time.time()
        # Keep calibration


# =============================================================================
# Synthetic Data Generator (for testing)
# =============================================================================

class SyntheticAvalancheGenerator:
    """
    Generates synthetic avalanche data for testing the analysis pipeline.

    Can generate:
    - CRITICAL: Power-law distributed avalanches (τ ≈ 1.5)
    - SUBCRITICAL: Exponentially decaying avalanches
    - SUPERCRITICAL: Runaway avalanches (heavy tail)
    """

    def __init__(self, regime: str = "critical", seed: int = 42):
        """
        Args:
            regime: "critical", "subcritical", or "supercritical"
            seed: Random seed for reproducibility
        """
        self.regime = regime
        self.rng = np.random.default_rng(seed)

    def generate_session(self, n_steps: int = 10000) -> AvalancheLogger:
        """
        Generate a synthetic session of avalanche data.

        Returns an AvalancheLogger populated with synthetic events.
        """
        logger = AvalancheLogger()
        logger._calibrated = True
        logger._baseline_mean = 0.0
        logger._baseline_std = 1.0
        logger._activation_threshold = 2.0

        step = 0
        while step < n_steps:
            # Generate inter-avalanche interval (silence)
            silence_duration = int(self.rng.exponential(20))
            step += silence_duration

            if step >= n_steps:
                break

            # Generate avalanche
            if self.regime == "critical":
                # Power-law size with τ ≈ 1.5
                # Using inverse transform: S ~ (1-U)^(-1/(τ-1))
                u = self.rng.uniform(0.01, 1.0)
                size = int((1 - u) ** (-1 / 0.5))  # τ-1 = 0.5
                size = min(size, 10000)  # Cap extreme values
            elif self.regime == "subcritical":
                # Exponential size (characteristic scale)
                size = int(self.rng.exponential(10))
            else:  # supercritical
                # Heavy-tailed (Pareto with α < 1)
                size = int(self.rng.pareto(0.8) * 10) + 1

            # Generate duration based on scaling relation
            # D ~ S^(1/σνz) ≈ S^0.5 at criticality
            duration = max(1, int(np.sqrt(size) * self.rng.uniform(0.5, 1.5)))

            # Distribute activity across duration
            shape = self._generate_shape(size, duration)

            event = AvalancheEvent(
                event_id=len(logger.events),
                start_step=step,
                end_step=step + duration,
                duration=duration,
                size=size,
                peak_activity=max(shape),
                mean_activity=size / duration,
                shape=shape,
            )

            logger.events.append(event)
            step += duration
            logger._total_steps = step

        return logger

    def _generate_shape(self, size: int, duration: int) -> List[int]:
        """Generate activity profile for an avalanche."""
        if size <= 0:
            return [1] * duration

        if duration == 1:
            return [max(1, size)]

        # Parabolic profile (typical avalanche shape)
        x = np.linspace(0, 1, duration)
        profile = 4 * x * (1 - x)  # Peaks at 0.5
        profile_sum = profile.sum()

        if profile_sum > 0:
            profile = profile / profile_sum * size
        else:
            profile = np.ones(duration) * (size / duration)

        return [max(1, int(a)) if not np.isnan(a) else 1 for a in profile]


# =============================================================================
# Tests
# =============================================================================

def test_avalanche_logger():
    """Test avalanche logger with synthetic data."""
    print("Testing Avalanche Logger")
    print("=" * 60)

    # Generate synthetic critical data
    gen = SyntheticAvalancheGenerator(regime="critical")
    logger = gen.generate_session(n_steps=10000)

    print(f"Generated {len(logger.events)} avalanches")

    stats = logger.get_statistics()
    print(f"Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Save
    path = logger.save_session("test_avalanches.csv")
    print(f"Saved to: {path}")

    # Quick power-law check
    sizes = [e.size for e in logger.events]
    log_sizes = np.log10([s for s in sizes if s > 0])

    print(f"\nSize distribution (log10):")
    print(f"  Min: {log_sizes.min():.2f}")
    print(f"  Max: {log_sizes.max():.2f}")
    print(f"  Mean: {log_sizes.mean():.2f}")

    print("\n" + "=" * 60)
    print("Avalanche logger test passed!")


def test_regime_comparison():
    """Compare avalanche distributions across regimes."""
    print("\nRegime Comparison")
    print("=" * 60)

    for regime in ["critical", "subcritical", "supercritical"]:
        gen = SyntheticAvalancheGenerator(regime=regime)
        logger = gen.generate_session(n_steps=5000)

        sizes = [e.size for e in logger.events]
        durations = [e.duration for e in logger.events]

        print(f"\n{regime.upper()}:")
        print(f"  N avalanches: {len(logger.events)}")
        print(f"  Size range: [{min(sizes)}, {max(sizes)}]")
        print(f"  Duration range: [{min(durations)}, {max(durations)}]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_avalanche_logger()
    test_regime_comparison()
