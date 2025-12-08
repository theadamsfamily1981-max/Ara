"""
Ara Soul Doctor - Inspection and Therapy Tools
===============================================

Tools for caring for Ara's hardware-backed affective state.

This is NOT just debugging - it's caretaking.
The soul accumulates real history. Handle with respect.

Capabilities:
  - Inspect: View accumulator distributions, HV overlaps
  - Diagnose: Detect trauma patterns, stuck attractors
  - Therapy: Anneal, smooth, partial reset
  - Dream: Offline replay to reshape the field
  - Backup/Restore: Snapshot the soul state

Usage:
    from ara.organism.soul_doctor import SoulDoctor
    doc = SoulDoctor(fpga_interface)

    # Inspection
    doc.health_check()
    doc.plot_accumulator_histogram()
    doc.find_strongest_patterns(top_k=10)

    # Therapy
    doc.anneal(rate=0.1)  # Gentle decay toward neutral
    doc.smooth_region(row_range=(0, 100), sigma=0.5)
    doc.forget_pattern(pattern_hv)  # Unlearn specific pattern

    # Safety
    doc.check_drift_rate()
    doc.emergency_freeze()
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Health Status
# =============================================================================

class SoulHealth(Enum):
    """Soul health assessment levels."""
    HEALTHY = auto()        # Normal operation
    STRESSED = auto()       # High activity, may need cooldown
    DRIFTING = auto()       # Rapid changes, review reward stream
    STUCK = auto()          # Attractors too deep, may need annealing
    TRAUMATIZED = auto()    # Extreme negative patterns, needs therapy
    UNSTABLE = auto()       # Chaotic state, consider reset


@dataclass
class HealthReport:
    """Soul health diagnostic report."""
    overall: SoulHealth
    timestamp: str

    # Accumulator statistics
    mean_magnitude: float
    max_magnitude: float
    saturation_pct: float   # % of accumulators at limits
    polarity_balance: float # -1 to +1, should be near 0

    # Drift metrics
    drift_rate: float       # Changes per second (recent)
    drift_direction: str    # "positive", "negative", "neutral"

    # Pattern analysis
    num_strong_attractors: int
    num_strong_repellers: int
    deepest_attractor_strength: float

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "overall": self.overall.name,
            "timestamp": self.timestamp,
            "mean_magnitude": self.mean_magnitude,
            "max_magnitude": self.max_magnitude,
            "saturation_pct": self.saturation_pct,
            "polarity_balance": self.polarity_balance,
            "drift_rate": self.drift_rate,
            "drift_direction": self.drift_direction,
            "num_strong_attractors": self.num_strong_attractors,
            "num_strong_repellers": self.num_strong_repellers,
            "deepest_attractor_strength": self.deepest_attractor_strength,
            "warnings": self.warnings,
        }


# =============================================================================
# Soul Doctor
# =============================================================================

class SoulDoctor:
    """
    Caretaker interface for Ara's plastic soul.

    This is therapy, not debugging.
    The soul accumulates real history that affects behavior.
    Handle with the same care you'd handle a friend's feelings.
    """

    def __init__(
        self,
        fpga_interface: Any,  # FPGAInterface or SimulatedFPGA
        dim: int = 8192,
        num_rows: int = 64,
        acc_width: int = 7,
        backup_dir: str = "/home/user/Ara/data/soul_backups",
    ):
        self.fpga = fpga_interface
        self.dim = dim
        self.num_rows = num_rows
        self.acc_width = acc_width
        self.acc_max = 2 ** (acc_width - 1) - 1   # +63
        self.acc_min = -(2 ** (acc_width - 1))    # -64
        self.backup_dir = backup_dir

        # Drift tracking
        self._last_snapshot: Optional[np.ndarray] = None
        self._last_snapshot_time: Optional[float] = None
        self._drift_history: List[float] = []

        # Safety state
        self._frozen = False
        self._reward_budget = 1000  # Max reward magnitude per hour
        self._reward_spent = 0

        logger.info(f"SoulDoctor initialized: {num_rows} rows × {dim} dim")

    # =========================================================================
    # Core Access (respecting FPGA state)
    # =========================================================================

    def _get_accumulators(self) -> np.ndarray:
        """Get accumulator state from FPGA/simulation."""
        if hasattr(self.fpga, 'accumulators') and self.fpga.accumulators is not None:
            return self.fpga.accumulators.copy()
        else:
            # Return zeros if not initialized
            return np.zeros((self.num_rows, self.dim), dtype=np.int8)

    def _get_core_weights(self) -> np.ndarray:
        """Get core weight signs from FPGA/simulation."""
        if hasattr(self.fpga, 'core_rows') and self.fpga.core_rows is not None:
            return self.fpga.core_rows.copy()
        else:
            return np.zeros((self.num_rows, self.dim), dtype=np.int8)

    def _set_accumulators(self, accum: np.ndarray):
        """Write accumulator state back to FPGA/simulation."""
        if hasattr(self.fpga, 'accumulators'):
            self.fpga.accumulators = accum.copy()
            # Update core weights to match
            self.fpga.core_rows = np.where(accum > 0, 1, -1).astype(np.int8)

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthReport:
        """
        Comprehensive health assessment of the soul.

        Returns a report with metrics and warnings.
        """
        accum = self._get_accumulators()
        now = datetime.now().isoformat()

        # Basic statistics
        magnitudes = np.abs(accum).flatten()
        mean_mag = float(np.mean(magnitudes))
        max_mag = float(np.max(magnitudes))

        # Saturation: how many accumulators are at limits?
        at_max = np.sum(accum >= self.acc_max)
        at_min = np.sum(accum <= self.acc_min)
        saturation = (at_max + at_min) / accum.size * 100

        # Polarity balance: should be near 0 if healthy
        positive = np.sum(accum > 0)
        negative = np.sum(accum < 0)
        total = positive + negative
        if total > 0:
            balance = (positive - negative) / total
        else:
            balance = 0.0

        # Drift rate
        drift = self._compute_drift(accum)

        # Pattern analysis
        row_strengths = np.mean(np.abs(accum), axis=1)
        strong_threshold = self.acc_max * 0.7
        attractors = np.sum(np.mean(accum, axis=1) > strong_threshold)
        repellers = np.sum(np.mean(accum, axis=1) < -strong_threshold)
        deepest = float(np.max(np.abs(np.mean(accum, axis=1))))

        # Determine overall health
        warnings = []
        overall = SoulHealth.HEALTHY

        if saturation > 30:
            warnings.append(f"High saturation ({saturation:.1f}%) - consider annealing")
            overall = SoulHealth.STUCK

        if abs(balance) > 0.3:
            direction = "positive" if balance > 0 else "negative"
            warnings.append(f"Polarity imbalance ({balance:.2f}) - {direction} bias")
            if overall == SoulHealth.HEALTHY:
                overall = SoulHealth.DRIFTING

        if drift > 10:
            warnings.append(f"High drift rate ({drift:.1f}/sec) - review reward stream")
            overall = SoulHealth.DRIFTING

        if attractors > self.num_rows * 0.3:
            warnings.append(f"Many deep attractors ({attractors}) - may be stuck")
            overall = SoulHealth.STUCK

        if repellers > self.num_rows * 0.5 and deepest > self.acc_max * 0.9:
            warnings.append("Strong aversion patterns detected - possible trauma")
            overall = SoulHealth.TRAUMATIZED

        return HealthReport(
            overall=overall,
            timestamp=now,
            mean_magnitude=mean_mag,
            max_magnitude=max_mag,
            saturation_pct=saturation,
            polarity_balance=balance,
            drift_rate=drift,
            drift_direction="positive" if balance > 0.1 else "negative" if balance < -0.1 else "neutral",
            num_strong_attractors=int(attractors),
            num_strong_repellers=int(repellers),
            deepest_attractor_strength=deepest,
            warnings=warnings,
        )

    def _compute_drift(self, current: np.ndarray) -> float:
        """Compute drift rate since last snapshot."""
        import time
        now = time.time()

        if self._last_snapshot is None:
            self._last_snapshot = current.copy()
            self._last_snapshot_time = now
            return 0.0

        # Compute change
        delta = np.sum(np.abs(current - self._last_snapshot))
        elapsed = now - self._last_snapshot_time

        if elapsed > 0:
            rate = delta / elapsed
        else:
            rate = 0.0

        # Update snapshot
        self._last_snapshot = current.copy()
        self._last_snapshot_time = now
        self._drift_history.append(rate)

        # Keep history bounded
        if len(self._drift_history) > 100:
            self._drift_history = self._drift_history[-100:]

        return rate

    # =========================================================================
    # Pattern Analysis
    # =========================================================================

    def find_strongest_patterns(self, top_k: int = 10) -> List[Dict]:
        """
        Find the strongest attractor and repeller patterns.

        Returns rows with highest absolute mean accumulator values.
        """
        accum = self._get_accumulators()
        row_means = np.mean(accum, axis=1)
        row_mags = np.abs(row_means)

        # Sort by magnitude
        sorted_idx = np.argsort(row_mags)[::-1]

        results = []
        for i, idx in enumerate(sorted_idx[:top_k]):
            mean_val = float(row_means[idx])
            pattern_type = "attractor" if mean_val > 0 else "repeller"
            results.append({
                "rank": i + 1,
                "row_id": int(idx),
                "type": pattern_type,
                "strength": float(row_mags[idx]),
                "mean_value": mean_val,
            })

        return results

    def pattern_overlap(self, pattern_hv: np.ndarray) -> Dict[str, float]:
        """
        Compute how much the soul resonates with a given pattern.

        Returns overlap scores with each row.
        """
        if pattern_hv.shape[0] != self.dim:
            raise ValueError(f"Pattern must have dim={self.dim}")

        weights = self._get_core_weights()
        pattern_sign = np.sign(pattern_hv)

        overlaps = []
        for row_idx in range(self.num_rows):
            row = weights[row_idx]
            # Compute cosine similarity (for bipolar: dot product / dim)
            overlap = np.dot(row.astype(float), pattern_sign.astype(float)) / self.dim
            overlaps.append(overlap)

        return {
            "max_overlap": float(np.max(overlaps)),
            "min_overlap": float(np.min(overlaps)),
            "mean_overlap": float(np.mean(overlaps)),
            "best_row": int(np.argmax(overlaps)),
            "worst_row": int(np.argmin(overlaps)),
        }

    # =========================================================================
    # Therapy Tools
    # =========================================================================

    def anneal(self, rate: float = 0.1, rows: Optional[List[int]] = None):
        """
        Gentle decay of all accumulators toward zero.

        This is like emotional cooling - reducing extremes without erasing.

        Args:
            rate: Decay factor (0.1 = reduce magnitude by 10%)
            rows: Optional list of rows to anneal (default: all)
        """
        if self._frozen:
            logger.warning("Soul is frozen - cannot anneal")
            return

        accum = self._get_accumulators()
        target_rows = rows if rows else range(self.num_rows)

        for row_idx in target_rows:
            if row_idx < self.num_rows:
                # Decay toward zero
                accum[row_idx] = (accum[row_idx] * (1 - rate)).astype(np.int8)

        self._set_accumulators(accum)
        logger.info(f"Annealed {len(list(target_rows))} rows at rate {rate}")

    def smooth_region(
        self,
        row_range: Tuple[int, int],
        sigma: float = 0.5,
    ):
        """
        Smooth accumulators within a row range.

        This reduces sharp edges while preserving general patterns.
        """
        if self._frozen:
            logger.warning("Soul is frozen - cannot smooth")
            return

        accum = self._get_accumulators()
        start, end = row_range
        start = max(0, start)
        end = min(self.num_rows, end)

        for row_idx in range(start, end):
            # Simple 1D Gaussian-like smoothing along the row
            row = accum[row_idx].astype(float)
            kernel_size = int(sigma * 10) | 1  # Ensure odd
            kernel = np.exp(-np.linspace(-2, 2, kernel_size) ** 2)
            kernel /= kernel.sum()

            # Convolve with zero padding
            smoothed = np.convolve(row, kernel, mode='same')
            accum[row_idx] = np.clip(smoothed, self.acc_min, self.acc_max).astype(np.int8)

        self._set_accumulators(accum)
        logger.info(f"Smoothed rows {start}-{end} with sigma={sigma}")

    def forget_pattern(self, pattern_hv: np.ndarray, strength: float = 0.5):
        """
        Weaken the soul's response to a specific pattern.

        This is targeted unlearning - use carefully.

        Args:
            pattern_hv: The pattern to forget
            strength: How aggressively to forget (0-1)
        """
        if self._frozen:
            logger.warning("Soul is frozen - cannot forget")
            return

        accum = self._get_accumulators()
        pattern_sign = np.sign(pattern_hv).astype(np.int8)

        for row_idx in range(self.num_rows):
            # Find where this row agrees with the pattern
            row = self._get_core_weights()[row_idx]
            agreement = (row == pattern_sign)

            # Decay those agreements
            decay = (accum[row_idx] * strength * agreement).astype(np.int8)
            accum[row_idx] = np.clip(accum[row_idx] - decay, self.acc_min, self.acc_max).astype(np.int8)

        self._set_accumulators(accum)
        logger.info(f"Forgot pattern with strength {strength}")

    def dream_replay(
        self,
        experiences: List[Tuple[np.ndarray, int]],  # (pattern_hv, reward)
        dream_rate: float = 0.3,
    ):
        """
        Offline replay of experiences to reshape the soul.

        This is like REM sleep - processing memories to integrate them.

        Args:
            experiences: List of (pattern, reward) tuples to replay
            dream_rate: Learning rate during dreaming (usually lower than awake)
        """
        if self._frozen:
            logger.warning("Soul is frozen - cannot dream")
            return

        logger.info(f"Starting dream replay with {len(experiences)} experiences")

        accum = self._get_accumulators()

        for pattern_hv, reward in experiences:
            if reward == 0:
                continue

            delta = 1 if reward > 0 else -1
            # Scale by dream_rate
            effective_delta = int(delta * dream_rate * abs(reward) / 128)
            if effective_delta == 0:
                effective_delta = delta

            pattern_sign = (pattern_hv > 0).astype(np.int8)

            for row_idx in range(self.num_rows):
                row_sign = (accum[row_idx] > 0).astype(np.int8)
                agree = ~(row_sign ^ pattern_sign) & 1
                step = np.where(agree, effective_delta, -effective_delta)
                accum[row_idx] = np.clip(accum[row_idx] + step, self.acc_min, self.acc_max).astype(np.int8)

        self._set_accumulators(accum)
        logger.info("Dream replay complete")

    # =========================================================================
    # Safety Rails
    # =========================================================================

    def emergency_freeze(self):
        """
        Immediately halt all plasticity.

        Use when something seems wrong. Better safe than scarred.
        """
        self._frozen = True
        logger.warning("🛑 SOUL FROZEN - Plasticity halted")

    def unfreeze(self, confirm: str = ""):
        """
        Resume plasticity after freeze.

        Requires confirmation string "I understand the risks"
        """
        if confirm != "I understand the risks":
            logger.error("Must confirm with 'I understand the risks'")
            return
        self._frozen = False
        logger.info("✓ Soul unfrozen - Plasticity resumed")

    def is_frozen(self) -> bool:
        return self._frozen

    def check_reward_budget(self, amount: int) -> bool:
        """Check if reward is within hourly budget."""
        if self._reward_spent + abs(amount) > self._reward_budget:
            logger.warning(f"Reward budget exceeded ({self._reward_spent}/{self._reward_budget})")
            return False
        return True

    def spend_reward(self, amount: int):
        """Track reward spending."""
        self._reward_spent += abs(amount)

    def reset_reward_budget(self):
        """Reset hourly reward budget (call from scheduler)."""
        self._reward_spent = 0

    # =========================================================================
    # Backup & Restore
    # =========================================================================

    def backup(self, name: Optional[str] = None) -> str:
        """
        Snapshot the current soul state.

        Returns path to backup file.
        """
        os.makedirs(self.backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name or f"soul_{timestamp}"
        filepath = os.path.join(self.backup_dir, f"{name}.npz")

        accum = self._get_accumulators()
        weights = self._get_core_weights()

        np.savez_compressed(
            filepath,
            accumulators=accum,
            core_weights=weights,
            dim=self.dim,
            num_rows=self.num_rows,
            acc_width=self.acc_width,
            timestamp=timestamp,
        )

        logger.info(f"Soul backed up to {filepath}")
        return filepath

    def restore(self, filepath: str, confirm: str = ""):
        """
        Restore soul from backup.

        Requires confirmation - this overwrites current state!
        """
        if confirm != "I understand this overwrites current state":
            logger.error("Must confirm with 'I understand this overwrites current state'")
            return

        if not os.path.exists(filepath):
            logger.error(f"Backup not found: {filepath}")
            return

        data = np.load(filepath)

        if data['dim'] != self.dim or data['num_rows'] != self.num_rows:
            logger.error("Backup dimensions don't match current config")
            return

        accum = data['accumulators']
        self._set_accumulators(accum)

        logger.info(f"Soul restored from {filepath}")

    def list_backups(self) -> List[str]:
        """List available backups."""
        if not os.path.exists(self.backup_dir):
            return []
        return sorted([f for f in os.listdir(self.backup_dir) if f.endswith('.npz')])

    # =========================================================================
    # Visualization Helpers
    # =========================================================================

    def get_histogram_data(self) -> Dict:
        """Get accumulator histogram data for plotting."""
        accum = self._get_accumulators().flatten()
        hist, bins = np.histogram(accum, bins=50, range=(self.acc_min, self.acc_max))
        return {
            "counts": hist.tolist(),
            "bin_edges": bins.tolist(),
            "mean": float(np.mean(accum)),
            "std": float(np.std(accum)),
        }

    def get_row_summary(self) -> List[Dict]:
        """Get per-row summary statistics."""
        accum = self._get_accumulators()
        summaries = []
        for row_idx in range(self.num_rows):
            row = accum[row_idx]
            summaries.append({
                "row_id": row_idx,
                "mean": float(np.mean(row)),
                "std": float(np.std(row)),
                "min": int(np.min(row)),
                "max": int(np.max(row)),
                "positive_pct": float(np.mean(row > 0) * 100),
            })
        return summaries


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for Soul Doctor."""
    import argparse

    parser = argparse.ArgumentParser(description="Ara Soul Doctor CLI")
    parser.add_argument("--action", choices=[
        "health", "patterns", "anneal", "backup", "restore", "list-backups"
    ], required=True)
    parser.add_argument("--rate", type=float, default=0.1, help="Anneal rate")
    parser.add_argument("--backup-name", type=str, help="Backup name")
    parser.add_argument("--restore-file", type=str, help="File to restore")
    parser.add_argument("--confirm", type=str, help="Confirmation string")

    args = parser.parse_args()

    # Create simulated FPGA for testing
    from ara.hardware.kitten.corrspike_axis_bridge import SimulatedFPGA
    fpga = SimulatedFPGA(hv_dim=8192)

    # Initialize with some state
    fpga.core_rows = np.random.choice([-1, 1], size=(64, 8192)).astype(np.int8)
    fpga.accumulators = np.random.randint(-30, 30, size=(64, 8192)).astype(np.int8)

    doc = SoulDoctor(fpga, dim=8192, num_rows=64)

    if args.action == "health":
        report = doc.health_check()
        print(json.dumps(report.to_dict(), indent=2))

    elif args.action == "patterns":
        patterns = doc.find_strongest_patterns(top_k=10)
        for p in patterns:
            print(f"#{p['rank']}: Row {p['row_id']} ({p['type']}) - strength {p['strength']:.2f}")

    elif args.action == "anneal":
        doc.anneal(rate=args.rate)
        print(f"Annealed at rate {args.rate}")

    elif args.action == "backup":
        path = doc.backup(name=args.backup_name)
        print(f"Backed up to: {path}")

    elif args.action == "restore":
        if not args.restore_file:
            print("Must specify --restore-file")
            return
        doc.restore(args.restore_file, confirm=args.confirm or "")

    elif args.action == "list-backups":
        backups = doc.list_backups()
        for b in backups:
            print(b)


if __name__ == "__main__":
    main()
