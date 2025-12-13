"""
ara.hgf.hud_client - HUD Integration for Real-Time HGF Visualization

Streams HGF metrics to the T-FAN GNOME HUD for real-time visualization.
Writes metrics to a JSON file that the HUD polls.

This bridges the HGF simulation with the spaceship cockpit visualization.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

from ara.hgf.core import HGFState, HGFParams
from ara.hgf.agents import HGFAgent, HGFTrajectory
from ara.hgf.tasks import TaskData


# Default metrics file location
RUNTIME_DIR = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
DEFAULT_METRICS_FILE = Path(RUNTIME_DIR) / "tfan_hud_metrics.json"


@dataclass
class HGFHudMetrics:
    """Metrics to send to the HUD."""
    # Trial info
    trial: int = 0
    observation: float = 0.0
    prediction: float = 0.5

    # Level 2 beliefs
    mu_2: float = 0.0
    sigma_2: float = 1.0

    # Level 3 beliefs (volatility)
    mu_3: float = 1.0
    sigma_3: float = 1.0

    # Prediction errors
    delta_1: float = 0.0
    delta_2: float = 0.0
    delta_3: float = 0.0

    # Precisions
    pi_1: float = 1.0
    pi_hat_2: float = 1.0

    # Derived metrics for HUD
    valence: float = 0.0  # Based on δ₁ sign
    drive_total: float = 0.3  # Based on |δ|

    # Volatility metrics
    volatility_estimate: float = 1.0

    # HGF layer health (0-1)
    hgf_l1_health: float = 1.0
    hgf_l2_health: float = 1.0
    hgf_l3_health: float = 1.0
    hgf_l4_health: float = 1.0
    hgf_l5_health: float = 1.0

    # Sanity metrics
    sanity_criticality: float = 0.1
    sanity_stability_margin: float = 0.8
    sanity_entropy_budget: float = 0.7

    # Antifragility
    antifragility_index: float = 0.3
    convexity: float = 0.1

    # DAU
    dau_active: bool = False
    dau_reason: str = ""

    # Timestamp
    last_update: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "trial": self.trial,
            "observation": self.observation,
            "prediction": self.prediction,
            "mu_2": self.mu_2,
            "sigma_2": self.sigma_2,
            "mu_3": self.mu_3,
            "sigma_3": self.sigma_3,
            "delta_1": self.delta_1,
            "delta_2": self.delta_2,
            "delta_3": self.delta_3,
            "pi_1": self.pi_1,
            "pi_hat_2": self.pi_hat_2,
            "valence": self.valence,
            "drive_total": self.drive_total,
            "volatility_estimate": self.volatility_estimate,
            "hgf_l1_health": self.hgf_l1_health,
            "hgf_l2_health": self.hgf_l2_health,
            "hgf_l3_health": self.hgf_l3_health,
            "hgf_l4_health": self.hgf_l4_health,
            "hgf_l5_health": self.hgf_l5_health,
            "sanity_criticality": self.sanity_criticality,
            "sanity_stability_margin": self.sanity_stability_margin,
            "sanity_entropy_budget": self.sanity_entropy_budget,
            "antifragility_index": self.antifragility_index,
            "convexity": self.convexity,
            "dau_active": self.dau_active,
            "dau_trigger_reason": self.dau_reason,
            "last_update": self.last_update,
        }


class HGFHudClient:
    """
    Client that streams HGF metrics to the GNOME HUD.

    Usage:
        client = HGFHudClient()
        client.start()

        # In your HGF simulation loop:
        for trial in range(n_trials):
            state = agent.update(observation)
            client.update(state, trial, observation)

        client.stop()
    """

    def __init__(
        self,
        metrics_file: Optional[Path] = None,
        model_id: str = "hgf_sim",
    ):
        """
        Initialize HUD client.

        Args:
            metrics_file: Path to metrics JSON file
            model_id: Identifier for this simulation
        """
        self.metrics_file = metrics_file or DEFAULT_METRICS_FILE
        self.model_id = model_id
        self.metrics = HGFHudMetrics()
        self._running = False

        # Rolling history for derived metrics
        self._delta_1_history: list = []
        self._delta_2_history: list = []
        self._sigma_2_history: list = []
        self._history_length = 20

    def start(self):
        """Start the HUD client."""
        self._running = True
        self._write_metrics()

    def stop(self):
        """Stop the HUD client."""
        self._running = False

    def update(
        self,
        state: HGFState,
        trial: int = 0,
        observation: float = 0.0,
        prediction: float = 0.5,
    ):
        """
        Update metrics from HGF state.

        Args:
            state: Current HGF state
            trial: Trial number
            observation: Current observation
            prediction: Model prediction before update
        """
        if not self._running:
            return

        # Update rolling history
        self._delta_1_history.append(state.delta_1)
        self._delta_2_history.append(state.delta_2)
        self._sigma_2_history.append(state.sigma_2)

        if len(self._delta_1_history) > self._history_length:
            self._delta_1_history.pop(0)
            self._delta_2_history.pop(0)
            self._sigma_2_history.pop(0)

        # Basic metrics
        self.metrics.trial = trial
        self.metrics.observation = observation
        self.metrics.prediction = prediction
        self.metrics.mu_2 = state.mu_2
        self.metrics.sigma_2 = state.sigma_2
        self.metrics.mu_3 = state.mu_3
        self.metrics.sigma_3 = state.sigma_3
        self.metrics.delta_1 = state.delta_1
        self.metrics.delta_2 = state.delta_2
        self.metrics.delta_3 = state.delta_3
        self.metrics.pi_1 = state.pi_1
        self.metrics.pi_hat_2 = state.pi_hat_2

        # Derived metrics
        self.metrics.valence = state.delta_1  # Positive PE = good surprise
        self.metrics.drive_total = min(1.0, abs(state.delta_1) + 0.5 * abs(state.delta_2))
        self.metrics.volatility_estimate = min(10.0, max(0.01, _safe_exp(state.mu_3)))

        # HGF layer health (inverse of recent variance/instability)
        if len(self._sigma_2_history) > 5:
            sigma_var = _variance(self._sigma_2_history[-10:])
            self.metrics.hgf_l2_health = max(0.0, 1.0 - min(1.0, sigma_var * 10))

            delta_1_var = _variance(self._delta_1_history[-10:])
            self.metrics.hgf_l1_health = max(0.0, 1.0 - min(1.0, delta_1_var * 5))

            delta_2_var = _variance(self._delta_2_history[-10:])
            self.metrics.hgf_l3_health = max(0.0, 1.0 - min(1.0, delta_2_var * 5))

        self.metrics.hgf_l4_health = 0.9  # Placeholder
        self.metrics.hgf_l5_health = max(0.5, 1.0 - abs(state.delta_3))

        # Sanity metrics
        self.metrics.sanity_criticality = min(1.0, self.metrics.drive_total * 1.5)
        self.metrics.sanity_stability_margin = max(0.0, 1.0 - self.metrics.sanity_criticality)
        self.metrics.sanity_entropy_budget = max(0.0, 1.0 - state.sigma_2 / 2.0)

        # Antifragility (simplified)
        # Positive if we're learning from volatility (high κ effect)
        self.metrics.antifragility_index = 0.5 - abs(state.delta_2) * 0.5
        self.metrics.convexity = -state.delta_2 * 0.1  # Simplified

        # DAU activation (defensive mode)
        if self.metrics.sanity_criticality > 0.7:
            self.metrics.dau_active = True
            self.metrics.dau_reason = f"High criticality: {self.metrics.sanity_criticality:.2f}"
        elif abs(state.delta_1) > 0.8:
            self.metrics.dau_active = True
            self.metrics.dau_reason = f"Large prediction error: δ₁={state.delta_1:.2f}"
        else:
            self.metrics.dau_active = False
            self.metrics.dau_reason = ""

        self.metrics.last_update = time.time()

        # Write to file
        self._write_metrics()

    def _write_metrics(self):
        """Write metrics to JSON file."""
        try:
            data = {
                "models": {
                    self.model_id: self.metrics.to_dict()
                }
            }
            self.metrics_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[HGFHudClient] Error writing metrics: {e}")

    def create_callback(self) -> Callable[[int, HGFState], None]:
        """
        Create a callback function for use with HGFAgent.run().

        Returns:
            Callback function
        """
        def callback(trial: int, state: HGFState):
            prediction = 1.0 / (1.0 + _safe_exp(-state.mu_2))
            self.update(state, trial, state.mu_1, prediction)

        return callback


def run_simulation_with_hud(
    task_data: TaskData,
    agent: HGFAgent,
    delay: float = 0.1,
    metrics_file: Optional[Path] = None,
) -> HGFTrajectory:
    """
    Run an HGF simulation with real-time HUD updates.

    Args:
        task_data: Task to simulate
        agent: HGF agent
        delay: Delay between trials (seconds) for visualization
        metrics_file: Path to metrics file

    Returns:
        HGF trajectory
    """
    client = HGFHudClient(metrics_file=metrics_file)
    client.start()

    try:
        trajectory = agent.run(
            task_data,
            generate_actions=True,
            callback=client.create_callback(),
        )

        # Add delay between trials for visualization
        import time
        for step in trajectory.steps:
            client.update(step.state, step.trial, step.observation, step.prediction)
            time.sleep(delay)

    finally:
        client.stop()

    return trajectory


# =============================================================================
# Helper Functions
# =============================================================================

def _safe_exp(x: float, max_val: float = 700.0) -> float:
    """Exponential with overflow protection."""
    import math
    return math.exp(min(x, max_val))


def _variance(x: list) -> float:
    """Compute variance of a list."""
    if len(x) < 2:
        return 0.0
    mean = sum(x) / len(x)
    return sum((xi - mean) ** 2 for xi in x) / len(x)
