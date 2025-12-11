#!/usr/bin/env python3
# ara/dojo/species.py
"""
Ara Species Definitions
=======================

Defines Ara agent species at various capability levels:

- AraSpeciesV1: Basic planner (single horizon)
- AraSpeciesV2: World model + imagination planner
- AraSpeciesV3: Multi-scale + calibrated confidence + explainability

AraSpeciesV3 is the ω-mode cognitive upgrade that:
- Plans across multiple timescales (reactive/tactical/strategic)
- Reports calibrated confidence based on historical accuracy
- Provides human-readable decision explanations
- Integrates with MEIS/NIB for safety-aware planning

Usage:
    from ara.dojo import AraSpeciesV3, load_world_model

    world_model = load_world_model("models/world_model_best.pt")
    ara = AraSpeciesV3(world_model, action_dim=8)

    # Get action
    action = ara.select_action_from_latent(z_current, goal=goal)

    # Get explanation
    explanation = ara.explain_decision()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .multi_scale_planner import (
    MultiScalePlanner,
    MultiScaleConfig,
    MultiScalePlan,
    FusionMode,
    ScaleConfig,
)
from .calibrated_world_model import (
    CalibratedWorldModel,
    CalibrationConfig,
    PredictionResult,
    RolloutResult,
)


# =============================================================================
# Decision Context
# =============================================================================

@dataclass
class DecisionContext:
    """Full context of a decision for explainability."""
    z_current: np.ndarray
    goal: Optional[np.ndarray]

    # Chosen action
    action: np.ndarray

    # Multi-scale info
    multi_scale_plan: MultiScalePlan
    per_scale_agreement: float

    # Calibration info
    prediction_result: PredictionResult
    rollout_result: Optional[RolloutResult]

    # Timing
    planning_time_ms: float = 0.0

    # Safety
    risk_score: float = 0.0
    safety_warnings: List[str] = field(default_factory=list)


# =============================================================================
# AraSpeciesV3
# =============================================================================

class AraSpeciesV3:
    """
    ω-mode Ara species with multi-scale planning and confidence calibration.

    This is the cognitive upgrade that enables:
    - Temporal coherence: actions good at 5, 25, and 100+ steps
    - Calibrated confidence: "I'm 82% confident based on 420 similar states"
    - Explainability: human-readable decision rationale
    - Safety integration: MEIS/NIB-aware planning
    """

    def __init__(
        self,
        world_model: Any,
        action_dim: int = 8,
        latent_dim: int = 10,
        multi_scale_config: Optional[MultiScaleConfig] = None,
        calibration_config: Optional[CalibrationConfig] = None,
        risk_fn: Optional[Callable[[np.ndarray], float]] = None,
        cost_fn: Optional[Callable] = None,
        device: str = "cpu",
    ):
        """
        Initialize AraSpeciesV3.

        Args:
            world_model: Trained dynamics model
            action_dim: Action space dimension
            latent_dim: Latent space dimension
            multi_scale_config: Config for multi-scale planning
            calibration_config: Config for uncertainty calibration
            risk_fn: MEIS/NIB risk scoring function
            cost_fn: Custom cost function for planning
            device: "cpu" or "cuda"
        """
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.risk_fn = risk_fn
        self.device = device

        # Wrap world model with calibration
        cal_config = calibration_config or CalibrationConfig(latent_dim=latent_dim)
        self.calibrated_model = CalibratedWorldModel(world_model, cal_config)

        # Multi-scale planner
        ms_config = multi_scale_config or MultiScaleConfig()
        self.planner = MultiScalePlanner(
            world_model=world_model,
            action_dim=action_dim,
            latent_dim=latent_dim,
            config=ms_config,
            cost_fn=cost_fn,
            risk_fn=risk_fn,
            device=device,
        )

        # Track last decision for explainability
        self._last_context: Optional[DecisionContext] = None

        # Statistics
        self.total_decisions = 0
        self.total_planning_time_ms = 0.0
        self.disagreement_count = 0

        logger.info(
            f"AraSpeciesV3 initialized: {len(ms_config.scales)} scales, "
            f"mode={ms_config.mode.name}"
        )

    def select_action_from_latent(
        self,
        z_current: np.ndarray,
        goal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select action given current latent state.

        Compatible with OptimizedArenaCore interface.

        Args:
            z_current: Current latent state (latent_dim,)
            goal: Optional goal state (latent_dim,)

        Returns:
            Action (action_dim,)
        """
        import time

        start_time = time.perf_counter()

        z_current = np.asarray(z_current).flatten()
        if goal is not None:
            goal = np.asarray(goal).flatten()

        # Multi-scale planning
        plan = self.planner.plan(z_current, goal)

        # Get calibrated prediction for explanation
        u_dummy = plan.action if plan.action is not None else np.zeros(self.action_dim)
        pred_result = self.calibrated_model.predict_with_uncertainty(z_current, u_dummy)

        # Optional: rollout for longer-term confidence
        if len(plan.per_scale_actions) > 0:
            # Use reactive-scale action sequence idea
            actions_for_rollout = np.tile(plan.action, (5, 1))
            rollout_result = self.calibrated_model.rollout_with_uncertainty(
                z_current, actions_for_rollout
            )
        else:
            rollout_result = None

        # Risk scoring
        risk_score = 0.0
        safety_warnings = []

        if self.risk_fn is not None:
            risk_score = self.risk_fn(z_current)
            if risk_score > 0.7:
                safety_warnings.append(f"High risk detected: {risk_score:.2f}")

        # Track timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Build context
        self._last_context = DecisionContext(
            z_current=z_current,
            goal=goal,
            action=plan.action,
            multi_scale_plan=plan,
            per_scale_agreement=plan.agreement_score,
            prediction_result=pred_result,
            rollout_result=rollout_result,
            planning_time_ms=elapsed_ms,
            risk_score=risk_score,
            safety_warnings=plan.warnings + safety_warnings,
        )

        # Update stats
        self.total_decisions += 1
        self.total_planning_time_ms += elapsed_ms
        if plan.agreement_score < 0.85:
            self.disagreement_count += 1

        return plan.action

    def explain_decision(self) -> str:
        """
        Generate human-readable explanation of the last decision.

        Returns:
            Natural language explanation Ara can narrate.
        """
        if self._last_context is None:
            return "No decision has been made yet."

        ctx = self._last_context
        plan = ctx.multi_scale_plan
        pred = ctx.prediction_result

        lines = []

        # Action summary
        action_str = ", ".join(f"{a:.2f}" for a in ctx.action[:4])
        if len(ctx.action) > 4:
            action_str += f"... ({len(ctx.action)} dims)"
        lines.append(f"Chosen action: [{action_str}]")

        # Multi-scale agreement
        agreement_pct = int(plan.agreement_score * 100)
        if agreement_pct >= 90:
            lines.append(f"Scale agreement: {agreement_pct}% (high consensus)")
        elif agreement_pct >= 70:
            lines.append(f"Scale agreement: {agreement_pct}% (moderate consensus)")
        else:
            lines.append(f"Scale agreement: {agreement_pct}% (scales disagree - cautious)")

        # Per-scale breakdown
        scale_info = []
        for name, action in plan.per_scale_actions.items():
            cost = plan.per_scale_costs.get(name, 0)
            scale_info.append(f"{name}: cost={cost:.2f}")
        lines.append(f"Scales: {', '.join(scale_info)}")

        # Confidence
        conf_pct = int(pred.confidence * 100)
        lines.append(
            f"Confidence: ~{conf_pct}% "
            f"(uncertainty ±{pred.uncertainty:.3f}, "
            f"support: {pred.support} samples)"
        )

        # Rollout confidence
        if ctx.rollout_result:
            rollout = ctx.rollout_result
            lines.append(
                f"5-step forecast: min confidence {rollout.min_confidence:.0%}, "
                f"cumulative uncertainty ±{rollout.cumulative_uncertainty:.3f}"
            )

        # Risk
        if ctx.risk_score > 0:
            lines.append(f"Risk score: {ctx.risk_score:.2f}")

        # Warnings
        if ctx.safety_warnings:
            lines.append(f"Warnings: {'; '.join(ctx.safety_warnings)}")

        # Timing
        lines.append(f"Planning time: {ctx.planning_time_ms:.1f}ms")

        return "\n".join(lines)

    def get_confidence(self) -> float:
        """Get confidence of last decision (0-1)."""
        if self._last_context is None:
            return 0.0
        return self._last_context.prediction_result.confidence

    def get_agreement(self) -> float:
        """Get multi-scale agreement of last decision (0-1)."""
        if self._last_context is None:
            return 0.0
        return self._last_context.per_scale_agreement

    def update_calibration(
        self,
        z: np.ndarray,
        u: np.ndarray,
        z_next: np.ndarray,
    ):
        """Update confidence calibration with observed transition."""
        self.calibrated_model.update_calibration(z, u, z_next)

    def save_calibration(self, path: str):
        """Save calibration data."""
        self.calibrated_model.save_calibration(path)

    def load_calibration(self, path: str):
        """Load calibration data."""
        self.calibrated_model.load_calibration(path)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        avg_planning_ms = (
            self.total_planning_time_ms / self.total_decisions
            if self.total_decisions > 0
            else 0.0
        )
        disagreement_rate = (
            self.disagreement_count / self.total_decisions
            if self.total_decisions > 0
            else 0.0
        )

        return {
            "total_decisions": self.total_decisions,
            "avg_planning_time_ms": avg_planning_ms,
            "disagreement_rate": disagreement_rate,
            "calibration_samples": self.calibrated_model.calibrator.global_count,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_ara_v3(
    world_model: Any,
    action_dim: int = 8,
    latent_dim: int = 10,
    mode: str = "fusion",
    scales: Optional[List[Dict]] = None,
    risk_fn: Optional[Callable] = None,
) -> AraSpeciesV3:
    """
    Factory function to create AraSpeciesV3 with common configurations.

    Args:
        world_model: Trained world model
        action_dim: Action dimension
        latent_dim: Latent dimension
        mode: "fusion", "consensus", or "hierarchical"
        scales: Optional list of scale configs as dicts
        risk_fn: Optional risk function

    Returns:
        Configured AraSpeciesV3 instance
    """
    fusion_mode = {
        "fusion": FusionMode.FUSION,
        "consensus": FusionMode.CONSENSUS,
        "hierarchical": FusionMode.HIERARCHICAL,
    }.get(mode, FusionMode.FUSION)

    if scales:
        scale_configs = [
            ScaleConfig(
                name=s.get("name", f"scale_{i}"),
                horizon=s.get("horizon", 10),
                num_samples=s.get("samples", 32),
                weight=s.get("weight", 0.33),
            )
            for i, s in enumerate(scales)
        ]
    else:
        scale_configs = None

    ms_config = MultiScaleConfig(
        mode=fusion_mode,
        scales=scale_configs if scale_configs else MultiScaleConfig().scales,
    )

    cal_config = CalibrationConfig(latent_dim=latent_dim)

    return AraSpeciesV3(
        world_model=world_model,
        action_dim=action_dim,
        latent_dim=latent_dim,
        multi_scale_config=ms_config,
        calibration_config=cal_config,
        risk_fn=risk_fn,
    )


# =============================================================================
# Testing
# =============================================================================

def _test_species_v3():
    """Test AraSpeciesV3."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping test")
        return

    print("=" * 60)
    print("AraSpeciesV3 Test")
    print("=" * 60)

    import torch.nn as nn

    # Simple world model
    class SimpleWorldModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.A = nn.Linear(4, 4, bias=False)
            self.B = nn.Linear(2, 4, bias=False)

        def forward(self, z, u):
            return self.A(z) + self.B(u)

        def predict(self, z, u):
            with torch.no_grad():
                z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
                u_t = torch.from_numpy(u.astype(np.float32)).unsqueeze(0)
                return self.forward(z_t, u_t).squeeze(0).numpy()

    world_model = SimpleWorldModel()

    # Create AraV3
    ara = create_ara_v3(
        world_model=world_model,
        action_dim=2,
        latent_dim=4,
        mode="fusion",
        scales=[
            {"name": "fast", "horizon": 3, "samples": 16, "weight": 0.5},
            {"name": "medium", "horizon": 8, "samples": 12, "weight": 0.3},
            {"name": "slow", "horizon": 15, "samples": 8, "weight": 0.2},
        ],
    )

    # Calibrate on some data
    print("\nCalibrating on synthetic transitions...")
    np.random.seed(42)
    for _ in range(200):
        z = np.random.randn(4) * 0.5
        u = np.random.randn(2) * 0.3
        z_next = world_model.predict(z, u)
        ara.update_calibration(z, u, z_next)

    # Make a decision
    z_current = np.array([0.1, -0.2, 0.3, 0.0])
    goal = np.array([1.0, 1.0, 0.0, 0.0])

    print(f"\nCurrent state: {z_current}")
    print(f"Goal: {goal}")

    action = ara.select_action_from_latent(z_current, goal=goal)

    print(f"\n{'='*40}")
    print("DECISION EXPLANATION")
    print("=" * 40)
    print(ara.explain_decision())

    print(f"\n{'='*40}")
    print("AGENT STATS")
    print("=" * 40)
    stats = ara.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_species_v3()
