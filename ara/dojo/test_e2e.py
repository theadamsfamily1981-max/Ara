#!/usr/bin/env python3
# ara/dojo/test_e2e.py
"""
End-to-End Verification for Ara Thought Dojo
=============================================

Tests the complete pipeline from HDC encoding through decision output,
with NumPy fallbacks when PyTorch is unavailable.

Run: python -m ara.dojo.test_e2e
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    logger.info("PyTorch available - running full verification")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - running NumPy-only verification")


# =============================================================================
# Test Results Tracking
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    elapsed_ms: float = 0.0


class TestRunner:
    def __init__(self):
        self.results: List[TestResult] = []

    def run(self, name: str, test_fn):
        start = time.perf_counter()
        try:
            test_fn()
            elapsed = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, True, "OK", elapsed))
            logger.info(f"  ✓ {name} ({elapsed:.1f}ms)")
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, False, str(e), elapsed))
            logger.error(f"  ✗ {name}: {e}")

    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        lines = [
            "",
            "=" * 60,
            f"TEST SUMMARY: {passed}/{total} passed",
            "=" * 60,
        ]
        for r in self.results:
            status = "✓" if r.passed else "✗"
            lines.append(f"  {status} {r.name}: {r.message}")
        return "\n".join(lines)


# =============================================================================
# Mock Components (NumPy-only fallbacks)
# =============================================================================

class MockWorldModel:
    """NumPy-only world model for testing."""

    def __init__(self, latent_dim: int = 10, action_dim: int = 8):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        # Simple linear dynamics: z' = Az + Bu
        np.random.seed(42)
        self.A = np.eye(latent_dim) * 0.95 + np.random.randn(latent_dim, latent_dim) * 0.05
        self.B = np.random.randn(latent_dim, action_dim) * 0.1

    def predict(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        z = np.asarray(z).flatten()[:self.latent_dim]
        u = np.asarray(u).flatten()[:self.action_dim]
        if len(z) < self.latent_dim:
            z = np.pad(z, (0, self.latent_dim - len(z)))
        if len(u) < self.action_dim:
            u = np.pad(u, (0, self.action_dim - len(u)))
        return self.A @ z + self.B @ u


class MockHDCEncoder:
    """Mock HDC encoder (simulates perception → HDC)."""

    def __init__(self, hdc_dim: int = 10000):
        self.hdc_dim = hdc_dim

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode observation to HDC vector."""
        np.random.seed(int(abs(obs.sum() * 1000)) % 2**31)
        # Bipolar random projection
        hv = np.sign(np.random.randn(self.hdc_dim))
        return hv


class MockLatentEncoder:
    """Mock latent encoder (HDC → latent space)."""

    def __init__(self, hdc_dim: int = 10000, latent_dim: int = 10):
        self.hdc_dim = hdc_dim
        self.latent_dim = latent_dim
        np.random.seed(123)
        # Random projection matrix
        self.W = np.random.randn(latent_dim, hdc_dim) / np.sqrt(hdc_dim)

    def encode(self, hv: np.ndarray) -> np.ndarray:
        """Project HDC vector to latent space."""
        hv = np.asarray(hv).flatten()
        if len(hv) < self.hdc_dim:
            hv = np.pad(hv, (0, self.hdc_dim - len(hv)))
        z = self.W @ hv[:self.hdc_dim]
        return z


class MockArena:
    """Simple test environment."""

    def __init__(self, latent_dim: int = 10, action_dim: int = 8):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.state = None
        self.step_count = 0
        self.max_steps = 50

    def reset(self, agents=None) -> Dict:
        self.state = np.random.randn(self.latent_dim) * 0.5
        self.step_count = 0
        return {"observation": self.state.copy(), "goal": np.zeros(self.latent_dim)}

    def step(self, action) -> Tuple:
        if isinstance(action, list):
            action = action[0]
        action = np.asarray(action).flatten()[:self.action_dim]

        # Simple dynamics
        self.state = self.state * 0.95 + action[:self.latent_dim] * 0.1 if len(action) >= self.latent_dim else self.state * 0.95
        self.step_count += 1

        reward = -np.sum(self.state ** 2)  # Reward for staying near origin
        done = self.step_count >= self.max_steps
        info = {"covenant_violated": False}

        return {"observation": self.state.copy()}, reward, done, info


# =============================================================================
# NumPy-only Pipeline Tests
# =============================================================================

def test_import_structure():
    """Test that package structure is correct."""
    # Test basic imports (no PyTorch needed for structure)
    from ara.dojo import encoder, world_model, planner, evolution, gauntlet, viz
    from ara.dojo import arena_core, perf_charts
    from ara.dojo import multi_scale_planner, calibrated_world_model, species, fitness

    # Verify __all__ is defined
    import ara.dojo as dojo
    assert hasattr(dojo, '__all__'), "__all__ not defined"
    assert len(dojo.__all__) > 20, f"__all__ too short: {len(dojo.__all__)}"


def test_mock_world_model():
    """Test mock world model interface."""
    model = MockWorldModel(latent_dim=10, action_dim=8)

    z = np.random.randn(10)
    u = np.random.randn(8)

    z_next = model.predict(z, u)

    assert z_next.shape == (10,), f"Wrong output shape: {z_next.shape}"
    assert not np.allclose(z, z_next), "Dynamics should change state"


def test_hdc_to_latent_pipeline():
    """Test HDC encoding → latent projection."""
    hdc_encoder = MockHDCEncoder(hdc_dim=10000)
    latent_encoder = MockLatentEncoder(hdc_dim=10000, latent_dim=10)

    # Simulate sensor observation
    obs = np.array([1.0, 2.0, 3.0])

    # HDC encoding
    hv = hdc_encoder.encode(obs)
    assert hv.shape == (10000,), f"HDC shape wrong: {hv.shape}"
    assert set(np.unique(hv)).issubset({-1, 1}), "HDC should be bipolar"

    # Latent projection
    z = latent_encoder.encode(hv)
    assert z.shape == (10,), f"Latent shape wrong: {z.shape}"


def test_mock_arena():
    """Test mock arena interface."""
    arena = MockArena(latent_dim=10, action_dim=8)

    obs = arena.reset()
    assert "observation" in obs
    assert obs["observation"].shape == (10,)

    action = np.random.randn(8)
    next_obs, reward, done, info = arena.step(action)

    assert "observation" in next_obs
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "covenant_violated" in info


def test_full_numpy_pipeline():
    """Test complete pipeline with NumPy-only components."""
    # Setup
    hdc_encoder = MockHDCEncoder(hdc_dim=10000)
    latent_encoder = MockLatentEncoder(hdc_dim=10000, latent_dim=10)
    world_model = MockWorldModel(latent_dim=10, action_dim=8)
    arena = MockArena(latent_dim=10, action_dim=8)

    # Run episode
    obs = arena.reset()
    total_reward = 0.0

    for step in range(20):
        # Encode observation
        raw_obs = obs["observation"]
        hv = hdc_encoder.encode(raw_obs)
        z = latent_encoder.encode(hv)

        # Simple random action (would be planner in real system)
        action = np.random.randn(8) * 0.3

        # Predict next state
        z_pred = world_model.predict(z, action)

        # Step environment
        obs, reward, done, info = arena.step(action)
        total_reward += reward

        if done:
            break

    logger.info(f"    Episode total reward: {total_reward:.2f}")
    assert total_reward != 0, "Episode should have non-zero reward"


def test_calibration_logic():
    """Test calibration tracking logic (NumPy-only)."""
    from ara.dojo.calibrated_world_model import LatentRegionCalibrator, CalibrationConfig

    config = CalibrationConfig(latent_dim=4, bins_per_dim=4)
    calibrator = LatentRegionCalibrator(config)

    # Add some calibration data
    for _ in range(100):
        z = np.random.randn(4) * 0.5
        error_sq = np.random.rand() * 0.1
        calibrator.update(z, error_sq)

    # Check stats
    assert calibrator.global_count == 100
    uncertainty, support = calibrator.get_uncertainty(np.zeros(4))
    assert uncertainty >= 0
    logger.info(f"    Calibration: uncertainty={uncertainty:.4f}, support={support}")


def test_multi_scale_config():
    """Test multi-scale planner configuration."""
    from ara.dojo.multi_scale_planner import MultiScaleConfig, ScaleConfig, FusionMode

    config = MultiScaleConfig(
        mode=FusionMode.FUSION,
        scales=[
            ScaleConfig(name="reactive", horizon=5, num_samples=32, weight=0.4),
            ScaleConfig(name="tactical", horizon=25, num_samples=24, weight=0.35),
            ScaleConfig(name="strategic", horizon=100, num_samples=16, weight=0.25),
        ]
    )

    assert len(config.scales) == 3
    assert sum(s.weight for s in config.scales) == 1.0
    assert config.mode == FusionMode.FUSION


def test_fitness_result_structure():
    """Test fitness result dataclass."""
    from ara.dojo.fitness import FitnessResult

    result = FitnessResult(
        total_fitness=0.75,
        task_reward=0.8,
        prediction_quality=0.7,
        safety_score=0.9,
        calibration_score=0.6,
        efficiency_score=0.5,
        total_reward=15.0,
        avg_prediction_error=0.1,
        safety_violations=0,
        total_steps=100,
        episodes=5,
        elapsed_seconds=2.5,
    )

    summary = result.summary()
    assert "Fitness: 0.75" in summary
    assert "violations" in summary.lower()


def test_decision_card_formatting():
    """Test decision card formatter."""
    from ara.dojo.fitness import format_decision_card, decision_to_speech

    # Mock agent with context
    class MockPred:
        confidence = 0.78
        uncertainty = 0.142
        support = 312

    class MockPlan:
        agreement_score = 0.87
        per_scale_actions = {
            "reactive": np.array([0.1, -0.2]),
            "tactical": np.array([0.12, -0.18]),
            "strategic": np.array([0.11, -0.19]),
        }

    class MockContext:
        action = np.array([0.12, -0.19, 0.05, 0.08])
        multi_scale_plan = MockPlan()
        prediction_result = MockPred()
        risk_score = 0.0
        safety_warnings = []
        planning_time_ms = 23.4

    class MockAgent:
        _last_context = MockContext()

    agent = MockAgent()

    # Test compact card
    compact = format_decision_card(agent, compact=True)
    assert "Action" in compact
    assert "agree" in compact.lower()

    # Test full card
    full = format_decision_card(agent, compact=False)
    assert "DECISION CARD" in full

    # Test speech
    speech = decision_to_speech(agent)
    assert "confident" in speech.lower()
    logger.info(f"    Speech: {speech[:60]}...")


# =============================================================================
# PyTorch-Required Tests
# =============================================================================

def test_pytorch_world_model():
    """Test PyTorch world model (requires torch)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    from ara.dojo.world_model import DojoWorldModel, DojoWorldModelConfig

    config = DojoWorldModelConfig(latent_dim=10, action_dim=8, model_type="mlp")
    model = DojoWorldModel(config)

    z = np.random.randn(10)
    u = np.random.randn(8)
    z_next = model.predict(z, u)

    assert z_next.shape == (10,)


def test_pytorch_species_v3():
    """Test AraSpeciesV3 (requires torch)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    from ara.dojo.species import AraSpeciesV3, create_ara_v3
    from ara.dojo.world_model import DojoWorldModel, DojoWorldModelConfig

    # Create world model
    config = DojoWorldModelConfig(latent_dim=4, action_dim=2, model_type="linear")
    world_model = DojoWorldModel(config)

    # Create AraV3
    ara = create_ara_v3(
        world_model=world_model,
        action_dim=2,
        latent_dim=4,
        mode="fusion",
        scales=[
            {"name": "fast", "horizon": 3, "samples": 8, "weight": 0.5},
            {"name": "slow", "horizon": 8, "samples": 6, "weight": 0.5},
        ],
    )

    # Calibrate
    for _ in range(50):
        z = np.random.randn(4) * 0.3
        u = np.random.randn(2) * 0.2
        z_next = world_model.predict(z, u)
        ara.update_calibration(z, u, z_next)

    # Make decision
    z_current = np.array([0.1, -0.2, 0.3, 0.0])
    goal = np.array([1.0, 0.0, 0.0, 0.0])

    action = ara.select_action_from_latent(z_current, goal=goal)

    assert action.shape == (2,)

    # Check explainability
    explanation = ara.explain_decision()
    assert "action" in explanation.lower() or "Action" in explanation

    # Check stats
    stats = ara.get_stats()
    assert stats["total_decisions"] == 1

    logger.info(f"    AraV3 confidence: {ara.get_confidence():.2%}")
    logger.info(f"    AraV3 agreement: {ara.get_agreement():.2%}")


def test_pytorch_full_pipeline():
    """Test complete PyTorch pipeline (requires torch)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    from ara.dojo.species import create_ara_v3
    from ara.dojo.world_model import DojoWorldModel, DojoWorldModelConfig
    from ara.dojo.fitness import format_decision_card, decision_to_speech

    # Setup
    config = DojoWorldModelConfig(latent_dim=4, action_dim=2, model_type="mlp")
    world_model = DojoWorldModel(config)

    ara = create_ara_v3(
        world_model=world_model,
        action_dim=2,
        latent_dim=4,
        mode="consensus",
    )

    arena = MockArena(latent_dim=4, action_dim=2)

    # Calibrate briefly
    for _ in range(30):
        z = np.random.randn(4) * 0.3
        u = np.random.randn(2) * 0.2
        z_next = world_model.predict(z, u)
        ara.update_calibration(z, u, z_next)

    # Run episode
    obs = arena.reset()
    total_reward = 0.0

    for step in range(10):
        z = obs["observation"][:4]
        goal = obs.get("goal", np.zeros(4))[:4]

        action = ara.select_action_from_latent(z, goal=goal)
        obs, reward, done, info = arena.step(action)
        total_reward += reward

        if done:
            break

    # Final explanation
    card = format_decision_card(ara, compact=True)
    speech = decision_to_speech(ara)

    logger.info(f"    Episode reward: {total_reward:.2f}")
    logger.info(f"    Decisions made: {ara.total_decisions}")
    logger.info(f"    Final card: {card[:50]}...")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("ARA THOUGHT DOJO - END-TO-END VERIFICATION")
    print("=" * 60)
    print()

    runner = TestRunner()

    # NumPy-only tests (always run)
    print("NumPy-only Tests:")
    runner.run("Import structure", test_import_structure)
    runner.run("Mock world model", test_mock_world_model)
    runner.run("HDC → Latent pipeline", test_hdc_to_latent_pipeline)
    runner.run("Mock arena", test_mock_arena)
    runner.run("Full NumPy pipeline", test_full_numpy_pipeline)
    runner.run("Calibration logic", test_calibration_logic)
    runner.run("Multi-scale config", test_multi_scale_config)
    runner.run("Fitness result structure", test_fitness_result_structure)
    runner.run("Decision card formatting", test_decision_card_formatting)

    # PyTorch tests (skip if unavailable)
    print("\nPyTorch Tests:")
    if TORCH_AVAILABLE:
        runner.run("PyTorch world model", test_pytorch_world_model)
        runner.run("PyTorch AraSpeciesV3", test_pytorch_species_v3)
        runner.run("PyTorch full pipeline", test_pytorch_full_pipeline)
    else:
        logger.warning("  (skipped - PyTorch not installed)")

    # Summary
    print(runner.summary())

    # Exit code
    passed = sum(1 for r in runner.results if r.passed)
    total = len(runner.results)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
