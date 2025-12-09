#!/usr/bin/env python3
"""
Sanity Test: Precision RL
==========================

Verifies that the precision weight learning works.

Success criteria:
- Positive reward increases weights
- Weight updates are bounded
- Export produces valid config

Run: python tests/sanity/test_precision_rl.py
"""

import sys
import numpy as np

# Add project root to path
sys.path.insert(0, '/home/user/Ara')

from research.rl_adaptation import (
    PrecisionWeightLearner,
    FeedbackSignal,
    WeightUpdate,
)


def test_learner_initializes():
    """Test that learner initializes with sensible defaults."""
    print("Test: Learner initializes...")

    modalities = ['speech', 'vision', 'intero', 'proprio']
    learner = PrecisionWeightLearner(modalities)

    # Check initial weights
    assert len(learner.omega) == 4, f"Expected 4 omegas, got {len(learner.omega)}"
    assert learner.kappa > 0, f"Kappa should be positive"

    for mod in modalities:
        assert mod in learner.omega, f"Missing omega for {mod}"
        assert 0 < learner.omega[mod] < 1, f"Omega[{mod}] out of range"

    print(f"  Initial omega: {learner.omega}")
    print(f"  Initial kappa: {learner.kappa:.3f}")
    print("  PASS: Initialization correct")


def test_feedback_recording():
    """Test that feedback signals are recorded."""
    print("Test: Feedback recording...")

    learner = PrecisionWeightLearner(['speech', 'vision'])

    # Record some feedback
    for i in range(5):
        signal = FeedbackSignal(
            timestamp=float(i),
            signal_type='dwell',
            value=30.0  # 30 seconds dwell
        )
        learner.record_feedback(signal)

    assert len(learner.episodes) == 5, f"Expected 5 episodes, got {len(learner.episodes)}"

    print(f"  Recorded {len(learner.episodes)} feedback signals")
    print("  PASS: Feedback recorded")


def test_reward_computation():
    """Test that reward is computed correctly."""
    print("Test: Reward computation...")

    learner = PrecisionWeightLearner(['speech'])

    # Test dwell reward
    dwell_signals = [
        FeedbackSignal(timestamp=0, signal_type='dwell', value=60.0),  # +2
    ]
    reward_dwell = learner.compute_reward(dwell_signals)
    assert abs(reward_dwell - 2.0) < 0.01, f"Expected ~2.0, got {reward_dwell}"

    # Test valence reward
    valence_signals = [
        FeedbackSignal(timestamp=0, signal_type='valence', value=0.5),  # +0.5
    ]
    reward_valence = learner.compute_reward(valence_signals)
    assert abs(reward_valence - 0.5) < 0.01, f"Expected ~0.5, got {reward_valence}"

    # Test HRV reward
    hrv_signals = [
        FeedbackSignal(timestamp=0, signal_type='hrv', value=1.0),  # +2 (improved)
    ]
    reward_hrv = learner.compute_reward(hrv_signals)
    assert reward_hrv == 2.0, f"Expected 2.0, got {reward_hrv}"

    print(f"  Dwell reward: {reward_dwell:.2f}")
    print(f"  Valence reward: {reward_valence:.2f}")
    print(f"  HRV reward: {reward_hrv:.2f}")
    print("  PASS: Rewards computed correctly")


def test_weight_update():
    """Test that weights are updated based on reward."""
    print("Test: Weight update...")

    learner = PrecisionWeightLearner(['speech', 'vision'])

    # Record enough positive feedback
    for i in range(20):
        signal = FeedbackSignal(
            timestamp=float(i),
            signal_type='dwell',
            value=60.0  # High dwell = positive
        )
        learner.record_feedback(signal)

    # Get initial weights
    initial_omega = dict(learner.omega)
    initial_kappa = learner.kappa

    # Update weights
    update = learner.update_weights()

    assert update is not None, "Should produce update with enough episodes"
    assert update.episode_count == 20, f"Expected 20 episodes, got {update.episode_count}"

    print(f"  Update confidence: {update.confidence:.2f}")
    print(f"  Omega deltas: {update.omega_deltas}")
    print(f"  Kappa delta: {update.kappa_delta:.4f}")

    # Deltas should be bounded
    for mod, delta in update.omega_deltas.items():
        assert abs(delta) < 0.1, f"Delta too large for {mod}: {delta}"

    assert abs(update.kappa_delta) < 0.05, f"Kappa delta too large: {update.kappa_delta}"

    print("  PASS: Update produced and bounded")


def test_apply_update():
    """Test that applying update changes weights."""
    print("Test: Apply update...")

    learner = PrecisionWeightLearner(['speech', 'vision'])

    # Create a manual update
    update = WeightUpdate(
        omega_deltas={'speech': 0.05, 'vision': -0.03},
        kappa_delta=0.01,
        confidence=0.8,
        episode_count=50
    )

    initial_speech = learner.omega['speech']
    initial_vision = learner.omega['vision']
    initial_kappa = learner.kappa

    # Apply it
    learner.apply_update(update)

    # Check changes
    assert learner.omega['speech'] > initial_speech, "Speech omega should increase"
    assert learner.omega['vision'] < initial_vision, "Vision omega should decrease"
    assert learner.kappa > initial_kappa, "Kappa should increase"

    print(f"  Speech: {initial_speech:.3f} → {learner.omega['speech']:.3f}")
    print(f"  Vision: {initial_vision:.3f} → {learner.omega['vision']:.3f}")
    print(f"  Kappa: {initial_kappa:.3f} → {learner.kappa:.3f}")
    print("  PASS: Update applied correctly")


def test_export_config():
    """Test that config export works."""
    print("Test: Export config...")

    learner = PrecisionWeightLearner(['speech', 'vision', 'intero'])

    # Make some updates
    for i in range(15):
        learner.record_feedback(FeedbackSignal(
            timestamp=float(i),
            signal_type='dwell',
            value=45.0
        ))

    learner.update_weights()

    # Export
    config = learner.export_config()

    assert 'precision_weights' in config, "Should have precision_weights key"
    assert 'omega' in config['precision_weights'], "Should have omega"
    assert 'kappa' in config['precision_weights'], "Should have kappa"

    print(f"  Exported config: {config}")
    print("  PASS: Config exported")


def main():
    """Run all precision RL sanity tests."""
    print("=" * 60)
    print("SANITY TEST: Precision RL")
    print("=" * 60)
    print()

    try:
        test_learner_initializes()
        test_feedback_recording()
        test_reward_computation()
        test_weight_update()
        test_apply_update()
        test_export_config()

        print()
        print("=" * 60)
        print("ALL PRECISION RL TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
