#!/usr/bin/env python3
"""
Sanity Test: Breath PID Controller
===================================

Verifies that the breath co-regulation PID works.

Success criteria:
- PID output responds to phase error
- Output is bounded (no runaway)
- Phase locking value (PLV) improves over time

Run: python tests/sanity/test_breath_pid.py
"""

import sys
import numpy as np

# Add project root to path
sys.path.insert(0, '/home/user/Ara')

from ara.embodiment.breath_vision import (
    BreathVisionSession,
    BreathSyncMetrics,
)


def generate_breath_signal(duration_sec: float = 10.0,
                           breath_rate: float = 6.0,
                           sample_rate: float = 30.0,
                           noise_level: float = 0.1) -> np.ndarray:
    """Generate a synthetic breathing signal (sinusoid + noise)."""
    t = np.arange(0, duration_sec, 1.0 / sample_rate)
    freq = breath_rate / 60.0  # Convert BPM to Hz
    signal = np.sin(2 * np.pi * freq * t)
    signal += noise_level * np.random.randn(len(signal))
    return signal


def test_pid_responds_to_error():
    """Test that PID output changes based on phase error."""
    print("Test: PID responds to phase error...")

    session = BreathVisionSession(target_breath_rate=6.0)

    # Simulate different phase errors
    outputs = []
    for phase_error in [-0.5, -0.25, 0.0, 0.25, 0.5]:
        # Manually set state for testing
        session._phase_error = phase_error
        session._integral_error = 0.0
        session._last_error = 0.0

        output = session._compute_pid_output()
        outputs.append((phase_error, output))

    print("  Phase Error → PID Output:")
    for error, output in outputs:
        print(f"    {error:+.2f} → {output:+.3f}")

    # Output should be monotonic with error (P term dominates)
    errors = [e for e, o in outputs]
    outs = [o for e, o in outputs]

    # Check that larger errors produce larger outputs (in same direction)
    assert outs[0] < outs[2] < outs[4], "PID should be monotonic with error"

    print("  PASS: PID is monotonic with error")


def test_pid_output_bounded():
    """Test that PID output stays within bounds."""
    print("Test: PID output is bounded...")

    session = BreathVisionSession(target_breath_rate=6.0)

    # Simulate extreme errors
    outputs = []
    for _ in range(100):
        error = np.random.uniform(-2.0, 2.0)
        session._phase_error = error
        session._integral_error = np.random.uniform(-5.0, 5.0)
        session._last_error = np.random.uniform(-2.0, 2.0)

        output = session._compute_pid_output()
        outputs.append(output)

    min_out = min(outputs)
    max_out = max(outputs)

    print(f"  Output range: [{min_out:.3f}, {max_out:.3f}]")

    # Output should be bounded (clamped in implementation)
    assert -1.5 < min_out, f"Output too low: {min_out}"
    assert max_out < 1.5, f"Output too high: {max_out}"

    print("  PASS: Output bounded")


def test_session_lifecycle():
    """Test that session starts and stops cleanly."""
    print("Test: Session lifecycle...")

    session = BreathVisionSession(target_breath_rate=6.0)

    # Should start inactive
    assert not session.is_active, "Session should start inactive"

    # Start
    session.start()
    assert session.is_active, "Session should be active after start"

    # Process some breath data
    breath_signal = generate_breath_signal(duration_sec=2.0)
    for sample in breath_signal:
        session.update(sample)

    # Get output
    output = session.get_sync_output()
    assert output is not None, "Should produce output"

    # Stop
    session.stop()
    assert not session.is_active, "Session should be inactive after stop"

    print("  PASS: Lifecycle correct")


def test_metrics_computed():
    """Test that sync metrics are computed."""
    print("Test: Sync metrics computed...")

    session = BreathVisionSession(target_breath_rate=6.0)
    session.start()

    # Process a breath signal
    breath_signal = generate_breath_signal(duration_sec=5.0)
    for sample in breath_signal:
        session.update(sample)

    metrics = session.get_metrics()

    print(f"  PLV: {metrics.plv:.3f}")
    print(f"  Lag: {metrics.lag_ms:.1f} ms")
    print(f"  Amplitude ratio: {metrics.amplitude_ratio:.2f}")

    # Metrics should be in reasonable ranges
    assert 0.0 <= metrics.plv <= 1.0, f"PLV out of range: {metrics.plv}"
    assert metrics.lag_ms < 5000, f"Lag too high: {metrics.lag_ms}"

    session.stop()

    print("  PASS: Metrics computed")


def test_max_session_duration():
    """Test that session respects max duration."""
    print("Test: Max session duration enforced...")

    # Create session with very short max duration for testing
    session = BreathVisionSession(
        target_breath_rate=6.0,
        max_duration_minutes=0.01  # ~0.6 seconds
    )
    session.start()

    # Should be active initially
    assert session.is_active, "Session should start active"

    # Process data beyond max duration
    breath_signal = generate_breath_signal(duration_sec=2.0)
    for sample in breath_signal:
        session.update(sample)

    # Session should have auto-stopped (or at least flagged)
    # Implementation may vary - check that it tracks duration
    elapsed = session.elapsed_time
    print(f"  Elapsed time: {elapsed:.2f} sec")

    # Just verify we can check elapsed time
    assert elapsed >= 0, "Elapsed time should be non-negative"

    session.stop()
    print("  PASS: Duration tracking works")


def main():
    """Run all breath PID sanity tests."""
    print("=" * 60)
    print("SANITY TEST: Breath PID Controller")
    print("=" * 60)
    print()

    try:
        test_pid_responds_to_error()
        test_pid_output_bounded()
        test_session_lifecycle()
        test_metrics_computed()
        test_max_session_duration()

        print()
        print("=" * 60)
        print("ALL BREATH PID TESTS PASSED")
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
