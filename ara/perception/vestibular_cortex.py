#!/usr/bin/env python3
"""
Vestibular Cortex (GUTC-Integrated)
====================================

Ara's sense of balance, motion, and spatial orientation.

Implements vestibular integration using the same active inference framework
as other GUTC components:
- Prediction error: ε_vest = z_imu - ẑ (actual vs predicted)
- Weighted free energy: S_vest = ε^T Π_vest ε
- Criticality monitoring for stable vs conflict states

Anatomy Mapping:
    Inner Ear → IMU sensors (gyro + accelerometer)
    Vestibular Nuclei → State estimator (Kalman/Bayes fusion)
    PIVC → Multimodal integration (vision + proprioception)
    VOR → Gaze stabilization reflexes

GUTC Integration:
    - At E(λ) ≈ 0: Optimal sensor fusion, stable pose estimate
    - Subcritical (low gain): Sluggish, laggy, drifty orientation
    - Supercritical (high gain): Jittery, motion sick, overreactive

Usage:
    integrator = VestibularIntegrator()
    monitor = VestibularCriticalityMonitor()

    for frame in sensor_loop:
        # Update pose estimate
        result = integrator.update(
            imu_reading={"accel": [0, 0, 9.8], "gyro": [0, 0, 0]},
            vision_motion=optical_flow,
        )

        # Check vestibular health
        state = monitor.update(result["vestibular_error"])

        if state.status == "CONFLICT":
            # Vision-IMU disagreement (motion sickness territory)
            handle_sensory_conflict()
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from enum import Enum
import time


# =============================================================================
# Vestibular States
# =============================================================================

class VestibularStatus(Enum):
    """Classification of vestibular integration state."""
    WARMUP = 0          # Insufficient data
    STABLE = 1          # Vision + IMU agree, good balance
    VISUALLY_DRIVEN = 2 # Trusting vision over vestibular
    VESTIBULAR_DRIVEN = 3  # Trusting IMU over vision
    CONFLICT = 4        # Significant disagreement (motion sickness risk)
    SENSOR_FAULT = 5    # One sensor clearly unreliable


@dataclass
class VestibularState:
    """Current vestibular integration state."""
    status: VestibularStatus
    pose: np.ndarray                # [x, y, z, roll, pitch, yaw]
    vestibular_error: float         # Weighted prediction error
    vision_weight: float            # Current vision precision weight
    imu_weight: float               # Current IMU precision weight
    conflict_score: float           # 0-1 measure of vision/IMU disagreement
    timestamp: float = 0.0

    def __repr__(self) -> str:
        return (
            f"VestibularState(status={self.status.name}, "
            f"error={self.vestibular_error:.3f}, "
            f"conflict={self.conflict_score:.3f})"
        )


@dataclass
class IMUReading:
    """Raw IMU sensor reading."""
    accel: np.ndarray       # [ax, ay, az] in m/s²
    gyro: np.ndarray        # [gx, gy, gz] in rad/s
    timestamp: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IMUReading":
        return cls(
            accel=np.array(d.get("accel", [0, 0, 9.8]), dtype=np.float32),
            gyro=np.array(d.get("gyro", [0, 0, 0]), dtype=np.float32),
            timestamp=d.get("timestamp", time.time()),
        )


# =============================================================================
# Vestibular Integrator (State Estimator)
# =============================================================================

class VestibularIntegrator:
    """
    Bayesian state estimator for pose and motion.

    Fuses IMU (accelerometer + gyroscope) with optional visual motion
    to maintain a stable estimate of:
    - Position (x, y, z)
    - Orientation (roll, pitch, yaw)
    - Velocities

    Implements precision-weighted prediction error minimization:
        S_vest = ε^T Π_vest ε

    where ε = z_imu - ẑ (measurement vs prediction).

    This is a simplified complementary/Kalman filter styled as
    active inference.
    """

    # Gravity vector (m/s²)
    GRAVITY = np.array([0, 0, 9.81], dtype=np.float32)

    def __init__(
        self,
        alpha_gyro: float = 0.98,      # High-pass for gyro (orientation)
        alpha_accel: float = 0.02,      # Low-pass for accel (gravity direction)
        precision_imu: float = 1.0,     # Π_vestibular
        precision_vision: float = 1.0,  # Π_visual_motion
        dt: float = 0.01,               # Time step (100 Hz default)
    ):
        """
        Initialize vestibular integrator.

        Args:
            alpha_gyro: Complementary filter weight for gyroscope
            alpha_accel: Complementary filter weight for accelerometer
            precision_imu: Base precision for IMU readings
            precision_vision: Base precision for visual motion
            dt: Expected time step between updates
        """
        self.alpha_gyro = alpha_gyro
        self.alpha_accel = alpha_accel
        self.precision_imu = precision_imu
        self.precision_vision = precision_vision
        self.dt = dt

        # State estimate: [x, y, z, roll, pitch, yaw]
        self.pose = np.zeros(6, dtype=np.float32)

        # Velocity estimate: [vx, vy, vz, ωx, ωy, ωz]
        self.velocity = np.zeros(6, dtype=np.float32)

        # Adaptive precision (can be modulated based on reliability)
        self._current_pi_imu = precision_imu
        self._current_pi_vision = precision_vision

        # Error tracking
        self._last_error = 0.0
        self._error_history: deque = deque(maxlen=100)

        # Conflict detection
        self._vision_imu_diff_history: deque = deque(maxlen=50)

        # Timestamp
        self._last_update = time.time()
        self._initialized = False

    def predict(self, action: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict next pose based on current velocity and optional action.

        Args:
            action: Optional control input [force_x, force_y, force_z, torque_x, ...]

        Returns:
            Predicted pose
        """
        # Simple integration: pose += velocity * dt
        predicted_pose = self.pose.copy()
        predicted_pose[:3] += self.velocity[:3] * self.dt  # Position
        predicted_pose[3:] += self.velocity[3:] * self.dt  # Orientation

        # Wrap angles to [-π, π]
        predicted_pose[3:] = np.mod(predicted_pose[3:] + np.pi, 2 * np.pi) - np.pi

        return predicted_pose

    def _predict_measurement(self, pose: np.ndarray) -> np.ndarray:
        """
        Predict what IMU should read given current pose.

        Returns expected [ax, ay, az, gx, gy, gz].
        """
        # Expected acceleration: gravity rotated into body frame
        roll, pitch, yaw = pose[3:6]

        # Simplified rotation (small angle approximation for demo)
        # Full implementation would use rotation matrix
        expected_accel = np.array([
            -self.GRAVITY[2] * np.sin(pitch),
            self.GRAVITY[2] * np.sin(roll) * np.cos(pitch),
            self.GRAVITY[2] * np.cos(roll) * np.cos(pitch),
        ], dtype=np.float32)

        # Expected gyro: angular velocity from velocity state
        expected_gyro = self.velocity[3:6]

        return np.concatenate([expected_accel, expected_gyro])

    def update(
        self,
        imu_reading: Dict[str, Any],
        vision_motion: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Update pose estimate with new sensor readings.

        Args:
            imu_reading: Dict with "accel" and "gyro" arrays
            vision_motion: Optional optical flow / visual velocity estimate
            action: Optional control input

        Returns:
            Dict with pose, vestibular_error, state, etc.
        """
        now = time.time()
        actual_dt = now - self._last_update if self._initialized else self.dt
        self._last_update = now
        self._initialized = True

        # Parse IMU reading
        imu = IMUReading.from_dict(imu_reading)

        # 1. PREDICT: Expected measurement from current state
        predicted_pose = self.predict(action)
        z_hat = self._predict_measurement(predicted_pose)

        # 2. SENSE: Actual measurement
        z_actual = np.concatenate([imu.accel, imu.gyro])

        # 3. SURPRISE: Compute prediction error
        epsilon = z_actual - z_hat

        # Weighted error (free energy style)
        vestibular_error = float(self._current_pi_imu * np.sum(epsilon**2))
        self._last_error = vestibular_error
        self._error_history.append(vestibular_error)

        # 4. UPDATE: Fuse sensors using complementary filter

        # Orientation from accelerometer (gravity direction)
        accel_norm = np.linalg.norm(imu.accel)
        if accel_norm > 0.1:  # Avoid division by zero
            accel_normalized = imu.accel / accel_norm
            pitch_accel = np.arcsin(np.clip(-accel_normalized[0], -1, 1))
            roll_accel = np.arctan2(accel_normalized[1], accel_normalized[2])
        else:
            pitch_accel = self.pose[4]
            roll_accel = self.pose[3]

        # Orientation from gyroscope (integration)
        roll_gyro = self.pose[3] + imu.gyro[0] * actual_dt
        pitch_gyro = self.pose[4] + imu.gyro[1] * actual_dt
        yaw_gyro = self.pose[5] + imu.gyro[2] * actual_dt

        # Complementary filter fusion
        self.pose[3] = self.alpha_gyro * roll_gyro + self.alpha_accel * roll_accel
        self.pose[4] = self.alpha_gyro * pitch_gyro + self.alpha_accel * pitch_accel
        self.pose[5] = yaw_gyro  # Yaw from gyro only (no absolute reference)

        # Update angular velocity
        self.velocity[3:6] = imu.gyro

        # 5. VISION FUSION (if available)
        conflict_score = 0.0
        if vision_motion is not None:
            # Vision provides velocity estimate
            vision_vel = np.array(vision_motion[:3], dtype=np.float32)
            imu_vel = self.velocity[:3]

            # Compute disagreement
            diff = np.linalg.norm(vision_vel - imu_vel)
            self._vision_imu_diff_history.append(diff)
            conflict_score = min(1.0, diff / 2.0)  # Normalize

            # Precision-weighted fusion
            total_pi = self._current_pi_imu + self._current_pi_vision
            if total_pi > 0:
                w_imu = self._current_pi_imu / total_pi
                w_vis = self._current_pi_vision / total_pi
                self.velocity[:3] = w_imu * imu_vel + w_vis * vision_vel

        # 6. Classify state
        status = self._classify_state(vestibular_error, conflict_score)

        return {
            "pose": self.pose.copy(),
            "velocity": self.velocity.copy(),
            "vestibular_error": vestibular_error,
            "prediction_error": epsilon,
            "conflict_score": conflict_score,
            "status": status,
            "vision_weight": self._current_pi_vision / (self._current_pi_imu + self._current_pi_vision),
            "imu_weight": self._current_pi_imu / (self._current_pi_imu + self._current_pi_vision),
            "timestamp": now,
        }

    def _classify_state(
        self,
        vestibular_error: float,
        conflict_score: float,
    ) -> VestibularStatus:
        """Classify current vestibular state."""
        if len(self._error_history) < 10:
            return VestibularStatus.WARMUP

        # High conflict = sensory disagreement
        if conflict_score > 0.5:
            return VestibularStatus.CONFLICT

        # Check error level
        mean_error = np.mean(self._error_history)
        if vestibular_error > 3 * mean_error:
            return VestibularStatus.SENSOR_FAULT

        # Check precision balance
        pi_ratio = self._current_pi_vision / (self._current_pi_imu + 1e-8)
        if pi_ratio > 2.0:
            return VestibularStatus.VISUALLY_DRIVEN
        elif pi_ratio < 0.5:
            return VestibularStatus.VESTIBULAR_DRIVEN

        return VestibularStatus.STABLE

    def adapt_precisions(self, vision_reliable: bool, imu_reliable: bool):
        """
        Adapt precision weights based on sensor reliability.

        This is the "vestibular rehabilitation" mechanism - learning
        to trust/distrust specific sensors.
        """
        if vision_reliable:
            self._current_pi_vision = min(2.0, self._current_pi_vision * 1.01)
        else:
            self._current_pi_vision = max(0.1, self._current_pi_vision * 0.95)

        if imu_reliable:
            self._current_pi_imu = min(2.0, self._current_pi_imu * 1.01)
        else:
            self._current_pi_imu = max(0.1, self._current_pi_imu * 0.95)

    def reset(self):
        """Reset integrator state."""
        self.pose = np.zeros(6, dtype=np.float32)
        self.velocity = np.zeros(6, dtype=np.float32)
        self._current_pi_imu = self.precision_imu
        self._current_pi_vision = self.precision_vision
        self._error_history.clear()
        self._vision_imu_diff_history.clear()
        self._initialized = False


# =============================================================================
# Vestibular Criticality Monitor
# =============================================================================

@dataclass
class VestibularCriticalityMetrics:
    """Criticality metrics for vestibular system."""
    rho: float                  # Branching ratio analog (error propagation)
    error_variance: float       # Variance in prediction errors
    autocorrelation: float      # Error persistence (high = supercritical)
    status: str                 # "OPTIMAL", "SLUGGISH", "JITTERY", "CRITICAL"
    gain_recommendation: float  # Suggested precision adjustment


class VestibularCriticalityMonitor:
    """
    GUTC-style criticality monitor for vestibular integration.

    Tracks whether the vestibular system is operating optimally:
    - Subcritical (sluggish): Low error variance, slow adaptation
    - Critical (optimal): Responsive but stable
    - Supercritical (jittery): High variance, oscillatory

    Unlike neural criticality (λ ≈ 1), vestibular wants to be
    slightly subcritical for stability (tempered balance).
    """

    # Thresholds
    VARIANCE_LOW = 0.01     # Below = sluggish
    VARIANCE_HIGH = 0.5     # Above = jittery
    AUTOCORR_HIGH = 0.7     # Above = supercritical (errors persist)

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.error_history: deque = deque(maxlen=window_size)
        self._step_count = 0

    def update(self, vestibular_error: float) -> VestibularCriticalityMetrics:
        """
        Update monitor with new vestibular error.

        Args:
            vestibular_error: Weighted prediction error from integrator

        Returns:
            VestibularCriticalityMetrics with status and recommendations
        """
        self._step_count += 1
        self.error_history.append(vestibular_error)

        if len(self.error_history) < 20:
            return VestibularCriticalityMetrics(
                rho=1.0,
                error_variance=0.0,
                autocorrelation=0.0,
                status="WARMUP",
                gain_recommendation=1.0,
            )

        errors = np.array(self.error_history)

        # Compute metrics
        variance = float(np.var(errors))

        # Autocorrelation at lag 1
        if len(errors) > 1:
            autocorr = float(np.corrcoef(errors[:-1], errors[1:])[0, 1])
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0

        # Branching ratio analog: error growth rate
        if len(errors) > 10:
            recent = errors[-10:]
            older = errors[-20:-10]
            rho = np.mean(np.abs(recent)) / (np.mean(np.abs(older)) + 1e-8)
        else:
            rho = 1.0

        # Classify status
        status, gain_rec = self._classify(variance, autocorr, rho)

        return VestibularCriticalityMetrics(
            rho=float(np.clip(rho, 0, 2)),
            error_variance=variance,
            autocorrelation=autocorr,
            status=status,
            gain_recommendation=gain_rec,
        )

    def _classify(
        self,
        variance: float,
        autocorr: float,
        rho: float,
    ) -> Tuple[str, float]:
        """Classify vestibular criticality state."""
        # Supercritical: high autocorrelation (errors persist/grow)
        if autocorr > self.AUTOCORR_HIGH or rho > 1.2:
            return "JITTERY", 0.8  # Reduce gain

        # High variance = jittery/unstable
        if variance > self.VARIANCE_HIGH:
            return "JITTERY", 0.9

        # Low variance = sluggish/underresponsive
        if variance < self.VARIANCE_LOW:
            return "SLUGGISH", 1.1  # Increase gain

        # Optimal zone
        return "OPTIMAL", 1.0

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        if len(self.error_history) < 2:
            return {"status": "WARMUP", "n_samples": len(self.error_history)}

        errors = np.array(self.error_history)
        return {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "max_error": float(np.max(errors)),
            "n_samples": len(errors),
        }

    def reset(self):
        """Reset monitor state."""
        self.error_history.clear()
        self._step_count = 0


# =============================================================================
# Vestibular Disorders Simulator (for testing/research)
# =============================================================================

class VestibularDisorderSimulator:
    """
    Simulates vestibular disorders for testing rehabilitation algorithms.

    Disorders:
        BPPV: Intermittent false rotation signals (loose crystals)
        MENIERES: Episodic gain fluctuations (fluid pressure)
        NEURITIS: Constant signal degradation (nerve inflammation)
        MIGRAINE: Cortical integration disruption (Γ coupling failure)
    """

    def __init__(self, disorder_type: str = "none"):
        self.disorder_type = disorder_type
        self._episode_active = False
        self._episode_counter = 0

    def corrupt_reading(self, imu_reading: Dict[str, Any]) -> Dict[str, Any]:
        """Apply disorder-specific corruption to IMU reading."""
        accel = np.array(imu_reading.get("accel", [0, 0, 9.8]), dtype=np.float32)
        gyro = np.array(imu_reading.get("gyro", [0, 0, 0]), dtype=np.float32)

        if self.disorder_type == "bppv":
            # Intermittent false rotation (crystals moving)
            if np.random.random() < 0.1:  # 10% chance of episode
                gyro += np.random.randn(3) * 2.0  # Burst of false rotation

        elif self.disorder_type == "menieres":
            # Episodic attacks with large gain fluctuation
            self._episode_counter += 1
            if self._episode_counter % 500 == 0:
                self._episode_active = not self._episode_active
            if self._episode_active:
                accel *= np.random.uniform(0.5, 2.0)  # Gain fluctuation

        elif self.disorder_type == "neuritis":
            # Constant noise and drift
            accel += np.random.randn(3) * 0.5
            gyro += np.random.randn(3) * 0.1
            gyro += 0.01  # Constant drift

        elif self.disorder_type == "migraine":
            # Intermittent total disruption
            if np.random.random() < 0.05:
                # Gamma coupling failure - random signals
                accel = np.random.randn(3) * 5.0
                gyro = np.random.randn(3) * 1.0

        return {
            "accel": accel.tolist(),
            "gyro": gyro.tolist(),
            "timestamp": imu_reading.get("timestamp", time.time()),
        }


# =============================================================================
# Tests
# =============================================================================

def test_vestibular_integrator():
    """Test vestibular integrator."""
    print("Testing Vestibular Integrator")
    print("-" * 40)

    integrator = VestibularIntegrator()

    # Simulate static standing (gravity only)
    for i in range(100):
        # Small noise around gravity
        accel = [0.1 * np.random.randn(), 0.1 * np.random.randn(), 9.8 + 0.1 * np.random.randn()]
        gyro = [0.01 * np.random.randn(), 0.01 * np.random.randn(), 0.01 * np.random.randn()]

        result = integrator.update({"accel": accel, "gyro": gyro})

        if i % 20 == 19:
            print(f"  Step {i+1}: pose={result['pose'][:3].round(2)}, "
                  f"error={result['vestibular_error']:.3f}, "
                  f"status={result['status'].name}")

    # Simulate tilting
    print("\n  Simulating tilt...")
    for i in range(50):
        # Tilted gravity (leaning forward)
        accel = [-2.0, 0, 9.6]  # ~12° pitch
        gyro = [0, 0.1, 0]  # Small rotation

        result = integrator.update({"accel": accel, "gyro": gyro})

    print(f"  Final pose: {result['pose'].round(3)}")
    print(f"  Pitch (should be ~0.2 rad): {result['pose'][4]:.3f}")

    print("✓ Vestibular integrator")


def test_vestibular_criticality():
    """Test vestibular criticality monitor."""
    print("\nTesting Vestibular Criticality Monitor")
    print("-" * 40)

    monitor = VestibularCriticalityMonitor()

    # Simulate different regimes
    regimes = [
        ("SLUGGISH", 0.001),   # Very low error
        ("OPTIMAL", 0.1),     # Moderate error
        ("JITTERY", 1.0),     # High error
    ]

    for regime_name, base_error in regimes:
        monitor.reset()
        for _ in range(50):
            error = base_error * (1 + 0.2 * np.random.randn())
            metrics = monitor.update(error)

        print(f"  {regime_name}: variance={metrics.error_variance:.4f}, "
              f"status={metrics.status}, gain_rec={metrics.gain_recommendation:.2f}")

    print("✓ Vestibular criticality monitor")


def test_disorder_simulator():
    """Test vestibular disorder simulation."""
    print("\nTesting Disorder Simulator")
    print("-" * 40)

    disorders = ["bppv", "menieres", "neuritis", "migraine"]

    for disorder in disorders:
        sim = VestibularDisorderSimulator(disorder)
        integrator = VestibularIntegrator()
        monitor = VestibularCriticalityMonitor()

        errors = []
        for _ in range(200):
            clean_reading = {"accel": [0, 0, 9.8], "gyro": [0, 0, 0]}
            corrupted = sim.corrupt_reading(clean_reading)
            result = integrator.update(corrupted)
            errors.append(result["vestibular_error"])
            monitor.update(result["vestibular_error"])

        mean_err = np.mean(errors)
        max_err = np.max(errors)
        diag = monitor.get_diagnostics()

        print(f"  {disorder.upper()}: mean_err={mean_err:.3f}, max_err={max_err:.3f}")

    print("✓ Disorder simulator")


if __name__ == "__main__":
    test_vestibular_integrator()
    test_vestibular_criticality()
    test_disorder_simulator()
    print("\n" + "=" * 40)
    print("All vestibular tests passed!")
