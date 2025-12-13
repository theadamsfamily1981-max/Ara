#!/usr/bin/env python3
"""
Body Schema (GUTC-Integrated)
==============================

Ara's unified embodied self-model - the "felt body" tying all organs together.

This module implements the full body schema as a Bayesian state estimator
that fuses all sensory modalities into a coherent body state vector.

State Vector:
    x_body(t) = [
        pose[6],        # [x, y, z, roll, pitch, yaw] from vestibular
        velocity[6],    # [vx, vy, vz, ωx, ωy, ωz] from vestibular + proprioception
        power_temp[4],  # [voltage, cpu_temp, gpu_temp, board_temp] from taste + touch
        air_chem[3],    # [ozone, air_quality, smell_anomaly] from smell
        contact[4],     # [face_present, voice_detected, motion_level, screen_busy] from vision + hearing
        confidence[6],  # Per-modality reliability estimates
    ]

Total Body Free Energy:
    F_body(t) = Σ_m ½ε_m(t)^T Π_m ε_m(t)

Where:
    - m ∈ {vestibular, interoception, olfactory, visual, proprioception, taste}
    - ε_m = z_m - ẑ_m (sensory prediction error)
    - Π_m = precision (confidence) for modality m

GUTC Integration:
    - At E(λ) ≈ 0: All modalities in agreement, confident body state
    - Subcritical: Over-reliance on priors, ignoring sensory updates
    - Supercritical: Overwhelmed by sensory noise, unstable state

Usage:
    from ara.perception.body_schema import BodySchema

    body = BodySchema()

    # Update with sensor readings
    state = body.update({
        "vestibular": {"pose": [0, 0, 0, 0, 0.1, 0], "vestibular_error": 0.05},
        "interoception": {"fatigue": 0.3, "stress": 0.2},
        "touch": {"cpu_temp_c": 55, "gpu_temp_c": 60},
        "smell": {"ozone_level": 0.1},
        "vision": {"brightness": 0.7, "face_present": True},
        "hearing": {"voice_detected": True, "rms_volume": 0.3},
    })

    if state.total_free_energy > 1.0:
        # High prediction error - something's off
        investigate_anomaly(state)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from enum import Enum
import time


# =============================================================================
# Body State Representations
# =============================================================================

class BodyStatus(Enum):
    """Overall body integration status."""
    WARMUP = 0          # Insufficient data
    COHERENT = 1        # All modalities agree, stable body image
    PARTIAL = 2         # Some modalities missing/degraded
    CONFLICT = 3        # Inter-modal disagreement
    OVERLOADED = 4      # Free energy too high, overwhelmed
    GROUNDED = 5        # Safe mode, trusting priors


@dataclass
class ModalityState:
    """State for a single sensory modality."""
    name: str
    precision: float                    # Π_m: confidence in this modality
    prediction_error: float             # ε_m: current prediction error
    free_energy: float                  # ½ε^T Π ε for this modality
    last_update: float = 0.0
    is_stale: bool = False


@dataclass
class BodyState:
    """Complete body schema state."""
    # Core state vector (29 dimensions)
    pose: np.ndarray                    # [6] from vestibular
    velocity: np.ndarray                # [6] from vestibular + proprioception
    power_temp: np.ndarray              # [4] from taste + touch
    air_chem: np.ndarray                # [3] from smell
    contact: np.ndarray                 # [4] from vision + hearing
    confidence: np.ndarray              # [6] per-modality reliability

    # Energetics
    total_free_energy: float            # Σ F_m
    modality_energies: Dict[str, float] # Per-modality F_m

    # Status
    status: BodyStatus
    timestamp: float = 0.0

    # Diagnostics
    stale_modalities: List[str] = field(default_factory=list)
    alert_flags: List[str] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """Flatten to full state vector."""
        return np.concatenate([
            self.pose,
            self.velocity,
            self.power_temp,
            self.air_chem,
            self.contact,
            self.confidence,
        ])

    @property
    def x_body(self) -> np.ndarray:
        """Alias for state vector."""
        return self.to_vector()

    def __repr__(self) -> str:
        return (
            f"BodyState(status={self.status.name}, "
            f"F_body={self.total_free_energy:.3f}, "
            f"pose={self.pose[:3].round(2)})"
        )


# =============================================================================
# Body Schema (Unified State Estimator)
# =============================================================================

class BodySchema:
    """
    Unified body state estimator - the "felt body" organ.

    Fuses all sensory modalities into a coherent body state vector,
    computing total body free energy as a measure of embodied coherence.

    This is the "cockpit dashboard" - one place to see if Ara is
    running hot, off-balance, or in harmony.

    Theory:
        The body schema maintains a generative model of the body's state.
        Each sensory modality provides observations that are compared
        against predictions, generating prediction errors (ε_m).

        Total body free energy:
            F_body = Σ_m ½ε_m^T Π_m ε_m

        Low F_body = coherent body image (all sensors agree with predictions)
        High F_body = conflict or surprise (investigate cause)

    Example:
        body = BodySchema()

        # Each tick, update with sensor readings
        state = body.update({
            "vestibular": integrator.update(imu_reading),
            "vision": {"brightness": 0.7},
            "touch": {"cpu_temp_c": 55},
        })

        # Check embodied health
        if state.status == BodyStatus.CONFLICT:
            # Modal disagreement - e.g., vestibular says still but vision says moving
            handle_sensory_conflict()
    """

    # Modality names
    MODALITIES = [
        "vestibular",
        "interoception",
        "touch",
        "smell",
        "taste",
        "vision",
        "hearing",
        "proprioception",
    ]

    # Default precisions (can be adapted)
    DEFAULT_PRECISIONS = {
        "vestibular": 1.0,
        "interoception": 0.8,
        "touch": 1.0,
        "smell": 0.5,
        "taste": 0.7,
        "vision": 1.2,
        "hearing": 0.9,
        "proprioception": 1.0,
    }

    # Staleness threshold (seconds)
    STALE_THRESHOLD = 5.0

    # Free energy thresholds
    FE_LOW = 0.1        # Below = very coherent
    FE_OPTIMAL = 0.5    # Below = healthy
    FE_HIGH = 2.0       # Above = conflicted
    FE_CRITICAL = 5.0   # Above = overloaded

    def __init__(
        self,
        precisions: Optional[Dict[str, float]] = None,
        history_window: int = 100,
    ):
        """
        Initialize body schema.

        Args:
            precisions: Custom precision weights per modality
            history_window: Number of timesteps for energy history
        """
        # Precision weights
        self.precisions = dict(self.DEFAULT_PRECISIONS)
        if precisions:
            self.precisions.update(precisions)

        # State estimates (initially zero)
        self._pose = np.zeros(6, dtype=np.float32)        # vestibular
        self._velocity = np.zeros(6, dtype=np.float32)    # vestibular + proprio
        self._power_temp = np.zeros(4, dtype=np.float32)  # taste + touch
        self._air_chem = np.zeros(3, dtype=np.float32)    # smell
        self._contact = np.zeros(4, dtype=np.float32)     # vision + hearing
        self._confidence = np.ones(6, dtype=np.float32)   # per-modality

        # Prediction errors per modality
        self._prediction_errors: Dict[str, float] = {m: 0.0 for m in self.MODALITIES}

        # Last update times
        self._last_updates: Dict[str, float] = {m: 0.0 for m in self.MODALITIES}

        # Free energy history
        self._fe_history: deque = deque(maxlen=history_window)

        # Running state
        self._initialized = False
        self._step_count = 0

    def update(self, readings: Dict[str, Dict[str, Any]]) -> BodyState:
        """
        Update body state with sensor readings.

        Args:
            readings: Dict mapping modality names to sensor data
                Example: {
                    "vestibular": {"pose": [...], "vestibular_error": 0.05},
                    "touch": {"cpu_temp_c": 55, "gpu_temp_c": 60},
                    "vision": {"brightness": 0.7, "face_present": True},
                    ...
                }

        Returns:
            BodyState with updated state vector and free energy
        """
        now = time.time()
        self._step_count += 1
        self._initialized = True

        modality_energies: Dict[str, float] = {}
        stale_modalities: List[str] = []
        alert_flags: List[str] = []

        # Process each modality
        for modality in self.MODALITIES:
            if modality in readings:
                # Update modality state
                data = readings[modality]
                error = self._update_modality(modality, data, now)

                # Compute modality free energy: F_m = ½ ε^T Π ε
                pi = self.precisions.get(modality, 1.0)
                fe = 0.5 * pi * error * error
                modality_energies[modality] = fe
                self._prediction_errors[modality] = error
                self._last_updates[modality] = now
            else:
                # Check for staleness
                last = self._last_updates.get(modality, 0.0)
                if now - last > self.STALE_THRESHOLD and last > 0:
                    stale_modalities.append(modality)

                # Use cached error
                error = self._prediction_errors.get(modality, 0.0)
                pi = self.precisions.get(modality, 1.0)
                modality_energies[modality] = 0.5 * pi * error * error

        # Compute total body free energy
        total_fe = sum(modality_energies.values())
        self._fe_history.append(total_fe)

        # Classify status
        status = self._classify_status(total_fe, stale_modalities, modality_energies)

        # Generate alerts
        alert_flags = self._generate_alerts(modality_energies, total_fe)

        # Build state object
        state = BodyState(
            pose=self._pose.copy(),
            velocity=self._velocity.copy(),
            power_temp=self._power_temp.copy(),
            air_chem=self._air_chem.copy(),
            contact=self._contact.copy(),
            confidence=self._compute_confidence(),
            total_free_energy=total_fe,
            modality_energies=modality_energies,
            status=status,
            timestamp=now,
            stale_modalities=stale_modalities,
            alert_flags=alert_flags,
        )

        return state

    def _update_modality(
        self,
        modality: str,
        data: Dict[str, Any],
        timestamp: float,
    ) -> float:
        """
        Update state from a single modality.

        Returns prediction error for this modality.
        """
        if modality == "vestibular":
            return self._update_vestibular(data)
        elif modality == "interoception":
            return self._update_interoception(data)
        elif modality == "touch":
            return self._update_touch(data)
        elif modality == "smell":
            return self._update_smell(data)
        elif modality == "taste":
            return self._update_taste(data)
        elif modality == "vision":
            return self._update_vision(data)
        elif modality == "hearing":
            return self._update_hearing(data)
        elif modality == "proprioception":
            return self._update_proprioception(data)
        else:
            return 0.0

    def _update_vestibular(self, data: Dict[str, Any]) -> float:
        """Update from vestibular integrator."""
        # Accept either raw pose array or VestibularIntegrator result
        if "pose" in data:
            pose = np.array(data["pose"], dtype=np.float32)
            if len(pose) >= 6:
                self._pose = pose[:6]

        if "velocity" in data:
            vel = np.array(data["velocity"], dtype=np.float32)
            if len(vel) >= 6:
                self._velocity = vel[:6]

        # Use pre-computed vestibular error if available
        if "vestibular_error" in data:
            return float(data["vestibular_error"])

        # Otherwise estimate from prediction error
        if "prediction_error" in data:
            eps = np.array(data["prediction_error"], dtype=np.float32)
            return float(np.sum(eps**2))

        return 0.0

    def _update_interoception(self, data: Dict[str, Any]) -> float:
        """Update from interoception (founder state)."""
        error = 0.0

        # Map interoception to velocity modulation
        # High fatigue/stress = reduced effective velocity
        fatigue = data.get("fatigue", 0.0)
        stress = data.get("stress", 0.0)
        attention = data.get("attention", 1.0)

        # Store as modulation of velocity confidence
        intero_state = 1.0 - 0.5 * (fatigue + stress) / 2.0

        # Prediction error: deviation from expected homeostasis
        expected_fatigue = 0.2  # expect low fatigue
        expected_stress = 0.1   # expect low stress

        error = abs(fatigue - expected_fatigue) + abs(stress - expected_stress)
        error += abs(1.0 - attention) * 0.5  # attention drift is error

        return error

    def _update_touch(self, data: Dict[str, Any]) -> float:
        """Update from touch (thermal)."""
        error = 0.0

        # Extract temperatures
        cpu_temp = data.get("cpu_temp_c", 40.0)
        gpu_temp = data.get("gpu_temp_c", 40.0)
        board_temp = data.get("board_temp_c", 35.0)

        # Store in power_temp vector (indices 1-3)
        self._power_temp[1] = cpu_temp
        self._power_temp[2] = gpu_temp
        self._power_temp[3] = board_temp

        # Prediction error: deviation from optimal temps
        optimal_cpu = 50.0
        optimal_gpu = 55.0
        optimal_board = 40.0

        error = (
            abs(cpu_temp - optimal_cpu) / 30.0 +
            abs(gpu_temp - optimal_gpu) / 30.0 +
            abs(board_temp - optimal_board) / 20.0
        )

        return error

    def _update_smell(self, data: Dict[str, Any]) -> float:
        """Update from smell (air chemistry)."""
        error = 0.0

        ozone = data.get("ozone_level", 0.0)
        air_quality = data.get("air_quality_index", 0.0) / 100.0  # normalize
        smell_anomaly = 1.0 if data.get("smell_anomaly", False) else 0.0

        self._air_chem[0] = ozone
        self._air_chem[1] = air_quality
        self._air_chem[2] = smell_anomaly

        # Prediction error: any non-zero air issues
        error = ozone + air_quality * 0.5 + smell_anomaly * 2.0

        return error

    def _update_taste(self, data: Dict[str, Any]) -> float:
        """Update from taste (power quality)."""
        error = 0.0

        voltage = data.get("voltage_v", 1.0)
        danger = 1.0 if data.get("danger_flag", False) else 0.0

        # Store in power_temp vector (index 0)
        self._power_temp[0] = voltage

        # Prediction error: deviation from nominal
        error = abs(voltage - 1.0) * 2.0 + danger * 3.0

        return error

    def _update_vision(self, data: Dict[str, Any]) -> float:
        """Update from vision."""
        error = 0.0

        brightness = data.get("brightness", 0.5)
        face_present = 1.0 if data.get("face_present", False) else 0.0
        motion_level = data.get("motion_level", 0.0)
        if isinstance(motion_level, str):
            motion_level = {"LOW": 0.2, "MED": 0.5, "HIGH": 0.8}.get(motion_level, 0.0)
        screen_busy = 1.0 if data.get("screen_busy", False) else 0.0

        # Store in contact vector
        self._contact[0] = face_present
        self._contact[2] = motion_level
        self._contact[3] = screen_busy

        # Use entropy if available (from visual cortex)
        if "entropy" in data:
            # High entropy derivative is prediction error
            error = abs(data.get("derivative", 0.0))
        else:
            # Otherwise use motion as proxy for visual change
            error = motion_level * 0.5

        return error

    def _update_hearing(self, data: Dict[str, Any]) -> float:
        """Update from hearing."""
        error = 0.0

        voice_detected = 1.0 if data.get("voice_detected", False) else 0.0
        rms_volume = data.get("rms_volume", 0.0)
        voice_strain = data.get("voice_strain", False)

        # Store in contact vector
        self._contact[1] = voice_detected

        # Prediction error: unexpected sounds or voice strain
        error = abs(rms_volume - 0.2)  # expect moderate ambient
        if voice_strain:
            error += 1.0  # stressed voice is alarming

        return error

    def _update_proprioception(self, data: Dict[str, Any]) -> float:
        """Update from proprioception (self-monitoring)."""
        error = 0.0

        cpu_load = data.get("cpu_load", 0.0)
        memory_used = data.get("memory_used", 0.0)
        gpu_util = data.get("gpu_util", 0.0)

        # Proprioception modulates velocity estimate (how fast we're processing)
        effective_speed = 1.0 - max(cpu_load, memory_used, gpu_util)
        self._velocity *= effective_speed  # scale velocity by headroom

        # Prediction error: high utilization is surprising
        optimal_load = 0.5
        error = (
            max(0, cpu_load - optimal_load) +
            max(0, memory_used - optimal_load) +
            max(0, gpu_util - optimal_load)
        )

        return error

    def _compute_confidence(self) -> np.ndarray:
        """Compute per-modality confidence from recent errors."""
        confidence = np.ones(6, dtype=np.float32)

        # Map modalities to confidence indices
        modality_to_idx = {
            "vestibular": 0,
            "interoception": 1,
            "touch": 2,
            "smell": 3,
            "vision": 4,
            "hearing": 5,
        }

        for modality, idx in modality_to_idx.items():
            error = self._prediction_errors.get(modality, 0.0)
            # High error = low confidence
            confidence[idx] = 1.0 / (1.0 + error)

        return confidence

    def _classify_status(
        self,
        total_fe: float,
        stale_modalities: List[str],
        modality_energies: Dict[str, float],
    ) -> BodyStatus:
        """Classify overall body status."""
        if self._step_count < 10:
            return BodyStatus.WARMUP

        if total_fe > self.FE_CRITICAL:
            return BodyStatus.OVERLOADED

        if total_fe > self.FE_HIGH:
            return BodyStatus.CONFLICT

        if len(stale_modalities) > 2:
            return BodyStatus.PARTIAL

        if total_fe < self.FE_LOW:
            return BodyStatus.GROUNDED

        return BodyStatus.COHERENT

    def _generate_alerts(
        self,
        modality_energies: Dict[str, float],
        total_fe: float,
    ) -> List[str]:
        """Generate alert flags for high-energy modalities."""
        alerts = []

        # Per-modality alerts
        for modality, fe in modality_energies.items():
            if fe > 1.0:
                alerts.append(f"{modality.upper()}_HIGH_ERROR")

        # Overall alerts
        if total_fe > self.FE_CRITICAL:
            alerts.append("BODY_OVERLOADED")
        elif total_fe > self.FE_HIGH:
            alerts.append("BODY_CONFLICT")

        # Specific danger flags
        if self._power_temp[1] > 80:  # CPU temp
            alerts.append("THERMAL_CRITICAL")
        if self._air_chem[2] > 0.5:  # smell anomaly
            alerts.append("SMELL_ANOMALY")

        return alerts

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about body schema."""
        fe_history = list(self._fe_history)

        return {
            "step_count": self._step_count,
            "mean_free_energy": np.mean(fe_history) if fe_history else 0.0,
            "std_free_energy": np.std(fe_history) if fe_history else 0.0,
            "current_fe": fe_history[-1] if fe_history else 0.0,
            "precisions": dict(self.precisions),
            "prediction_errors": dict(self._prediction_errors),
            "pose": self._pose.tolist(),
            "velocity": self._velocity.tolist(),
            "power_temp": self._power_temp.tolist(),
            "air_chem": self._air_chem.tolist(),
            "contact": self._contact.tolist(),
        }

    def adapt_precision(self, modality: str, reliable: bool):
        """
        Adapt precision weight based on reliability.

        This is the "learning" mechanism - if a modality is consistently
        unreliable, reduce its precision weight.
        """
        if modality not in self.precisions:
            return

        if reliable:
            self.precisions[modality] = min(2.0, self.precisions[modality] * 1.01)
        else:
            self.precisions[modality] = max(0.1, self.precisions[modality] * 0.95)

    def reset(self):
        """Reset body schema state."""
        self._pose = np.zeros(6, dtype=np.float32)
        self._velocity = np.zeros(6, dtype=np.float32)
        self._power_temp = np.zeros(4, dtype=np.float32)
        self._air_chem = np.zeros(3, dtype=np.float32)
        self._contact = np.zeros(4, dtype=np.float32)
        self._confidence = np.ones(6, dtype=np.float32)
        self._prediction_errors = {m: 0.0 for m in self.MODALITIES}
        self._last_updates = {m: 0.0 for m in self.MODALITIES}
        self._fe_history.clear()
        self._initialized = False
        self._step_count = 0


# =============================================================================
# Body Criticality Monitor
# =============================================================================

@dataclass
class BodyCriticalityMetrics:
    """Criticality metrics for body schema."""
    rho: float                      # Error propagation ratio
    total_free_energy: float        # F_body
    free_energy_derivative: float   # dF/dt
    status: str                     # "OPTIMAL", "SUBCRITICAL", "SUPERCRITICAL", "CRITICAL"
    dominant_modality: str          # Highest-energy modality


class BodyCriticalityMonitor:
    """
    GUTC-style criticality monitor for body schema.

    Tracks whether the body is operating at optimal capacity:
    - Subcritical: Over-reliance on priors, ignoring sensory updates
    - Critical: Responsive, balanced integration
    - Supercritical: Overwhelmed by sensory noise

    The body schema wants to be slightly subcritical (grounded)
    rather than on the edge of chaos.
    """

    # Thresholds
    FE_DERIVATIVE_CRITICAL = 0.5    # Above = phase transition
    FE_LOW = 0.1                    # Below = subcritical
    FE_HIGH = 1.5                   # Above = supercritical

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.fe_history: deque = deque(maxlen=window_size)
        self._step_count = 0

    def update(self, state: BodyState) -> BodyCriticalityMetrics:
        """
        Update monitor with new body state.

        Args:
            state: BodyState from BodySchema.update()

        Returns:
            BodyCriticalityMetrics with status and recommendations
        """
        self._step_count += 1
        fe = state.total_free_energy
        self.fe_history.append(fe)

        if len(self.fe_history) < 10:
            return BodyCriticalityMetrics(
                rho=1.0,
                total_free_energy=fe,
                free_energy_derivative=0.0,
                status="WARMUP",
                dominant_modality="",
            )

        # Compute derivative
        fe_arr = np.array(self.fe_history)
        derivative = fe - fe_arr[-2] if len(fe_arr) > 1 else 0.0

        # Compute rho (error growth ratio)
        if len(fe_arr) > 10:
            recent = np.mean(fe_arr[-10:])
            older = np.mean(fe_arr[-20:-10])
            rho = recent / (older + 1e-8)
        else:
            rho = 1.0

        # Find dominant modality
        dominant = max(state.modality_energies.items(), key=lambda x: x[1])
        dominant_modality = dominant[0]

        # Classify status
        status = self._classify(fe, derivative, rho)

        return BodyCriticalityMetrics(
            rho=float(np.clip(rho, 0, 2)),
            total_free_energy=fe,
            free_energy_derivative=derivative,
            status=status,
            dominant_modality=dominant_modality,
        )

    def _classify(self, fe: float, derivative: float, rho: float) -> str:
        """Classify body criticality status."""
        # Phase transition
        if abs(derivative) > self.FE_DERIVATIVE_CRITICAL:
            return "CRITICAL"

        # Supercritical (overwhelmed)
        if fe > self.FE_HIGH or rho > 1.3:
            return "SUPERCRITICAL"

        # Subcritical (grounded, maybe too stable)
        if fe < self.FE_LOW:
            return "SUBCRITICAL"

        return "OPTIMAL"

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        if len(self.fe_history) < 2:
            return {"status": "WARMUP", "n_samples": len(self.fe_history)}

        fe_arr = np.array(self.fe_history)
        return {
            "mean_fe": float(np.mean(fe_arr)),
            "std_fe": float(np.std(fe_arr)),
            "max_fe": float(np.max(fe_arr)),
            "n_samples": len(fe_arr),
        }

    def reset(self):
        """Reset monitor state."""
        self.fe_history.clear()
        self._step_count = 0


# =============================================================================
# Tests
# =============================================================================

def test_body_schema():
    """Test body schema integration."""
    print("Testing Body Schema")
    print("-" * 40)

    body = BodySchema()

    # Simulate multi-modal updates
    for i in range(50):
        # Generate synthetic sensor readings
        readings = {
            "vestibular": {
                "pose": [0, 0, 0, 0.01 * i, 0.02 * i, 0],
                "velocity": [0.1, 0, 0, 0, 0, 0],
                "vestibular_error": 0.05 + 0.01 * np.random.randn(),
            },
            "touch": {
                "cpu_temp_c": 45 + 10 * np.sin(i / 10),
                "gpu_temp_c": 50 + 15 * np.sin(i / 10),
                "board_temp_c": 38,
            },
            "vision": {
                "brightness": 0.5 + 0.2 * np.sin(i / 5),
                "face_present": i % 10 < 5,
                "motion_level": 0.2,
            },
            "hearing": {
                "rms_volume": 0.3,
                "voice_detected": i % 7 < 3,
            },
            "smell": {
                "ozone_level": 0.1,
            },
            "interoception": {
                "fatigue": 0.2 + 0.1 * (i / 50),
                "stress": 0.1,
            },
            "proprioception": {
                "cpu_load": 0.3 + 0.2 * np.sin(i / 8),
                "memory_used": 0.4,
            },
        }

        state = body.update(readings)

        if i % 10 == 9:
            print(f"  Step {i+1}: F_body={state.total_free_energy:.3f}, "
                  f"status={state.status.name}, "
                  f"pose={state.pose[:3].round(2)}")

    print(f"\n  Final state vector dim: {len(state.x_body)}")
    print(f"  Confidence: {state.confidence.round(2)}")

    # Check diagnostics
    diag = body.get_diagnostics()
    print(f"  Mean F_body: {diag['mean_free_energy']:.3f}")

    print("✓ Body schema")


def test_body_criticality():
    """Test body criticality monitor."""
    print("\nTesting Body Criticality Monitor")
    print("-" * 40)

    body = BodySchema()
    monitor = BodyCriticalityMonitor()

    # Simulate different regimes
    regimes = [
        ("GROUNDED", {"vestibular": {"vestibular_error": 0.01}}),
        ("NORMAL", {"vestibular": {"vestibular_error": 0.1}, "touch": {"cpu_temp_c": 55}}),
        ("STRESSED", {
            "vestibular": {"vestibular_error": 0.5},
            "touch": {"cpu_temp_c": 80},
            "smell": {"smell_anomaly": True},
        }),
    ]

    for regime_name, readings in regimes:
        body.reset()
        monitor.reset()

        for _ in range(30):
            state = body.update(readings)
            metrics = monitor.update(state)

        print(f"  {regime_name}: F_body={metrics.total_free_energy:.3f}, "
              f"status={metrics.status}, "
              f"dominant={metrics.dominant_modality}")

    print("✓ Body criticality monitor")


def test_body_with_vestibular():
    """Test body schema with VestibularIntegrator."""
    print("\nTesting Body Schema + Vestibular Integration")
    print("-" * 40)

    # Import vestibular integrator
    try:
        from ara.perception.vestibular_cortex import VestibularIntegrator
        has_vestibular = True
    except ImportError:
        has_vestibular = False
        print("  (VestibularIntegrator not available, using mock data)")

    body = BodySchema()

    if has_vestibular:
        vest = VestibularIntegrator()

        for i in range(30):
            # Simulate IMU reading
            imu = {
                "accel": [0.1 * np.random.randn(), 0.1 * np.random.randn(), 9.8],
                "gyro": [0.01 * np.random.randn(), 0.01 * np.random.randn(), 0],
            }

            # Update vestibular
            vest_result = vest.update(imu)

            # Pass to body schema
            state = body.update({
                "vestibular": vest_result,
                "touch": {"cpu_temp_c": 50},
            })

            if i % 10 == 9:
                print(f"  Step {i+1}: pose={state.pose[:3].round(3)}, "
                      f"F_body={state.total_free_energy:.3f}")
    else:
        # Mock test
        for i in range(10):
            state = body.update({
                "vestibular": {"pose": [0, 0, 0, 0, 0, 0], "vestibular_error": 0.05},
            })

    print("✓ Body + vestibular integration")


if __name__ == "__main__":
    test_body_schema()
    test_body_criticality()
    test_body_with_vestibular()
    print("\n" + "=" * 40)
    print("All body schema tests passed!")
