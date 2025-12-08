"""
L1 Hardware Reflex Adapter
==========================

Adapter sitting next to the FPGA daemon.
Converts telemetry into L1 HV and reads L9 bias from Axis.

This is the "nervous system" layer - fastest reflexes, hardware pain/pleasure,
and the ability to override higher layers in emergencies.

Features:
- Encodes hardware telemetry (temp, voltage, errors, load) as HV
- Reads "creativity/risk" bias from L9 via Axis
- Computes threshold scaling for CorrSpike-HDC parameters
- Can trigger emergency overrides (reflex arcs)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

from ara.system.axis import AxisMundi

logger = logging.getLogger(__name__)


@dataclass
class TelemetryPacket:
    """Hardware telemetry packet."""
    temp_c: float = 45.0       # Temperature in Celsius
    volt_v: float = 1.0        # Core voltage
    error_rate: float = 0.0    # Error rate (CRC, ECC, etc.)
    load: float = 0.5          # Compute load [0, 1]
    fpga_temp: float = 50.0    # FPGA die temperature
    memory_pressure: float = 0.0  # Memory pressure [0, 1]


@dataclass
class L1State:
    """Current state of the L1 adapter."""
    thresh_scale: float = 1.0   # Threshold scaling for CorrSpike
    pain_level: float = 0.0     # Aggregate "pain" [0, 1]
    bias_val: float = 0.0       # Current bias from L9
    is_emergency: bool = False  # Emergency override active


class L1HardwareReflex:
    """
    L1 Hardware Reflex Adapter.

    Sits next to the FPGA daemon and:
    1. Encodes telemetry -> L1 HV -> writes to Axis
    2. Reads L9 bias from Axis -> threshold scaling
    3. Triggers reflex arcs on emergency conditions

    Usage:
        axis = AxisMundi()
        l1 = L1HardwareReflex(axis)

        # In control loop:
        telemetry = TelemetryPacket(temp_c=65, load=0.9)
        state = l1.step(telemetry)

        # Apply to FPGA parameters:
        new_thresh = base_thresh * state.thresh_scale
    """

    # Emergency thresholds
    TEMP_CRITICAL = 85.0      # Celsius
    TEMP_WARNING = 75.0
    ERROR_CRITICAL = 0.05     # 5% error rate
    LOAD_CRITICAL = 0.95

    def __init__(
        self,
        axis: AxisMundi,
        bias_band_start: int = 0,
        bias_band_end: int = 128,
    ):
        """
        Initialize L1 adapter.

        Args:
            axis: The AxisMundi global bus
            bias_band_start: Start of bias band in HV (L9->L1 bias)
            bias_band_end: End of bias band in HV
        """
        self.axis = axis
        self.hv_dim = axis.dim
        self.bias_band = (bias_band_start, bias_band_end)

        # Feature encoding keys (random, fixed)
        rng = np.random.default_rng(42)
        self.k_temp = self._rand_unit_hv(rng)
        self.k_volt = self._rand_unit_hv(rng)
        self.k_errors = self._rand_unit_hv(rng)
        self.k_load = self._rand_unit_hv(rng)
        self.k_fpga_temp = self._rand_unit_hv(rng)
        self.k_memory = self._rand_unit_hv(rng)

        # Current state
        self.state = L1State()
        self.last_telemetry: Optional[TelemetryPacket] = None

        # Smoothing
        self._bias_ema = 0.0
        self._bias_alpha = 0.3

        logger.info("L1HardwareReflex initialized")

    def _rand_unit_hv(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a random unit-normalized HV."""
        hv = rng.standard_normal(self.hv_dim).astype(np.float32)
        hv /= np.linalg.norm(hv) + 1e-8
        return hv

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """Normalize to unit length."""
        n = np.linalg.norm(v)
        if n < 1e-8:
            return v
        return v / n

    def encode_telemetry(self, telemetry: TelemetryPacket) -> np.ndarray:
        """
        Encode telemetry into a hypervector.

        Each metric is normalized to [-1, 1] and bound with its feature key.
        """
        # Normalize metrics to [-1, 1] range
        t = np.tanh((telemetry.temp_c - 50.0) / 30.0)  # 50C is "normal"
        v = np.tanh((telemetry.volt_v - 1.0) / 0.15)   # 1.0V is nominal
        e = np.tanh(telemetry.error_rate * 20.0)       # Amplify errors
        l = np.tanh(telemetry.load * 2.0 - 1.0)        # 0.5 is "normal"
        ft = np.tanh((telemetry.fpga_temp - 60.0) / 25.0)
        m = np.tanh(telemetry.memory_pressure * 2.0 - 1.0)

        # Bind each metric with its key and sum
        hv = (
            self.k_temp * t +
            self.k_volt * v +
            self.k_errors * e +
            self.k_load * l +
            self.k_fpga_temp * ft +
            self.k_memory * m
        )

        return self._normalize(hv)

    def compute_pain_level(self, telemetry: TelemetryPacket) -> float:
        """
        Compute aggregate "pain" level from telemetry.

        Pain = weighted combination of stress indicators.
        """
        # Temperature pain (increases sharply above warning)
        temp_pain = 0.0
        if telemetry.temp_c > self.TEMP_WARNING:
            temp_pain = (telemetry.temp_c - self.TEMP_WARNING) / (self.TEMP_CRITICAL - self.TEMP_WARNING)
            temp_pain = min(1.0, temp_pain)

        # FPGA temperature pain
        fpga_pain = 0.0
        if telemetry.fpga_temp > 70.0:
            fpga_pain = (telemetry.fpga_temp - 70.0) / 20.0
            fpga_pain = min(1.0, fpga_pain)

        # Error pain
        error_pain = min(1.0, telemetry.error_rate / self.ERROR_CRITICAL)

        # Load pain (above 90% is stressed)
        load_pain = max(0.0, (telemetry.load - 0.9) / 0.1)

        # Memory pressure
        mem_pain = telemetry.memory_pressure

        # Weighted combination
        pain = (
            0.3 * temp_pain +
            0.2 * fpga_pain +
            0.25 * error_pain +
            0.15 * load_pain +
            0.1 * mem_pain
        )

        return float(np.clip(pain, 0.0, 1.0))

    def check_emergency(self, telemetry: TelemetryPacket) -> bool:
        """Check if emergency conditions are met."""
        return (
            telemetry.temp_c >= self.TEMP_CRITICAL or
            telemetry.fpga_temp >= 90.0 or
            telemetry.error_rate >= self.ERROR_CRITICAL or
            telemetry.load >= self.LOAD_CRITICAL
        )

    def step(self, telemetry: TelemetryPacket) -> L1State:
        """
        Process one control frame of telemetry.

        1. Encode telemetry as HV
        2. Write to Axis as L1 state
        3. Read L9 bias from Axis
        4. Compute threshold scaling
        5. Check for emergency conditions

        Args:
            telemetry: Hardware telemetry packet

        Returns:
            L1State with threshold scaling and status
        """
        self.last_telemetry = telemetry

        # 1. Encode telemetry as HV
        hv_state = self.encode_telemetry(telemetry)

        # 2. Write to Axis
        pain = self.compute_pain_level(telemetry)
        strength = 0.5 + 0.5 * pain  # More pain = stronger signal
        self.axis.write_layer_state(layer_id=1, raw_state_hv=hv_state, strength=strength)

        # 3. Read world view with L1 lens (includes L9 bias)
        world_view = self.axis.read_layer_state(1)

        # Extract bias from designated band
        bias_band = world_view[self.bias_band[0]:self.bias_band[1]]
        raw_bias = float(np.mean(bias_band))

        # EMA smoothing
        self._bias_ema = self._bias_alpha * raw_bias + (1 - self._bias_alpha) * self._bias_ema
        bias_val = self._bias_ema

        # 4. Map bias to threshold scaling
        # Positive bias → more creative (lower thresholds, more exploratory)
        # Negative bias → more conservative (higher thresholds, safer)
        thresh_scale = 1.0 - 0.3 * np.clip(bias_val, -1.0, 1.0)  # [0.7, 1.3]

        # 5. Check emergency
        is_emergency = self.check_emergency(telemetry)
        if is_emergency:
            # Override: go conservative
            thresh_scale = 1.3
            logger.warning(f"L1 EMERGENCY: temp={telemetry.temp_c}, errors={telemetry.error_rate}")

        # Update state
        self.state = L1State(
            thresh_scale=float(thresh_scale),
            pain_level=pain,
            bias_val=bias_val,
            is_emergency=is_emergency,
        )

        return self.state

    def get_status(self) -> Dict[str, Any]:
        """Get current L1 status."""
        return {
            "thresh_scale": self.state.thresh_scale,
            "pain_level": self.state.pain_level,
            "bias_val": self.state.bias_val,
            "is_emergency": self.state.is_emergency,
            "last_telemetry": {
                "temp_c": self.last_telemetry.temp_c if self.last_telemetry else None,
                "load": self.last_telemetry.load if self.last_telemetry else None,
                "error_rate": self.last_telemetry.error_rate if self.last_telemetry else None,
            },
        }


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate L1 Hardware Reflex."""
    print("=" * 60)
    print("L1 Hardware Reflex Demo")
    print("=" * 60)

    axis = AxisMundi(dim=1024, num_layers=9)
    l1 = L1HardwareReflex(axis)

    # Simulate telemetry progression
    scenarios = [
        ("Normal operation", TelemetryPacket(temp_c=45, load=0.3, error_rate=0.001)),
        ("Moderate load", TelemetryPacket(temp_c=55, load=0.6, error_rate=0.002)),
        ("High load", TelemetryPacket(temp_c=65, load=0.85, error_rate=0.005)),
        ("Thermal warning", TelemetryPacket(temp_c=78, load=0.9, error_rate=0.01)),
        ("Critical!", TelemetryPacket(temp_c=86, load=0.96, error_rate=0.06)),
    ]

    for name, telemetry in scenarios:
        print(f"\n--- {name} ---")
        state = l1.step(telemetry)
        print(f"  Temp: {telemetry.temp_c}C, Load: {telemetry.load:.1%}")
        print(f"  Pain: {state.pain_level:.3f}")
        print(f"  Thresh scale: {state.thresh_scale:.3f}")
        print(f"  Emergency: {state.is_emergency}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
