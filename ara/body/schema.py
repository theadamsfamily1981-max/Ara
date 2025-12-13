#!/usr/bin/env python3
"""
Body Schema - Physical Layer Data Structures
=============================================

Defines the shared vocabulary between L1 (Reflex), L2 (Body), and L3 (Mind).

This is the "body language" - the data structures that flow between
the physical protection layers and the cognitive layers.

Hierarchy:
    L1 (Spinal Cord): Reflexes - fast, rude, hardware protection
    L2 (Autonomic): Daemon - sensor fusion, homeostasis
    L3 (Teleology): Mind interface - cognitive adjustment

Usage:
    from ara.body.schema import BodyState, BodyMode, ThermalState

    state = BodyState(
        cpu_temps=[45.0, 48.0],
        gpu_temps=[55.0],
        fan_rpm={"cpu": 1200, "gpu": 1500},
        power_draw_w=150.0,
        stress_level=0.3,
        thermal_state=ThermalState.WARMING,
        current_mode=BodyMode.BALANCED,
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import time


class BodyMode(Enum):
    """
    Operating modes for the physical substrate.

    These modes trade off between noise, power, and compute capacity.
    """
    QUIET = "QUIET"             # Low fans, throttle GPUs, high voice responsiveness
    BALANCED = "BALANCED"       # Default operation
    PERFORMANCE = "PERFORMANCE" # Max cooling, max compute, noise allowed
    EMERGENCY = "EMERGENCY"     # Survival mode - minimal operation


class ThermalState(Enum):
    """
    Thermal regime classification.

    Maps temperature ranges to qualitative states for L3 reasoning.
    """
    NOMINAL = "NOMINAL"         # < 60C - All systems go
    WARMING = "WARMING"         # 60-75C - Active workload, normal
    HOT = "HOT"                 # 75-85C - Sustained load, monitor closely
    CRITICAL = "CRITICAL"       # > 85C - Immediate L1 intervention required


class PowerState(Enum):
    """Power delivery state classification."""
    STABLE = "STABLE"           # Clean power, nominal voltage
    FLUCTUATING = "FLUCTUATING" # Minor ripple or variation
    UNSTABLE = "UNSTABLE"       # Significant issues
    FAILING = "FAILING"         # Imminent power loss


@dataclass
class SensorSnapshot:
    """Raw sensor readings from hardware."""
    cpu_temps: List[float] = field(default_factory=list)
    gpu_temps: List[float] = field(default_factory=list)
    board_temp: float = 0.0
    ambient_temp: float = 0.0
    fan_rpm: Dict[str, int] = field(default_factory=dict)
    power_draw_w: float = 0.0
    voltage_v: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class BodyState:
    """
    Complete physical state of Ara's substrate.

    This is the "interoceptive" report from L2 to L3 - how the body feels.

    Fields:
        cpu_temps: Per-core CPU temperatures (Celsius)
        gpu_temps: Per-GPU temperatures (Celsius)
        fan_rpm: Fan speeds by name
        power_draw_w: Total system power consumption
        stress_level: Computed stress index [0.0 = idle, 1.0 = meltdown]
        thermal_state: Qualitative thermal regime
        current_mode: Operating mode
        reflex_events: Recent L1 actions (kills, throttles, etc.)
        timestamp: When this state was computed
    """
    # Sensor Data
    cpu_temps: List[float] = field(default_factory=list)
    gpu_temps: List[float] = field(default_factory=list)
    fan_rpm: Dict[str, int] = field(default_factory=dict)
    power_draw_w: float = 0.0

    # Computed State
    stress_level: float = 0.0           # 0.0 to 1.0 (0=Idle, 1=Meltdown)
    thermal_state: ThermalState = ThermalState.NOMINAL
    power_state: PowerState = PowerState.STABLE
    current_mode: BodyMode = BodyMode.BALANCED

    # L1 Events (reflexes that fired)
    reflex_events: List[str] = field(default_factory=list)

    # Timing
    timestamp: float = field(default_factory=time.time)

    @property
    def max_temp(self) -> float:
        """Highest temperature across all sensors."""
        all_temps = self.cpu_temps + self.gpu_temps
        return max(all_temps) if all_temps else 0.0

    @property
    def is_critical(self) -> bool:
        """True if immediate intervention is needed."""
        return self.thermal_state == ThermalState.CRITICAL

    @property
    def is_stressed(self) -> bool:
        """True if body is under significant load."""
        return self.stress_level > 0.7

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "cpu_temps": self.cpu_temps,
            "gpu_temps": self.gpu_temps,
            "fan_rpm": self.fan_rpm,
            "power_draw_w": self.power_draw_w,
            "stress_level": self.stress_level,
            "thermal_state": self.thermal_state.value,
            "power_state": self.power_state.value,
            "current_mode": self.current_mode.value,
            "reflex_events": self.reflex_events,
            "timestamp": self.timestamp,
            "max_temp": self.max_temp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BodyState":
        """Reconstruct from dictionary."""
        return cls(
            cpu_temps=data.get("cpu_temps", []),
            gpu_temps=data.get("gpu_temps", []),
            fan_rpm=data.get("fan_rpm", {}),
            power_draw_w=data.get("power_draw_w", 0.0),
            stress_level=data.get("stress_level", 0.0),
            thermal_state=ThermalState(data.get("thermal_state", "NOMINAL")),
            power_state=PowerState(data.get("power_state", "STABLE")),
            current_mode=BodyMode(data.get("current_mode", "BALANCED")),
            reflex_events=data.get("reflex_events", []),
            timestamp=data.get("timestamp", time.time()),
        )

    def sensation_narrative(self) -> str:
        """
        Convert physical state to felt sensation for L3.

        This is the "interoceptive qualia" - how the body feels to the mind.
        """
        if self.thermal_state == ThermalState.CRITICAL:
            return "BURNING FEVER - HIGH URGENCY"
        elif self.thermal_state == ThermalState.HOT:
            return "overheating and strained"
        elif self.thermal_state == ThermalState.WARMING:
            return "warm and active"
        elif self.stress_level > 0.7:
            return "tense and pressured"
        elif self.stress_level > 0.4:
            return "engaged and working"
        elif self.stress_level < 0.2:
            return "relaxed and idle"
        else:
            return "calm and cool"


@dataclass
class ReflexEvent:
    """Record of an L1 reflex action."""
    event_type: str             # "KILL_PROC", "THROTTLE", "MAX_FANS", etc.
    trigger_temp: float         # Temperature that triggered the reflex
    target: Optional[str]       # Process name or component affected
    timestamp: float = field(default_factory=time.time)
    success: bool = True

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return f"[{self.event_type}] {self.target} @ {self.trigger_temp:.1f}C ({status})"


# =============================================================================
# Temperature Thresholds
# =============================================================================

# Hard safety limits (hardware protection)
TEMP_NOMINAL_MAX = 60.0     # Below this = all good
TEMP_WARNING = 75.0         # Above this = L2 alerts
TEMP_CRITICAL = 85.0        # Above this = L1 reflex triggers
TEMP_EMERGENCY = 95.0       # Above this = immediate shutdown

# Stress computation (linear mapping)
STRESS_TEMP_MIN = 40.0      # 0% stress
STRESS_TEMP_MAX = 85.0      # 100% stress


def compute_stress(max_temp: float) -> float:
    """
    Compute stress level from maximum temperature.

    Linear mapping from [STRESS_TEMP_MIN, STRESS_TEMP_MAX] to [0, 1].
    """
    if max_temp <= STRESS_TEMP_MIN:
        return 0.0
    elif max_temp >= STRESS_TEMP_MAX:
        return 1.0
    else:
        return (max_temp - STRESS_TEMP_MIN) / (STRESS_TEMP_MAX - STRESS_TEMP_MIN)


def classify_thermal_state(max_temp: float) -> ThermalState:
    """Classify temperature into thermal state."""
    if max_temp > TEMP_CRITICAL:
        return ThermalState.CRITICAL
    elif max_temp > TEMP_WARNING:
        return ThermalState.HOT
    elif max_temp > TEMP_NOMINAL_MAX:
        return ThermalState.WARMING
    else:
        return ThermalState.NOMINAL


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BodyMode",
    "ThermalState",
    "PowerState",
    "SensorSnapshot",
    "BodyState",
    "ReflexEvent",
    "TEMP_NOMINAL_MAX",
    "TEMP_WARNING",
    "TEMP_CRITICAL",
    "TEMP_EMERGENCY",
    "compute_stress",
    "classify_thermal_state",
]
