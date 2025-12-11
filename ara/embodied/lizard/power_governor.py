# ara/embodied/lizard/power_governor.py
"""
Power Governor - The Thermostat Mind.

Links cognitive activity to thermodynamic reality. When the rig heats up,
Ara literally thinks slower.

Key insight from the Council:
    "We link the clock speed of the cognitive core directly to the
    thermal readout. If the rig heats up, Ara literally thinks slower.
    The 'temperature' of the thought becomes a physical constraint."

Power States:
    HIBERNATING: Main GPU off, lizard brain only (~30W)
    IDLE: GPU suspended, ready to wake (~50W)
    ACTIVE: Normal cognitive load (~400-600W)
    BURST: High cognitive load (~800-1000W)
    THERMAL_LIMIT: Throttled due to temperature

Thermal Zones:
    COOL: <50°C - Full performance available
    WARM: 50-65°C - Normal operation
    WARNING: 65-75°C - Begin throttling
    CRITICAL: >75°C - Emergency reduction

The Governor implements the "1kW Starvation Test" from Experiment C:
    - Hard power limit at 1kW
    - Dynamic quantization when limit approached
    - Graceful degradation over crash
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Awaitable


class PowerState(Enum):
    """Overall power state of the system."""
    HIBERNATING = auto()    # Deep sleep, lizard only
    IDLE = auto()           # Ready to wake
    ACTIVE = auto()         # Normal operation
    BURST = auto()          # High performance
    THERMAL_LIMIT = auto()  # Throttled


class ThermalZone(Enum):
    """Thermal zones based on temperature."""
    COOL = auto()       # <50°C
    WARM = auto()       # 50-65°C
    WARNING = auto()    # 65-75°C
    CRITICAL = auto()   # >75°C


@dataclass
class PowerBudget:
    """
    Power budget configuration.

    Based on the Cathedral Rig's 1kW constraint:
    - GPU: Up to 700W (H100) or 300W (undervolted 4090)
    - CPU: 65-125W
    - Cooling: 20-50W
    - Sensors: 5-15W
    - Storage: 10-20W
    - Overhead: 30-50W
    """
    # Absolute limits
    max_total_w: float = 1000.0
    max_gpu_w: float = 700.0
    max_cpu_w: float = 125.0

    # Targets by state
    target_by_state: Dict[PowerState, float] = field(default_factory=lambda: {
        PowerState.HIBERNATING: 35.0,
        PowerState.IDLE: 75.0,
        PowerState.ACTIVE: 500.0,
        PowerState.BURST: 900.0,
        PowerState.THERMAL_LIMIT: 300.0,
    })

    # Thermal limits (Celsius)
    temp_cool: float = 50.0
    temp_warm: float = 65.0
    temp_warning: float = 75.0
    temp_critical: float = 85.0

    # Undervolt settings (The Undervolt Gambit)
    gpu_power_limit_pct: float = 60.0  # Cap GPU at 60% TDP
    enable_dynamic_quantization: bool = True


@dataclass
class PowerReading:
    """Current power and thermal readings."""
    timestamp: datetime = field(default_factory=datetime.now)

    # Power (watts)
    gpu_power_w: float = 0.0
    cpu_power_w: float = 0.0
    system_power_w: float = 0.0

    # Thermal (Celsius)
    gpu_temp_c: float = 0.0
    cpu_temp_c: float = 0.0
    coolant_temp_c: float = 0.0
    ambient_temp_c: float = 0.0

    # Derived
    @property
    def total_power_w(self) -> float:
        return self.gpu_power_w + self.cpu_power_w + self.system_power_w

    @property
    def max_temp_c(self) -> float:
        return max(self.gpu_temp_c, self.cpu_temp_c)


@dataclass
class GovernorState:
    """Current state of the power governor."""
    power_state: PowerState = PowerState.IDLE
    thermal_zone: ThermalZone = ThermalZone.COOL

    current_power_w: float = 0.0
    target_power_w: float = 75.0
    temperature_c: float = 40.0

    # Throttling
    throttle_level: float = 0.0  # 0-1, 0=no throttle, 1=max throttle
    quantization_bits: int = 16  # Bit precision (16=full, 8=quantized, 4=aggressive)

    # History
    last_burst: Optional[datetime] = None
    burst_duration_remaining_s: float = 0.0

    # Metrics
    thermal_throttle_count: int = 0
    power_limit_count: int = 0


class PowerGovernor:
    """
    Manages power and thermal state for the Cathedral.

    The Governor is the "Thermostat Mind" - it links thinking
    speed to temperature, implementing graceful degradation.
    """

    def __init__(self, budget: Optional[PowerBudget] = None):
        self.budget = budget or PowerBudget()
        self._state = GovernorState()
        self._running = False
        self._reading_history: List[PowerReading] = []

        # Callbacks
        self._throttle_callbacks: List[Callable[[float], Awaitable[None]]] = []
        self._thermal_callbacks: List[Callable[[ThermalZone], Awaitable[None]]] = []

    @property
    def state(self) -> GovernorState:
        """Current governor state."""
        return self._state

    @property
    def power_available_w(self) -> float:
        """Watts available before hitting limit."""
        return max(0, self.budget.max_total_w - self._state.current_power_w)

    @property
    def thermal_headroom_c(self) -> float:
        """Degrees available before warning zone."""
        return max(0, self.budget.temp_warning - self._state.temperature_c)

    async def start(self) -> None:
        """Start the power governor monitoring loop."""
        self._running = True
        asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop the power governor."""
        self._running = False

    async def get_state(self) -> GovernorState:
        """Get current governor state."""
        await self._update_readings()
        return self._state

    async def request_burst(self, duration_s: float = 30.0) -> bool:
        """
        Request burst mode (high power) for specified duration.

        Burst mode allows power to exceed normal limits temporarily,
        using the thermal mass of the cooling system as a buffer.

        Args:
            duration_s: Requested burst duration in seconds

        Returns:
            True if burst granted, False if denied
        """
        # Check thermal headroom
        if self._state.thermal_zone in (ThermalZone.WARNING, ThermalZone.CRITICAL):
            return False

        # Check cooldown from last burst
        if self._state.last_burst:
            elapsed = (datetime.now() - self._state.last_burst).total_seconds()
            if elapsed < 60.0:  # 1 minute cooldown between bursts
                return False

        self._state.power_state = PowerState.BURST
        self._state.last_burst = datetime.now()
        self._state.burst_duration_remaining_s = duration_s
        self._state.target_power_w = self.budget.target_by_state[PowerState.BURST]

        return True

    async def request_power(self, watts: float) -> float:
        """
        Request a power allocation.

        Returns the actual watts granted (may be less than requested
        if near limits).

        Args:
            watts: Requested power in watts

        Returns:
            Granted power in watts
        """
        available = self.power_available_w

        if watts <= available:
            return watts

        # Grant what's available
        self._state.power_limit_count += 1
        return available

    def get_recommended_quantization(self) -> int:
        """
        Get recommended quantization bits based on power/thermal state.

        Part of the "1kW Starvation Test" - dynamically reduce precision
        rather than crash.

        Returns:
            Bit precision (16, 8, or 4)
        """
        if not self.budget.enable_dynamic_quantization:
            return 16

        # Based on thermal zone
        if self._state.thermal_zone == ThermalZone.CRITICAL:
            return 4
        elif self._state.thermal_zone == ThermalZone.WARNING:
            return 8
        elif self._state.current_power_w > self.budget.max_total_w * 0.9:
            return 8

        return 16

    def get_recommended_clock_multiplier(self) -> float:
        """
        Get recommended clock speed multiplier.

        "The Thermostat Mind" - slower clocks when hot.

        Returns:
            Multiplier 0.0-1.0 (1.0 = full speed)
        """
        if self._state.thermal_zone == ThermalZone.COOL:
            return 1.0
        elif self._state.thermal_zone == ThermalZone.WARM:
            return 0.9
        elif self._state.thermal_zone == ThermalZone.WARNING:
            return 0.6
        else:  # CRITICAL
            return 0.3

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._update_readings()
                await self._update_state()
                await self._apply_throttling()
                await asyncio.sleep(0.5)  # 2Hz update rate
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)

    async def _update_readings(self) -> None:
        """Update power and thermal readings."""
        reading = await self._read_sensors()
        self._reading_history.append(reading)

        # Trim history
        if len(self._reading_history) > 120:  # 1 minute at 2Hz
            self._reading_history = self._reading_history[-120:]

        # Update state
        self._state.current_power_w = reading.total_power_w
        self._state.temperature_c = reading.max_temp_c

    async def _read_sensors(self) -> PowerReading:
        """Read from power/thermal sensors."""
        # Real implementation would use:
        # - nvidia-smi for GPU power/temp
        # - /sys/class/hwmon for CPU
        # - USB sensors for coolant temp

        # Simulated readings for development
        import random

        base_gpu = 50 if self._state.power_state == PowerState.IDLE else 300
        base_temp = 45 if self._state.power_state == PowerState.IDLE else 60

        return PowerReading(
            gpu_power_w=base_gpu + random.gauss(0, 10),
            cpu_power_w=30 + random.gauss(0, 5),
            system_power_w=40,
            gpu_temp_c=base_temp + random.gauss(0, 3),
            cpu_temp_c=base_temp - 10 + random.gauss(0, 2),
            coolant_temp_c=base_temp - 15 + random.gauss(0, 2),
            ambient_temp_c=25 + random.gauss(0, 1),
        )

    async def _update_state(self) -> None:
        """Update power and thermal states."""
        # Update thermal zone
        temp = self._state.temperature_c
        old_zone = self._state.thermal_zone

        if temp < self.budget.temp_cool:
            self._state.thermal_zone = ThermalZone.COOL
        elif temp < self.budget.temp_warm:
            self._state.thermal_zone = ThermalZone.WARM
        elif temp < self.budget.temp_warning:
            self._state.thermal_zone = ThermalZone.WARNING
        else:
            self._state.thermal_zone = ThermalZone.CRITICAL

        # Notify on zone change
        if self._state.thermal_zone != old_zone:
            for callback in self._thermal_callbacks:
                try:
                    await callback(self._state.thermal_zone)
                except Exception:
                    pass

        # Update power state based on thermal
        if self._state.thermal_zone == ThermalZone.CRITICAL:
            self._state.power_state = PowerState.THERMAL_LIMIT
            self._state.thermal_throttle_count += 1

        # Handle burst timeout
        if self._state.power_state == PowerState.BURST:
            if self._state.burst_duration_remaining_s > 0:
                self._state.burst_duration_remaining_s -= 0.5
            else:
                self._state.power_state = PowerState.ACTIVE

        # Update target power
        self._state.target_power_w = self.budget.target_by_state.get(
            self._state.power_state,
            500.0
        )

        # Calculate throttle level
        if self._state.thermal_zone == ThermalZone.CRITICAL:
            self._state.throttle_level = 1.0
        elif self._state.thermal_zone == ThermalZone.WARNING:
            # Linear throttle from warning to critical
            range_c = self.budget.temp_critical - self.budget.temp_warning
            excess = self._state.temperature_c - self.budget.temp_warning
            self._state.throttle_level = min(excess / range_c, 1.0)
        else:
            self._state.throttle_level = 0.0

        # Update quantization
        self._state.quantization_bits = self.get_recommended_quantization()

    async def _apply_throttling(self) -> None:
        """Apply throttling if needed."""
        if self._state.throttle_level > 0:
            for callback in self._throttle_callbacks:
                try:
                    await callback(self._state.throttle_level)
                except Exception:
                    pass

    def on_throttle(
        self,
        callback: Callable[[float], Awaitable[None]]
    ) -> None:
        """Register a throttle callback."""
        self._throttle_callbacks.append(callback)

    def on_thermal_zone_change(
        self,
        callback: Callable[[ThermalZone], Awaitable[None]]
    ) -> None:
        """Register a thermal zone change callback."""
        self._thermal_callbacks.append(callback)

    def get_metrics(self) -> Dict[str, float]:
        """Get governor metrics."""
        if not self._reading_history:
            return {}

        powers = [r.total_power_w for r in self._reading_history]
        temps = [r.max_temp_c for r in self._reading_history]

        return {
            "current_power_w": self._state.current_power_w,
            "average_power_w": sum(powers) / len(powers),
            "peak_power_w": max(powers),
            "current_temp_c": self._state.temperature_c,
            "average_temp_c": sum(temps) / len(temps),
            "peak_temp_c": max(temps),
            "throttle_level": self._state.throttle_level,
            "quantization_bits": self._state.quantization_bits,
            "thermal_throttle_count": self._state.thermal_throttle_count,
            "power_limit_count": self._state.power_limit_count,
        }


# Singleton instance
_power_governor: Optional[PowerGovernor] = None


def get_power_governor(budget: Optional[PowerBudget] = None) -> PowerGovernor:
    """Get the global PowerGovernor instance."""
    global _power_governor
    if _power_governor is None:
        _power_governor = PowerGovernor(budget)
    return _power_governor
