# ara/embodied/lizard/cortex.py
"""
Lizard Brain Cortex - The always-on sensory processing loop.

This is the central coordinator for low-power vigilance. It runs
continuously while the main Cathedral sleeps, processing sensor
streams for salience and managing power states.

Architecture:
    ┌─────────────────────────────────────────────────┐
    │                  LIZARD BRAIN                    │
    │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
    │  │ Sensors  │→ │ Salience │→ │ Wake Protocol │  │
    │  └──────────┘  └──────────┘  └──────────────┘  │
    │       ↑              ↓              ↓          │
    │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
    │  │ Thermal  │← │ Governor │← │   Attractor   │  │
    │  └──────────┘  └──────────┘  └──────────────┘  │
    └─────────────────────────────────────────────────┘
                          ↓
                   [WAKE CATHEDRAL]

The cortex maintains three operational modes:
    VIGILANT: Low-power monitoring, sensors active
    PROCESSING: Salience detected, analyzing
    WAKING: Triggering Cathedral wake sequence
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Any,
    Awaitable,
)

from .wake_protocol import WakeProtocol, WakeEvent, WakeEventType, WakeCriteria
from .power_governor import PowerGovernor, PowerState, ThermalZone
from .attractor_monitor import AttractorMonitor, BasinType


class LizardState(Enum):
    """Operational states of the Lizard Brain."""
    DORMANT = auto()       # Not running
    VIGILANT = auto()      # Low-power monitoring
    PROCESSING = auto()    # Analyzing detected salience
    WAKING = auto()        # Triggering Cathedral wake
    DREAMING = auto()      # Sleep cycle memory consolidation


@dataclass
class LizardConfig:
    """Configuration for the Lizard Brain."""
    # Power budgets (watts)
    idle_power_w: float = 35.0
    processing_power_w: float = 65.0
    max_power_w: float = 100.0  # Before triggering Cathedral

    # Timing
    sensor_poll_hz: float = 10.0      # How often to check sensors
    salience_window_s: float = 5.0    # Window for salience aggregation
    wake_cooldown_s: float = 30.0     # Minimum time between wake events

    # Thresholds
    salience_threshold: float = 0.6   # 0-1, when to wake Cathedral
    thermal_throttle_c: float = 70.0  # Start throttling at this temp
    thermal_shutdown_c: float = 85.0  # Emergency shutdown

    # Features
    enable_dreaming: bool = True
    dream_interval_hours: float = 6.0
    dream_duration_minutes: float = 30.0

    # Callbacks
    on_wake: Optional[Callable[[], Awaitable[None]]] = None
    on_sleep: Optional[Callable[[], Awaitable[None]]] = None
    on_thermal_warning: Optional[Callable[[float], Awaitable[None]]] = None


@dataclass
class SensorReading:
    """Aggregated sensor reading for salience detection."""
    timestamp: datetime
    audio_level: float = 0.0          # 0-1 normalized
    motion_detected: bool = False
    light_change: float = 0.0         # Delta from baseline
    network_activity: bool = False
    user_proximity: bool = False
    prediction_error: float = 0.0     # From internal models

    # Raw sensor data for detailed analysis
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LizardMetrics:
    """Runtime metrics for the Lizard Brain."""
    uptime_seconds: float = 0.0
    wake_events_triggered: int = 0
    salience_events_processed: int = 0
    average_power_w: float = 0.0
    peak_power_w: float = 0.0
    thermal_throttle_events: int = 0
    current_basin: BasinType = BasinType.HOMEOSTATIC
    last_wake_time: Optional[datetime] = None
    last_dream_time: Optional[datetime] = None


class LizardBrain:
    """
    The always-on vigilance cortex.

    Runs continuously at low power, monitoring sensors and deciding
    when to wake the full Cathedral system.
    """

    def __init__(self, config: Optional[LizardConfig] = None):
        self.config = config or LizardConfig()
        self._state = LizardState.DORMANT
        self._metrics = LizardMetrics()

        # Subsystems
        self._wake_protocol = WakeProtocol()
        self._power_governor = PowerGovernor()
        self._attractor_monitor = AttractorMonitor()

        # Runtime state
        self._running = False
        self._start_time: Optional[datetime] = None
        self._last_wake: Optional[datetime] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._sensor_buffer: List[SensorReading] = []

        # Callbacks
        self._event_callbacks: List[Callable[[WakeEvent], Awaitable[None]]] = []

    @property
    def state(self) -> LizardState:
        """Current operational state."""
        return self._state

    @property
    def metrics(self) -> LizardMetrics:
        """Current runtime metrics."""
        if self._start_time:
            self._metrics.uptime_seconds = (
                datetime.now() - self._start_time
            ).total_seconds()
        return self._metrics

    @property
    def is_running(self) -> bool:
        """Check if lizard brain is active."""
        return self._running

    async def start(self) -> None:
        """
        Start the Lizard Brain.

        Begins low-power monitoring loop.
        """
        if self._running:
            return

        self._running = True
        self._start_time = datetime.now()
        self._state = LizardState.VIGILANT

        # Start subsystems
        await self._power_governor.start()
        await self._attractor_monitor.start()

        # Start main loop
        asyncio.create_task(self._vigilance_loop())
        asyncio.create_task(self._dream_scheduler())

    async def stop(self) -> None:
        """Stop the Lizard Brain."""
        self._running = False
        self._state = LizardState.DORMANT

        await self._power_governor.stop()
        await self._attractor_monitor.stop()

    async def events(self) -> AsyncIterator[WakeEvent]:
        """
        Stream wake events from the Lizard Brain.

        Yields events when salience is detected or state changes.
        """
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                yield event
            except asyncio.TimeoutError:
                continue

    def subscribe(
        self,
        callback: Callable[[WakeEvent], Awaitable[None]]
    ) -> None:
        """Subscribe to wake events."""
        self._event_callbacks.append(callback)

    async def force_wake(self, reason: str = "manual") -> WakeEvent:
        """Force a wake event (for testing or manual override)."""
        event = WakeEvent(
            event_type=WakeEventType.MANUAL,
            salience=1.0,
            source="manual",
            reason=reason,
        )
        await self._trigger_wake(event)
        return event

    async def enter_dream_state(self) -> None:
        """Enter dreaming/consolidation mode."""
        if self._state == LizardState.DORMANT:
            return

        self._state = LizardState.DREAMING
        self._metrics.last_dream_time = datetime.now()

        # Reduce sensor polling during dreams
        await self._run_dream_cycle()

        self._state = LizardState.VIGILANT

    async def _vigilance_loop(self) -> None:
        """Main monitoring loop."""
        poll_interval = 1.0 / self.config.sensor_poll_hz

        while self._running:
            try:
                # Check power/thermal state
                power_state = await self._power_governor.get_state()

                if power_state.thermal_zone == ThermalZone.CRITICAL:
                    await self._handle_thermal_emergency()
                    continue

                if power_state.thermal_zone == ThermalZone.WARNING:
                    self._metrics.thermal_throttle_events += 1
                    if self.config.on_thermal_warning:
                        await self.config.on_thermal_warning(
                            power_state.temperature_c
                        )

                # Read sensors
                reading = await self._read_sensors()
                self._sensor_buffer.append(reading)

                # Trim buffer to window
                cutoff = datetime.now() - timedelta(
                    seconds=self.config.salience_window_s
                )
                self._sensor_buffer = [
                    r for r in self._sensor_buffer
                    if r.timestamp > cutoff
                ]

                # Compute salience
                salience = self._compute_salience()

                # Update attractor monitor
                await self._attractor_monitor.update(
                    power_w=power_state.current_power_w,
                    prediction_error=reading.prediction_error,
                    sensor_variance=self._compute_sensor_variance(),
                )
                self._metrics.current_basin = self._attractor_monitor.current_basin

                # Check wake criteria
                if salience >= self.config.salience_threshold:
                    await self._handle_salience(salience, reading)

                # Update metrics
                self._metrics.average_power_w = (
                    self._metrics.average_power_w * 0.99 +
                    power_state.current_power_w * 0.01
                )
                self._metrics.peak_power_w = max(
                    self._metrics.peak_power_w,
                    power_state.current_power_w
                )

                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but keep running
                await asyncio.sleep(poll_interval)

    async def _read_sensors(self) -> SensorReading:
        """Read from available sensors."""
        # This integrates with ara.embodied.sensors and ara.perception
        # For now, return simulated data

        import random

        return SensorReading(
            timestamp=datetime.now(),
            audio_level=random.random() * 0.3,  # Usually quiet
            motion_detected=random.random() < 0.05,  # Rare motion
            light_change=random.gauss(0, 0.1),
            network_activity=random.random() < 0.2,
            user_proximity=random.random() < 0.1,
            prediction_error=random.random() * 0.2,
        )

    def _compute_salience(self) -> float:
        """Compute aggregate salience from sensor buffer."""
        if not self._sensor_buffer:
            return 0.0

        # Weight different sensor modalities
        weights = {
            "audio": 0.3,
            "motion": 0.4,
            "light": 0.1,
            "network": 0.1,
            "proximity": 0.5,
            "prediction_error": 0.3,
        }

        salience = 0.0

        for reading in self._sensor_buffer[-5:]:  # Last 5 readings
            if reading.motion_detected:
                salience += weights["motion"]
            if reading.audio_level > 0.5:
                salience += weights["audio"] * reading.audio_level
            if abs(reading.light_change) > 0.3:
                salience += weights["light"]
            if reading.network_activity:
                salience += weights["network"]
            if reading.user_proximity:
                salience += weights["proximity"]
            salience += weights["prediction_error"] * reading.prediction_error

        # Normalize to 0-1
        return min(salience / len(self._sensor_buffer[-5:]), 1.0)

    def _compute_sensor_variance(self) -> float:
        """Compute variance in sensor readings (for attractor monitor)."""
        if len(self._sensor_buffer) < 2:
            return 0.0

        audio_levels = [r.audio_level for r in self._sensor_buffer]
        mean = sum(audio_levels) / len(audio_levels)
        variance = sum((x - mean) ** 2 for x in audio_levels) / len(audio_levels)

        return variance

    async def _handle_salience(
        self,
        salience: float,
        reading: SensorReading
    ) -> None:
        """Handle detected salience."""
        self._state = LizardState.PROCESSING
        self._metrics.salience_events_processed += 1

        # Check wake cooldown
        if self._last_wake:
            cooldown = timedelta(seconds=self.config.wake_cooldown_s)
            if datetime.now() - self._last_wake < cooldown:
                self._state = LizardState.VIGILANT
                return

        # Determine wake event type
        event_type = WakeEventType.SALIENCE

        if reading.motion_detected:
            event_type = WakeEventType.MOTION
        elif reading.audio_level > 0.7:
            event_type = WakeEventType.AUDIO
        elif reading.user_proximity:
            event_type = WakeEventType.PROXIMITY

        # Create wake event
        event = WakeEvent(
            event_type=event_type,
            salience=salience,
            source="lizard_cortex",
            reason=f"Salience {salience:.2f} exceeded threshold",
            sensor_data=reading.raw_data,
        )

        await self._trigger_wake(event)

    async def _trigger_wake(self, event: WakeEvent) -> None:
        """Trigger a wake event."""
        self._state = LizardState.WAKING
        self._last_wake = datetime.now()
        self._metrics.wake_events_triggered += 1
        self._metrics.last_wake_time = datetime.now()

        # Queue event
        await self._event_queue.put(event)

        # Notify callbacks
        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception:
                pass

        # Call config callback
        if self.config.on_wake:
            await self.config.on_wake()

        self._state = LizardState.VIGILANT

    async def _handle_thermal_emergency(self) -> None:
        """Handle critical thermal state."""
        # Emergency throttle - reduce all activity
        self._state = LizardState.DORMANT

        if self.config.on_thermal_warning:
            await self.config.on_thermal_warning(
                self.config.thermal_shutdown_c
            )

        # Wait for cooldown
        await asyncio.sleep(30.0)
        self._state = LizardState.VIGILANT

    async def _dream_scheduler(self) -> None:
        """Schedule periodic dream cycles."""
        if not self.config.enable_dreaming:
            return

        interval = self.config.dream_interval_hours * 3600

        while self._running:
            await asyncio.sleep(interval)
            if self._running and self._state == LizardState.VIGILANT:
                await self.enter_dream_state()

    async def _run_dream_cycle(self) -> None:
        """
        Run memory consolidation / dreaming.

        During dreams:
        - Reduce sensor polling
        - Run generative replay on stored experiences
        - Update internal models
        - Garbage collect old memories
        """
        duration = self.config.dream_duration_minutes * 60

        # Simplified dream cycle
        # Real implementation would:
        # 1. Buffer recent experiences
        # 2. Generate replay sequences
        # 3. Update weights via sleep-aware learning
        # 4. Prune low-value memories

        await asyncio.sleep(min(duration, 10.0))  # Abbreviated for dev


# Singleton instance
_lizard_brain: Optional[LizardBrain] = None


def get_lizard_brain(config: Optional[LizardConfig] = None) -> LizardBrain:
    """
    Get the global LizardBrain instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        Global LizardBrain instance
    """
    global _lizard_brain
    if _lizard_brain is None:
        _lizard_brain = LizardBrain(config)
    return _lizard_brain
