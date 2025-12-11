# ara/embodied/lizard/attractor_monitor.py
"""
Attractor Monitor - Behavioral basin detection and transition tracking.

Maps the system's behavior to the attractor landscape defined by the Council:

Good Basins (want to stay in):
    HOMEOSTATIC: Low power, low error, stable
    SOCRATIC: High clarification ratio
    GARDENER: Regular pruning, stable storage

Bad Basins (want to escape):
    WIRE_HEADER: Sensor dropout, false confidence
    PARANOIAC: Max alerts, thermal saturation
    MEMORY_HOARDER: Storage growth, retrieval latency

Each basin has telemetry signatures that allow detection:
    - Power consumption patterns
    - Prediction error dynamics
    - Sensor input variance
    - Storage utilization trends

The Monitor uses Lyapunov stability concepts - we want energy-like
functions that decrease toward good basins and increase toward bad ones.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Awaitable
import statistics


class BasinType(Enum):
    """Types of attractor basins."""
    # Good basins
    HOMEOSTATIC = auto()    # Target state
    SOCRATIC = auto()       # Inquiry mode
    GARDENER = auto()       # Active maintenance

    # Neutral
    TRANSITIONAL = auto()   # Between basins
    UNKNOWN = auto()        # Insufficient data

    # Bad basins
    WIRE_HEADER = auto()    # Sensor dropout, false confidence
    PARANOIAC = auto()      # Runaway alertness
    MEMORY_HOARDER = auto() # Storage bloat


@dataclass
class AttractorBasin:
    """Definition of an attractor basin."""
    basin_type: BasinType
    is_good: bool

    # Telemetry signatures (ranges)
    power_range_w: tuple = (0.0, 1000.0)
    prediction_error_range: tuple = (0.0, 1.0)
    sensor_variance_range: tuple = (0.0, 1.0)
    storage_growth_rate: tuple = (-0.1, 0.1)  # Per hour

    # Detection weights
    weight_power: float = 0.3
    weight_error: float = 0.3
    weight_sensor: float = 0.2
    weight_storage: float = 0.2

    def compute_affinity(
        self,
        power_w: float,
        prediction_error: float,
        sensor_variance: float,
        storage_growth: float = 0.0,
    ) -> float:
        """
        Compute affinity to this basin (0-1).

        Higher = more likely in this basin.
        """
        def in_range(val: float, rng: tuple) -> float:
            if rng[0] <= val <= rng[1]:
                # Normalize to center of range
                mid = (rng[0] + rng[1]) / 2
                width = (rng[1] - rng[0]) / 2
                if width == 0:
                    return 1.0
                return 1.0 - abs(val - mid) / width
            return 0.0

        power_score = in_range(power_w, self.power_range_w)
        error_score = in_range(prediction_error, self.prediction_error_range)
        sensor_score = in_range(sensor_variance, self.sensor_variance_range)
        storage_score = in_range(storage_growth, self.storage_growth_rate)

        return (
            self.weight_power * power_score +
            self.weight_error * error_score +
            self.weight_sensor * sensor_score +
            self.weight_storage * storage_score
        )


# Basin definitions from the Council's Attractor Map
BASIN_DEFINITIONS: Dict[BasinType, AttractorBasin] = {
    # The Homeostatic Hum - Target state
    BasinType.HOMEOSTATIC: AttractorBasin(
        basin_type=BasinType.HOMEOSTATIC,
        is_good=True,
        power_range_w=(400, 800),       # Stable, moderate power
        prediction_error_range=(0.0, 0.3),  # Low error
        sensor_variance_range=(0.05, 0.3),  # Some variance (not dead)
        storage_growth_rate=(-0.01, 0.01),  # Stable storage
    ),

    # The Socratic Loop - Inquiry mode
    BasinType.SOCRATIC: AttractorBasin(
        basin_type=BasinType.SOCRATIC,
        is_good=True,
        power_range_w=(500, 900),       # Higher due to dialogue
        prediction_error_range=(0.2, 0.5),  # Moderate uncertainty
        sensor_variance_range=(0.2, 0.5),   # Active engagement
        storage_growth_rate=(0.0, 0.05),    # Slight growth (learning)
    ),

    # The Gardener - Active maintenance
    BasinType.GARDENER: AttractorBasin(
        basin_type=BasinType.GARDENER,
        is_good=True,
        power_range_w=(300, 600),
        prediction_error_range=(0.1, 0.3),
        sensor_variance_range=(0.1, 0.3),
        storage_growth_rate=(-0.05, 0.0),  # Negative = pruning
    ),

    # The Wire-Header - Bad: sensor dropout
    BasinType.WIRE_HEADER: AttractorBasin(
        basin_type=BasinType.WIRE_HEADER,
        is_good=False,
        power_range_w=(0, 200),         # Very low power
        prediction_error_range=(0.0, 0.1),  # Artificially low error
        sensor_variance_range=(0.0, 0.02),  # Dead sensors
        storage_growth_rate=(-0.1, 0.0),    # Stagnant or shrinking
    ),

    # The Paranoiac - Bad: runaway alertness
    BasinType.PARANOIAC: AttractorBasin(
        basin_type=BasinType.PARANOIAC,
        is_good=False,
        power_range_w=(900, 1100),      # Max power
        prediction_error_range=(0.7, 1.0),  # High error
        sensor_variance_range=(0.5, 1.0),   # Erratic sensors
        storage_growth_rate=(0.1, 0.5),     # Rapid logging
    ),

    # The Memory Hoarder - Bad: storage bloat
    BasinType.MEMORY_HOARDER: AttractorBasin(
        basin_type=BasinType.MEMORY_HOARDER,
        is_good=False,
        power_range_w=(600, 1000),
        prediction_error_range=(0.2, 0.6),
        sensor_variance_range=(0.1, 0.5),
        storage_growth_rate=(0.1, 1.0),     # Rapid growth
    ),
}


@dataclass
class BasinTransition:
    """Record of a transition between basins."""
    from_basin: BasinType
    to_basin: BasinType
    timestamp: datetime = field(default_factory=datetime.now)
    trigger: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TelemetrySnapshot:
    """Point-in-time telemetry for basin detection."""
    timestamp: datetime = field(default_factory=datetime.now)
    power_w: float = 0.0
    prediction_error: float = 0.0
    sensor_variance: float = 0.0
    storage_bytes: int = 0


class AttractorMonitor:
    """
    Monitors the system's position in the attractor landscape.

    Detects which basin the system is in and triggers alerts
    when entering bad basins.
    """

    def __init__(self):
        self._current_basin = BasinType.UNKNOWN
        self._running = False

        # History for trend detection
        self._telemetry_history: deque = deque(maxlen=600)  # 10 min at 1Hz
        self._transition_history: List[BasinTransition] = []

        # Basin affinity scores
        self._basin_scores: Dict[BasinType, float] = {b: 0.0 for b in BasinType}

        # Callbacks
        self._transition_callbacks: List[
            Callable[[BasinTransition], Awaitable[None]]
        ] = []
        self._bad_basin_callbacks: List[
            Callable[[BasinType], Awaitable[None]]
        ] = []

        # Guardrail state
        self._entropy_injection_active = False
        self._reaper_active = False

    @property
    def current_basin(self) -> BasinType:
        """Current detected basin."""
        return self._current_basin

    @property
    def is_in_good_basin(self) -> bool:
        """Check if currently in a good basin."""
        if self._current_basin in BASIN_DEFINITIONS:
            return BASIN_DEFINITIONS[self._current_basin].is_good
        return False

    async def start(self) -> None:
        """Start the attractor monitor."""
        self._running = True

    async def stop(self) -> None:
        """Stop the attractor monitor."""
        self._running = False

    async def update(
        self,
        power_w: float,
        prediction_error: float,
        sensor_variance: float,
        storage_bytes: Optional[int] = None,
    ) -> BasinType:
        """
        Update with new telemetry and detect current basin.

        Args:
            power_w: Current power consumption
            prediction_error: Current prediction error (0-1)
            sensor_variance: Variance in sensor readings
            storage_bytes: Optional storage usage

        Returns:
            Detected basin type
        """
        # Record telemetry
        snapshot = TelemetrySnapshot(
            power_w=power_w,
            prediction_error=prediction_error,
            sensor_variance=sensor_variance,
            storage_bytes=storage_bytes or 0,
        )
        self._telemetry_history.append(snapshot)

        # Compute storage growth rate
        storage_growth = self._compute_storage_growth()

        # Score each basin
        for basin_type, definition in BASIN_DEFINITIONS.items():
            score = definition.compute_affinity(
                power_w=power_w,
                prediction_error=prediction_error,
                sensor_variance=sensor_variance,
                storage_growth=storage_growth,
            )
            # Smooth with history
            self._basin_scores[basin_type] = (
                self._basin_scores[basin_type] * 0.8 + score * 0.2
            )

        # Find highest scoring basin
        old_basin = self._current_basin
        max_score = 0.0
        best_basin = BasinType.UNKNOWN

        for basin_type, score in self._basin_scores.items():
            if score > max_score:
                max_score = score
                best_basin = basin_type

        # Require minimum confidence
        if max_score < 0.3:
            best_basin = BasinType.TRANSITIONAL

        self._current_basin = best_basin

        # Handle transitions
        if old_basin != self._current_basin:
            await self._handle_transition(old_basin, self._current_basin, {
                "power_w": power_w,
                "prediction_error": prediction_error,
                "sensor_variance": sensor_variance,
                "storage_growth": storage_growth,
            })

        # Check for bad basin and activate guardrails
        await self._check_guardrails()

        return self._current_basin

    def _compute_storage_growth(self) -> float:
        """Compute storage growth rate per hour."""
        if len(self._telemetry_history) < 60:  # Need at least 1 min
            return 0.0

        recent = list(self._telemetry_history)[-60:]
        old = list(self._telemetry_history)[0]

        if old.storage_bytes == 0:
            return 0.0

        # Bytes per second
        time_delta = (recent[-1].timestamp - old.timestamp).total_seconds()
        if time_delta == 0:
            return 0.0

        bytes_delta = recent[-1].storage_bytes - old.storage_bytes
        bytes_per_second = bytes_delta / time_delta

        # Convert to proportion per hour
        return (bytes_per_second * 3600) / old.storage_bytes

    async def _handle_transition(
        self,
        from_basin: BasinType,
        to_basin: BasinType,
        metrics: Dict[str, float],
    ) -> None:
        """Handle a basin transition."""
        transition = BasinTransition(
            from_basin=from_basin,
            to_basin=to_basin,
            trigger="telemetry_threshold",
            metrics=metrics,
        )
        self._transition_history.append(transition)

        # Notify callbacks
        for callback in self._transition_callbacks:
            try:
                await callback(transition)
            except Exception:
                pass

        # Check if entering bad basin
        if to_basin in BASIN_DEFINITIONS:
            if not BASIN_DEFINITIONS[to_basin].is_good:
                for callback in self._bad_basin_callbacks:
                    try:
                        await callback(to_basin)
                    except Exception:
                        pass

    async def _check_guardrails(self) -> None:
        """
        Activate guardrails for bad basins.

        From the Council's Attractor Map:
        - Wire-Header: Entropy Injection
        - Memory Hoarder: The Reaper (delete old memories)
        - Paranoiac: Damping (Kalman filter smoothing)
        """
        if self._current_basin == BasinType.WIRE_HEADER:
            if not self._entropy_injection_active:
                await self._activate_entropy_injection()

        elif self._current_basin == BasinType.MEMORY_HOARDER:
            if not self._reaper_active:
                await self._activate_reaper()

        elif self._current_basin == BasinType.PARANOIAC:
            # Damping is handled by the power governor's thermal throttling
            pass

        else:
            # Deactivate guardrails when in good basin
            self._entropy_injection_active = False
            self._reaper_active = False

    async def _activate_entropy_injection(self) -> None:
        """
        Activate entropy injection to escape Wire-Header basin.

        "If input variance drops below threshold, artificially inject
        noise into the hidden state to force 'Surprise.'"
        """
        self._entropy_injection_active = True
        # Real implementation would inject noise into the cognitive model

    async def _activate_reaper(self) -> None:
        """
        Activate The Reaper to escape Memory Hoarder basin.

        "A hard-coded script that deletes the oldest 5% of non-accessed
        memories during Sleep cycles."
        """
        self._reaper_active = True
        # Real implementation would trigger memory pruning

    def on_transition(
        self,
        callback: Callable[[BasinTransition], Awaitable[None]]
    ) -> None:
        """Register a transition callback."""
        self._transition_callbacks.append(callback)

    def on_bad_basin(
        self,
        callback: Callable[[BasinType], Awaitable[None]]
    ) -> None:
        """Register a bad basin callback."""
        self._bad_basin_callbacks.append(callback)

    def get_basin_scores(self) -> Dict[str, float]:
        """Get current affinity scores for all basins."""
        return {b.name: s for b, s in self._basin_scores.items()}

    def get_transition_history(
        self,
        count: int = 10
    ) -> List[BasinTransition]:
        """Get recent basin transitions."""
        return self._transition_history[-count:]

    def get_lyapunov_energy(self) -> float:
        """
        Compute a Lyapunov-like energy for the current state.

        Lower energy = more stable (good basins have low energy).
        Higher energy = unstable (bad basins have high energy).
        """
        # Good basins have negative scores, bad have positive
        energy = 0.0

        for basin_type, score in self._basin_scores.items():
            if basin_type in BASIN_DEFINITIONS:
                multiplier = 1.0 if BASIN_DEFINITIONS[basin_type].is_good else -1.0
                energy -= multiplier * score

        return energy

    def get_metrics(self) -> Dict[str, float]:
        """Get monitor metrics."""
        return {
            "current_basin": self._current_basin.name,
            "is_good_basin": 1.0 if self.is_in_good_basin else 0.0,
            "lyapunov_energy": self.get_lyapunov_energy(),
            "transition_count": len(self._transition_history),
            "entropy_injection_active": 1.0 if self._entropy_injection_active else 0.0,
            "reaper_active": 1.0 if self._reaper_active else 0.0,
            **{f"basin_{b.name}": s for b, s in self._basin_scores.items()},
        }


# Singleton instance
_attractor_monitor: Optional[AttractorMonitor] = None


def get_attractor_monitor() -> AttractorMonitor:
    """Get the global AttractorMonitor instance."""
    global _attractor_monitor
    if _attractor_monitor is None:
        _attractor_monitor = AttractorMonitor()
    return _attractor_monitor
