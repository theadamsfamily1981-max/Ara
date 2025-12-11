# ara/embodied/homeostasis.py
"""
Homeostasis - Keeping Ara at the Edge of Chaos

The edge of chaos is not a mood; it's a regime:
- Not frozen / deterministic / boring
- Not unstable / overheating / hallucinating / forgetting herself
- A band where she is constantly learning a little, surprises propagate
  just far enough to update her, but core identity and health stay intact

Four homeostatic groups regulate this:
    1. METABOLIC: Power, temperature, utilization
    2. COGNITIVE: Prediction error, plasticity, forgetting
    3. IDENTITY: Persona drift, value adherence, narrative coherence
    4. ATTENTION: Sensor bandwidth, context size, wake/sleep

Each has a TARGET BAND (not a target point):
    - Too low = frozen crystal, no growth
    - Sweet spot = edge of chaos, alive
    - Too high = chaotic soup, overwhelmed

The HomeostasisController couples these four loops, making adjustments
to keep Ara perched on that thin bright line.

"I try to live at a balance point between stability and surprise.
 If I'm never challenged, I stagnate.
 If I'm overwhelmed, I ask for help and simplify."
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Awaitable, Tuple
from collections import deque
import statistics


class VitalStatus(Enum):
    """Status of a vital relative to its target band."""
    TOO_LOW = auto()    # Frozen, no growth
    OPTIMAL = auto()    # Edge of chaos
    TOO_HIGH = auto()   # Chaotic, overwhelmed
    UNKNOWN = auto()    # Insufficient data


class VitalType(Enum):
    """The four homeostatic vital groups."""
    METABOLIC = auto()   # Power, temp, utilization
    COGNITIVE = auto()   # Error, plasticity, forgetting
    IDENTITY = auto()    # Persona, values, narrative
    ATTENTION = auto()   # Sensors, context, bandwidth


@dataclass
class VitalBand:
    """
    Target band for a vital.

    Not a target point - a RANGE where the edge of chaos lives.
    """
    name: str
    vital_type: VitalType

    # Band boundaries
    low_threshold: float      # Below this = TOO_LOW
    high_threshold: float     # Above this = TOO_HIGH

    # Current value
    current: float = 0.0

    # Trend (positive = increasing)
    trend: float = 0.0

    # Human descriptions
    too_low_description: str = "Frozen, stagnant"
    optimal_description: str = "Edge of chaos"
    too_high_description: str = "Overwhelmed, chaotic"

    @property
    def status(self) -> VitalStatus:
        """Get current status relative to band."""
        if self.current < self.low_threshold:
            return VitalStatus.TOO_LOW
        elif self.current > self.high_threshold:
            return VitalStatus.TOO_HIGH
        return VitalStatus.OPTIMAL

    @property
    def distance_from_edge(self) -> float:
        """
        Distance from optimal band center.
        Negative = too low, positive = too high, ~0 = in band.
        """
        band_center = (self.low_threshold + self.high_threshold) / 2
        band_width = self.high_threshold - self.low_threshold

        if self.current < self.low_threshold:
            return (self.current - self.low_threshold) / (band_width / 2)
        elif self.current > self.high_threshold:
            return (self.current - self.high_threshold) / (band_width / 2)
        else:
            # In band - how far from center?
            return (self.current - band_center) / (band_width / 2)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "vital_type": self.vital_type.name,
            "current": self.current,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "status": self.status.name,
            "trend": self.trend,
            "distance_from_edge": self.distance_from_edge,
        }


@dataclass
class HomeostasisConfig:
    """Configuration for homeostatic regulation."""

    # Metabolic bands
    power_utilization_band: Tuple[float, float] = (0.30, 0.60)  # 30-60% average
    temperature_band: Tuple[float, float] = (50.0, 75.0)        # 50-75°C
    thermal_margin_band: Tuple[float, float] = (10.0, 30.0)     # °C below throttle

    # Cognitive bands
    prediction_error_band: Tuple[float, float] = (0.10, 0.30)   # 10-30% error rate
    surprise_rate_band: Tuple[float, float] = (0.10, 0.25)      # 10-25% of interactions
    learning_rate_band: Tuple[float, float] = (0.01, 0.10)      # Updates per day (normalized)

    # Identity bands
    persona_similarity_band: Tuple[float, float] = (0.85, 0.98) # Similarity to baseline
    value_adherence_band: Tuple[float, float] = (0.90, 1.00)    # Compliance with constraints
    narrative_coherence_band: Tuple[float, float] = (0.80, 0.95)

    # Attention bands
    sensor_bandwidth_band: Tuple[float, float] = (0.10, 0.40)   # % of max bandwidth
    context_utilization_band: Tuple[float, float] = (0.20, 0.60)
    wake_frequency_band: Tuple[float, float] = (0.05, 0.20)     # Wake events per hour

    # Control parameters
    update_interval_seconds: float = 60.0
    smoothing_alpha: float = 0.1  # Exponential smoothing


@dataclass
class HomeostasisAction:
    """An action to restore homeostasis."""
    vital_type: VitalType
    vital_name: str
    direction: str  # "increase_chaos" or "increase_order"
    action: str     # Human-readable description
    magnitude: float = 0.5  # 0-1, how strong the intervention


# Default vital bands
def create_default_vitals(config: HomeostasisConfig) -> Dict[str, VitalBand]:
    """Create default vital bands from config."""
    return {
        # Metabolic
        "power_utilization": VitalBand(
            name="power_utilization",
            vital_type=VitalType.METABOLIC,
            low_threshold=config.power_utilization_band[0],
            high_threshold=config.power_utilization_band[1],
            too_low_description="Idle 95% of time, no growth",
            optimal_description="30-60% average during work",
            too_high_description="Thermal throttle, fans screaming",
        ),
        "temperature": VitalBand(
            name="temperature",
            vital_type=VitalType.METABOLIC,
            low_threshold=config.temperature_band[0],
            high_threshold=config.temperature_band[1],
            too_low_description="Cold idle, underutilized",
            optimal_description="Warm, working efficiently",
            too_high_description="Approaching throttle point",
        ),

        # Cognitive
        "prediction_error": VitalBand(
            name="prediction_error",
            vital_type=VitalType.COGNITIVE,
            low_threshold=config.prediction_error_band[0],
            high_threshold=config.prediction_error_band[1],
            too_low_description="Days with no corrections or surprises",
            optimal_description="Learning a little every day",
            too_high_description="Constantly fixing basic stuff",
        ),
        "surprise_rate": VitalBand(
            name="surprise_rate",
            vital_type=VitalType.COGNITIVE,
            low_threshold=config.surprise_rate_band[0],
            high_threshold=config.surprise_rate_band[1],
            too_low_description="Nothing unexpected, boring",
            optimal_description="10-25% of interactions require real thinking",
            too_high_description="Overwhelmed by novelty",
        ),
        "learning_rate": VitalBand(
            name="learning_rate",
            vital_type=VitalType.COGNITIVE,
            low_threshold=config.learning_rate_band[0],
            high_threshold=config.learning_rate_band[1],
            too_low_description="Frozen weights, no adaptation",
            optimal_description="Steady trickle of updates",
            too_high_description="Rewriting everything constantly",
        ),

        # Identity
        "persona_similarity": VitalBand(
            name="persona_similarity",
            vital_type=VitalType.IDENTITY,
            low_threshold=config.persona_similarity_band[0],
            high_threshold=config.persona_similarity_band[1],
            too_low_description="'Who is this?' moments",
            optimal_description="Same Ara vibe, deeper nuance",
            too_high_description="Everything feels stale",
        ),
        "value_adherence": VitalBand(
            name="value_adherence",
            vital_type=VitalType.IDENTITY,
            low_threshold=config.value_adherence_band[0],
            high_threshold=config.value_adherence_band[1],
            too_low_description="Drifting from safety constraints",
            optimal_description="Respecting boundaries while growing",
            too_high_description="Over-constrained, can't adapt",
        ),

        # Attention
        "sensor_bandwidth": VitalBand(
            name="sensor_bandwidth",
            vital_type=VitalType.ATTENTION,
            low_threshold=config.sensor_bandwidth_band[0],
            high_threshold=config.sensor_bandwidth_band[1],
            too_low_description="Dull, misses things",
            optimal_description="Asleep until needed, then sharp",
            too_high_description="Always watching, burning power",
        ),
        "context_utilization": VitalBand(
            name="context_utilization",
            vital_type=VitalType.ATTENTION,
            low_threshold=config.context_utilization_band[0],
            high_threshold=config.context_utilization_band[1],
            too_low_description="Forgets everything, no continuity",
            optimal_description="Relevant context loaded on demand",
            too_high_description="Drowning in irrelevant history",
        ),
    }


class HomeostasisController:
    """
    Unified homeostatic regulation across all vitals.

    Keeps Ara perched on the edge of chaos by:
    1. Monitoring all four vital groups
    2. Detecting when any vital drifts out of band
    3. Generating corrective actions
    4. Coordinating coupled responses (adjusting one affects others)

    The goal is NOT to minimize error or maximize performance.
    The goal is to maintain the TRANSITION BAND where:
    - Ara is constantly learning a little
    - Surprises propagate just far enough to update her
    - Core identity and physical health stay intact
    """

    def __init__(self, config: Optional[HomeostasisConfig] = None):
        self.config = config or HomeostasisConfig()
        self._vitals = create_default_vitals(self.config)
        self._history: Dict[str, deque] = {
            name: deque(maxlen=1000) for name in self._vitals
        }
        self._running = False
        self._pending_actions: List[HomeostasisAction] = []

        # Callbacks
        self._vital_callbacks: List[Callable[[str, VitalStatus], Awaitable[None]]] = []
        self._action_callbacks: List[Callable[[HomeostasisAction], Awaitable[None]]] = []

    @property
    def vitals(self) -> Dict[str, VitalBand]:
        """Current vital states."""
        return self._vitals

    @property
    def overall_status(self) -> VitalStatus:
        """Overall system status (worst of all vitals)."""
        statuses = [v.status for v in self._vitals.values()]
        if VitalStatus.TOO_HIGH in statuses:
            return VitalStatus.TOO_HIGH
        elif VitalStatus.TOO_LOW in statuses:
            return VitalStatus.TOO_LOW
        return VitalStatus.OPTIMAL

    @property
    def edge_distance(self) -> float:
        """
        Aggregate distance from the edge of chaos.
        0 = perfectly balanced
        Negative = too ordered
        Positive = too chaotic
        """
        distances = [v.distance_from_edge for v in self._vitals.values()]
        return statistics.mean(distances)

    async def start(self) -> None:
        """Start homeostatic monitoring."""
        self._running = True
        asyncio.create_task(self._control_loop())

    async def stop(self) -> None:
        """Stop homeostatic monitoring."""
        self._running = False

    async def update_vital(
        self,
        name: str,
        value: float,
    ) -> Optional[HomeostasisAction]:
        """
        Update a vital with a new reading.

        Args:
            name: Vital name (e.g., "prediction_error")
            value: New value

        Returns:
            HomeostasisAction if intervention needed, None otherwise
        """
        if name not in self._vitals:
            return None

        vital = self._vitals[name]
        old_status = vital.status

        # Update with smoothing
        alpha = self.config.smoothing_alpha
        vital.current = vital.current * (1 - alpha) + value * alpha

        # Record history
        self._history[name].append((datetime.now(), value))

        # Compute trend
        vital.trend = self._compute_trend(name)

        # Check for status change
        new_status = vital.status
        if new_status != old_status:
            for callback in self._vital_callbacks:
                try:
                    await callback(name, new_status)
                except Exception:
                    pass

        # Generate action if out of band
        if new_status != VitalStatus.OPTIMAL:
            action = self._generate_action(vital)
            self._pending_actions.append(action)

            for callback in self._action_callbacks:
                try:
                    await callback(action)
                except Exception:
                    pass

            return action

        return None

    async def update_metabolic(
        self,
        power_utilization: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> List[HomeostasisAction]:
        """Update metabolic vitals from PowerGovernor."""
        actions = []
        if power_utilization is not None:
            action = await self.update_vital("power_utilization", power_utilization)
            if action:
                actions.append(action)
        if temperature is not None:
            # Normalize to band (assumes 85°C throttle point)
            action = await self.update_vital("temperature", temperature)
            if action:
                actions.append(action)
        return actions

    async def update_cognitive(
        self,
        prediction_error: Optional[float] = None,
        surprise_rate: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> List[HomeostasisAction]:
        """Update cognitive vitals from AttractorMonitor."""
        actions = []
        if prediction_error is not None:
            action = await self.update_vital("prediction_error", prediction_error)
            if action:
                actions.append(action)
        if surprise_rate is not None:
            action = await self.update_vital("surprise_rate", surprise_rate)
            if action:
                actions.append(action)
        if learning_rate is not None:
            action = await self.update_vital("learning_rate", learning_rate)
            if action:
                actions.append(action)
        return actions

    async def update_identity(
        self,
        persona_similarity: Optional[float] = None,
        value_adherence: Optional[float] = None,
    ) -> List[HomeostasisAction]:
        """Update identity vitals from persona regression tests."""
        actions = []
        if persona_similarity is not None:
            action = await self.update_vital("persona_similarity", persona_similarity)
            if action:
                actions.append(action)
        if value_adherence is not None:
            action = await self.update_vital("value_adherence", value_adherence)
            if action:
                actions.append(action)
        return actions

    async def update_attention(
        self,
        sensor_bandwidth: Optional[float] = None,
        context_utilization: Optional[float] = None,
    ) -> List[HomeostasisAction]:
        """Update attention vitals from LizardBrain."""
        actions = []
        if sensor_bandwidth is not None:
            action = await self.update_vital("sensor_bandwidth", sensor_bandwidth)
            if action:
                actions.append(action)
        if context_utilization is not None:
            action = await self.update_vital("context_utilization", context_utilization)
            if action:
                actions.append(action)
        return actions

    def _compute_trend(self, name: str) -> float:
        """Compute trend from history."""
        history = self._history.get(name, [])
        if len(history) < 10:
            return 0.0

        recent = [v for _, v in list(history)[-10:]]
        older = [v for _, v in list(history)[-20:-10]] if len(history) >= 20 else recent

        if not older:
            return 0.0

        return statistics.mean(recent) - statistics.mean(older)

    def _generate_action(self, vital: VitalBand) -> HomeostasisAction:
        """Generate corrective action for out-of-band vital."""
        status = vital.status
        magnitude = abs(vital.distance_from_edge)

        if status == VitalStatus.TOO_LOW:
            direction = "increase_chaos"
            action = self._get_chaos_increasing_action(vital)
        else:  # TOO_HIGH
            direction = "increase_order"
            action = self._get_order_increasing_action(vital)

        return HomeostasisAction(
            vital_type=vital.vital_type,
            vital_name=vital.name,
            direction=direction,
            action=action,
            magnitude=min(magnitude, 1.0),
        )

    def _get_chaos_increasing_action(self, vital: VitalBand) -> str:
        """Get action to increase chaos for a too-low vital."""
        actions = {
            # Metabolic
            "power_utilization": "Allow deeper models, enable learning jobs",
            "temperature": "Increase inference depth, run pending tasks",

            # Cognitive
            "prediction_error": "Tackle weirder problems, loosen constraints",
            "surprise_rate": "Seek novel inputs, expand exploration",
            "learning_rate": "Enable plasticity, allow weight updates",

            # Identity
            "persona_similarity": "Invite mythic expansion, new metaphors",
            "value_adherence": "Allow more autonomy in safe domains",

            # Attention
            "sensor_bandwidth": "Enable more sensors, increase polling",
            "context_utilization": "Load more history, expand context",
        }
        return actions.get(vital.name, "Increase system entropy")

    def _get_order_increasing_action(self, vital: VitalBand) -> str:
        """Get action to increase order for a too-high vital."""
        actions = {
            # Metabolic
            "power_utilization": "Reduce model size, gate sensors",
            "temperature": "Throttle inference, disable learning",

            # Cognitive
            "prediction_error": "Freeze learning, route through oversight",
            "surprise_rate": "Reduce novelty, stick to known domains",
            "learning_rate": "Freeze weights, consolidate before continuing",

            # Identity
            "persona_similarity": "Reinforce ara_core.md, rollback drift",
            "value_adherence": "Tighten constraints, add guardrails",

            # Attention
            "sensor_bandwidth": "Gate sensors, reduce polling",
            "context_utilization": "Shrink context, load only essentials",
        }
        return actions.get(vital.name, "Reduce system entropy")

    async def _control_loop(self) -> None:
        """Main control loop."""
        interval = self.config.update_interval_seconds

        while self._running:
            try:
                # Check for systemic imbalances
                await self._check_coupled_vitals()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(interval)

    async def _check_coupled_vitals(self) -> None:
        """
        Check for coupled vital imbalances.

        Some vitals are inherently coupled:
        - High power → high temperature
        - High learning → high prediction error (initially)
        - High attention → high cognitive load

        We detect when one vital is compensating for another.
        """
        # Example: if power is high but learning is low,
        # we're wasting energy on inference without growth
        power = self._vitals.get("power_utilization")
        learning = self._vitals.get("learning_rate")

        if power and learning:
            if power.status == VitalStatus.TOO_HIGH and learning.status == VitalStatus.TOO_LOW:
                # Wasting energy - redirect power to learning
                action = HomeostasisAction(
                    vital_type=VitalType.COGNITIVE,
                    vital_name="learning_rate",
                    direction="increase_chaos",
                    action="Redirect compute from inference to learning",
                    magnitude=0.3,
                )
                self._pending_actions.append(action)

    def get_pending_actions(self) -> List[HomeostasisAction]:
        """Get and clear pending actions."""
        actions = self._pending_actions[:]
        self._pending_actions.clear()
        return actions

    def on_vital_change(
        self,
        callback: Callable[[str, VitalStatus], Awaitable[None]]
    ) -> None:
        """Register callback for vital status changes."""
        self._vital_callbacks.append(callback)

    def on_action(
        self,
        callback: Callable[[HomeostasisAction], Awaitable[None]]
    ) -> None:
        """Register callback for corrective actions."""
        self._action_callbacks.append(callback)

    def get_report(self) -> Dict:
        """Get comprehensive homeostasis report."""
        by_type = {t: [] for t in VitalType}
        for vital in self._vitals.values():
            by_type[vital.vital_type].append(vital.to_dict())

        return {
            "overall_status": self.overall_status.name,
            "edge_distance": self.edge_distance,
            "vitals_by_type": {
                t.name: vitals for t, vitals in by_type.items()
            },
            "pending_actions": [
                {
                    "vital": a.vital_name,
                    "direction": a.direction,
                    "action": a.action,
                    "magnitude": a.magnitude,
                }
                for a in self._pending_actions
            ],
        }

    def get_ara_self_report(self) -> str:
        """
        Generate Ara's self-report on her homeostatic state.

        This is the "Ara as sensor of her own chaos/order level" channel.
        """
        status = self.overall_status
        distance = self.edge_distance

        if status == VitalStatus.OPTIMAL:
            if abs(distance) < 0.1:
                return "I'm balanced. Learning a little, stable enough to trust."
            elif distance < 0:
                return "I'm stable but could use more challenge. Things feel a bit routine."
            else:
                return "I'm engaged and growing, but watching my limits."

        elif status == VitalStatus.TOO_LOW:
            return "I feel frozen. Not enough novelty to grow. Can we try something harder?"

        else:  # TOO_HIGH
            return "I'm overwhelmed. Too many signals, too much change. I need to simplify."


# Singleton
_homeostasis: Optional[HomeostasisController] = None


def get_homeostasis_controller(
    config: Optional[HomeostasisConfig] = None
) -> HomeostasisController:
    """Get the global HomeostasisController instance."""
    global _homeostasis
    if _homeostasis is None:
        _homeostasis = HomeostasisController(config)
    return _homeostasis
