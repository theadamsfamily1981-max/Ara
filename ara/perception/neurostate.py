# ara/perception/neurostate.py
"""
Neurostate - State extraction from brainlink signals.

This is the middle layer between hardware (brainlink) and behavior
(intent_modulator):

    brainlink (hardware) → neurostate (state extraction) → intent_modulator (behavior)

NeuroState aggregates raw signals into meaningful cognitive/physiological
states that can influence Ara's behavior:

    - Attention level (are they focused or distracted?)
    - Stress level (fight/flight or parasympathetic?)
    - Engagement (are they actively processing or passive?)
    - Fatigue (cognitive resource depletion?)
    - Emotional valence (positive/negative affect?)

These states feed into the intent modulator which can:
    - Adjust response verbosity based on attention
    - Slow down when user is stressed
    - Offer breaks when fatigue detected
    - Adapt tone to emotional state

Usage:
    from ara.perception.neurostate import NeuroState, get_neurostate

    neuro = get_neurostate()
    await neuro.connect_brainlink("muse")

    state = await neuro.get_current_state()
    print(f"Attention: {state.attention}")
    print(f"Stress: {state.stress}")

    # Or stream continuously
    async for state in neuro.stream_states():
        adapt_response(state)
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
from collections import deque
import statistics

# Import brainlink
from ara.embodied.brainlink import (
    BrainlinkProtocol,
    BrainlinkReading,
    BrainlinkStatus,
    get_brainlink,
)


class CognitiveState(Enum):
    """High-level cognitive state categories."""
    FOCUSED = auto()      # Deep work, high concentration
    ENGAGED = auto()      # Active but not deep focus
    RELAXED = auto()      # Calm, receptive
    DROWSY = auto()       # Low alertness, fatigue
    STRESSED = auto()     # Elevated arousal, fight/flight
    FLOW = auto()         # Optimal performance state
    UNKNOWN = auto()      # Insufficient data


class EmotionalValence(Enum):
    """Emotional tone direction."""
    POSITIVE = auto()
    NEUTRAL = auto()
    NEGATIVE = auto()


@dataclass
class NeuroStateReading:
    """
    Aggregated cognitive/physiological state.

    This is the main output of the neurostate system - a snapshot
    of the user's current mental state that Ara can respond to.
    """
    # Core dimensions (0-1 scales)
    attention: float = 0.5       # How focused is the user?
    stress: float = 0.5          # Sympathetic activation level
    engagement: float = 0.5      # Active processing vs passive
    fatigue: float = 0.5         # Cognitive resource depletion
    arousal: float = 0.5         # Overall activation level
    valence: float = 0.5         # Emotional positivity (0=neg, 1=pos)

    # Derived states
    cognitive_state: CognitiveState = CognitiveState.UNKNOWN
    emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL

    # Confidence in the reading (based on signal quality)
    confidence: float = 0.0

    # Trends (change over time)
    attention_trend: float = 0.0    # Positive = improving
    stress_trend: float = 0.0       # Positive = increasing
    fatigue_trend: float = 0.0      # Positive = increasing

    # Raw source metrics (for debugging)
    source_metrics: Dict[str, float] = field(default_factory=dict)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    source_device: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if state has sufficient confidence."""
        return self.confidence >= 0.3

    @property
    def needs_break(self) -> bool:
        """Suggest a break based on fatigue and stress."""
        return self.fatigue > 0.7 or (self.stress > 0.7 and self.fatigue > 0.5)

    @property
    def is_receptive(self) -> bool:
        """Is user in a good state to receive information?"""
        return (
            self.attention > 0.4
            and self.stress < 0.6
            and self.fatigue < 0.7
        )

    @property
    def optimal_response_length(self) -> str:
        """Suggest response length based on state."""
        if self.attention < 0.3 or self.fatigue > 0.7:
            return "brief"
        elif self.attention > 0.7 and self.engagement > 0.6:
            return "detailed"
        else:
            return "moderate"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "attention": self.attention,
            "stress": self.stress,
            "engagement": self.engagement,
            "fatigue": self.fatigue,
            "arousal": self.arousal,
            "valence": self.valence,
            "cognitive_state": self.cognitive_state.name,
            "emotional_valence": self.emotional_valence.name,
            "confidence": self.confidence,
            "attention_trend": self.attention_trend,
            "stress_trend": self.stress_trend,
            "fatigue_trend": self.fatigue_trend,
            "timestamp": self.timestamp.isoformat(),
            "source_device": self.source_device,
            "is_valid": self.is_valid,
            "needs_break": self.needs_break,
            "is_receptive": self.is_receptive,
            "optimal_response_length": self.optimal_response_length,
        }


@dataclass
class NeuroStateConfig:
    """Configuration for NeuroState."""
    # Time windows for analysis
    short_window_seconds: float = 10.0   # For immediate state
    long_window_seconds: float = 300.0   # For trends (5 min)

    # Thresholds for state classification
    focus_threshold: float = 0.6
    stress_threshold: float = 0.6
    fatigue_threshold: float = 0.7

    # Update rate
    update_interval_seconds: float = 1.0

    # Callbacks
    on_state_change: Optional[Callable[[NeuroStateReading], Awaitable[None]]] = None
    on_threshold_crossed: Optional[Callable[[str, float], Awaitable[None]]] = None


class NeuroState:
    """
    Cognitive/physiological state aggregator.

    Connects to brainlink hardware and extracts meaningful states
    from raw signals.
    """

    def __init__(self, config: Optional[NeuroStateConfig] = None):
        self.config = config or NeuroStateConfig()
        self._brainlink: Optional[BrainlinkProtocol] = None
        self._reading_history: deque = deque(maxlen=1000)  # ~15 min at 1Hz
        self._last_state: Optional[NeuroStateReading] = None
        self._running: bool = False
        self._callbacks: List[Callable[[NeuroStateReading], Awaitable[None]]] = []

    @property
    def is_connected(self) -> bool:
        """Check if brainlink is connected."""
        return self._brainlink is not None and self._brainlink.is_connected

    @property
    def last_state(self) -> Optional[NeuroStateReading]:
        """Get most recent state reading."""
        return self._last_state

    async def connect_brainlink(
        self,
        backend: str = "mock",
        **kwargs
    ) -> bool:
        """
        Connect to a brainlink backend.

        Args:
            backend: One of "mock", "physio", "muse", "openbci"
            **kwargs: Additional config for brainlink

        Returns:
            True if connected successfully
        """
        self._brainlink = get_brainlink(backend, **kwargs)
        return await self._brainlink.connect()

    async def disconnect(self) -> None:
        """Disconnect from brainlink."""
        self._running = False
        if self._brainlink:
            await self._brainlink.disconnect()
            self._brainlink = None

    async def get_current_state(self) -> NeuroStateReading:
        """
        Get the current aggregated neurostate.

        This is the main API - call this to get a snapshot of
        the user's current cognitive/physiological state.

        Returns:
            NeuroStateReading with current state
        """
        if not self.is_connected:
            return NeuroStateReading(
                cognitive_state=CognitiveState.UNKNOWN,
                confidence=0.0,
            )

        # Get latest brainlink reading
        reading = await self._brainlink.read()
        self._reading_history.append(reading)

        # Aggregate into neurostate
        state = self._aggregate_state()
        self._last_state = state

        return state

    async def stream_states(self) -> AsyncIterator[NeuroStateReading]:
        """
        Stream continuous neurostate updates.

        Yields:
            NeuroStateReading at configured interval
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to brainlink")

        self._running = True
        interval = self.config.update_interval_seconds

        async for reading in self._brainlink.stream():
            if not self._running:
                break

            self._reading_history.append(reading)
            state = self._aggregate_state()

            # Check for state changes and thresholds
            await self._check_callbacks(state)

            self._last_state = state
            yield state

            await asyncio.sleep(interval)

    def subscribe(
        self,
        callback: Callable[[NeuroStateReading], Awaitable[None]]
    ) -> None:
        """Subscribe to state updates."""
        self._callbacks.append(callback)

    def unsubscribe(
        self,
        callback: Callable[[NeuroStateReading], Awaitable[None]]
    ) -> None:
        """Unsubscribe from state updates."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _aggregate_state(self) -> NeuroStateReading:
        """Aggregate raw readings into a NeuroStateReading."""
        if not self._reading_history:
            return NeuroStateReading(confidence=0.0)

        # Get recent readings
        now = datetime.now()
        short_window = timedelta(seconds=self.config.short_window_seconds)
        long_window = timedelta(seconds=self.config.long_window_seconds)

        recent = [
            r for r in self._reading_history
            if now - r.timestamp < short_window
        ]
        historical = [
            r for r in self._reading_history
            if now - r.timestamp < long_window
        ]

        if not recent:
            return NeuroStateReading(confidence=0.0)

        # Extract metrics from recent readings
        metrics = self._extract_metrics(recent)
        historical_metrics = self._extract_metrics(historical) if len(historical) > 10 else metrics

        # Compute state dimensions
        attention = self._compute_attention(metrics)
        stress = self._compute_stress(metrics)
        engagement = self._compute_engagement(metrics)
        fatigue = self._compute_fatigue(metrics, historical_metrics)
        arousal = self._compute_arousal(metrics)
        valence = self._compute_valence(metrics)

        # Compute trends
        attention_trend = self._compute_trend("focus_score", recent, historical)
        stress_trend = self._compute_trend("stress_index", recent, historical)
        fatigue_trend = fatigue - 0.5  # Simplified

        # Classify cognitive state
        cognitive_state = self._classify_cognitive_state(
            attention, stress, engagement, fatigue, arousal
        )

        # Classify emotional valence
        emotional_valence = self._classify_valence(valence)

        # Confidence based on signal quality and data amount
        confidence = self._compute_confidence(recent)

        return NeuroStateReading(
            attention=attention,
            stress=stress,
            engagement=engagement,
            fatigue=fatigue,
            arousal=arousal,
            valence=valence,
            cognitive_state=cognitive_state,
            emotional_valence=emotional_valence,
            confidence=confidence,
            attention_trend=attention_trend,
            stress_trend=stress_trend,
            fatigue_trend=fatigue_trend,
            source_metrics=metrics,
            source_device=recent[-1].device_type if recent else "",
        )

    def _extract_metrics(self, readings: List[BrainlinkReading]) -> Dict[str, float]:
        """Extract averaged metrics from readings."""
        if not readings:
            return {}

        # Collect all metrics
        all_metrics: Dict[str, List[float]] = {}
        for reading in readings:
            for key, value in reading.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        # Average them
        return {
            key: statistics.mean(values)
            for key, values in all_metrics.items()
            if values
        }

    def _compute_attention(self, metrics: Dict[str, float]) -> float:
        """Compute attention level from metrics."""
        # Primary: focus_score from EEG (beta/alpha)
        if "focus_score" in metrics:
            return min(max(metrics["focus_score"], 0), 1)

        # Fallback: inverse of stress
        if "stress_index" in metrics:
            return 1.0 - metrics["stress_index"]

        return 0.5

    def _compute_stress(self, metrics: Dict[str, float]) -> float:
        """Compute stress level from metrics."""
        stress_indicators = []

        # HRV-based stress
        if "stress_index" in metrics:
            stress_indicators.append(metrics["stress_index"])

        # GSR-based arousal (high GSR = high stress)
        if "arousal_level" in metrics:
            stress_indicators.append(metrics["arousal_level"])

        # HR elevation
        if "hr_bpm" in metrics:
            hr = metrics["hr_bpm"]
            # Normalize: 60-100 bpm normal, higher = more stress
            hr_stress = min(max((hr - 60) / 60, 0), 1)
            stress_indicators.append(hr_stress)

        if stress_indicators:
            return statistics.mean(stress_indicators)
        return 0.5

    def _compute_engagement(self, metrics: Dict[str, float]) -> float:
        """Compute engagement level from metrics."""
        engagement_indicators = []

        # Beta power indicates active processing
        if "band_beta" in metrics:
            beta = metrics["band_beta"]
            engagement_indicators.append(min(beta / 20, 1))

        # Focus score
        if "focus_score" in metrics:
            engagement_indicators.append(metrics["focus_score"])

        if engagement_indicators:
            return statistics.mean(engagement_indicators)
        return 0.5

    def _compute_fatigue(
        self,
        current: Dict[str, float],
        historical: Dict[str, float]
    ) -> float:
        """Compute fatigue level from metrics and trends."""
        fatigue_indicators = []

        # Theta increase indicates fatigue
        if "band_theta" in current:
            theta = current["band_theta"]
            fatigue_indicators.append(min(theta / 25, 1))

        # HRV decrease indicates fatigue
        if "hrv_rmssd" in current and "hrv_rmssd" in historical:
            current_hrv = current["hrv_rmssd"]
            baseline_hrv = historical["hrv_rmssd"]
            if baseline_hrv > 0:
                hrv_ratio = current_hrv / baseline_hrv
                fatigue_indicators.append(1.0 - min(hrv_ratio, 1))

        # Attention decrease over time
        if "focus_score" in current and "focus_score" in historical:
            focus_drop = historical["focus_score"] - current["focus_score"]
            if focus_drop > 0:
                fatigue_indicators.append(min(focus_drop * 2, 1))

        if fatigue_indicators:
            return statistics.mean(fatigue_indicators)
        return 0.5

    def _compute_arousal(self, metrics: Dict[str, float]) -> float:
        """Compute overall arousal level."""
        arousal_indicators = []

        if "arousal_level" in metrics:
            arousal_indicators.append(metrics["arousal_level"])

        # HR-based
        if "hr_bpm" in metrics:
            hr = metrics["hr_bpm"]
            hr_arousal = min(max((hr - 50) / 80, 0), 1)
            arousal_indicators.append(hr_arousal)

        # Beta power
        if "band_beta" in metrics:
            arousal_indicators.append(min(metrics["band_beta"] / 20, 1))

        if arousal_indicators:
            return statistics.mean(arousal_indicators)
        return 0.5

    def _compute_valence(self, metrics: Dict[str, float]) -> float:
        """Compute emotional valence (positive/negative)."""
        # Simplified: based on alpha asymmetry and HRV
        # Higher alpha = more positive
        # Higher HRV = more positive

        valence_indicators = []

        if "calm_score" in metrics:
            valence_indicators.append(metrics["calm_score"])

        if "hrv_rmssd" in metrics:
            hrv = metrics["hrv_rmssd"]
            # Higher HRV generally associated with positive affect
            valence_indicators.append(min(hrv / 60, 1))

        # Inverse of stress
        if "stress_index" in metrics:
            valence_indicators.append(1.0 - metrics["stress_index"])

        if valence_indicators:
            return statistics.mean(valence_indicators)
        return 0.5

    def _compute_trend(
        self,
        metric: str,
        recent: List[BrainlinkReading],
        historical: List[BrainlinkReading]
    ) -> float:
        """Compute trend for a metric (positive = increasing)."""
        recent_values = [r.metrics.get(metric, 0) for r in recent if metric in r.metrics]
        historical_values = [r.metrics.get(metric, 0) for r in historical if metric in r.metrics]

        if not recent_values or not historical_values:
            return 0.0

        recent_mean = statistics.mean(recent_values)
        historical_mean = statistics.mean(historical_values)

        if historical_mean == 0:
            return 0.0

        return (recent_mean - historical_mean) / historical_mean

    def _classify_cognitive_state(
        self,
        attention: float,
        stress: float,
        engagement: float,
        fatigue: float,
        arousal: float,
    ) -> CognitiveState:
        """Classify into a discrete cognitive state."""
        # Flow state: high attention, moderate arousal, low stress
        if attention > 0.7 and 0.3 < arousal < 0.7 and stress < 0.4:
            return CognitiveState.FLOW

        # Focused: high attention, high engagement
        if attention > 0.6 and engagement > 0.5:
            return CognitiveState.FOCUSED

        # Stressed: high stress
        if stress > 0.7:
            return CognitiveState.STRESSED

        # Drowsy: high fatigue, low arousal
        if fatigue > 0.6 or arousal < 0.3:
            return CognitiveState.DROWSY

        # Relaxed: low stress, moderate-low arousal
        if stress < 0.4 and arousal < 0.5:
            return CognitiveState.RELAXED

        # Engaged: moderate everything
        if engagement > 0.4:
            return CognitiveState.ENGAGED

        return CognitiveState.UNKNOWN

    def _classify_valence(self, valence: float) -> EmotionalValence:
        """Classify emotional valence."""
        if valence > 0.6:
            return EmotionalValence.POSITIVE
        elif valence < 0.4:
            return EmotionalValence.NEGATIVE
        return EmotionalValence.NEUTRAL

    def _compute_confidence(self, readings: List[BrainlinkReading]) -> float:
        """Compute confidence based on data quality and quantity."""
        if not readings:
            return 0.0

        # Factor 1: Data quantity (more readings = higher confidence)
        quantity_factor = min(len(readings) / 10, 1.0)

        # Factor 2: Signal quality
        from ara.embodied.brainlink import SignalQuality
        quality_scores = {
            SignalQuality.EXCELLENT: 1.0,
            SignalQuality.GOOD: 0.8,
            SignalQuality.FAIR: 0.6,
            SignalQuality.POOR: 0.3,
            SignalQuality.BAD: 0.1,
            SignalQuality.NO_SIGNAL: 0.0,
        }
        avg_quality = statistics.mean(
            quality_scores.get(r.quality, 0.5) for r in readings
        )

        return quantity_factor * avg_quality

    async def _check_callbacks(self, state: NeuroStateReading) -> None:
        """Check for state changes and threshold crossings."""
        prev = self._last_state

        # Notify general callbacks
        for callback in self._callbacks:
            try:
                await callback(state)
            except Exception:
                pass

        # Check state change
        if prev and state.cognitive_state != prev.cognitive_state:
            if self.config.on_state_change:
                await self.config.on_state_change(state)

        # Check threshold crossings
        if self.config.on_threshold_crossed:
            if state.stress > self.config.stress_threshold:
                if not prev or prev.stress <= self.config.stress_threshold:
                    await self.config.on_threshold_crossed("stress", state.stress)

            if state.fatigue > self.config.fatigue_threshold:
                if not prev or prev.fatigue <= self.config.fatigue_threshold:
                    await self.config.on_threshold_crossed("fatigue", state.fatigue)

    async def __aenter__(self) -> NeuroState:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


# Singleton instance
_neurostate: Optional[NeuroState] = None


def get_neurostate(config: Optional[NeuroStateConfig] = None) -> NeuroState:
    """
    Get the global NeuroState instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        Global NeuroState instance
    """
    global _neurostate
    if _neurostate is None:
        _neurostate = NeuroState(config)
    return _neurostate


async def get_current_state() -> NeuroStateReading:
    """
    Convenience function to get current neurostate.

    Returns:
        Current NeuroStateReading (or default if not connected)
    """
    neuro = get_neurostate()
    if neuro.is_connected:
        return await neuro.get_current_state()
    return NeuroStateReading(confidence=0.0)
