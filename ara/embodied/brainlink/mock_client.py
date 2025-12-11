# ara/embodied/brainlink/mock_client.py
"""
Mock Brainlink Client - Simulated data for development and testing.

This provides realistic mock data without requiring actual hardware.
Useful for:
    - Development and testing
    - UI prototyping
    - Demonstrations
    - CI/CD pipelines

The mock generates physiologically plausible signals that respond
to configurable "scenarios" (e.g., relaxed, focused, stressed).
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import AsyncIterator, Dict, Optional

from .base import (
    BrainlinkProtocol,
    BrainlinkReading,
    BrainlinkStatus,
    BrainlinkConfig,
    SignalQuality,
    ChannelData,
)


class MockScenario(Enum):
    """Predefined physiological scenarios."""
    RELAXED = auto()      # High alpha, low beta, steady HR
    FOCUSED = auto()      # High beta, moderate alpha
    STRESSED = auto()     # Low HRV, high GSR, elevated HR
    DROWSY = auto()       # High theta, low beta
    MEDITATIVE = auto()   # High theta+alpha, low beta
    RANDOM = auto()       # Random fluctuations


@dataclass
class MockConfig:
    """Configuration for mock data generation."""
    scenario: MockScenario = MockScenario.RELAXED
    noise_level: float = 0.1          # 0-1
    drift_rate: float = 0.01          # How fast values drift
    include_eeg: bool = True
    include_physio: bool = True
    sample_rate_hz: float = 10.0      # Output rate


class MockBrainlinkClient(BrainlinkProtocol):
    """
    Mock client that generates simulated brainlink data.

    Useful for development and testing without hardware.
    """

    def __init__(
        self,
        config: Optional[BrainlinkConfig] = None,
        mock_config: Optional[MockConfig] = None,
    ):
        super().__init__(config)
        self.mock_config = mock_config or MockConfig()
        self._state = self._init_state()
        self._time = 0.0

    @property
    def device_type(self) -> str:
        return "mock"

    def _init_state(self) -> Dict[str, float]:
        """Initialize internal state based on scenario."""
        scenario = self.mock_config.scenario

        if scenario == MockScenario.RELAXED:
            return {
                "hr": 65, "hrv_rmssd": 45, "gsr": 3.0,
                "alpha": 20, "beta": 8, "theta": 10, "delta": 12, "gamma": 4,
            }
        elif scenario == MockScenario.FOCUSED:
            return {
                "hr": 72, "hrv_rmssd": 35, "gsr": 5.0,
                "alpha": 12, "beta": 18, "theta": 8, "delta": 10, "gamma": 6,
            }
        elif scenario == MockScenario.STRESSED:
            return {
                "hr": 85, "hrv_rmssd": 20, "gsr": 12.0,
                "alpha": 8, "beta": 15, "theta": 6, "delta": 8, "gamma": 5,
            }
        elif scenario == MockScenario.DROWSY:
            return {
                "hr": 58, "hrv_rmssd": 50, "gsr": 2.0,
                "alpha": 15, "beta": 5, "theta": 18, "delta": 20, "gamma": 2,
            }
        elif scenario == MockScenario.MEDITATIVE:
            return {
                "hr": 60, "hrv_rmssd": 55, "gsr": 2.5,
                "alpha": 22, "beta": 6, "theta": 16, "delta": 14, "gamma": 3,
            }
        else:  # RANDOM
            return {
                "hr": 70, "hrv_rmssd": 40, "gsr": 5.0,
                "alpha": 15, "beta": 12, "theta": 10, "delta": 12, "gamma": 5,
            }

    def set_scenario(self, scenario: MockScenario) -> None:
        """Change the active scenario."""
        self.mock_config.scenario = scenario
        target = self._init_state()
        # Smoothly transition to new scenario
        for key in self._state:
            if key in target:
                self._state[key] = target[key]

    async def connect(self) -> bool:
        """Simulate connection."""
        self._status = BrainlinkStatus.CONNECTING
        await asyncio.sleep(0.1)
        self._device_id = "mock_device_001"
        self._status = BrainlinkStatus.CONNECTED

        if self.config.on_connect:
            await self.config.on_connect()

        return True

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._status = BrainlinkStatus.DISCONNECTED
        self._device_id = None

        if self.config.on_disconnect:
            await self.config.on_disconnect()

    async def read(self) -> BrainlinkReading:
        """Generate a single mock reading."""
        self._update_state()
        reading = self._generate_reading()
        await self._notify_callbacks(reading)
        return reading

    async def stream(self) -> AsyncIterator[BrainlinkReading]:
        """Stream mock readings."""
        self._status = BrainlinkStatus.STREAMING
        interval = 1.0 / self.mock_config.sample_rate_hz

        while self._status == BrainlinkStatus.STREAMING:
            try:
                reading = await self.read()
                yield reading
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    def _update_state(self) -> None:
        """Update internal state with drift and noise."""
        drift = self.mock_config.drift_rate
        noise = self.mock_config.noise_level
        target = self._init_state()

        for key in self._state:
            # Drift toward target
            diff = target[key] - self._state[key]
            self._state[key] += diff * drift

            # Add noise
            self._state[key] += random.gauss(0, noise * abs(target[key]))

            # Clamp to reasonable ranges
            if key == "hr":
                self._state[key] = max(40, min(150, self._state[key]))
            elif key == "hrv_rmssd":
                self._state[key] = max(5, min(100, self._state[key]))
            elif key == "gsr":
                self._state[key] = max(0.5, min(25, self._state[key]))
            elif key in ("alpha", "beta", "theta", "delta", "gamma"):
                self._state[key] = max(1, min(40, self._state[key]))

        self._time += 1.0 / self.mock_config.sample_rate_hz

    def _generate_reading(self) -> BrainlinkReading:
        """Generate reading from current state."""
        channels: Dict[str, ChannelData] = {}
        metrics: Dict[str, float] = {}

        # Physio channels
        if self.mock_config.include_physio:
            channels["HR"] = ChannelData(
                name="HR",
                values=[self._state["hr"]],
                sample_rate=1.0,
                unit="bpm",
            )
            channels["GSR"] = ChannelData(
                name="GSR",
                values=[self._state["gsr"]],
                sample_rate=4.0,
                unit="uS",
            )

            metrics["hr_bpm"] = self._state["hr"]
            metrics["hrv_rmssd"] = self._state["hrv_rmssd"]
            metrics["gsr_conductance"] = self._state["gsr"]

            # Computed stress index
            stress = 1.0 - min(self._state["hrv_rmssd"] / 60, 1.0)
            metrics["stress_index"] = stress

        # EEG channels
        if self.mock_config.include_eeg:
            for band in ["alpha", "beta", "theta", "delta", "gamma"]:
                channels[f"band_{band}"] = ChannelData(
                    name=f"band_{band}",
                    values=[self._state[band]],
                    sample_rate=10.0,
                    unit="dB",
                )
                metrics[f"band_{band}"] = self._state[band]

            # Focus score (beta/alpha)
            alpha = max(self._state["alpha"], 0.01)
            beta = self._state["beta"]
            focus_ratio = beta / alpha
            metrics["focus_score"] = min(focus_ratio / 2.5, 1.0)

            # Calm score (alpha prominence)
            total = sum(self._state[b] for b in ["alpha", "beta", "theta", "delta", "gamma"])
            metrics["calm_score"] = self._state["alpha"] / max(total, 0.01)

            # Meditation score
            metrics["meditation_score"] = (
                self._state["theta"] + self._state["alpha"]
            ) / max(total * 2, 0.01)

        return BrainlinkReading(
            timestamp=datetime.now(),
            device_id=self._device_id or "mock",
            device_type="mock",
            channels=channels,
            metrics=metrics,
            quality=SignalQuality.GOOD,
            extras={"scenario": self.mock_config.scenario.name},
        )


def get_mock_client(
    scenario: MockScenario = MockScenario.RELAXED,
    **kwargs
) -> MockBrainlinkClient:
    """
    Factory function to get a MockBrainlinkClient.

    Args:
        scenario: Initial scenario to simulate
        **kwargs: Additional mock config options

    Returns:
        Configured MockBrainlinkClient
    """
    mock_config = MockConfig(scenario=scenario, **kwargs)
    return MockBrainlinkClient(mock_config=mock_config)
