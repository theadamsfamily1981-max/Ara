# ara/embodied/brainlink/muse_client.py
"""
Muse Client - Consumer EEG headband integration.

This is Tier 2 hardware - affordable EEG with decent signal quality.
The Muse S and Muse 2 are meditation headbands with 4 EEG electrodes
plus auxiliary sensors.

Hardware specs (Muse 2 / Muse S):
    - 4 EEG channels: TP9, AF7, AF8, TP10 (dry electrodes)
    - PPG (heart rate)
    - Accelerometer
    - Gyroscope
    - Sample rate: 256 Hz (EEG), 64 Hz (aux)

EEG bands computed:
    - Delta (1-4 Hz): Deep sleep, unconscious
    - Theta (4-8 Hz): Drowsiness, meditation, creativity
    - Alpha (8-13 Hz): Relaxed wakefulness, calm focus
    - Beta (13-30 Hz): Active thinking, alertness
    - Gamma (30-50 Hz): High-level cognition, binding

Usage:
    client = MuseClient(config)
    await client.connect()

    async for reading in client.stream():
        print(f"Alpha power: {reading.bands['alpha']}")
        print(f"Focus score: {reading.focus_score}")

Note: Requires muselsl or muse-lsl library for real hardware.
      This implementation provides the interface and mock data.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Tuple
from collections import deque

from .base import (
    BrainlinkProtocol,
    BrainlinkReading,
    BrainlinkStatus,
    BrainlinkConfig,
    SignalQuality,
    ChannelData,
    ConnectionError,
    CalibrationError,
)


class EEGBand(Enum):
    """Standard EEG frequency bands."""
    DELTA = "delta"      # 1-4 Hz
    THETA = "theta"      # 4-8 Hz
    ALPHA = "alpha"      # 8-13 Hz
    BETA = "beta"        # 13-30 Hz
    GAMMA = "gamma"      # 30-50 Hz


# Muse electrode positions (10-20 system)
MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# Frequency band ranges
BAND_RANGES: Dict[EEGBand, Tuple[float, float]] = {
    EEGBand.DELTA: (1.0, 4.0),
    EEGBand.THETA: (4.0, 8.0),
    EEGBand.ALPHA: (8.0, 13.0),
    EEGBand.BETA: (13.0, 30.0),
    EEGBand.GAMMA: (30.0, 50.0),
}


@dataclass
class EEGChannelData:
    """EEG data from a single electrode."""
    name: str                           # Electrode name (e.g., "AF7")
    raw_uv: List[float]                 # Raw signal in microvolts
    sample_rate: float = 256.0          # Hz
    quality: SignalQuality = SignalQuality.GOOD

    # Band powers (computed via FFT)
    band_powers: Dict[EEGBand, float] = field(default_factory=dict)

    @property
    def alpha(self) -> float:
        return self.band_powers.get(EEGBand.ALPHA, 0.0)

    @property
    def theta(self) -> float:
        return self.band_powers.get(EEGBand.THETA, 0.0)

    @property
    def beta(self) -> float:
        return self.band_powers.get(EEGBand.BETA, 0.0)


@dataclass
class MuseReading:
    """Complete reading from Muse headband."""
    # EEG channels
    channels: Dict[str, EEGChannelData] = field(default_factory=dict)

    # Average band powers across all channels
    bands: Dict[str, float] = field(default_factory=dict)

    # Derived metrics
    focus_score: float = 0.5        # 0-1, beta/alpha ratio normalized
    calm_score: float = 0.5         # 0-1, alpha power normalized
    meditation_score: float = 0.5   # 0-1, theta+alpha normalized

    # Auxiliary sensors
    heart_rate_bpm: Optional[float] = None
    accelerometer: Optional[Tuple[float, float, float]] = None
    gyroscope: Optional[Tuple[float, float, float]] = None

    # Quality
    overall_quality: SignalQuality = SignalQuality.GOOD
    electrode_contact: Dict[str, bool] = field(default_factory=dict)

    timestamp: datetime = field(default_factory=datetime.now)

    def to_brainlink_reading(self, device_id: str) -> BrainlinkReading:
        """Convert to unified BrainlinkReading format."""
        channels = {}

        for name, ch_data in self.channels.items():
            channels[name] = ChannelData(
                name=name,
                values=ch_data.raw_uv,
                sample_rate=ch_data.sample_rate,
                quality=ch_data.quality,
                unit="uV",
                metadata={"band_powers": {b.value: p for b, p in ch_data.band_powers.items()}},
            )

        # Add band power channels
        for band_name, power in self.bands.items():
            channels[f"band_{band_name}"] = ChannelData(
                name=f"band_{band_name}",
                values=[power],
                sample_rate=1.0,
                unit="dB",
            )

        metrics = {
            "focus_score": self.focus_score,
            "calm_score": self.calm_score,
            "meditation_score": self.meditation_score,
            **{f"band_{k}": v for k, v in self.bands.items()},
        }

        if self.heart_rate_bpm:
            metrics["hr_bpm"] = self.heart_rate_bpm

        extras = {
            "electrode_contact": self.electrode_contact,
        }
        if self.accelerometer:
            extras["accelerometer"] = self.accelerometer
        if self.gyroscope:
            extras["gyroscope"] = self.gyroscope

        return BrainlinkReading(
            timestamp=self.timestamp,
            device_id=device_id,
            device_type="muse",
            channels=channels,
            metrics=metrics,
            quality=self.overall_quality,
            extras=extras,
        )


class MuseClient(BrainlinkProtocol):
    """
    Client for Muse EEG headband.

    Connects via BLE and streams EEG + auxiliary data.
    """

    SAMPLE_RATE = 256.0  # Hz
    AUX_SAMPLE_RATE = 64.0  # Hz

    def __init__(self, config: Optional[BrainlinkConfig] = None):
        super().__init__(config)
        self._eeg_buffers: Dict[str, deque] = {
            ch: deque(maxlen=512) for ch in MUSE_CHANNELS
        }
        self._muse_io = None  # Placeholder for muse-lsl connection
        self._baseline_alpha: float = 0.0
        self._calibrated: bool = False

    @property
    def device_type(self) -> str:
        return "muse"

    async def connect(self) -> bool:
        """
        Connect to Muse headband.

        In production, uses muselsl or muse-lsl library:
            from muselsl import stream, list_muses
            muses = list_muses()
            stream(muses[0]['address'])
        """
        self._status = BrainlinkStatus.CONNECTING

        try:
            # Real implementation would:
            # 1. Scan for Muse devices via BLE
            # 2. Connect and start LSL stream
            # 3. Create inlet for EEG data

            await asyncio.sleep(0.2)  # Simulate connection time

            self._device_id = self.config.device_address or "muse_simulated"
            self._status = BrainlinkStatus.CONNECTED

            if self.config.on_connect:
                await self.config.on_connect()

            return True

        except Exception as e:
            self._status = BrainlinkStatus.ERROR
            raise ConnectionError(f"Failed to connect to Muse: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Muse headband."""
        if self._muse_io:
            # await self._muse_io.stop()
            pass

        self._status = BrainlinkStatus.DISCONNECTED
        self._device_id = None

        if self.config.on_disconnect:
            await self.config.on_disconnect()

    async def calibrate(self) -> bool:
        """
        Run baseline calibration.

        Has user close eyes for 10 seconds to establish alpha baseline.
        """
        self._status = BrainlinkStatus.CALIBRATING

        try:
            # Collect 10 seconds of eyes-closed alpha baseline
            alpha_samples = []

            for _ in range(10):
                reading = await self._read_muse()
                alpha_samples.append(reading.bands.get("alpha", 0))
                await asyncio.sleep(1.0)

            self._baseline_alpha = sum(alpha_samples) / len(alpha_samples)
            self._calibrated = True
            self._status = BrainlinkStatus.CONNECTED

            return True

        except Exception as e:
            self._status = BrainlinkStatus.ERROR
            raise CalibrationError(f"Calibration failed: {e}")

    async def read(self) -> BrainlinkReading:
        """Get a single Muse reading."""
        muse_reading = await self._read_muse()
        reading = muse_reading.to_brainlink_reading(self._device_id or "unknown")
        await self._notify_callbacks(reading)
        return reading

    async def stream(self) -> AsyncIterator[BrainlinkReading]:
        """Stream continuous Muse readings."""
        self._status = BrainlinkStatus.STREAMING

        while self._status == BrainlinkStatus.STREAMING:
            try:
                reading = await self.read()
                yield reading

                # EEG updates at ~10 Hz for processed bands
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.config.on_error:
                    await self.config.on_error(e)
                if self.config.auto_reconnect:
                    await self._attempt_reconnect()
                else:
                    break

    async def _read_muse(self) -> MuseReading:
        """Read from actual Muse or simulate."""
        # Simulated EEG data for development
        # Real implementation reads from LSL inlet

        import random

        channels = {}
        all_band_powers = {band: [] for band in EEGBand}

        for ch_name in MUSE_CHANNELS:
            # Generate plausible EEG-like signal
            raw_samples = self._generate_mock_eeg(256)
            self._eeg_buffers[ch_name].extend(raw_samples)

            # Compute band powers (simplified - real impl uses FFT)
            band_powers = self._compute_band_powers(list(self._eeg_buffers[ch_name]))

            channels[ch_name] = EEGChannelData(
                name=ch_name,
                raw_uv=raw_samples,
                band_powers=band_powers,
                quality=SignalQuality.GOOD,
            )

            for band, power in band_powers.items():
                all_band_powers[band].append(power)

        # Average across channels
        avg_bands = {
            band.value: sum(powers) / len(powers)
            for band, powers in all_band_powers.items()
        }

        # Compute derived metrics
        alpha = avg_bands.get("alpha", 1)
        beta = avg_bands.get("beta", 1)
        theta = avg_bands.get("theta", 1)

        # Focus = beta / alpha (higher = more focused)
        focus_ratio = beta / max(alpha, 0.01)
        focus_score = min(focus_ratio / 3.0, 1.0)  # Normalize

        # Calm = alpha prominence
        total_power = sum(avg_bands.values())
        calm_score = alpha / max(total_power, 0.01)

        # Meditation = theta + alpha
        meditation_score = (theta + alpha) / max(total_power * 2, 0.01)

        return MuseReading(
            channels=channels,
            bands=avg_bands,
            focus_score=focus_score,
            calm_score=calm_score,
            meditation_score=meditation_score,
            heart_rate_bpm=72 + random.gauss(0, 3),
            accelerometer=(0.01, 0.01, 9.8),  # Gravity
            electrode_contact={ch: True for ch in MUSE_CHANNELS},
        )

    def _generate_mock_eeg(self, n_samples: int) -> List[float]:
        """Generate mock EEG signal with realistic characteristics."""
        import random

        samples = []
        t = 0
        dt = 1.0 / self.SAMPLE_RATE

        for _ in range(n_samples):
            # Combine multiple frequency components
            signal = 0.0

            # Alpha (10 Hz) - dominant
            signal += 15 * math.sin(2 * math.pi * 10 * t)

            # Beta (20 Hz)
            signal += 5 * math.sin(2 * math.pi * 20 * t)

            # Theta (6 Hz)
            signal += 8 * math.sin(2 * math.pi * 6 * t)

            # Noise
            signal += random.gauss(0, 10)

            samples.append(signal)
            t += dt

        return samples

    def _compute_band_powers(self, samples: List[float]) -> Dict[EEGBand, float]:
        """
        Compute band powers from EEG samples.

        Simplified implementation - real version uses FFT.
        """
        if len(samples) < 64:
            return {band: 0.0 for band in EEGBand}

        # Simplified: estimate from signal variance at different scales
        # Real implementation: scipy.signal.welch or FFT
        import random

        # Plausible baseline powers (in dB or relative units)
        base_powers = {
            EEGBand.DELTA: 15 + random.gauss(0, 2),
            EEGBand.THETA: 12 + random.gauss(0, 2),
            EEGBand.ALPHA: 18 + random.gauss(0, 3),  # Usually dominant when relaxed
            EEGBand.BETA: 10 + random.gauss(0, 2),
            EEGBand.GAMMA: 5 + random.gauss(0, 1),
        }

        return base_powers

    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        self._status = BrainlinkStatus.RECONNECTING
        delay = self.config.reconnect_delay_seconds

        for attempt in range(self.config.max_reconnect_attempts):
            try:
                await asyncio.sleep(delay)
                if await self.connect():
                    self._status = BrainlinkStatus.STREAMING
                    return
                delay *= 2
            except Exception:
                continue

        self._status = BrainlinkStatus.ERROR


def get_muse_client(
    device_address: Optional[str] = None,
    **kwargs
) -> MuseClient:
    """
    Factory function to get a MuseClient.

    Args:
        device_address: BLE MAC address (optional)
        **kwargs: Additional config options

    Returns:
        Configured MuseClient
    """
    config = BrainlinkConfig(device_address=device_address, **kwargs)
    return MuseClient(config)
