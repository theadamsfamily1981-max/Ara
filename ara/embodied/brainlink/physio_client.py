# ara/embodied/brainlink/physio_client.py
"""
Physio Client - Heart rate, HRV, and GSR sensor integration.

This is Tier 1 hardware - cheap, reliable, and provides useful biofeedback
without the complexity of EEG.

Supported hardware:
    - Polar H10/H9 chest strap (HR, HRV via BLE)
    - Garmin HRM-Pro (HR, HRV via ANT+)
    - Generic pulse oximeters (HR via BLE)
    - Grove GSR sensor (via Arduino/serial)
    - Empatica E4 (HR, GSR, temp - research grade)

Data provided:
    - Heart Rate (HR): beats per minute
    - Heart Rate Variability (HRV): RMSSD, SDNN, pNN50
    - Galvanic Skin Response (GSR): skin conductance in microsiemens

Usage:
    client = PhysioClient(config)
    await client.connect()

    async for reading in client.stream():
        print(f"HR: {reading.heart_rate.bpm} bpm")
        print(f"HRV RMSSD: {reading.hrv.rmssd_ms} ms")
        print(f"GSR: {reading.gsr.conductance_us} uS")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Any
from collections import deque
import math

from .base import (
    BrainlinkProtocol,
    BrainlinkReading,
    BrainlinkStatus,
    BrainlinkConfig,
    SignalQuality,
    ChannelData,
    ConnectionError,
)


@dataclass
class HeartRateData:
    """Heart rate measurement."""
    bpm: float                          # Beats per minute
    rr_intervals_ms: List[float]        # R-R intervals in milliseconds
    confidence: float = 1.0             # 0-1 confidence in measurement
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        """Check if HR is in physiological range."""
        return 30 <= self.bpm <= 220


@dataclass
class HRVData:
    """Heart rate variability metrics."""
    # Time domain
    rmssd_ms: float = 0.0      # Root mean square of successive differences
    sdnn_ms: float = 0.0       # Standard deviation of NN intervals
    pnn50: float = 0.0         # Percentage of successive RR > 50ms

    # Frequency domain (requires longer recording)
    lf_power: Optional[float] = None   # Low frequency power (0.04-0.15 Hz)
    hf_power: Optional[float] = None   # High frequency power (0.15-0.4 Hz)
    lf_hf_ratio: Optional[float] = None

    # Stress indicator
    stress_index: float = 0.5  # 0-1, computed from HRV metrics

    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_rr_intervals(cls, rr_intervals_ms: List[float]) -> HRVData:
        """Compute HRV metrics from R-R intervals."""
        if len(rr_intervals_ms) < 2:
            return cls()

        # RMSSD
        diffs = [
            rr_intervals_ms[i + 1] - rr_intervals_ms[i]
            for i in range(len(rr_intervals_ms) - 1)
        ]
        rmssd = math.sqrt(sum(d ** 2 for d in diffs) / len(diffs))

        # SDNN
        mean_rr = sum(rr_intervals_ms) / len(rr_intervals_ms)
        variance = sum((rr - mean_rr) ** 2 for rr in rr_intervals_ms) / len(rr_intervals_ms)
        sdnn = math.sqrt(variance)

        # pNN50
        nn50_count = sum(1 for d in diffs if abs(d) > 50)
        pnn50 = (nn50_count / len(diffs)) * 100

        # Simple stress index (inverse of HRV - higher HRV = lower stress)
        # Normalized: assume healthy RMSSD range 20-100ms
        normalized_rmssd = min(max((rmssd - 20) / 80, 0), 1)
        stress_index = 1.0 - normalized_rmssd

        return cls(
            rmssd_ms=rmssd,
            sdnn_ms=sdnn,
            pnn50=pnn50,
            stress_index=stress_index,
        )


@dataclass
class GSRData:
    """Galvanic skin response (electrodermal activity)."""
    conductance_us: float = 0.0     # Microsiemens (typical 1-20 uS)
    scl: float = 0.0                # Skin conductance level (tonic)
    scr_count: int = 0              # Skin conductance responses (phasic)
    arousal_level: float = 0.5      # 0-1 normalized arousal

    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_conductance(
        cls,
        values: List[float],
        baseline: float = 2.0
    ) -> GSRData:
        """Compute GSR metrics from raw conductance values."""
        if not values:
            return cls()

        current = values[-1]
        scl = sum(values) / len(values)

        # Count phasic responses (peaks above threshold)
        threshold = scl * 1.1  # 10% above tonic level
        scr_count = sum(1 for v in values if v > threshold)

        # Arousal: normalize against typical range
        # Low: 1-3 uS, High: 10-20 uS
        normalized = min(max((current - 1) / 15, 0), 1)

        return cls(
            conductance_us=current,
            scl=scl,
            scr_count=scr_count,
            arousal_level=normalized,
        )


@dataclass
class PhysioReading:
    """Combined physiological reading."""
    heart_rate: HeartRateData
    hrv: HRVData
    gsr: Optional[GSRData] = None
    temperature_c: Optional[float] = None  # Skin temp if available

    timestamp: datetime = field(default_factory=datetime.now)

    def to_brainlink_reading(self, device_id: str) -> BrainlinkReading:
        """Convert to unified BrainlinkReading format."""
        channels = {
            "HR": ChannelData(
                name="HR",
                values=[self.heart_rate.bpm],
                sample_rate=1.0,
                unit="bpm",
                quality=SignalQuality.GOOD if self.heart_rate.is_valid else SignalQuality.POOR,
            ),
        }

        if self.heart_rate.rr_intervals_ms:
            channels["RR"] = ChannelData(
                name="RR",
                values=self.heart_rate.rr_intervals_ms,
                sample_rate=1.0,
                unit="ms",
            )

        if self.gsr:
            channels["GSR"] = ChannelData(
                name="GSR",
                values=[self.gsr.conductance_us],
                sample_rate=4.0,  # Typical GSR sample rate
                unit="uS",
            )

        metrics = {
            "hr_bpm": self.heart_rate.bpm,
            "hrv_rmssd": self.hrv.rmssd_ms,
            "hrv_sdnn": self.hrv.sdnn_ms,
            "hrv_pnn50": self.hrv.pnn50,
            "stress_index": self.hrv.stress_index,
        }

        if self.gsr:
            metrics["gsr_conductance"] = self.gsr.conductance_us
            metrics["arousal_level"] = self.gsr.arousal_level

        if self.temperature_c:
            metrics["skin_temp_c"] = self.temperature_c

        return BrainlinkReading(
            timestamp=self.timestamp,
            device_id=device_id,
            device_type="physio",
            channels=channels,
            metrics=metrics,
            extras={
                "lf_hf_ratio": self.hrv.lf_hf_ratio,
            },
        )


class PhysioClient(BrainlinkProtocol):
    """
    Client for physiological sensors (HR, HRV, GSR).

    Connects to BLE heart rate monitors and optional GSR sensors.
    """

    def __init__(self, config: Optional[BrainlinkConfig] = None):
        super().__init__(config)
        self._rr_buffer: deque = deque(maxlen=120)  # 2 min at 60bpm
        self._gsr_buffer: deque = deque(maxlen=256)
        self._last_hr: float = 0.0
        self._ble_client = None  # Placeholder for bleak client

    @property
    def device_type(self) -> str:
        return "physio"

    async def connect(self) -> bool:
        """
        Connect to physio sensors.

        Currently a stub - real implementation would use bleak for BLE.
        """
        self._status = BrainlinkStatus.CONNECTING

        try:
            # In real implementation:
            # 1. Scan for BLE devices with HR service UUID
            # 2. Connect to specified device or first found
            # 3. Subscribe to HR measurement characteristic

            # For now, simulate connection
            await asyncio.sleep(0.1)

            self._device_id = self.config.device_address or "physio_simulated"
            self._status = BrainlinkStatus.CONNECTED

            if self.config.on_connect:
                await self.config.on_connect()

            return True

        except Exception as e:
            self._status = BrainlinkStatus.ERROR
            raise ConnectionError(f"Failed to connect to physio sensors: {e}")

    async def disconnect(self) -> None:
        """Disconnect from physio sensors."""
        if self._ble_client:
            # await self._ble_client.disconnect()
            pass

        self._status = BrainlinkStatus.DISCONNECTED
        self._device_id = None

        if self.config.on_disconnect:
            await self.config.on_disconnect()

    async def read(self) -> BrainlinkReading:
        """Get a single physio reading."""
        physio = await self._read_physio()
        reading = physio.to_brainlink_reading(self._device_id or "unknown")
        await self._notify_callbacks(reading)
        return reading

    async def stream(self) -> AsyncIterator[BrainlinkReading]:
        """Stream continuous physio readings."""
        self._status = BrainlinkStatus.STREAMING

        while self._status == BrainlinkStatus.STREAMING:
            try:
                reading = await self.read()
                yield reading
                await asyncio.sleep(1.0)  # ~1 Hz update rate

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.config.on_error:
                    await self.config.on_error(e)
                if self.config.auto_reconnect:
                    await self._attempt_reconnect()
                else:
                    break

    async def _read_physio(self) -> PhysioReading:
        """Read from actual hardware or simulate."""
        # Simulated data for development
        # In production, this reads from BLE characteristics

        import random

        # Simulate HR with slight variation
        base_hr = 72 if self._last_hr == 0 else self._last_hr
        hr = base_hr + random.gauss(0, 2)
        hr = max(50, min(120, hr))  # Clamp to reasonable range
        self._last_hr = hr

        # Generate RR intervals
        mean_rr = 60000 / hr  # ms
        rr_interval = mean_rr + random.gauss(0, 30)
        self._rr_buffer.append(rr_interval)

        # Compute HRV
        rr_list = list(self._rr_buffer)
        hrv = HRVData.from_rr_intervals(rr_list)

        # Simulate GSR
        gsr_value = 5.0 + random.gauss(0, 0.5)
        self._gsr_buffer.append(gsr_value)
        gsr = GSRData.from_conductance(list(self._gsr_buffer))

        return PhysioReading(
            heart_rate=HeartRateData(
                bpm=hr,
                rr_intervals_ms=rr_list[-10:],  # Last 10 intervals
            ),
            hrv=hrv,
            gsr=gsr,
        )

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


def get_physio_client(
    device_address: Optional[str] = None,
    **kwargs
) -> PhysioClient:
    """
    Factory function to get a PhysioClient.

    Args:
        device_address: BLE MAC address (optional, will scan if not provided)
        **kwargs: Additional config options

    Returns:
        Configured PhysioClient
    """
    config = BrainlinkConfig(device_address=device_address, **kwargs)
    return PhysioClient(config)
