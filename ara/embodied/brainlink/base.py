# ara/embodied/brainlink/base.py
"""
Base protocol and data structures for Brainlink hardware abstraction.

All brainlink clients implement BrainlinkProtocol, allowing uniform
access to different hardware backends.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    AsyncIterator,
    Dict,
    List,
    Optional,
    Any,
    Callable,
    Awaitable,
)


class BrainlinkStatus(Enum):
    """Connection status for brainlink hardware."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    CALIBRATING = auto()
    STREAMING = auto()
    ERROR = auto()
    RECONNECTING = auto()


class SignalQuality(Enum):
    """Signal quality indicator."""
    EXCELLENT = auto()  # >95% good data
    GOOD = auto()       # 80-95%
    FAIR = auto()       # 60-80%
    POOR = auto()       # 40-60%
    BAD = auto()        # <40%
    NO_SIGNAL = auto()  # No contact or disconnected


@dataclass
class ChannelData:
    """
    Data from a single sensor channel.

    Generic structure that works for EEG electrodes, physio sensors, etc.
    """
    name: str                          # Channel identifier (e.g., "AF7", "HR", "GSR")
    values: List[float]                # Raw sample values (buffer)
    sample_rate: float                 # Hz
    quality: SignalQuality = SignalQuality.GOOD
    unit: str = ""                     # Unit of measurement (uV, bpm, uS, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def latest(self) -> float:
        """Get most recent value."""
        return self.values[-1] if self.values else 0.0

    @property
    def mean(self) -> float:
        """Get mean of buffer."""
        return sum(self.values) / len(self.values) if self.values else 0.0


@dataclass
class BrainlinkReading:
    """
    Unified reading from any brainlink hardware.

    This is the common format that flows through the system regardless
    of the underlying hardware (Muse, physio sensors, OpenBCI, etc.).
    """
    timestamp: datetime
    device_id: str
    device_type: str  # "muse", "physio", "openbci", "mock"

    # Core data channels
    channels: Dict[str, ChannelData] = field(default_factory=dict)

    # Derived metrics (computed by device or client)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Overall signal quality
    quality: SignalQuality = SignalQuality.GOOD

    # Device-specific extras
    extras: Dict[str, Any] = field(default_factory=dict)

    def get_channel(self, name: str) -> Optional[ChannelData]:
        """Get a specific channel by name."""
        return self.channels.get(name)

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get a derived metric value."""
        return self.metrics.get(name, default)

    @property
    def is_valid(self) -> bool:
        """Check if reading has usable data."""
        return (
            self.quality not in (SignalQuality.BAD, SignalQuality.NO_SIGNAL)
            and len(self.channels) > 0
        )


@dataclass
class BrainlinkConfig:
    """Configuration for brainlink clients."""
    # Connection
    device_address: Optional[str] = None  # MAC address or serial port
    auto_reconnect: bool = True
    reconnect_delay_seconds: float = 2.0
    max_reconnect_attempts: int = 5

    # Streaming
    buffer_size: int = 256          # Samples to buffer
    sample_rate: Optional[float] = None  # Override device default

    # Quality
    min_signal_quality: SignalQuality = SignalQuality.FAIR

    # Callbacks
    on_connect: Optional[Callable[[], Awaitable[None]]] = None
    on_disconnect: Optional[Callable[[], Awaitable[None]]] = None
    on_error: Optional[Callable[[Exception], Awaitable[None]]] = None

    # Filtering
    apply_notch_filter: bool = True   # Remove 50/60Hz line noise
    apply_bandpass: bool = True       # Standard EEG bandpass
    bandpass_low: float = 1.0         # Hz
    bandpass_high: float = 50.0       # Hz


class BrainlinkError(Exception):
    """Base exception for brainlink errors."""
    pass


class ConnectionError(BrainlinkError):
    """Failed to connect to device."""
    pass


class CalibrationError(BrainlinkError):
    """Device calibration failed."""
    pass


class BrainlinkProtocol(ABC):
    """
    Protocol that all brainlink clients must implement.

    This provides a uniform interface for:
    - Connection management
    - Single readings
    - Continuous streaming
    - Status monitoring

    Usage:
        async with get_brainlink("muse") as link:
            async for reading in link.stream():
                print(reading.metrics)
    """

    def __init__(self, config: Optional[BrainlinkConfig] = None):
        self.config = config or BrainlinkConfig()
        self._status = BrainlinkStatus.DISCONNECTED
        self._device_id: Optional[str] = None
        self._callbacks: List[Callable[[BrainlinkReading], Awaitable[None]]] = []

    @property
    def status(self) -> BrainlinkStatus:
        """Current connection status."""
        return self._status

    @property
    def device_id(self) -> Optional[str]:
        """Connected device identifier."""
        return self._device_id

    @property
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._status in (
            BrainlinkStatus.CONNECTED,
            BrainlinkStatus.CALIBRATING,
            BrainlinkStatus.STREAMING,
        )

    @property
    @abstractmethod
    def device_type(self) -> str:
        """Return the device type identifier."""
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the hardware device.

        Returns:
            True if connection successful, False otherwise

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the hardware device."""
        pass

    @abstractmethod
    async def read(self) -> BrainlinkReading:
        """
        Get a single reading from the device.

        Returns:
            BrainlinkReading with current sensor data
        """
        pass

    @abstractmethod
    async def stream(self) -> AsyncIterator[BrainlinkReading]:
        """
        Stream continuous readings from the device.

        Yields:
            BrainlinkReading for each sample/batch
        """
        pass

    async def calibrate(self) -> bool:
        """
        Run device calibration (if supported).

        Default implementation does nothing. Override for devices
        that need calibration (e.g., EEG impedance check).

        Returns:
            True if calibration successful
        """
        return True

    def subscribe(
        self,
        callback: Callable[[BrainlinkReading], Awaitable[None]]
    ) -> None:
        """Subscribe to readings with a callback."""
        self._callbacks.append(callback)

    def unsubscribe(
        self,
        callback: Callable[[BrainlinkReading], Awaitable[None]]
    ) -> None:
        """Unsubscribe a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def _notify_callbacks(self, reading: BrainlinkReading) -> None:
        """Notify all subscribed callbacks."""
        for callback in self._callbacks:
            try:
                await callback(reading)
            except Exception:
                pass  # Don't let callback errors break the stream

    async def __aenter__(self) -> BrainlinkProtocol:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


# Type alias for reading handlers
ReadingHandler = Callable[[BrainlinkReading], Awaitable[None]]
