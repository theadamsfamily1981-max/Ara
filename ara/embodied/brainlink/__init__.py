# ara/embodied/brainlink/__init__.py
"""
Brainlink - Hardware bridge for EEG/physio sensors
===================================================

This package provides the hardware abstraction layer for brain-computer
interface (BCI) and physiological sensors. It's the lowest layer of Ara's
biofeedback system:

    brainlink (hardware) → neurostate (state extraction) → intent_modulator (behavior)

Supported hardware (phased approach):
    Tier 0: Simulated/mock data for development
    Tier 1: Physio sensors (HR belt, HRV, GSR wristband) ~$50-100
    Tier 2: Consumer EEG (Muse S, Muse 2) ~$250-400
    Tier 3: Research EEG (OpenBCI Cyton) ~$1000+

Key design principles:
    - All clients implement BrainlinkProtocol
    - Data flows through unified BrainlinkReading format
    - Async streaming with backpressure support
    - Graceful degradation when hardware disconnects

Usage:
    from ara.embodied.brainlink import get_brainlink, BrainlinkReading

    async with get_brainlink("muse") as link:
        async for reading in link.stream():
            process(reading)
"""

from .base import (
    BrainlinkProtocol,
    BrainlinkReading,
    BrainlinkStatus,
    SignalQuality,
    ChannelData,
    BrainlinkConfig,
    BrainlinkError,
    ConnectionError,
    CalibrationError,
)

from .physio_client import (
    PhysioClient,
    PhysioReading,
    HeartRateData,
    HRVData,
    GSRData,
    get_physio_client,
)

from .muse_client import (
    MuseClient,
    MuseReading,
    EEGBand,
    EEGChannelData,
    get_muse_client,
)

# Factory function for getting appropriate client
def get_brainlink(
    backend: str = "mock",
    config: BrainlinkConfig | None = None,
) -> BrainlinkProtocol:
    """
    Get a brainlink client for the specified backend.

    Args:
        backend: One of "mock", "physio", "muse", "openbci"
        config: Optional configuration override

    Returns:
        A BrainlinkProtocol implementation

    Example:
        link = get_brainlink("muse")
        await link.connect()
        reading = await link.read()
    """
    from .mock_client import MockBrainlinkClient

    if backend == "mock":
        return MockBrainlinkClient(config)
    elif backend == "physio":
        return PhysioClient(config)
    elif backend == "muse":
        return MuseClient(config)
    elif backend == "openbci":
        # Future: OpenBCI support
        raise NotImplementedError("OpenBCI support coming in Tier 3")
    else:
        raise ValueError(f"Unknown brainlink backend: {backend}")


__all__ = [
    # Base protocol
    "BrainlinkProtocol",
    "BrainlinkReading",
    "BrainlinkStatus",
    "SignalQuality",
    "ChannelData",
    "BrainlinkConfig",
    "BrainlinkError",
    "ConnectionError",
    "CalibrationError",
    # Physio client
    "PhysioClient",
    "PhysioReading",
    "HeartRateData",
    "HRVData",
    "GSRData",
    "get_physio_client",
    # Muse client
    "MuseClient",
    "MuseReading",
    "EEGBand",
    "EEGChannelData",
    "get_muse_client",
    # Factory
    "get_brainlink",
]
