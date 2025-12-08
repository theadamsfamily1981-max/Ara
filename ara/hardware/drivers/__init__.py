"""
FPGA Hardware Drivers
=====================

Python drivers for communicating with Ara's FPGA accelerators.

Drivers:
    CorrSpikeDriver: Driver for CorrSpike-HDC kernel
    SpikeBlockDriver: Driver for SpikingBrain-style tile (TODO)
"""

from ara.hardware.drivers.corr_spike_driver import (
    CorrSpikeDriver,
    CorrSpikeConfig,
    CorrSpikeResult,
    encode_telemetry_to_hv,
    HV_DIM,
    N_PROTOS
)

__all__ = [
    'CorrSpikeDriver',
    'CorrSpikeConfig',
    'CorrSpikeResult',
    'encode_telemetry_to_hv',
    'HV_DIM',
    'N_PROTOS'
]
