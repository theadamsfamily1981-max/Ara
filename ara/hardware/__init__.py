"""
Hardware Module - FPGA Neuromorphic Blocks
==========================================

RTL and HLS specifications for neuromorphic FPGA tiles.

Key components:
    spike_block/: SpikingBrain-style spiking attention tile
    hls/: High-Level Synthesis C++ kernels
        - spike_block_kernel.cpp: Spiking attention HLS
        - corr_spike_hdc.cpp: CorrSpike-HDC correlation kernel
    drivers/: Python drivers for host-FPGA communication
    kitten/: Kitten Fabric Tile (CorrSpike-HDC + AxisMundi integration)
        - kitten_fabric_tile.sv: SystemVerilog tile with ping-pong buffers
        - corrspike_axis_bridge.py: Python bridge for wavelength multiplexing

The goal: adapt SpikingBrain/Dragon-Hatchling principles to
FPGA fabric (Stratix-10, VU9P, etc.) as a path off the NVIDIA treadmill.

Architecture:
    - Spiking neurons with dynamic thresholds
    - Linear attention (no full QK^T)
    - Hebbian plasticity for on-chip learning
    - Sparse, event-driven computation
    - Hyperdimensional correlation (CorrSpike-HDC)
    - Wavelength multiplexing for logical layer sharing
"""

# Kitten Fabric - CorrSpike-HDC ↔ AxisMundi integration
from .kitten.corrspike_axis_bridge import (
    CorrSpikeAxisBridge,
    EmotionalSubcortex,
    WavelengthCode,
    FPGAInterface,
    SimulatedFPGA,
    generate_wavelength_codes,
)

__all__ = [
    # Kitten Fabric
    "CorrSpikeAxisBridge",
    "EmotionalSubcortex",
    "WavelengthCode",
    "FPGAInterface",
    "SimulatedFPGA",
    "generate_wavelength_codes",
]
