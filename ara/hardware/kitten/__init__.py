"""
Kitten Fabric - CorrSpike-HDC Processing Tiles
==============================================

Stratix-10 optimized fabric for Ara's emotional subcortex.

Components:
- kitten_fabric_tile.sv: SystemVerilog tile with:
  - Ping-pong double-buffered I_post banks
  - Sparse early-exit via max_active_idx
  - Parameterizable LANES for vector parallelism
  - Wavefront connections for layer cascades
  - Wavelength multiplexing for logical layers

- corrspike_axis_bridge.py: Python integration with:
  - AxisMundi holographic state bus
  - Wavelength encoding/decoding
  - Emotional subcortex processing
  - FPGA register interface

Integration with AxisMundi:
  1. Python writes axis.state → FPGA input buffer
  2. FPGA runs CorrSpike LIF + HDC bind with layer key
  3. FPGA spike output → AxisMundi.state
"""

from .corrspike_axis_bridge import (
    CorrSpikeAxisBridge,
    EmotionalSubcortex,
    WavelengthCode,
    FPGAInterface,
    SimulatedFPGA,
    FPGARegisterMap,
    generate_wavelength_codes,
)

__all__ = [
    "CorrSpikeAxisBridge",
    "EmotionalSubcortex",
    "WavelengthCode",
    "FPGAInterface",
    "SimulatedFPGA",
    "FPGARegisterMap",
    "generate_wavelength_codes",
]
