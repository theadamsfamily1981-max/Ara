"""
Kitten Fabric - CorrSpike-HDC Processing Tiles
==============================================

Stratix-10 optimized fabric for Ara's emotional subcortex.

Components:

RTL (SystemVerilog):
- kitten_fabric_tile.sv: Base processing tile with:
  - Ping-pong double-buffered I_post banks
  - Sparse early-exit via max_active_idx
  - Parameterizable LANES for vector parallelism
  - Wavefront connections for layer cascades
  - Wavelength multiplexing for logical layers

- temporal_loom.sv: Multi-layer wavefront scheduler with:
  - 3-axis parallelism (spatial, temporal, wavelength)
  - Ping-pong bank controller for 2x throughput
  - Wavefront pipeline across NUM_TILES
  - HDC superposition for wavelength multiplexing
  - Master FSM for layer sweep orchestration
  - Performance counters (spikes, early exits)

- plasticity_engine.sv: Reward-modulated Hebbian learning with:
  - Chunked streaming (512-bit/cycle, NOT "1 clock fully unrolled")
  - plasticity_row_engine: Per-row accumulator update
  - plasticity_controller: Row scheduling based on active mask
  - REALISTIC performance: ~1.7 µs per emotional event

Python Integration:
- corrspike_axis_bridge.py: Python ↔ FPGA bridge with:
  - AxisMundi holographic state bus
  - Wavelength encoding/decoding
  - Emotional subcortex processing
  - FPGA register interface
  - PlasticityInterface: Reward-driven learning API

Parallelism Strategy (Elegant Maximization):
  1. TEMPORAL:    Ping-pong → 2x (read/write overlap)
  2. SPATIAL:     Wavefront → NUM_TILES concurrent
  3. WAVELENGTH:  HDC codes → LAYERS_PER_TICK logical layers

  Combined: 9 logical layers in 3 physical ticks
            24x speedup over naive sequential

Plasticity (REALITY CHECK):
  - NOT "33.5M bit updates in one clock" (that was poetry)
  - IS "512-bit chunks, ~53 ns/row, ~1.7 µs/event"
  - Still instantaneous at human/emotional timescale
"""

from .corrspike_axis_bridge import (
    CorrSpikeAxisBridge,
    EmotionalSubcortex,
    WavelengthCode,
    FPGAInterface,
    SimulatedFPGA,
    FPGARegisterMap,
    generate_wavelength_codes,
    PlasticityConfig,
    PlasticityInterface,
)

__all__ = [
    "CorrSpikeAxisBridge",
    "EmotionalSubcortex",
    "WavelengthCode",
    "FPGAInterface",
    "SimulatedFPGA",
    "FPGARegisterMap",
    "generate_wavelength_codes",
    "PlasticityConfig",
    "PlasticityInterface",
]
