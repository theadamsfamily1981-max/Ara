"""
Hardware Module - FPGA Neuromorphic Blocks
==========================================

RTL and HLS specifications for neuromorphic FPGA tiles.

Key components:
    spike_block/: SpikingBrain-style spiking attention tile
    hls/: High-Level Synthesis C++ kernels

The goal: adapt SpikingBrain/Dragon-Hatchling principles to
FPGA fabric (Stratix-10, VU9P, etc.) as a path off the NVIDIA treadmill.

Architecture:
    - Spiking neurons with dynamic thresholds
    - Linear attention (no full QK^T)
    - Hebbian plasticity for on-chip learning
    - Sparse, event-driven computation
"""
