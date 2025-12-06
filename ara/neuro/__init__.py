"""
Ara Neuromorphic Package
========================

Hardware-aware neural computation layers:
    - binary/: 1-bit neurons for cheap, massive correlation
    - (future) snn/: Spiking neural networks
    - (future) tfan/: Topological Feature Attention Networks

The neuromorphic stack follows a principle:
    "Let cheap ops shape data before expensive ops process it."

Typical pipeline:
    Raw Input → BinaryFrontEnd (XNOR+popcount) → TopologicalSketch
                        ↓
              BinaryMemory (pattern recall)
                        ↓
              SNN/T-FAN Core (high-precision reasoning)
"""

from ara.neuro import binary

__all__ = ['binary']
