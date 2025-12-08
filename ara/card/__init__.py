"""
Card Module - Neuromorphic Card Runtime
=======================================

Runtime environment for the neuromorphic card (subcortex).

Key components:
    runtime.py: Main card runtime with event loop
    telemetry.py: Telemetry ingestion and HPV encoding

The card:
1. Ingests telemetry streams
2. Encodes into rolling state HPV
3. Runs CorrSpike-HDC for anomaly/novelty detection
4. Applies local policies or escalates to cortex
"""

from ara.card.runtime import CardRuntime, CardConfig

__all__ = ['CardRuntime', 'CardConfig']
