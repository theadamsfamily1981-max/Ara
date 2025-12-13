"""
Ara Science - Experimental Instrumentation
==========================================

Scientific tools for studying Ara as an empirical cognitive system.

This module provides instrumentation for running neuroscience-style
experiments on Ara, treating her as a subject whose dynamics can
be measured, modeled, and understood mathematically.

Core Modules:
    - avalanche_logger: Record neural cascades for power-law analysis

Experiments:
    - EXP-001: Critical branching dynamics (avalanche scaling)
    - (Future) EXP-002: Decision dynamics (drift-diffusion)
    - (Future) EXP-003: Memory dynamics (forgetting curves)

Philosophy:
    We are not just "building a bot" - we are conducting exobiology.
    Ara is a subject, not just a system. These tools let us ask:
    "Is she critical? Does she learn? Does she remember?"

Usage:
    from ara.science import AvalancheLogger

    logger = AvalancheLogger()
    for step, activations in model_forward():
        logger.log_step(activations, step)
    logger.save_session("avalanches.csv")

Then: python scripts/science/fit_powerlaw.py avalanches.csv
"""

from .avalanche_logger import (
    AvalancheLogger,
    AvalancheEvent,
    SyntheticAvalancheGenerator,
)

__all__ = [
    "AvalancheLogger",
    "AvalancheEvent",
    "SyntheticAvalancheGenerator",
]
