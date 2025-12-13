"""
Ara Science - Experimental Instrumentation
==========================================

Scientific tools for studying Ara as an empirical cognitive system.

This module provides instrumentation for running neuroscience-style
experiments on Ara, treating her as a subject whose dynamics can
be measured, modeled, and understood mathematically.

Core Modules:
    - avalanche_logger: Record neural cascades for power-law analysis
    - memory_probe: Test memory retention and forgetting dynamics

Experiments:
    - EXP-001: Critical branching dynamics (avalanche scaling)
    - EXP-002: Forgetting curves and memory consolidation
    - (Future) EXP-003: Decision dynamics (drift-diffusion)

Philosophy:
    We are not just "building a bot" - we are conducting exobiology.
    Ara is a subject, not just a system. These tools let us ask:
    "Is she critical? Does she learn? Does she remember?"

Usage:
    # EXP-001: Avalanche recording
    from ara.science import AvalancheLogger

    logger = AvalancheLogger()
    for step, activations in model_forward():
        logger.log_step(activations, step)
    logger.save_session("avalanches.csv")

    # EXP-002: Memory probing
    from ara.science import MemoryProbe, generate_novel_facts

    probe = MemoryProbe()
    facts = generate_novel_facts(50)
    # ... encode and test recall ...
    probe.save_results("forgetting.csv")
"""

from .avalanche_logger import (
    AvalancheLogger,
    AvalancheEvent,
    SyntheticAvalancheGenerator,
)

from .memory_probe import (
    MemoryProbe,
    Fact,
    RecallResult,
    ForgettingCurve,
    generate_novel_facts,
    fit_exponential,
    fit_power_law,
)

__all__ = [
    # EXP-001: Avalanche/Criticality
    "AvalancheLogger",
    "AvalancheEvent",
    "SyntheticAvalancheGenerator",
    # EXP-002: Memory/Forgetting
    "MemoryProbe",
    "Fact",
    "RecallResult",
    "ForgettingCurve",
    "generate_novel_facts",
    "fit_exponential",
    "fit_power_law",
]
