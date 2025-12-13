"""
Brain Remodulator - Precision Thermostat for Neural Dynamics

A computational psychiatry framework for correcting aberrant precision
weighting, inspired by predictive processing models of schizophrenia
and autism spectrum disorder.

THEORETICAL BASIS
=================

The brain as a prediction machine:
- Maintains generative models that predict sensory input
- Computes prediction errors (PE) = observed - predicted
- Weights PEs by precision (Π) = confidence in the signal

The Delusion Index (D):
    D = Π_prior / Π_sensory

    D = 1.0  → Healthy balance between beliefs and evidence
    D >> 1   → Prior-dominated: beliefs override evidence (psychosis)
    D << 1   → Sensory-dominated: overwhelmed by input (autism-like)

DISORDER MODELS
===============

Schizophrenia Spectrum (D >> 1):
- Aberrantly high precision on priors/predictions
- Reduced precision on sensory prediction errors
- False perceptions treated as real (hallucinations)
- Beliefs resistant to contradicting evidence (delusions)

Autism Spectrum (D << 1):
- Aberrantly high precision on sensory input
- Reduced precision on priors/predictions
- Sensory overwhelm, difficulty filtering
- Hyper-literal, difficulty with abstraction

THE REMODULATOR
===============

A control system that monitors D and ρ (criticality), detects drift
from the healthy "critical corridor", and computes corrective
interventions to rebalance precision weighting.

Intervention modalities:
- DirectPrecisionModality: Simulation/theoretical
- NeurofeedbackModality: Maps to EEG training targets
- PharmacologicalModality: Models drug effects

Usage:
    from ara.neuro.remodulator import BrainRemodulator

    # Create remodulator
    remodulator = BrainRemodulator(D_target=1.0, rho_target=0.88)

    # Update with sensor data
    remodulator.update(pi_prior=1.8, pi_sensory=0.6, rho=0.95)

    # Get diagnosis and interventions
    diagnosis = remodulator.get_diagnosis()
    interventions = remodulator.get_pending_interventions()

    # Apply corrections (simulation mode)
    remodulator.apply_interventions()

Run simulation demo:
    python -m ara.neuro.remodulator.simulation --pattern all --steps 100
"""

from .core import (
    # Enums
    DisorderPattern,
    CriticalityRegime,
    InterventionType,
    # Data structures
    PrecisionState,
    HierarchicalPrecisionState,  # v2.0: D_low, D_high, ΔH
    CriticalityState,
    BrainState,
    Intervention,
    # Control
    PrecisionControlLaw,
    # Modalities
    InterventionModality,
    DirectPrecisionModality,
    NeurofeedbackModality,
    PharmacologicalModality,
    # Main class
    BrainRemodulator,
)

from .telemetry_bridge import (
    RemodulatorTelemetryBridge,
    create_bridge,
    TELEMETRY_AVAILABLE,
)

__all__ = [
    # Enums
    "DisorderPattern",
    "CriticalityRegime",
    "InterventionType",
    # Data structures
    "PrecisionState",
    "HierarchicalPrecisionState",  # v2.0
    "CriticalityState",
    "BrainState",
    "Intervention",
    # Control
    "PrecisionControlLaw",
    # Modalities
    "InterventionModality",
    "DirectPrecisionModality",
    "NeurofeedbackModality",
    "PharmacologicalModality",
    # Main class
    "BrainRemodulator",
    # Telemetry
    "RemodulatorTelemetryBridge",
    "create_bridge",
    "TELEMETRY_AVAILABLE",
]

__version__ = "0.1.0"
