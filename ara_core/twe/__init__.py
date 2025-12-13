"""
Teleoplastic Weave Engine (TWE)
===============================

Closed-loop, criticality-tuned organ manufacturing platform.

The TWE is not just a bioprinter - it's a physical instance of the
(λ, Π) control manifold: a near-critical, precision-tuned inference
machine that outputs organs.

Architecture:
    Meta-Brain (architect) → CADD (safety) → Kitten Fabric (control) → Hardware

Control Stack:
    1. Strategic (Meta-Brain): Design organs as antifragile dynamical objects
    2. Tactical (CADD + Supervisor): Safety gate, enforce diversity
    3. Reflexive (Kitten Fabric): Real-time closed-loop on viability

GUTC Integration:
    λ_fabric: Global gain on SNN controllers (target ≈ 1)
    λ_tissue: Estimated from sensor avalanches (target ≈ 1)
    Π_sensory: Gain on viability errors (what sensors say)
    Π_prior: Gain on blueprint conformance (what the plan says)

Usage:
    from ara_core.twe import TeleoplasticWeaveEngine
    from ara_core.cadd import CADDSentinel

    # Create TWE with safety sentinel
    sentinel = CADDSentinel()
    twe = TeleoplasticWeaveEngine(sentinel=sentinel)
    twe.initialize()

    # Plan organ with diversity checks
    blueprint = twe.plan_blueprint(patient_profile)

    # Print with closed-loop control
    job = twe.print_organ(blueprint)

    twe.shutdown()

Quick Demo:
    from ara_core.twe import run_demo_print
    job = run_demo_print()
"""

# Blueprint structures
from .blueprint import (
    # Enums
    MaterialType,
    CellType,
    VascularState,
    ScheduleEventType,
    # Voxel and fields
    Voxel,
    ScaffoldField,
    CellField,
    # Vascular graph
    VascularNode,
    VascularEdge,
    VascularGraph,
    # Schedule
    ScheduleEvent,
    TemporalSchedule,
    # Blueprint
    OrganBlueprint,
    PrintLayer,
    # Factory
    create_demo_vascular_patch,
)

# Kitten Fabric (SNN control)
from .fabric import (
    # Sensor state
    SensorTile,
    SensorGrid,
    # Avalanche analysis
    Avalanche,
    AvalancheDetector,
    # Precision control
    PrecisionState,
    # Actuator commands
    ActuatorCommand,
    # Controller
    KittenFabric,
)

# Hardware abstraction
from .hardware import (
    # State
    PrinterState,
    NozzleState,
    StageState,
    ChamberState,
    # Interface
    TWEHardware,
    SimulatedHardware,
)

# Main engine
from .engine import (
    # Job management
    JobState,
    PrintJob,
    # Meta-Brain
    MetaBrainInterface,
    # Engine
    TeleoplasticWeaveEngine,
    # Demo
    run_demo_print,
)

__all__ = [
    # Blueprint
    "MaterialType",
    "CellType",
    "VascularState",
    "ScheduleEventType",
    "Voxel",
    "ScaffoldField",
    "CellField",
    "VascularNode",
    "VascularEdge",
    "VascularGraph",
    "ScheduleEvent",
    "TemporalSchedule",
    "OrganBlueprint",
    "PrintLayer",
    "create_demo_vascular_patch",
    # Fabric
    "SensorTile",
    "SensorGrid",
    "Avalanche",
    "AvalancheDetector",
    "PrecisionState",
    "ActuatorCommand",
    "KittenFabric",
    # Hardware
    "PrinterState",
    "NozzleState",
    "StageState",
    "ChamberState",
    "TWEHardware",
    "SimulatedHardware",
    # Engine
    "JobState",
    "PrintJob",
    "MetaBrainInterface",
    "TeleoplasticWeaveEngine",
    "run_demo_print",
]
