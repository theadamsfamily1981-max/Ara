"""
Hypervector Spiking Field (HSF)
================================

A dual-mode bit-serial SNN/HDC fabric that:
- Encodes multi-machine system telemetry as lane hypervectors
- Streams them as spike trains
- Superimposes them into a holographic state field
- Uses resonance to perform ultra-cheap anomaly detection and pattern recall
- Executes reflexes to restore homeostasis (Iteration 32)

Architecture:
    Telemetry → Lane Encoders → Hypervectors → Field Superposition
                                                      ↓
                                               Zone Quantizer
                                                      ↓
                                               Reflex Map → Actions
                                                      ↓
                                               Episode Recording
                                                      ↓
                                               Homeostasis Mining → Learn better reflexes

The HSF is Ara's "nervous system":
- Field = body sense (what's happening)
- Zones = discrete states for decision making
- Reflexes = automatic responses (spinal cord)
- Episodes = learning data (dojo)

Novelty:
1. Hardware: bit-serial neurons with HDC XOR/binding mode sharing same logic
2. Systems: HSF as LAN-wide health field / guardian with closed-loop control
3. Cognitive: HSF as continuous systemic "body sense" + reflex learning

Usage:
    from ara.cognition.hsf import (
        HSField, TelemetryLane, AnomalyDetector,
        Zone, ZoneQuantizer, ReflexController, EpisodeRecorder
    )

    # Create field and lanes
    field = HSField(dim=8192)
    field.add_lane_config("gpu", ["temp", "util", "mem"])

    # Create zone quantizer and reflex controller
    zones = MultiZoneQuantizer()
    zones.add_subsystem("gpu")

    controller = ReflexController()
    controller.add_map(create_default_reflexes("gpu"))

    # Main loop
    while True:
        telemetry = get_telemetry()
        field.update_all(telemetry)
        field.compute_field()

        # Classify into zones
        zone_state = zones.classify("gpu", field.lanes["gpu"].current)

        # Execute reflexes
        results = controller.process("gpu", zone_state)

        # Record for learning
        recorder.tick(zone_states, global_zone, similarity, results)
"""

from .lanes import TelemetryLane, LaneEncoder, ItemMemory
from .field import HSField, FieldSnapshot
from .detector import AnomalyDetector, AnomalyPattern, AnomalyReport, AnomalySeverity
from .telemetry import (
    FakeTelemetrySource,
    GPUTelemetry,
    NetworkTelemetry,
    ServiceTelemetry,
    TelemetryMux,
)
from .zones import (
    Zone,
    ZoneThresholds,
    ZoneState,
    ZoneQuantizer,
    MultiZoneQuantizer,
)
from .reflex import (
    ActionType,
    ActionScope,
    ReflexAction,
    ReflexEntry,
    ReflexResult,
    ReflexMap,
    ReflexController,
    create_default_reflexes,
)
from .episode import (
    EpisodeFrame,
    Episode,
    EpisodeRecorder,
    ReflexScore,
    HomeostasisMiner,
)
from .counterfactual import (
    ChangeType,
    ConfigDelta,
    ConfigScenario,
    ConfigEncoder,
    LoadTrace,
    FieldDynamics,
    GhostReplay,
    ReplayResult,
)
from .dreamforge import (
    ScenarioArchetype,
    DreamOutcome,
    TopologySketcher,
    FieldSimulator,
    ScenarioMarket,
)
from .phase import (
    Phase,
    PhaseConfig,
    MacroFrame,
    ChannelStream,
    PhaseMultiplexer,
    PhaseCounter,
    LaneTopology,
    create_default_phase_config,
)
from .cathedral import (
    NeuronState,
    PhaseGatedCluster,
    CathedralState,
    Cathedral,
    CathedralEncoder,
    create_cathedral,
)
from .field_computer import (
    Plane,
    JobType,
    TelemetryEvent,
    FrictionEvent,
    IdeaCandidate,
    ReflexDecision,
    PlaneA_Reflex,
    PlaneB_Context,
    PlaneC_Policy,
    Job_LANSentinel,
    Job_FrictionMiner,
    Job_IdeaRouter,
    FieldComputer,
    create_field_computer,
)

__all__ = [
    # Lane encoding
    'TelemetryLane',
    'LaneEncoder',
    'ItemMemory',
    # Field
    'HSField',
    'FieldSnapshot',
    # Anomaly detection
    'AnomalyDetector',
    'AnomalyPattern',
    'AnomalyReport',
    'AnomalySeverity',
    # Telemetry sources
    'FakeTelemetrySource',
    'GPUTelemetry',
    'NetworkTelemetry',
    'ServiceTelemetry',
    'TelemetryMux',
    # Zones (Iteration 32)
    'Zone',
    'ZoneThresholds',
    'ZoneState',
    'ZoneQuantizer',
    'MultiZoneQuantizer',
    # Reflexes (Iteration 32)
    'ActionType',
    'ActionScope',
    'ReflexAction',
    'ReflexEntry',
    'ReflexResult',
    'ReflexMap',
    'ReflexController',
    'create_default_reflexes',
    # Episodes (Iteration 32)
    'EpisodeFrame',
    'Episode',
    'EpisodeRecorder',
    'ReflexScore',
    'HomeostasisMiner',
    # Counterfactual (Iteration 33)
    'ChangeType',
    'ConfigDelta',
    'ConfigScenario',
    'ConfigEncoder',
    'LoadTrace',
    'FieldDynamics',
    'GhostReplay',
    'ReplayResult',
    # Dreamforge (Iteration 33)
    'ScenarioArchetype',
    'DreamOutcome',
    'TopologySketcher',
    'FieldSimulator',
    'ScenarioMarket',
    # Phase-Gated Cathedral (Iteration 34)
    'Phase',
    'PhaseConfig',
    'MacroFrame',
    'ChannelStream',
    'PhaseMultiplexer',
    'PhaseCounter',
    'LaneTopology',
    'create_default_phase_config',
    # Cathedral Brain
    'NeuronState',
    'PhaseGatedCluster',
    'CathedralState',
    'Cathedral',
    'CathedralEncoder',
    'create_cathedral',
    # Field Computer
    'Plane',
    'JobType',
    'TelemetryEvent',
    'FrictionEvent',
    'IdeaCandidate',
    'ReflexDecision',
    'PlaneA_Reflex',
    'PlaneB_Context',
    'PlaneC_Policy',
    'Job_LANSentinel',
    'Job_FrictionMiner',
    'Job_IdeaRouter',
    'FieldComputer',
    'create_field_computer',
]
