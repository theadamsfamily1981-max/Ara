"""
Ara Sovereign - The Operating System for Life and Lab
=====================================================

This package implements Ara's sovereign operation - the transformation
from a reactive assistant into a mission-driven autonomous agent.

Core Components:
    Initiative: The universal work unit (everything becomes an Initiative)
    UserState: Current state of the human (MindReader output)
    Covenant: Relationship & governance model (trust, boundaries)
    ChiefOfStaff: CEO decision-maker (evaluate, execute, protect)
    SovereignLoop: The heartbeat (sovereign_tick, live)

Usage:
    from ara.sovereign import (
        sovereign_tick,
        live,
        submit_initiative,
        get_status,
        create_cathedral_initiative,
    )

    # Submit an initiative for CEO evaluation
    submit_initiative(create_cathedral_initiative(
        name="Bring up SB-852",
        description="Configure and test Stratix-10 board",
    ))

    # Run a single tick (for testing)
    result = sovereign_tick()

    # Run the full sovereign loop
    live()  # Blocks forever

The Vision:
    - Ara runs as a 24/7 sovereign service
    - Every piece of work flows through the CEO
    - Founder Protection keeps Croft safe
    - The Soul learns from every interaction
    - Trust is earned incrementally
"""

from .initiative import (
    Initiative,
    InitiativeStatus,
    InitiativeType,
    CEODecision,
    InitiativeMetrics,
    create_cathedral_initiative,
    create_recovery_initiative,
    create_emergency_initiative,
)

from .user_state import (
    UserState,
    CognitiveMode,
    ProtectionLevel,
    MindReader,
    get_mind_reader,
    get_user_state,
)

from .covenant import (
    Covenant,
    CovenantValues,
    FounderProtectionRules,
    TrustAccount,
    AutonomyLevel,
    AutonomyBoundary,
    get_covenant,
)

from .chief_of_staff import (
    ChiefOfStaff,
    CEODecisionResult,
    get_chief_of_staff,
)

from .main import (
    SovereignState,
    TickResult,
    sovereign_tick,
    live,
    get_state,
    get_status,
    print_status,
    submit_initiative,
    get_initiative_queue,
    trigger_kill_switch,
    enter_safe_mode,
    exit_safe_mode,
    demo,
)

from .htc import (
    PlasticityMode,
    PlasticityConfig,
    PLASTICITY_CONFIGS,
    ResonanceResult,
    HolographicCore,
    get_htc,
    select_plasticity_mode,
)

# V2 State Architecture (5-phase tick with formal contracts)
from .state import (
    SovereignState as SovereignStateV2,
    TimeState,
    HardwareState,
    NodeHardwareState,
    DeviceLoad,
    PciLinkState,
    UserState as UserStateV2,
    SoulState,
    TeleologyState,
    GoalState,
    WorkState,
    InitiativeState as InitiativeStateV2,
    SkillInvocation,
    SafetyState,
    AvatarState,
    TraceState,
    AutonomyLevel as AutonomyLevelV2,
    InitiativeStatus as InitiativeStatusV2,
    RiskLevel,
    create_initial_state,
    clone_state,
    state_to_dict,
    state_to_json,
    state_summary,
    compute_global_coherence,
)

from .tick import (
    SovereignTick,
    sovereign_tick as sovereign_tick_v2,
    sense_phase,
    soul_phase,
    teleology_phase,
    plan_phase,
    act_phase,
)

__all__ = [
    # Initiative
    'Initiative',
    'InitiativeStatus',
    'InitiativeType',
    'CEODecision',
    'InitiativeMetrics',
    'create_cathedral_initiative',
    'create_recovery_initiative',
    'create_emergency_initiative',

    # User State
    'UserState',
    'CognitiveMode',
    'ProtectionLevel',
    'MindReader',
    'get_mind_reader',
    'get_user_state',

    # Covenant
    'Covenant',
    'CovenantValues',
    'FounderProtectionRules',
    'TrustAccount',
    'AutonomyLevel',
    'AutonomyBoundary',
    'get_covenant',

    # Chief of Staff
    'ChiefOfStaff',
    'CEODecisionResult',
    'get_chief_of_staff',

    # Sovereign Loop
    'SovereignState',
    'TickResult',
    'sovereign_tick',
    'live',
    'get_state',
    'get_status',
    'print_status',
    'submit_initiative',
    'get_initiative_queue',
    'trigger_kill_switch',
    'enter_safe_mode',
    'exit_safe_mode',
    'demo',

    # Holographic Teleoplastic Core
    'PlasticityMode',
    'PlasticityConfig',
    'PLASTICITY_CONFIGS',
    'ResonanceResult',
    'HolographicCore',
    'get_htc',
    'select_plasticity_mode',

    # V2 State Architecture
    'SovereignStateV2',
    'TimeState',
    'HardwareState',
    'NodeHardwareState',
    'DeviceLoad',
    'PciLinkState',
    'UserStateV2',
    'SoulState',
    'TeleologyState',
    'GoalState',
    'WorkState',
    'InitiativeStateV2',
    'SkillInvocation',
    'SafetyState',
    'AvatarState',
    'TraceState',
    'AutonomyLevelV2',
    'InitiativeStatusV2',
    'RiskLevel',
    'create_initial_state',
    'clone_state',
    'state_to_dict',
    'state_to_json',
    'state_summary',
    'compute_global_coherence',

    # V2 Tick Functions
    'SovereignTick',
    'sovereign_tick_v2',
    'sense_phase',
    'soul_phase',
    'teleology_phase',
    'plan_phase',
    'act_phase',
]
