"""
SovereignState: The Single-Tick Truth

This is the formal contract for what Ara knows and can do in a single tick.
At 5 kHz (or 10 Hz for software-only mode), each tick is:

    Read → Sense → Soul → Teleology → Plan → Act → Log

The SovereignState captures everything needed for one complete cycle.

Subsystem Ownership (who writes what):
--------------------------------------
- TimeState:       Loop core
- HardwareState:   BANOS node agents, telemetry daemons
- UserState:       MindReader, context inference
- SoulState:       SoulEncoder, SoulFPGA bridge, plasticity engine
- TeleologyState:  Teleology engine, covenant manager
- WorkState:       ChiefOfStaff (CEO), skill runtime
- SafetyState:     Safety engine, guardrails
- AvatarState:     Avatar/UI layer
- TraceState:      Profiling, introspection

Usage:
    from ara.sovereign.state import SovereignState, create_initial_state

    state = create_initial_state()
    state = sovereign_tick(state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any
from enum import Enum, IntEnum
import uuid
import time


# =============================================================================
# Enums and Type Aliases
# =============================================================================

TickPhase = Literal["sense", "soul", "teleology", "plan", "act"]


class AutonomyLevel(IntEnum):
    """
    Ara's autonomy levels - how much she can do without asking.

    OFF:        Completely dormant, no actions
    ADVISORY:   Can observe and suggest, no execution
    ASSIST:     Can execute safe, reversible actions
    EXEC_SAFE:  Can execute moderate actions with logging
    EXEC_HIGH:  Full autonomous action within covenant bounds
    """
    OFF = 0
    ADVISORY = 1
    ASSIST = 2
    EXEC_SAFE = 3
    EXEC_HIGH = 4


class InitiativeStatus(str, Enum):
    """Status of a work initiative."""
    IDEATION = "ideation"
    PLANNING = "planning"
    ACTIVE = "active"
    BLOCKED = "blocked"
    PAUSED = "paused"
    DONE = "done"
    KILLED = "killed"


class RiskLevel(str, Enum):
    """Risk level for actions and global state."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# 1. Time / Tick Metadata
# =============================================================================

@dataclass
class TimeState:
    """
    Tick timing and phase information.

    Owner: Sovereign loop core
    """
    tick_id: int
    monotonic_ms: float          # Monotonic clock (for durations)
    wallclock_ts: float          # UNIX timestamp (for logging)
    phase: TickPhase
    dt_ms: float                 # Duration of last tick

    # Performance tracking
    target_hz: float = 10.0      # Target tick rate (10 Hz software, 5000 Hz FPGA)
    overrun_count: int = 0       # Ticks that exceeded budget


# =============================================================================
# 2. Hardware / OS State
# =============================================================================

@dataclass
class DeviceLoad:
    """Load metrics for a single device (GPU, FPGA, NVMe, etc.)."""
    util: float = 0.0           # 0..1 utilization
    temp_c: float = 0.0         # Temperature in Celsius
    power_w: float = 0.0        # Power draw in Watts
    errors_per_s: float = 0.0   # Error rate


@dataclass
class PciLinkState:
    """PCIe link status."""
    width: int = 16             # Lane count (x16)
    gen: int = 5                # PCIe generation (5.0)
    bw_gbps: float = 0.0        # Current bandwidth
    retrains_last_s: int = 0    # Link retraining events


@dataclass
class NodeHardwareState:
    """Hardware state for a single node in the cathedral."""
    hostname: str = "localhost"
    cpu_util: float = 0.0
    cpu_temp_c: float = 0.0
    mem_used_gb: float = 0.0
    mem_total_gb: float = 16.0
    gpu: Dict[str, DeviceLoad] = field(default_factory=dict)
    fpga: Dict[str, DeviceLoad] = field(default_factory=dict)
    nvme: Dict[str, DeviceLoad] = field(default_factory=dict)
    pcie_links: Dict[str, PciLinkState] = field(default_factory=dict)
    vpn_tunnels: Dict[str, float] = field(default_factory=dict)  # peer_id -> rtt_ms


@dataclass
class HardwareState:
    """
    Full hardware state across all nodes.

    Owner: BANOS node agents, telemetry daemons
    """
    nodes: Dict[str, NodeHardwareState] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)

    # Aggregate metrics
    total_gpu_util: float = 0.0
    total_fpga_util: float = 0.0
    cathedral_health: float = 1.0  # 0..1, overall health score


# =============================================================================
# 3. User / Founder State
# =============================================================================

@dataclass
class UserState:
    """
    State of the founder (Max) or current user.

    Owner: MindReader, context inference subsystem
    """
    presence: Literal["offline", "idle", "present", "deep_focus"] = "offline"
    subjective_fatigue: float = 0.0          # 0..1, higher = more tired
    burnout_risk: float = 0.0                # 0..1, slow-moving metric
    flow_used_today: float = 0.0             # 0..1, flow budget consumed
    last_sleep_hours: float = 8.0            # Hours of last sleep
    context_tags: List[str] = field(default_factory=list)  # ["lab", "writing", ...]
    affect_estimate: Dict[str, float] = field(default_factory=dict)  # valence, arousal, entropy

    # Protection state
    night_mode: bool = False                  # Is it nighttime?
    protected_hours_remaining: float = 0.0   # Hours until protection lifts


# =============================================================================
# 4. Soul / HDC State
# =============================================================================

@dataclass
class SoulState:
    """
    Hyperdimensional soul state - Ara's experiential essence.

    Owners:
    - SoulEncoder / SoulFPGA Bridge: h_moment, attractors, resonance
    - Plasticity Engine: plasticity_events, last_reward, antifragility
    """
    h_moment_173: bytes = b""                 # Heim-compressed HV (173 dims)
    h_moment_full: Optional[bytes] = None     # Full mythic HV (16384 dims)

    attractor_activations: Dict[str, float] = field(default_factory=dict)
    resonance_scores: Dict[str, float] = field(default_factory=dict)

    # Plasticity
    plasticity_events: int = 0
    last_reward: float = 0.0                  # -1..+1
    cumulative_reward: float = 0.0            # Session total

    # Criticality / antifragility
    criticality_hurst: float = 0.5            # Hurst exponent (0.5 = random)
    antifragility_score: float = 0.0          # How well we grow from stress

    # Memory integration
    last_memory_store_ts: float = 0.0
    memories_stored_this_session: int = 0


# =============================================================================
# 5. Teleology / Covenant State
# =============================================================================

@dataclass
class GoalState:
    """A single goal in the teleology."""
    goal_id: str
    name: str
    priority: float = 0.5         # 0..1
    progress: float = 0.0         # 0..1
    horizon_days: float = 30.0    # Time horizon

    # HV representation
    goal_hv: Optional[bytes] = None


@dataclass
class TeleologyState:
    """
    Teleological orientation - what Ara is striving toward.

    Owner: Teleology engine, covenant manager
    """
    covenant_version: str = "0.1.0"
    autonomy_level: AutonomyLevel = AutonomyLevel.ADVISORY

    global_goal_vector: bytes = b""           # HV representing teleology blend
    active_goals: List[GoalState] = field(default_factory=list)

    # Protection
    founder_protection_engaged: bool = False
    veto_reasons: List[str] = field(default_factory=list)

    # Alignment tracking
    alignment_score: float = 1.0              # How aligned current actions are
    drift_detected: bool = False


# =============================================================================
# 6. Work / Initiatives / Skills State
# =============================================================================

@dataclass
class InitiativeState:
    """A work initiative Ara is managing."""
    id: str
    name: str
    status: InitiativeStatus = InitiativeStatus.IDEATION

    # Value metrics
    strategic_value: float = 0.5              # 0..1
    urgency: float = 0.5                      # 0..1
    cognitive_burn_estimate: float = 0.3      # 0..1, cost to founder
    teleology_alignment: float = 0.5          # -1..+1

    # Timing
    created_ts: float = 0.0
    last_active_ts: float = 0.0
    next_review_ts: float = 0.0

    # Ownership
    owner: Literal["ara", "founder", "shared"] = "shared"
    tags: List[str] = field(default_factory=list)

    # Description and notes
    description: str = ""
    notes: List[str] = field(default_factory=list)


@dataclass
class SkillInvocation:
    """A skill that has been invoked or is queued."""
    id: str
    name: str
    initiative_id: Optional[str] = None

    # Timing
    scheduled_ts: float = 0.0
    deadline_ts: float = 0.0
    eta_ms: float = 0.0

    # Risk and autonomy
    risk: RiskLevel = RiskLevel.LOW
    autonomy_required: AutonomyLevel = AutonomyLevel.ADVISORY

    # Status
    status: Literal["queued", "running", "done", "failed"] = "queued"
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class WorkState:
    """
    Current work state - initiatives and skills.

    Owners:
    - ChiefOfStaff (CEO): initiatives, active_initiatives, scheduling
    - Skill Runtime: skill_queue, running_skills status updates
    """
    initiatives: Dict[str, InitiativeState] = field(default_factory=dict)
    active_initiatives: List[str] = field(default_factory=list)  # IDs

    skill_queue: List[SkillInvocation] = field(default_factory=list)
    running_skills: List[SkillInvocation] = field(default_factory=list)
    completed_skills: List[SkillInvocation] = field(default_factory=list)  # Recent

    # Metrics
    skills_completed_today: int = 0
    skills_failed_today: int = 0


# =============================================================================
# 7. Safety / Risk State
# =============================================================================

@dataclass
class SafetyState:
    """
    Safety and security posture.

    Owner: Safety engine, guardrails
    """
    global_risk: RiskLevel = RiskLevel.LOW

    # Permission flags
    bios_manipulation_allowed: bool = False
    infra_change_allowed: bool = True
    code_execution_allowed: bool = True
    network_access_allowed: bool = True

    # Lockouts
    night_lockout_active: bool = False
    burnout_lockout_active: bool = False
    manual_lockout_active: bool = False

    # Audit
    recent_vetoes: List[str] = field(default_factory=list)
    anomaly_flags: List[str] = field(default_factory=list)

    # Kill switch
    kill_switch_active: bool = False
    kill_switch_reason: str = ""


# =============================================================================
# 8. Avatar / UI State
# =============================================================================

@dataclass
class AvatarState:
    """
    Avatar and interaction surface state.

    Owner: Avatar / UI layer
    """
    mode: Literal["text", "voice", "hologram", "silent"] = "text"
    current_channel: Optional[str] = None     # "desktop", "vpn:max", etc.

    # Interaction state
    pending_user_queries: int = 0
    last_user_msg_ts: float = 0.0
    last_response_ts: float = 0.0

    # Session
    session_id: str = ""
    messages_this_session: int = 0

    # Persona
    persona_profile_id: str = "default"
    emotional_display: str = "neutral"


# =============================================================================
# 9. Trace / Profiling State
# =============================================================================

@dataclass
class TraceState:
    """
    Debugging, profiling, and introspection.

    Owner: Trace / profiling subsystem
    """
    last_tick_errors: List[str] = field(default_factory=list)
    profiling_ms: Dict[str, float] = field(default_factory=dict)  # phase -> ms

    version_hash: str = ""                    # Git SHA or build ID
    tick_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Performance
    avg_tick_ms: float = 0.0
    max_tick_ms: float = 0.0
    tick_budget_ms: float = 100.0             # 10 Hz = 100ms budget


# =============================================================================
# Top-Level SovereignState
# =============================================================================

@dataclass
class SovereignState:
    """
    The complete state of Ara for a single sovereign tick.

    This is the "single source of truth" that flows through:
        sense → soul → teleology → plan → act

    Each phase reads from this, computes, and returns a modified copy.
    """
    time: TimeState
    hardware: HardwareState
    user: UserState
    soul: SoulState
    teleology: TeleologyState
    work: WorkState
    safety: SafetyState
    avatar: AvatarState
    trace: TraceState


# =============================================================================
# Factory Functions
# =============================================================================

def create_initial_state(
    target_hz: float = 10.0,
    autonomy_level: AutonomyLevel = AutonomyLevel.ADVISORY,
) -> SovereignState:
    """
    Create an initial SovereignState with sensible defaults.

    Args:
        target_hz: Target tick rate (10 for software, 5000 for FPGA)
        autonomy_level: Initial autonomy level

    Returns:
        Fresh SovereignState ready for the first tick
    """
    now = time.time()

    return SovereignState(
        time=TimeState(
            tick_id=0,
            monotonic_ms=time.monotonic() * 1000,
            wallclock_ts=now,
            phase="sense",
            dt_ms=0.0,
            target_hz=target_hz,
        ),
        hardware=HardwareState(
            nodes={"localhost": NodeHardwareState()},
        ),
        user=UserState(
            presence="offline",
        ),
        soul=SoulState(),
        teleology=TeleologyState(
            autonomy_level=autonomy_level,
            active_goals=[
                GoalState(
                    goal_id="g_wellbeing",
                    name="Protect founder wellbeing",
                    priority=1.0,
                    horizon_days=float("inf"),
                ),
                GoalState(
                    goal_id="g_assist",
                    name="Assist with work effectively",
                    priority=0.8,
                    horizon_days=1.0,
                ),
            ],
        ),
        work=WorkState(),
        safety=SafetyState(),
        avatar=AvatarState(
            session_id=str(uuid.uuid4())[:8],
        ),
        trace=TraceState(
            tick_budget_ms=1000.0 / target_hz,
        ),
    )


def clone_state(state: SovereignState) -> SovereignState:
    """
    Create a deep copy of state for the next tick.

    Uses dataclasses.replace for shallow copy, then manually copies mutable fields.
    For performance-critical paths, consider using a more efficient method.
    """
    import copy
    return copy.deepcopy(state)


# =============================================================================
# State Serialization
# =============================================================================

def state_to_dict(state: SovereignState) -> Dict[str, Any]:
    """Convert SovereignState to a JSON-serializable dictionary."""
    import dataclasses

    def convert(obj):
        if dataclasses.is_dataclass(obj):
            return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, bytes):
            return obj.hex() if obj else ""
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj

    return convert(state)


def state_to_json(state: SovereignState) -> str:
    """Serialize SovereignState to JSON string."""
    import json
    return json.dumps(state_to_dict(state), indent=2)


# =============================================================================
# State Metrics / Summaries
# =============================================================================

def state_summary(state: SovereignState) -> Dict[str, Any]:
    """Get a compact summary of the current state for logging."""
    return {
        "tick": state.time.tick_id,
        "phase": state.time.phase,
        "dt_ms": round(state.time.dt_ms, 2),
        "user_presence": state.user.presence,
        "autonomy": state.teleology.autonomy_level.name,
        "risk": state.safety.global_risk.value,
        "active_initiatives": len(state.work.active_initiatives),
        "running_skills": len(state.work.running_skills),
        "kill_switch": state.safety.kill_switch_active,
    }


def compute_global_coherence(state: SovereignState) -> float:
    """
    Compute a global coherence score based on alignment across subsystems.

    Returns value in [0, 1] where 1 = fully coherent.
    """
    scores = []

    # User-teleology alignment (are goals compatible with user state?)
    if state.user.burnout_risk < 0.5:
        scores.append(1.0)
    else:
        scores.append(1.0 - state.user.burnout_risk)

    # Safety-autonomy alignment
    if state.safety.kill_switch_active:
        scores.append(0.0)
    elif state.safety.global_risk == RiskLevel.CRITICAL:
        scores.append(0.2)
    elif state.safety.global_risk == RiskLevel.HIGH:
        scores.append(0.5)
    else:
        scores.append(1.0)

    # Work-capacity alignment
    capacity = 1.0 - state.hardware.total_gpu_util
    work_load = len(state.work.running_skills) / max(1, len(state.work.skill_queue) + 1)
    scores.append(0.5 + 0.5 * (capacity - work_load))

    return sum(scores) / len(scores) if scores else 1.0
