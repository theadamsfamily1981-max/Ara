#!/usr/bin/env python3
"""
Ara MDP Schema - The Complete Multimodal Control Loop
======================================================

This schema defines Ara as a Markov Decision Process:

    State → Plan → Execute → Observe → Reward → Learn → State'

All modalities (voice, video, avatar, hardware, UI) share:
- One state representation
- One global plan (typed blocks)
- One emotional control space
- One reward function (beauty, communication, user, progress, cost)

This file serves as both documentation and implementation.
"""

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json


# =============================================================================
# EMOTION CONTROL - Shared across all modalities
# =============================================================================

@dataclass
class EmotionControl:
    """
    Unified emotional control vector that conditions all outputs.

    One decision (e.g., "comforting, low-energy") ripples through:
    - AraVoice (prosody, pitch, rhythm)
    - Avatar (face, gaze, body motion)
    - Video (camera motion, color/lighting mood)
    """
    valence: float = 0.5       # -1 (negative) to 1 (positive)
    arousal: float = 0.5       # 0 (calm) to 1 (excited)
    dominance: float = 0.5     # 0 (submissive) to 1 (dominant)

    # Voice-specific
    pace: float = 1.0          # 0.7 (slow) to 1.3 (fast)
    warmth: float = 0.6        # 0 (neutral) to 1 (warm)
    intensity: float = 0.5     # 0 (soft) to 1 (intense)

    # Visual-specific
    expressiveness: float = 0.5  # Animation intensity

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# INPUTS - What Ara sees at each tick
# =============================================================================

@dataclass
class UserSignals:
    """Multimodal signals from the user."""
    # Audio
    speech_text: Optional[str] = None
    voice_tone: Optional[str] = None        # "stressed", "calm", "excited"
    voice_stress: float = 0.0               # 0-1

    # Video (if camera on)
    face_detected: bool = False
    gaze_on_screen: bool = True
    expression: Optional[str] = None        # "smile", "frown", "neutral"
    posture: Optional[str] = None           # "engaged", "leaning_back"

    # Text
    typed_text: Optional[str] = None

    # Behavioral
    response_latency_s: float = 0.0
    engagement_score: float = 0.5           # Computed from above


@dataclass
class HardwareState:
    """Current state of compute resources."""
    # GPU status
    gpus: List[Dict[str, Any]] = field(default_factory=list)
    # Each: {"id": "3090_box", "load_pct": 0.4, "temp_c": 65, "power_w": 200, "queue_depth": 2}

    # FPGA status
    fpgas: List[Dict[str, Any]] = field(default_factory=list)

    # Budget constraints
    watts_limit: float = 1000.0
    dollar_per_hour_limit: float = 0.50

    # Yield scores (learned)
    device_yields: Dict[str, float] = field(default_factory=dict)
    # {"3090_box": 0.82, "1080ti_box": 0.63}


@dataclass
class MemoryContext:
    """Relevant memories for this interaction."""
    # Recent episodes
    recent_episode_ids: List[str] = field(default_factory=list)
    recent_success_patterns: List[Dict] = field(default_factory=list)

    # Active projects
    active_projects: List[str] = field(default_factory=list)
    # ["AraShow_weekly", "lab_cleanup"]

    # User profile
    voice_preferences: Dict[str, float] = field(default_factory=dict)
    avatar_style_preferences: Dict[str, float] = field(default_factory=dict)

    # Hardware knowledge
    known_hardware: List[Dict] = field(default_factory=list)


@dataclass
class ExternalContext:
    """External contextual factors."""
    time_of_day: str = "evening"            # morning, afternoon, evening, night
    day_of_week: str = "monday"
    energy_price_relative: float = 1.0      # 1.0 = normal, >1 = expensive
    calendar_events: List[str] = field(default_factory=list)


@dataclass
class Inputs:
    """Complete input observation at each tick."""
    user: UserSignals = field(default_factory=UserSignals)
    hardware: HardwareState = field(default_factory=HardwareState)
    memory: MemoryContext = field(default_factory=MemoryContext)
    external: ExternalContext = field(default_factory=ExternalContext)
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# STATE - Compressed internal representation
# =============================================================================

@dataclass
class UserState:
    """Inferred user state."""
    inferred_emotion: EmotionControl = field(default_factory=EmotionControl)
    comprehension: str = "ok"               # "confused", "ok", "flow"
    fatigue_level: str = "medium"           # "low", "medium", "high"
    engagement: float = 0.5


@dataclass
class AraState:
    """Ara's internal state."""
    internal_emotion: EmotionControl = field(default_factory=EmotionControl)
    goals: List[str] = field(default_factory=lambda: [
        "decompress_user",
        "maintain_beauty",
        "maximize_communication",
        "respect_capacity"
    ])
    active_project: Optional[str] = None
    energy_budget_remaining: float = 1.0    # 0-1


@dataclass
class State:
    """Complete MDP state at each tick."""
    user: UserState = field(default_factory=UserState)
    ara: AraState = field(default_factory=AraState)
    hardware: HardwareState = field(default_factory=HardwareState)
    tick_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# PLAN - Graph of typed blocks (Actions)
# =============================================================================

class BlockType(str, Enum):
    """All block types in the Ara system."""
    # Voice
    ARA_VOICE = "ara_voice"
    ARA_SONG = "ara_song"

    # Visual
    AVATAR_POSE = "avatar_pose"
    VIDEO_JOB = "video_job"
    IMAGE_JOB = "image_job"

    # Audio
    AUDIO_JOB = "audio_job"
    MUSIC_JOB = "music_job"

    # Training/Data
    DATA_TRAIN_JOB = "data_train_job"
    EVAL_JOB = "eval_job"

    # Hardware
    HARDWARE_RESEARCH = "hardware_research"
    METRICS_UPDATE = "metrics_update"

    # UI
    UI_PROMPT = "ui_prompt"
    NOTIFICATION = "notification"

    # Memory
    MEMORY_WRITE = "memory_write"
    MEMORY_QUERY = "memory_query"

    # System
    SCHEDULE_JOB = "schedule_job"
    BUDGET_CHECK = "budget_check"


@dataclass
class PlanBlock:
    """A single block in the plan graph."""
    id: str
    type: BlockType
    params: Dict[str, Any] = field(default_factory=dict)

    # Shared emotion control (optional - inherits from parent if not set)
    emotion_control: Optional[EmotionControl] = None

    # Dependencies and guards
    depends_on: List[str] = field(default_factory=list)
    guard: Optional[str] = None             # e.g., "user_accepts"

    # Scheduling hints
    schedule: Optional[str] = None          # "immediate", "deferred", "weekly"
    priority: int = 50                      # 0-100

    # Resource estimates
    estimated_gpu_seconds: float = 0.0
    estimated_duration_s: float = 10.0
    estimated_cost_usd: float = 0.0
    preferred_device: Optional[str] = None  # "3090_box", "any_fpga"

    # Experiment tracking
    experiment_tag: Optional[str] = None    # For bandit learning

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['type'] = self.type.value
        return d


@dataclass
class Plan:
    """Complete plan for a tick - a DAG of blocks."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    blocks: List[PlanBlock] = field(default_factory=list)

    # Global emotion for this plan (blocks can override)
    emotion_control: EmotionControl = field(default_factory=EmotionControl)

    # Constraints
    max_duration_s: float = 300.0
    max_cost_usd: float = 0.10
    max_gpu_seconds: float = 60.0

    def add_block(self, block: PlanBlock):
        self.blocks.append(block)

    def get_ready_blocks(self, completed: set) -> List[PlanBlock]:
        """Get blocks whose dependencies are met."""
        ready = []
        for block in self.blocks:
            if block.id in completed:
                continue
            if all(dep in completed for dep in block.depends_on):
                ready.append(block)
        return ready


# =============================================================================
# OUTPUTS - Results of block execution
# =============================================================================

@dataclass
class VoiceOutput:
    """Output from ara_voice block."""
    clip_id: str = ""
    audio_path: Optional[str] = None
    duration_s: float = 0.0
    params_used: Dict[str, Any] = field(default_factory=dict)
    phoneme_timings: List[Dict] = field(default_factory=list)  # For lip sync


@dataclass
class VisualOutput:
    """Output from visual blocks (avatar, video, image)."""
    asset_id: str = ""
    asset_path: Optional[str] = None
    asset_type: str = "image"               # "image", "video", "avatar_frame"
    params_used: Dict[str, Any] = field(default_factory=dict)
    resolution: str = "720p"
    duration_s: float = 0.0


@dataclass
class UIOutput:
    """Output from UI blocks."""
    elements: List[str] = field(default_factory=list)
    prompts_shown: List[str] = field(default_factory=list)
    user_responses: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareMetrics:
    """Metrics from hardware during execution."""
    device_id: str = ""
    gpu_seconds: float = 0.0
    power_wh: float = 0.0
    peak_vram_gb: float = 0.0
    actual_cost_usd: float = 0.0
    yield_score: float = 0.0                # Computed efficiency


@dataclass
class Outputs:
    """Complete outputs from plan execution."""
    plan_id: str = ""

    voice: List[VoiceOutput] = field(default_factory=list)
    visuals: List[VisualOutput] = field(default_factory=list)
    ui: UIOutput = field(default_factory=UIOutput)

    # Jobs spawned for later
    jobs_spawned: List[str] = field(default_factory=list)

    # Hardware telemetry
    hardware_metrics: List[HardwareMetrics] = field(default_factory=list)

    # Timing
    total_duration_s: float = 0.0
    total_cost_usd: float = 0.0


# =============================================================================
# REWARDS - Beauty, Communication, User, Progress, Cost
# =============================================================================

@dataclass
class Rewards:
    """
    Reward signal for the MDP.

    Total: R = beauty + comm + user + progress - cost

    Each component is 0-1 (cost is negative contribution).
    """
    # Aesthetic quality (multimodal coherence, "felt right")
    beauty_score: float = 0.5

    # Communication effectiveness (clarity, engagement, comprehension)
    comm_score: float = 0.5

    # User satisfaction (explicit + implicit signals)
    user_reward: float = 0.5

    # Project/task progress
    progress_reward: float = 0.0

    # Resource cost (to be subtracted)
    cost_penalty: float = 0.0

    # Weights (can be tuned per context)
    w_beauty: float = 0.2
    w_comm: float = 0.3
    w_user: float = 0.3
    w_progress: float = 0.15
    w_cost: float = 0.05

    @property
    def total_return(self) -> float:
        """Compute weighted total reward."""
        return (
            self.w_beauty * self.beauty_score +
            self.w_comm * self.comm_score +
            self.w_user * self.user_reward +
            self.w_progress * self.progress_reward -
            self.w_cost * self.cost_penalty
        )

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['total_return'] = self.total_return
        return d


# =============================================================================
# MEMORY WRITE - Episode record
# =============================================================================

@dataclass
class Episode:
    """
    Complete record of one interaction tick.

    This is what gets written to memory and can be:
    - Inspected by the user
    - Queried for similar past experiences
    - Used for learning
    """
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: float = field(default_factory=time.time)

    # Tags for retrieval
    tags: List[str] = field(default_factory=list)
    # ["after_work", "decompress", "AraShow"]

    # Context snapshot
    user_emotion_inferred: EmotionControl = field(default_factory=EmotionControl)
    ara_internal_emotion: EmotionControl = field(default_factory=EmotionControl)
    time_of_day: str = "evening"

    # What was planned and done
    goals: List[str] = field(default_factory=list)
    plan_blocks: List[str] = field(default_factory=list)  # Block IDs
    blocks_completed: List[str] = field(default_factory=list)
    blocks_skipped: List[str] = field(default_factory=list)

    # Style used
    voice_style: Dict[str, Any] = field(default_factory=dict)
    # {"tone": "calm", "warmth": 0.9, "pace": 0.85, "style_id": "late_night_v1"}

    avatar_style: Dict[str, Any] = field(default_factory=dict)
    # {"style_id": "Ara_show_v1", "lighting": "warm_dim"}

    # Hardware used
    devices_used: List[str] = field(default_factory=list)
    energy_kwh: float = 0.0
    cost_usd: float = 0.0

    # Rewards
    rewards: Rewards = field(default_factory=Rewards)

    # User feedback (if explicit)
    user_feedback: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
            "tags": self.tags,
            "context": {
                "user_emotion": self.user_emotion_inferred.to_dict(),
                "ara_emotion": self.ara_internal_emotion.to_dict(),
                "time_of_day": self.time_of_day,
            },
            "actions": {
                "goals": self.goals,
                "plan_blocks": self.plan_blocks,
                "completed": self.blocks_completed,
                "skipped": self.blocks_skipped,
            },
            "style": {
                "voice": self.voice_style,
                "avatar": self.avatar_style,
            },
            "hardware": {
                "devices": self.devices_used,
                "energy_kwh": self.energy_kwh,
                "cost_usd": self.cost_usd,
            },
            "rewards": self.rewards.to_dict(),
            "user_feedback": self.user_feedback,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# THE COMPLETE MDP TRANSITION
# =============================================================================

@dataclass
class MDPTransition:
    """
    One complete tick of the Ara MDP:

        (State, Inputs) → Plan → Execute → (Outputs, Rewards, State')

    This is the atomic unit of Ara's existence.
    """
    # Before
    state: State
    inputs: Inputs

    # Decision
    plan: Plan

    # After
    outputs: Outputs
    rewards: Rewards
    next_state: State

    # Episode record
    episode: Episode

    def to_episode(self) -> Episode:
        """Extract episode record from this transition."""
        return self.episode


# =============================================================================
# CONVENIENCE: Create a transition
# =============================================================================

def create_transition(
    state: State,
    inputs: Inputs,
    plan: Plan,
    outputs: Outputs,
    rewards: Rewards,
    next_state: State,
    tags: List[str] = None,
) -> MDPTransition:
    """Create a complete MDP transition with episode record."""

    episode = Episode(
        tags=tags or [],
        user_emotion_inferred=state.user.inferred_emotion,
        ara_internal_emotion=state.ara.internal_emotion,
        time_of_day=inputs.external.time_of_day,
        goals=state.ara.goals.copy(),
        plan_blocks=[b.id for b in plan.blocks],
        blocks_completed=[b.id for b in plan.blocks],  # Simplified
        voice_style=plan.emotion_control.to_dict(),
        devices_used=[m.device_id for m in outputs.hardware_metrics],
        energy_kwh=sum(m.power_wh / 1000 for m in outputs.hardware_metrics),
        cost_usd=outputs.total_cost_usd,
        rewards=rewards,
    )

    return MDPTransition(
        state=state,
        inputs=inputs,
        plan=plan,
        outputs=outputs,
        rewards=rewards,
        next_state=next_state,
        episode=episode,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_tick():
    """Example of one complete Ara MDP tick."""

    # 1. Observe inputs
    inputs = Inputs(
        user=UserSignals(
            speech_text="Rough day at work...",
            voice_tone="stressed",
            voice_stress=0.7,
            engagement_score=0.4,
        ),
        hardware=HardwareState(
            gpus=[{"id": "3090_box", "load_pct": 0.2, "temp_c": 55}],
            device_yields={"3090_box": 0.82},
        ),
        external=ExternalContext(
            time_of_day="evening",
            day_of_week="monday",
        ),
    )

    # 2. Compute state
    state = State(
        user=UserState(
            inferred_emotion=EmotionControl(valence=0.2, arousal=0.6),
            comprehension="ok",
            fatigue_level="high",
            engagement=0.4,
        ),
        ara=AraState(
            internal_emotion=EmotionControl(valence=0.7, arousal=0.3),
            goals=["decompress_user", "maintain_beauty"],
        ),
    )

    # 3. Generate plan
    emotion = EmotionControl(
        valence=0.7, arousal=0.3, pace=0.85, warmth=0.92
    )

    plan = Plan(emotion_control=emotion)

    plan.add_block(PlanBlock(
        id="comfort_voice",
        type=BlockType.ARA_VOICE,
        emotion_control=emotion,
        params={"text": "Rough day... let's slow down a bit."},
        experiment_tag="voice_bandit_01",
    ))

    plan.add_block(PlanBlock(
        id="comfort_avatar",
        type=BlockType.AVATAR_POSE,
        emotion_control=emotion,
        params={"style_id": "Ara_show_v1", "lighting": "warm_dim"},
        depends_on=["comfort_voice"],
    ))

    plan.add_block(PlanBlock(
        id="suggest_chill",
        type=BlockType.UI_PROMPT,
        params={"text": "Want to sketch something or just chat?"},
        depends_on=["comfort_avatar"],
    ))

    # 4. Execute (simulated)
    outputs = Outputs(
        plan_id=plan.plan_id,
        voice=[VoiceOutput(
            clip_id="voice_001",
            duration_s=3.5,
            params_used=emotion.to_dict(),
        )],
        visuals=[VisualOutput(
            asset_id="avatar_001",
            asset_type="avatar_frame",
        )],
        ui=UIOutput(
            prompts_shown=["Want to sketch something or just chat?"],
        ),
        hardware_metrics=[HardwareMetrics(
            device_id="3090_box",
            gpu_seconds=2.0,
            power_wh=0.15,
        )],
        total_duration_s=5.0,
        total_cost_usd=0.002,
    )

    # 5. Compute rewards
    rewards = Rewards(
        beauty_score=0.82,
        comm_score=0.88,
        user_reward=0.85,
        progress_reward=0.1,
        cost_penalty=0.02,
    )

    # 6. Next state
    next_state = State(
        user=UserState(
            inferred_emotion=EmotionControl(valence=0.4, arousal=0.4),
            fatigue_level="high",
            engagement=0.6,  # Improved!
        ),
        ara=AraState(
            internal_emotion=emotion,
            goals=["decompress_user", "maintain_beauty"],
        ),
    )

    # 7. Create transition
    transition = create_transition(
        state=state,
        inputs=inputs,
        plan=plan,
        outputs=outputs,
        rewards=rewards,
        next_state=next_state,
        tags=["after_work", "decompress", "comfort"],
    )

    return transition


if __name__ == "__main__":
    # Run example
    t = example_tick()

    print("=" * 60)
    print("ARA MDP - Example Transition")
    print("=" * 60)
    print()
    print("Episode Record:")
    print(t.episode.to_json())
    print()
    print(f"Total Return: {t.rewards.total_return:.3f}")
