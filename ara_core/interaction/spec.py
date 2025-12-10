#!/usr/bin/env python3
"""
Ara Interaction Spec - Block Language Definition
=================================================

Defines the InteractionSpec schema and Block types for Ara's
cost-aware, inspectable multimodal behavior system.

This is Ara's "interaction block language" - turning multimodal
behavior into a controllable MDP instead of vibes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal
from enum import Enum
import json
import yaml


class BlockType(str, Enum):
    """Available block types in the interaction system."""
    ARA_VOICE = "ara_voice"       # Vocal output (TTS or synth)
    ARA_SONG = "ara_song"         # Music generation
    UI_PROMPT = "ui_prompt"       # UI interaction
    VIDEO_JOB = "video_job"       # Hive video generation job
    AUDIO_JOB = "audio_job"       # Hive audio generation job
    METRICS_UPDATE = "metrics_update"  # Hardware/performance metrics
    MEMORY_WRITE = "memory_write"      # Write to memory fabric
    MEMORY_READ = "memory_read"        # Read from memory fabric


@dataclass
class Block:
    """
    A single action block in an interaction plan.

    Blocks are the atomic units of Ara's behavior - each one is
    typed, has parameters, and can have guard conditions.
    """
    id: str
    type: BlockType
    params: Dict[str, Any] = field(default_factory=dict)
    guard: Optional[str] = None  # Condition that must be true to execute
    depends_on: List[str] = field(default_factory=list)  # Block IDs this depends on

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "params": self.params,
            "guard": self.guard,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Block":
        return cls(
            id=data["id"],
            type=BlockType(data["type"]),
            params=data.get("params", {}),
            guard=data.get("guard"),
            depends_on=data.get("depends_on", []),
        )


@dataclass
class InputSpec:
    """Specification of inputs for an interaction."""
    audio: Optional[str] = None      # Audio source (e.g., "mic_stream")
    video: Optional[str] = None      # Video source (e.g., "webcam_stream")
    system: Dict[str, Any] = field(default_factory=dict)   # System metrics
    memory: Dict[str, Any] = field(default_factory=dict)   # Memory queries
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context


@dataclass
class StateSpec:
    """State snapshot for an interaction."""
    emotion: str = "neutral"
    energy: float = 0.5
    goals: List[str] = field(default_factory=list)
    capacity: Dict[str, float] = field(default_factory=dict)  # Resource budgets
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputSpec:
    """Expected outputs from an interaction."""
    voice: Optional[str] = None      # Text to speak
    ui: Optional[str] = None         # UI action/display
    jobs: List[str] = field(default_factory=list)  # Job IDs to enqueue
    audio: Optional[bytes] = None    # Raw audio output
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryWriteSpec:
    """What to remember from this interaction."""
    episode_tags: List[str] = field(default_factory=list)
    emotion: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None


@dataclass
class InteractionSpec:
    """
    Complete specification for a multimodal interaction.

    This is the core schema that makes Ara's behavior inspectable
    and controllable. Every interaction follows this structure:

    1. Inputs: What sensors/data to read
    2. State: Current emotional/goal state
    3. Plan: Sequence of typed blocks to execute
    4. Outputs: What to produce
    5. Memory: What to remember
    """
    interaction_id: str
    inputs: InputSpec = field(default_factory=InputSpec)
    state: StateSpec = field(default_factory=StateSpec)
    plan: List[Block] = field(default_factory=list)
    outputs: OutputSpec = field(default_factory=OutputSpec)
    memory_write: MemoryWriteSpec = field(default_factory=MemoryWriteSpec)

    # Metadata
    version: str = "1.0.0"
    description: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "interaction_id": self.interaction_id,
            "version": self.version,
            "description": self.description,
            "inputs": {
                "audio": self.inputs.audio,
                "video": self.inputs.video,
                "system": self.inputs.system,
                "memory": self.inputs.memory,
                "context": self.inputs.context,
            },
            "state": {
                "emotion": self.state.emotion,
                "energy": self.state.energy,
                "goals": self.state.goals,
                "capacity": self.state.capacity,
                "context": self.state.context,
            },
            "plan": [b.to_dict() for b in self.plan],
            "outputs": {
                "voice": self.outputs.voice,
                "ui": self.outputs.ui,
                "jobs": self.outputs.jobs,
                "artifacts": self.outputs.artifacts,
            },
            "memory_write": {
                "episode_tags": self.memory_write.episode_tags,
                "emotion": self.memory_write.emotion,
                "artifacts": self.memory_write.artifacts,
                "hardware": self.memory_write.hardware,
                "summary": self.memory_write.summary,
            },
        }

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> "InteractionSpec":
        inputs_data = data.get("inputs", {})
        inputs = InputSpec(
            audio=inputs_data.get("audio"),
            video=inputs_data.get("video"),
            system=inputs_data.get("system", {}),
            memory=inputs_data.get("memory", {}),
            context=inputs_data.get("context", {}),
        )

        state_data = data.get("state", {})
        state = StateSpec(
            emotion=state_data.get("emotion", "neutral"),
            energy=state_data.get("energy", 0.5),
            goals=state_data.get("goals", []),
            capacity=state_data.get("capacity", {}),
            context=state_data.get("context", {}),
        )

        plan = [Block.from_dict(b) for b in data.get("plan", [])]

        outputs_data = data.get("outputs", {})
        outputs = OutputSpec(
            voice=outputs_data.get("voice"),
            ui=outputs_data.get("ui"),
            jobs=outputs_data.get("jobs", []),
            artifacts=outputs_data.get("artifacts", {}),
        )

        memory_data = data.get("memory_write", {})
        memory_write = MemoryWriteSpec(
            episode_tags=memory_data.get("episode_tags", []),
            emotion=memory_data.get("emotion"),
            artifacts=memory_data.get("artifacts", {}),
            hardware=memory_data.get("hardware", {}),
            summary=memory_data.get("summary"),
        )

        return cls(
            interaction_id=data["interaction_id"],
            version=data.get("version", "1.0.0"),
            description=data.get("description"),
            inputs=inputs,
            state=state,
            plan=plan,
            outputs=outputs,
            memory_write=memory_write,
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "InteractionSpec":
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, json_str: str) -> "InteractionSpec":
        data = json.loads(json_str)
        return cls.from_dict(data)


# =============================================================================
# Voice Block Parameters
# =============================================================================

@dataclass
class AraVoiceParams:
    """Parameters for ara_voice blocks."""
    text: str = ""
    tone: str = "neutral"      # calm, warm, excited, etc.
    warmth: float = 0.5        # 0-1
    pace: float = 1.0          # Speech rate multiplier
    emotion: str = "neutral"
    pitch_shift: float = 0.0   # Semitones
    use_tts: bool = False      # True = real TTS, False = synth voice

    @classmethod
    def from_dict(cls, params: Dict) -> "AraVoiceParams":
        return cls(
            text=params.get("text", ""),
            tone=params.get("tone", "neutral"),
            warmth=params.get("warmth", 0.5),
            pace=params.get("pace", 1.0),
            emotion=params.get("emotion", "neutral"),
            pitch_shift=params.get("pitch_shift", 0.0),
            use_tts=params.get("use_tts", False),
        )


@dataclass
class AraSongParams:
    """Parameters for ara_song blocks."""
    song_path: Optional[str] = None  # Path to song JSON
    mood: str = "neutral"
    energy: float = 0.5
    duration_s: Optional[float] = None
    with_vocals: bool = True

    @classmethod
    def from_dict(cls, params: Dict) -> "AraSongParams":
        return cls(
            song_path=params.get("song_path"),
            mood=params.get("mood", "neutral"),
            energy=params.get("energy", 0.5),
            duration_s=params.get("duration_s"),
            with_vocals=params.get("with_vocals", True),
        )


@dataclass
class VideoJobParams:
    """Parameters for video_job blocks."""
    prompt: str = ""
    duration_s: float = 10.0
    resolution: str = "720p"
    style: Optional[str] = None
    model: str = "default"

    @classmethod
    def from_dict(cls, params: Dict) -> "VideoJobParams":
        return cls(
            prompt=params.get("prompt", ""),
            duration_s=params.get("duration_s", 10.0),
            resolution=params.get("resolution", "720p"),
            style=params.get("style"),
            model=params.get("model", "default"),
        )
