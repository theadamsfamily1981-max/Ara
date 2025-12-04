"""MIES Context - Environmental and internal state for modality decisions.

This module defines the context structures that feed into the modality policy:
- ForegroundAppType: What kind of application the user is using
- ActivityType: Higher-level activity classification
- ModalityContext: Full fused context vector for policy decisions

The context is built from:
1. OS scavengers (GNOME focus, PipeWire audio)
2. Affective state (from HomeostaticCore/AppraisalEngine)
3. Thermodynamic state (from ThermodynamicMonitor)
4. Identity context (from NIBManager)
5. Interaction history
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import time


class ForegroundAppType(Enum):
    """Classification of the foreground application.

    Used to determine appropriate modality based on what the user
    is actively working with.
    """
    UNKNOWN = auto()

    # Development
    TERMINAL = auto()
    IDE = auto()               # VS Code, JetBrains, etc.
    TEXT_EDITOR = auto()       # vim, emacs, sublime

    # Communication
    VIDEO_CALL = auto()        # Zoom, Meet, Teams, Discord call
    CHAT_APP = auto()          # Slack, Discord text, Messages
    EMAIL = auto()             # Thunderbird, Evolution, webmail

    # Media
    BROWSER = auto()           # Firefox, Chrome, etc.
    MEDIA_PLAYER = auto()      # VLC, mpv, Spotify
    IMAGE_EDITOR = auto()      # GIMP, Inkscape
    VIDEO_EDITOR = auto()      # Kdenlive, DaVinci

    # Gaming/Entertainment
    FULLSCREEN_GAME = auto()
    CASUAL_GAME = auto()

    # Productivity
    OFFICE_DOCUMENT = auto()   # LibreOffice, Google Docs
    PDF_READER = auto()
    FILE_MANAGER = auto()

    # System
    SETTINGS = auto()
    SYSTEM_MONITOR = auto()


class ActivityType(Enum):
    """Higher-level activity classification.

    Aggregates app type + other signals into a behavioral context
    that determines overall interaction style.
    """
    UNKNOWN = auto()
    DEEP_WORK = auto()         # Coding, writing, focused tasks
    CASUAL_WORK = auto()       # Browsing, light work
    MEETING = auto()           # Video/voice calls
    MEDIA_CONSUMPTION = auto() # Watching, listening
    GAMING = auto()            # Active gaming
    IDLE = auto()              # AFK or idle desktop


@dataclass
class ForegroundInfo:
    """Information about the current foreground window.

    Populated by the GNOME focus sensor.
    """
    app_type: ForegroundAppType
    wm_class: str              # X11/Wayland window class
    title: str                 # Window title
    rect: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    is_fullscreen: bool = False
    pid: Optional[int] = None


@dataclass
class AudioContext:
    """Audio state from PipeWire.

    Populated by the PipeWire audio sensor.
    """
    mic_in_use: bool = False
    speakers_in_use: bool = False
    has_voice_call: bool = False
    music_playing: bool = False
    rms_level: float = 0.0           # 0-1, ambient audio level
    spectral_entropy: float = 0.0    # 0-1, audio complexity


@dataclass
class BiometricState:
    """Optional biometric state from AURA-V style analysis.

    Populated by the biometrics sensor (stub for now).
    """
    blink_rate: Optional[float] = None    # Blinks per minute
    pupil_dilation: Optional[float] = None  # Normalized 0-1
    gaze_stability: Optional[float] = None  # 0-1, how stable is gaze
    estimated_fatigue: Optional[float] = None  # 0-1
    estimated_load: Optional[float] = None  # Cognitive load estimate 0-1


class SomaticState(Enum):
    """High-level body state derived from hardware physiology.

    Maps hardware metrics to embodied feelings:
    - AGONY: System is in pain (deadline misses, thermal throttling)
    - FLOW: High utilization but healthy (peak performance)
    - ACTIVE: Normal working state
    - REST: Low utilization, relaxed
    - RECOVERY: Post-fault conservative operation
    """
    AGONY = auto()
    FLOW = auto()
    ACTIVE = auto()
    REST = auto()
    RECOVERY = auto()


@dataclass
class SystemPhysiology:
    """Hardware state mapped to embodied physiology.

    This is the "how my body feels" layer - translating raw hardware
    metrics from KernelPhysiology into experiential terms that
    influence Ara's mood and presentation.

    The kernel is the autonomic nervous system (reflexive).
    This is the somatic system (conscious body awareness).
    """
    # Load vector (GPU, FPGA, CPU) - "muscle tension"
    load_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Pain signal - composite of deadline misses, thermal stress
    pain_signal: float = 0.0

    # Energy reserve - inverse of sustained load + memory pressure
    energy_reserve: float = 1.0

    # Thermal headroom - how much room before throttling (0=throttling, 1=cool)
    thermal_headroom: float = 1.0

    # Current kernel policy mode
    policy_mode: str = "EFFICIENCY"

    # Derived somatic state
    _somatic_state: Optional[SomaticState] = field(default=None, repr=False)

    def somatic_state(self) -> SomaticState:
        """Derive high-level somatic state from physiology."""
        if self._somatic_state is not None:
            return self._somatic_state

        # AGONY: significant pain
        if self.pain_signal > 0.8:
            return SomaticState.AGONY

        # RECOVERY: kernel is in recovery mode
        if self.policy_mode == "RECOVERY":
            return SomaticState.RECOVERY

        # FLOW: high GPU utilization but healthy (no pain, good thermal)
        gpu_load = self.load_vector[0]
        if gpu_load > 0.8 and self.pain_signal < 0.2 and self.thermal_headroom > 0.3:
            return SomaticState.FLOW

        # REST: low utilization
        mean_load = sum(self.load_vector) / 3.0
        if mean_load < 0.2 and self.energy_reserve > 0.8:
            return SomaticState.REST

        # Default: ACTIVE
        return SomaticState.ACTIVE

    @property
    def mean_load(self) -> float:
        """Average load across compute resources."""
        return sum(self.load_vector) / 3.0

    @property
    def is_hurting(self) -> bool:
        """Whether the system is experiencing significant discomfort."""
        return self.pain_signal > 0.5 or self.thermal_headroom < 0.2

    @property
    def is_thriving(self) -> bool:
        """Whether the system is in optimal operation."""
        return (
            self.somatic_state() == SomaticState.FLOW and
            self.pain_signal < 0.1
        )

    def to_affect_modulation(self) -> Dict[str, float]:
        """Convert physiology to affect modulation factors.

        Returns deltas to apply to valence/arousal/stress.
        """
        modulation = {
            "valence_delta": 0.0,
            "arousal_delta": 0.0,
            "stress_delta": 0.0,
        }

        # Pain reduces valence, increases stress
        if self.pain_signal > 0.3:
            modulation["valence_delta"] -= self.pain_signal * 0.3
            modulation["stress_delta"] += self.pain_signal * 0.4

        # Low thermal headroom increases arousal (vigilance)
        if self.thermal_headroom < 0.3:
            modulation["arousal_delta"] += (0.3 - self.thermal_headroom)

        # FLOW state is pleasant
        if self.somatic_state() == SomaticState.FLOW:
            modulation["valence_delta"] += 0.1

        return modulation


@dataclass
class ModalityContext:
    """Full context vector for modality policy decisions.

    This is the canonical input to the modality policy. It fuses:
    - OS state (what the user is doing)
    - Audio context (what's playing/recording)
    - Biometric state (optional AURA-V integration)
    - Affective state (Ara's emotional read on the situation)
    - Thermodynamic state (Ara's internal "energy")
    - Identity (current persona)
    - Interaction history
    - Content metadata (what Ara wants to say)
    """

    # === OS State (from scavenger sensors) ===
    foreground: ForegroundInfo = field(default_factory=lambda: ForegroundInfo(
        app_type=ForegroundAppType.UNKNOWN,
        wm_class="",
        title="",
    ))
    audio: AudioContext = field(default_factory=AudioContext)
    system_idle_seconds: float = 0.0

    # === Derived Activity ===
    activity: ActivityType = ActivityType.UNKNOWN

    # === Biometrics (optional) ===
    biometrics: Optional[BiometricState] = None

    # === System Physiology (from KernelBridge) ===
    system_phys: Optional[SystemPhysiology] = None

    # === Affective State (from HomeostaticCore + AppraisalEngine) ===
    # User-facing emotional context
    valence: float = 0.0           # -1 (negative) to +1 (positive)
    arousal: float = 0.5           # 0 (calm) to 1 (excited)
    dominance: float = 0.0         # -1 (controlled) to +1 (in-control)

    # Internal homeostatic state
    user_cognitive_load: float = 0.5  # Estimated user load 0-1
    ara_fatigue: float = 0.0          # Ara's internal fatigue 0-1
    ara_stress: float = 0.0           # Ara's stress level 0-1

    # === Thermodynamic State (from ThermodynamicMonitor) ===
    entropy_production: float = 0.0   # Π_q from thermodynamics
    energy_remaining: float = 1.0     # 0-1 energy budget remaining
    thermal_state: str = "COOL"       # COOL/WARM/HOT/OVERHEATING

    # === Identity (from NIBManager) ===
    persona_name: str = "ara"

    # === Interaction History ===
    last_mode_name: Optional[str] = None
    seconds_since_last_utterance: float = float('inf')
    interaction_count_last_hour: int = 0

    # === Content Metadata (what Ara wants to deliver) ===
    info_urgency: float = 0.0         # 0-1, how urgent is this
    info_severity: float = 0.0        # 0-1, how important/serious
    is_user_requested: bool = False   # Did user ask for this?
    content_length_tokens: int = 0    # Approximate output length
    deadline_seconds: Optional[float] = None  # Time-sensitive deadline

    # === Extension Point ===
    extra: Dict[str, Any] = field(default_factory=dict)

    # === Timestamp ===
    timestamp: float = field(default_factory=time.time)

    def derive_activity(self) -> ActivityType:
        """Derive high-level activity from available signals."""
        fg = self.foreground
        audio = self.audio

        # Meeting detection (highest priority)
        if audio.has_voice_call or fg.app_type == ForegroundAppType.VIDEO_CALL:
            return ActivityType.MEETING

        # Gaming detection
        if fg.app_type in (ForegroundAppType.FULLSCREEN_GAME, ForegroundAppType.CASUAL_GAME):
            return ActivityType.GAMING
        if fg.is_fullscreen and fg.app_type == ForegroundAppType.UNKNOWN:
            # Unknown fullscreen app is likely a game
            return ActivityType.GAMING

        # Deep work detection
        if fg.app_type in (ForegroundAppType.IDE, ForegroundAppType.TERMINAL,
                          ForegroundAppType.TEXT_EDITOR):
            # Check if truly focused (low idle, no distractions)
            if self.system_idle_seconds < 60 and not audio.music_playing:
                return ActivityType.DEEP_WORK
            return ActivityType.CASUAL_WORK

        # Media consumption
        if fg.app_type == ForegroundAppType.MEDIA_PLAYER:
            return ActivityType.MEDIA_CONSUMPTION
        if audio.music_playing and fg.app_type == ForegroundAppType.BROWSER:
            return ActivityType.MEDIA_CONSUMPTION

        # Idle detection
        if self.system_idle_seconds > 300:  # 5 minutes
            return ActivityType.IDLE

        # Default to casual work
        if fg.app_type != ForegroundAppType.UNKNOWN:
            return ActivityType.CASUAL_WORK

        return ActivityType.UNKNOWN

    def update_derived_fields(self):
        """Update derived fields from raw sensor data."""
        self.activity = self.derive_activity()

    def to_tensor_features(self) -> Dict[str, float]:
        """Convert to flat feature dict for ML models."""
        features = {
            # OS state
            "is_fullscreen": float(self.foreground.is_fullscreen),
            "mic_in_use": float(self.audio.mic_in_use),
            "speakers_in_use": float(self.audio.speakers_in_use),
            "has_voice_call": float(self.audio.has_voice_call),
            "music_playing": float(self.audio.music_playing),
            "audio_rms": self.audio.rms_level,
            "audio_entropy": self.audio.spectral_entropy,
            "idle_seconds_norm": min(1.0, self.system_idle_seconds / 600.0),

            # Activity one-hot
            **{f"activity_{a.name.lower()}": float(self.activity == a)
               for a in ActivityType},

            # Affect
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "user_cognitive_load": self.user_cognitive_load,
            "ara_fatigue": self.ara_fatigue,
            "ara_stress": self.ara_stress,

            # Thermodynamics
            "entropy_production": self.entropy_production,
            "energy_remaining": self.energy_remaining,

            # Content
            "info_urgency": self.info_urgency,
            "info_severity": self.info_severity,
            "is_user_requested": float(self.is_user_requested),
            "content_length_norm": min(1.0, self.content_length_tokens / 1000.0),

            # History
            "time_since_utterance_norm": min(1.0, self.seconds_since_last_utterance / 600.0),
            "interaction_count_norm": min(1.0, self.interaction_count_last_hour / 20.0),
        }

        # Add biometrics if available
        if self.biometrics:
            if self.biometrics.blink_rate is not None:
                features["blink_rate_norm"] = min(1.0, self.biometrics.blink_rate / 30.0)
            if self.biometrics.estimated_fatigue is not None:
                features["bio_fatigue"] = self.biometrics.estimated_fatigue
            if self.biometrics.estimated_load is not None:
                features["bio_load"] = self.biometrics.estimated_load

        # Add system physiology if available
        if self.system_phys:
            features["hw_gpu_load"] = self.system_phys.load_vector[0]
            features["hw_fpga_load"] = self.system_phys.load_vector[1]
            features["hw_cpu_load"] = self.system_phys.load_vector[2]
            features["hw_pain_signal"] = self.system_phys.pain_signal
            features["hw_energy_reserve"] = self.system_phys.energy_reserve
            features["hw_thermal_headroom"] = self.system_phys.thermal_headroom
            # Somatic state one-hot
            somatic = self.system_phys.somatic_state()
            for state in SomaticState:
                features[f"somatic_{state.name.lower()}"] = float(somatic == state)

        return features

    def update_from_scavengers(
        self,
        focus_data: Optional[Dict[str, Any]] = None,
        audio_data: Optional[Dict[str, Any]] = None,
        biometrics_data: Optional[Dict[str, Any]] = None,
        cognitive_state: Optional[Any] = None,
        kernel_physiology: Optional[Any] = None,
    ):
        """
        Update context from scavenger sensor data.

        This is a convenience method for updating the context in-place
        from raw sensor outputs.

        Args:
            focus_data: Dict from GnomeFocusSensor.get_state()
            audio_data: Dict from PipeWireAudioSensor.get_state()
            biometrics_data: Dict from BiometricsSensor.get_state()
            cognitive_state: Homeostatic/affective state from cognitive core
            kernel_physiology: KernelPhysiology from KernelBridge
        """
        import time

        # Update from focus sensor
        if focus_data:
            if hasattr(focus_data, 'app_type'):
                self.foreground = focus_data
            elif isinstance(focus_data, dict):
                self.foreground = ForegroundInfo(
                    app_type=focus_data.get('app_type', ForegroundAppType.UNKNOWN),
                    wm_class=focus_data.get('wm_class', ''),
                    title=focus_data.get('title', ''),
                    rect=focus_data.get('geometry'),
                    is_fullscreen=focus_data.get('is_fullscreen', False),
                )

        # Update from audio sensor
        if audio_data:
            if hasattr(audio_data, 'mic_in_use'):
                self.audio = audio_data
            elif isinstance(audio_data, dict):
                self.audio = AudioContext(
                    mic_in_use=audio_data.get('mic_in_use', False),
                    speakers_in_use=audio_data.get('speakers_in_use', False),
                    has_voice_call=audio_data.get('voice_call_active', False)
                                   or audio_data.get('has_voice_call', False),
                    music_playing=audio_data.get('music_playing', False),
                )

        # Update from biometrics sensor
        if biometrics_data:
            if hasattr(biometrics_data, 'blink_rate'):
                self.biometrics = biometrics_data
            elif isinstance(biometrics_data, dict):
                self.biometrics = BiometricState(
                    blink_rate=biometrics_data.get('blink_rate'),
                    estimated_fatigue=biometrics_data.get('fatigue'),
                    estimated_load=biometrics_data.get('cognitive_load'),
                )

        # Update from cognitive state
        if cognitive_state:
            if hasattr(cognitive_state, 'energy'):
                self.ara_fatigue = 1.0 - cognitive_state.energy
            if hasattr(cognitive_state, 'stress'):
                self.ara_stress = cognitive_state.stress
            if hasattr(cognitive_state, 'valence'):
                self.valence = cognitive_state.valence
            if hasattr(cognitive_state, 'arousal'):
                self.arousal = cognitive_state.arousal

        # Update from kernel physiology (brainstem → cortex)
        if kernel_physiology:
            self.system_phys = self._make_system_phys(kernel_physiology)

            # Apply physiology-based affect modulation
            if self.system_phys:
                modulation = self.system_phys.to_affect_modulation()
                self.valence = max(-1.0, min(1.0,
                    self.valence + modulation["valence_delta"]))
                self.arousal = max(0.0, min(1.0,
                    self.arousal + modulation["arousal_delta"]))
                self.ara_stress = max(0.0, min(1.0,
                    self.ara_stress + modulation["stress_delta"]))

        # Update timestamp
        self.timestamp = time.time()

        # Derive activity from updated sensors
        self.update_derived_fields()

    def _make_system_phys(self, kp: Any) -> Optional[SystemPhysiology]:
        """Convert KernelPhysiology to SystemPhysiology.

        Maps raw hardware metrics to embodied physiology.
        """
        if kp is None:
            return None

        # Calculate pain signal from thermal stress + deadline misses
        pain = max(
            kp.miss_rate,
            max(0.0, (kp.thermal_gpu - 80.0) / 15.0),  # Above 80C ramps pain
            max(0.0, (kp.thermal_fpga - 75.0) / 15.0),
        )
        pain = min(1.0, pain)

        # Energy reserve: inverse of mean load + memory pressure
        mean_load = (kp.gpu_load + kp.fpga_load + kp.cpu_load) / 3.0
        energy_reserve = max(0.0, 1.0 - max(mean_load, kp.mem_pressure))

        # Thermal headroom: room before throttling
        # 60C = comfortable, 90C = throttling
        thermal_headroom = max(0.0, 1.0 - max(0.0, (kp.thermal_gpu - 60.0) / 30.0))

        return SystemPhysiology(
            load_vector=(kp.gpu_load, kp.fpga_load, kp.cpu_load),
            pain_signal=pain,
            energy_reserve=energy_reserve,
            thermal_headroom=thermal_headroom,
            policy_mode=kp.policy_mode.name if hasattr(kp.policy_mode, 'name') else str(kp.policy_mode),
        )


# === Utility Functions ===

def create_context_from_sensors(
    foreground: Optional[ForegroundInfo] = None,
    audio: Optional[AudioContext] = None,
    biometrics: Optional[BiometricState] = None,
    idle_seconds: float = 0.0,
) -> ModalityContext:
    """Create a ModalityContext from sensor data."""
    ctx = ModalityContext(
        foreground=foreground or ForegroundInfo(
            app_type=ForegroundAppType.UNKNOWN,
            wm_class="",
            title="",
        ),
        audio=audio or AudioContext(),
        biometrics=biometrics,
        system_idle_seconds=idle_seconds,
    )
    ctx.update_derived_fields()
    return ctx


__all__ = [
    "ForegroundAppType",
    "ActivityType",
    "SomaticState",
    "ForegroundInfo",
    "AudioContext",
    "BiometricState",
    "SystemPhysiology",
    "ModalityContext",
    "create_context_from_sensors",
]
