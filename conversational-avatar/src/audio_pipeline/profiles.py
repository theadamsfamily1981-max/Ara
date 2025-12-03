"""Audio pipeline profiles for fast/deep mode switching."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import json
from pathlib import Path

from .advanced_tts import TTSConfig
from .audio_mastering import MasteringConfig
from .enhanced_asr import ASRConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProfileMode(Enum):
    """Available profile modes."""
    FAST = "fast"           # Low latency, good quality
    BALANCED = "balanced"   # Balance of speed and quality
    DEEP = "deep"           # Highest quality, more latency
    CUSTOM = "custom"       # User-defined settings


@dataclass
class AudioProfile:
    """Complete audio pipeline profile."""
    name: str
    mode: ProfileMode
    description: str = ""

    # Component configurations
    tts_config: TTSConfig = field(default_factory=TTSConfig)
    mastering_config: MasteringConfig = field(default_factory=MasteringConfig)
    asr_config: ASRConfig = field(default_factory=ASRConfig)

    # Voice samples
    voice_samples: List[str] = field(default_factory=list)

    # Custom vocabulary
    vocabulary: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "mode": self.mode.value,
            "description": self.description,
            "tts": {
                "model_name": self.tts_config.model_name,
                "speed": self.tts_config.speed,
                "max_chunk_chars": self.tts_config.max_chunk_chars,
                "language": self.tts_config.language,
            },
            "mastering": {
                "enable_compressor": self.mastering_config.enable_compressor,
                "enable_reverb": self.mastering_config.enable_reverb,
                "enable_limiter": self.mastering_config.enable_limiter,
                "reverb_wet_level": self.mastering_config.reverb_wet_level,
            },
            "asr": {
                "model_size": self.asr_config.model_size,
                "beam_size": self.asr_config.beam_size,
                "language": self.asr_config.language,
            },
            "voice_samples": self.voice_samples,
            "vocabulary": self.vocabulary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioProfile":
        """Create profile from dictionary."""
        # TTS config
        tts_data = data.get("tts", {})
        tts_config = TTSConfig(
            model_name=tts_data.get("model_name", TTSConfig.model_name),
            speed=tts_data.get("speed", TTSConfig.speed),
            max_chunk_chars=tts_data.get("max_chunk_chars", TTSConfig.max_chunk_chars),
            language=tts_data.get("language", TTSConfig.language),
        )

        # Mastering config
        master_data = data.get("mastering", {})
        mastering_config = MasteringConfig(
            enable_compressor=master_data.get("enable_compressor", True),
            enable_reverb=master_data.get("enable_reverb", True),
            enable_limiter=master_data.get("enable_limiter", True),
            reverb_wet_level=master_data.get("reverb_wet_level", 0.08),
        )

        # ASR config
        asr_data = data.get("asr", {})
        asr_config = ASRConfig(
            model_size=asr_data.get("model_size", "base"),
            beam_size=asr_data.get("beam_size", 5),
            language=asr_data.get("language", "en"),
        )

        return cls(
            name=data.get("name", "custom"),
            mode=ProfileMode(data.get("mode", "custom")),
            description=data.get("description", ""),
            tts_config=tts_config,
            mastering_config=mastering_config,
            asr_config=asr_config,
            voice_samples=data.get("voice_samples", []),
            vocabulary=data.get("vocabulary", []),
        )


# Predefined profiles
FAST_PROFILE = AudioProfile(
    name="Fast",
    mode=ProfileMode.FAST,
    description="Low latency mode for real-time interaction",
    tts_config=TTSConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        speed=1.1,  # Slightly faster
        max_chunk_chars=150,  # Smaller chunks for faster synthesis
        min_chunk_chars=30,
    ),
    mastering_config=MasteringConfig(
        enable_compressor=True,
        enable_reverb=False,  # Skip reverb for speed
        enable_limiter=True,
        enable_deesser=False,  # Skip de-esser for speed
        enable_highpass=True,
    ),
    asr_config=ASRConfig(
        model_size="tiny",  # Fastest model
        beam_size=1,  # Greedy decoding
        best_of=1,
        temperature=0.0,
    ),
)

BALANCED_PROFILE = AudioProfile(
    name="Balanced",
    mode=ProfileMode.BALANCED,
    description="Balanced mode for good quality with reasonable latency",
    tts_config=TTSConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        speed=1.0,
        max_chunk_chars=220,
        min_chunk_chars=40,
    ),
    mastering_config=MasteringConfig(
        enable_compressor=True,
        enable_reverb=True,
        reverb_wet_level=0.06,  # Subtle
        enable_limiter=True,
        enable_deesser=True,
        enable_highpass=True,
    ),
    asr_config=ASRConfig(
        model_size="base",  # Good balance
        beam_size=3,
        best_of=3,
        temperature=0.0,
    ),
)

DEEP_PROFILE = AudioProfile(
    name="Deep",
    mode=ProfileMode.DEEP,
    description="Highest quality mode for broadcast-quality output",
    tts_config=TTSConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        speed=0.95,  # Slightly slower for clarity
        max_chunk_chars=300,  # Larger chunks for better prosody
        min_chunk_chars=50,
    ),
    mastering_config=MasteringConfig(
        enable_compressor=True,
        comp_ratio=2.5,  # Gentler compression
        comp_knee_db=8.0,  # Softer knee
        enable_reverb=True,
        reverb_room_size=0.2,
        reverb_wet_level=0.10,
        enable_limiter=True,
        enable_deesser=True,
        deesser_reduction_db=4.0,  # Gentler de-essing
        enable_highpass=True,
        target_loudness_db=-14.0,  # Slightly louder target
    ),
    asr_config=ASRConfig(
        model_size="medium",  # High accuracy
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=True,
    ),
)


class ProfileManager:
    """Manage audio pipeline profiles."""

    # Built-in profiles
    BUILTIN_PROFILES = {
        ProfileMode.FAST: FAST_PROFILE,
        ProfileMode.BALANCED: BALANCED_PROFILE,
        ProfileMode.DEEP: DEEP_PROFILE,
    }

    def __init__(self, profiles_dir: Optional[Path] = None):
        """Initialize profile manager.

        Args:
            profiles_dir: Directory for custom profile storage
        """
        self.profiles_dir = profiles_dir or Path("~/.ara/audio_profiles").expanduser()
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self.active_profile: Optional[AudioProfile] = None
        self.custom_profiles: Dict[str, AudioProfile] = {}

        # Load custom profiles
        self._load_custom_profiles()

    def get_profile(self, mode: ProfileMode) -> AudioProfile:
        """Get a profile by mode.

        Args:
            mode: Profile mode

        Returns:
            AudioProfile
        """
        if mode in self.BUILTIN_PROFILES:
            return self.BUILTIN_PROFILES[mode]
        elif mode == ProfileMode.CUSTOM and self.active_profile:
            return self.active_profile
        else:
            # Default to balanced
            return self.BUILTIN_PROFILES[ProfileMode.BALANCED]

    def get_profile_by_name(self, name: str) -> Optional[AudioProfile]:
        """Get a profile by name.

        Args:
            name: Profile name

        Returns:
            AudioProfile or None
        """
        # Check built-in profiles
        name_lower = name.lower()
        for mode, profile in self.BUILTIN_PROFILES.items():
            if profile.name.lower() == name_lower:
                return profile

        # Check custom profiles
        return self.custom_profiles.get(name)

    def set_active_profile(self, profile: AudioProfile) -> None:
        """Set the active profile.

        Args:
            profile: Profile to activate
        """
        self.active_profile = profile
        logger.info(f"Activated audio profile: {profile.name} ({profile.mode.value})")

    def set_active_by_mode(self, mode: ProfileMode) -> AudioProfile:
        """Set active profile by mode.

        Args:
            mode: Profile mode

        Returns:
            Activated profile
        """
        profile = self.get_profile(mode)
        self.set_active_profile(profile)
        return profile

    def create_custom_profile(
        self,
        name: str,
        base_mode: ProfileMode = ProfileMode.BALANCED,
        **overrides
    ) -> AudioProfile:
        """Create a custom profile based on an existing one.

        Args:
            name: Name for the custom profile
            base_mode: Base profile mode to start from
            **overrides: Configuration overrides

        Returns:
            New custom profile
        """
        base = self.get_profile(base_mode)

        # Create new configs with overrides
        tts_overrides = overrides.get("tts", {})
        tts_config = TTSConfig(
            model_name=tts_overrides.get("model_name", base.tts_config.model_name),
            speed=tts_overrides.get("speed", base.tts_config.speed),
            max_chunk_chars=tts_overrides.get("max_chunk_chars", base.tts_config.max_chunk_chars),
            language=tts_overrides.get("language", base.tts_config.language),
            voice_samples=overrides.get("voice_samples", base.voice_samples),
        )

        master_overrides = overrides.get("mastering", {})
        mastering_config = MasteringConfig(
            enable_compressor=master_overrides.get("enable_compressor", base.mastering_config.enable_compressor),
            enable_reverb=master_overrides.get("enable_reverb", base.mastering_config.enable_reverb),
            reverb_wet_level=master_overrides.get("reverb_wet_level", base.mastering_config.reverb_wet_level),
            enable_limiter=master_overrides.get("enable_limiter", base.mastering_config.enable_limiter),
        )

        asr_overrides = overrides.get("asr", {})
        asr_config = ASRConfig(
            model_size=asr_overrides.get("model_size", base.asr_config.model_size),
            beam_size=asr_overrides.get("beam_size", base.asr_config.beam_size),
            language=asr_overrides.get("language", base.asr_config.language),
            custom_vocabulary=overrides.get("vocabulary", base.vocabulary),
        )

        profile = AudioProfile(
            name=name,
            mode=ProfileMode.CUSTOM,
            description=overrides.get("description", f"Custom profile based on {base.name}"),
            tts_config=tts_config,
            mastering_config=mastering_config,
            asr_config=asr_config,
            voice_samples=overrides.get("voice_samples", base.voice_samples),
            vocabulary=overrides.get("vocabulary", base.vocabulary),
        )

        self.custom_profiles[name] = profile
        logger.info(f"Created custom profile: {name}")

        return profile

    def save_profile(self, profile: AudioProfile) -> Path:
        """Save a custom profile to disk.

        Args:
            profile: Profile to save

        Returns:
            Path to saved profile file
        """
        filename = f"{profile.name.lower().replace(' ', '_')}.json"
        filepath = self.profiles_dir / filename

        with open(filepath, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)

        logger.info(f"Saved profile to {filepath}")
        return filepath

    def load_profile(self, filepath: Path) -> AudioProfile:
        """Load a profile from disk.

        Args:
            filepath: Path to profile file

        Returns:
            Loaded profile
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)

        profile = AudioProfile.from_dict(data)
        self.custom_profiles[profile.name] = profile

        logger.info(f"Loaded profile: {profile.name}")
        return profile

    def _load_custom_profiles(self) -> None:
        """Load all custom profiles from disk."""
        if not self.profiles_dir.exists():
            return

        for filepath in self.profiles_dir.glob("*.json"):
            try:
                self.load_profile(filepath)
            except Exception as e:
                logger.warning(f"Failed to load profile {filepath}: {e}")

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all available profiles.

        Returns:
            List of profile summaries
        """
        profiles = []

        # Built-in profiles
        for mode, profile in self.BUILTIN_PROFILES.items():
            profiles.append({
                "name": profile.name,
                "mode": mode.value,
                "description": profile.description,
                "builtin": True,
            })

        # Custom profiles
        for name, profile in self.custom_profiles.items():
            profiles.append({
                "name": profile.name,
                "mode": profile.mode.value,
                "description": profile.description,
                "builtin": False,
            })

        return profiles

    def delete_profile(self, name: str) -> bool:
        """Delete a custom profile.

        Args:
            name: Profile name

        Returns:
            True if deleted, False if not found
        """
        if name not in self.custom_profiles:
            return False

        # Remove from memory
        del self.custom_profiles[name]

        # Remove from disk
        filename = f"{name.lower().replace(' ', '_')}.json"
        filepath = self.profiles_dir / filename
        if filepath.exists():
            filepath.unlink()

        logger.info(f"Deleted profile: {name}")
        return True

    def get_recommended_profile(
        self,
        latency_sensitive: bool = False,
        quality_priority: bool = False,
        hardware_gpu: bool = True
    ) -> AudioProfile:
        """Get a recommended profile based on requirements.

        Args:
            latency_sensitive: Prioritize low latency
            quality_priority: Prioritize audio quality
            hardware_gpu: Whether GPU is available

        Returns:
            Recommended AudioProfile
        """
        if latency_sensitive and not quality_priority:
            return self.get_profile(ProfileMode.FAST)
        elif quality_priority and not latency_sensitive:
            return self.get_profile(ProfileMode.DEEP)
        elif not hardware_gpu:
            # Use fast profile on CPU
            return self.get_profile(ProfileMode.FAST)
        else:
            return self.get_profile(ProfileMode.BALANCED)
