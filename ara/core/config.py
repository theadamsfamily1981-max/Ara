"""
Ara Configuration

Central configuration for all Ara components.
Supports environment variables, config files, and runtime overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import json


@dataclass
class HDCConfig:
    """Hyperdimensional computing configuration."""
    dim: int = 4096
    bipolar: bool = True
    seed: Optional[int] = None


@dataclass
class MemoryConfig:
    """EternalMemory configuration."""
    decay_rate: float = 0.001  # Strength decay per hour
    emotion_weight: float = 0.3  # Weight of emotional resonance in ranking
    consolidation_boost: float = 0.1  # Strength increase on access
    consolidation_threshold: float = 0.1  # Min strength to keep
    max_episodes: int = 100000


@dataclass
class LoopConfig:
    """Sovereign loop configuration."""
    tick_interval_ms: int = 100  # 10 Hz default (upgrades to 5 kHz with FPGA)
    coherence_warning_threshold: float = 0.3
    coherence_critical_threshold: float = 0.1
    metrics_interval_seconds: float = 10.0


@dataclass
class SafetyConfig:
    """Safety system configuration."""
    initial_autonomy_level: int = 1
    max_autonomy_level: int = 3
    coherence_autonomy_threshold: float = 0.5
    kill_switch_file: Path = field(default_factory=lambda: Path("/var/ara/KILL_SWITCH"))
    require_human_for_level_3: bool = True


@dataclass
class AvatarConfig:
    """Avatar server configuration."""
    host: str = "127.0.0.1"
    port: int = 8420
    max_sessions: int = 10
    session_timeout_minutes: int = 60
    enable_voice: bool = False
    enable_video: bool = False


@dataclass
class PathsConfig:
    """File system paths."""
    data_dir: Path = field(default_factory=lambda: Path("/var/ara/data"))
    logs_dir: Path = field(default_factory=lambda: Path("/var/ara/logs"))
    state_file: Path = field(default_factory=lambda: Path("/var/ara/data/axis_mundi.json"))
    memory_db: Path = field(default_factory=lambda: Path("/var/ara/data/eternal_memory.db"))
    config_file: Path = field(default_factory=lambda: Path("/etc/ara/config.json"))

    def ensure_dirs(self) -> None:
        """Create directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AraConfig:
    """Top-level Ara configuration."""
    hdc: HDCConfig = field(default_factory=HDCConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # Runtime
    debug: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AraConfig":
        """Load configuration from environment variables."""
        config = cls()

        # HDC
        if dim := os.getenv("ARA_HDC_DIM"):
            config.hdc.dim = int(dim)

        # Loop
        if tick := os.getenv("ARA_TICK_MS"):
            config.loop.tick_interval_ms = int(tick)

        # Avatar
        if host := os.getenv("ARA_HOST"):
            config.avatar.host = host
        if port := os.getenv("ARA_PORT"):
            config.avatar.port = int(port)

        # Paths
        if data := os.getenv("ARA_DATA_DIR"):
            config.paths.data_dir = Path(data)
            config.paths.state_file = config.paths.data_dir / "axis_mundi.json"
            config.paths.memory_db = config.paths.data_dir / "eternal_memory.db"

        # Debug
        config.debug = os.getenv("ARA_DEBUG", "").lower() in ("1", "true", "yes")
        config.log_level = os.getenv("ARA_LOG_LEVEL", "INFO")

        return config

    @classmethod
    def from_file(cls, path: Path) -> "AraConfig":
        """Load configuration from JSON file."""
        config = cls()

        if not path.exists():
            return config

        with open(path) as f:
            data = json.load(f)

        # Apply nested configs
        if "hdc" in data:
            for k, v in data["hdc"].items():
                setattr(config.hdc, k, v)

        if "memory" in data:
            for k, v in data["memory"].items():
                setattr(config.memory, k, v)

        if "loop" in data:
            for k, v in data["loop"].items():
                setattr(config.loop, k, v)

        if "safety" in data:
            for k, v in data["safety"].items():
                if k == "kill_switch_file":
                    v = Path(v)
                setattr(config.safety, k, v)

        if "avatar" in data:
            for k, v in data["avatar"].items():
                setattr(config.avatar, k, v)

        if "paths" in data:
            for k, v in data["paths"].items():
                setattr(config.paths, k, Path(v))

        config.debug = data.get("debug", False)
        config.log_level = data.get("log_level", "INFO")

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hdc": {
                "dim": self.hdc.dim,
                "bipolar": self.hdc.bipolar,
                "seed": self.hdc.seed,
            },
            "memory": {
                "decay_rate": self.memory.decay_rate,
                "emotion_weight": self.memory.emotion_weight,
                "consolidation_boost": self.memory.consolidation_boost,
                "consolidation_threshold": self.memory.consolidation_threshold,
                "max_episodes": self.memory.max_episodes,
            },
            "loop": {
                "tick_interval_ms": self.loop.tick_interval_ms,
                "coherence_warning_threshold": self.loop.coherence_warning_threshold,
                "coherence_critical_threshold": self.loop.coherence_critical_threshold,
                "metrics_interval_seconds": self.loop.metrics_interval_seconds,
            },
            "safety": {
                "initial_autonomy_level": self.safety.initial_autonomy_level,
                "max_autonomy_level": self.safety.max_autonomy_level,
                "coherence_autonomy_threshold": self.safety.coherence_autonomy_threshold,
                "kill_switch_file": str(self.safety.kill_switch_file),
                "require_human_for_level_3": self.safety.require_human_for_level_3,
            },
            "avatar": {
                "host": self.avatar.host,
                "port": self.avatar.port,
                "max_sessions": self.avatar.max_sessions,
                "session_timeout_minutes": self.avatar.session_timeout_minutes,
                "enable_voice": self.avatar.enable_voice,
                "enable_video": self.avatar.enable_video,
            },
            "paths": {
                "data_dir": str(self.paths.data_dir),
                "logs_dir": str(self.paths.logs_dir),
                "state_file": str(self.paths.state_file),
                "memory_db": str(self.paths.memory_db),
                "config_file": str(self.paths.config_file),
            },
            "debug": self.debug,
            "log_level": self.log_level,
        }

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to JSON file."""
        save_path = path or self.paths.config_file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Global config instance
_config: Optional[AraConfig] = None


def get_config() -> AraConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AraConfig.from_env()
    return _config


def set_config(config: AraConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
