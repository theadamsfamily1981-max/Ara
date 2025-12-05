"""Meta Configuration - Settings for Ara's meta-learning layer.

Handles loading and accessing configuration for:
- Logging settings
- Analysis thresholds
- Auto-apply rules
- Research agenda paths
"""

from __future__ import annotations

import os
import yaml
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Configuration for meta logging."""

    enabled: bool = True
    log_path: str = "~/.ara/meta/interactions.jsonl"
    min_quality_to_log: float = 0.0
    buffer_size: int = 100
    rotate_after_mb: float = 50.0


@dataclass
class AnalysisConfig:
    """Configuration for pattern analysis."""

    default_window_days: int = 30
    min_samples_for_suggestion: int = 5
    golden_path_threshold: float = 0.8
    min_occurrences_for_pattern: int = 3
    auto_refresh_hours: float = 1.0


@dataclass
class SuggestionConfig:
    """Configuration for suggestions."""

    enabled: bool = True
    auto_apply_min_confidence: float = 0.9
    require_user_confirmation: bool = True
    max_pending_suggestions: int = 20


@dataclass
class ResearchConfig:
    """Configuration for research agenda."""

    agenda_path: str = "~/.ara/meta/research_agenda.json"
    default_goal: str = "Become a better collaborator and learner"


@dataclass
class MetaConfig:
    """Complete meta-learning configuration."""

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    suggestions: SuggestionConfig = field(default_factory=SuggestionConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)

    # Additional settings
    data_dir: str = "~/.ara/meta"
    verbose: bool = False

    def get_data_path(self) -> Path:
        """Get expanded data directory path."""
        return Path(os.path.expanduser(self.data_dir))

    def get_log_path(self) -> Path:
        """Get expanded log file path."""
        return Path(os.path.expanduser(self.logging.log_path))

    def get_agenda_path(self) -> Path:
        """Get expanded research agenda path."""
        return Path(os.path.expanduser(self.research.agenda_path))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "logging": {
                "enabled": self.logging.enabled,
                "log_path": self.logging.log_path,
                "min_quality_to_log": self.logging.min_quality_to_log,
                "buffer_size": self.logging.buffer_size,
                "rotate_after_mb": self.logging.rotate_after_mb,
            },
            "analysis": {
                "default_window_days": self.analysis.default_window_days,
                "min_samples_for_suggestion": self.analysis.min_samples_for_suggestion,
                "golden_path_threshold": self.analysis.golden_path_threshold,
                "min_occurrences_for_pattern": self.analysis.min_occurrences_for_pattern,
                "auto_refresh_hours": self.analysis.auto_refresh_hours,
            },
            "suggestions": {
                "enabled": self.suggestions.enabled,
                "auto_apply_min_confidence": self.suggestions.auto_apply_min_confidence,
                "require_user_confirmation": self.suggestions.require_user_confirmation,
                "max_pending_suggestions": self.suggestions.max_pending_suggestions,
            },
            "research": {
                "agenda_path": self.research.agenda_path,
                "default_goal": self.research.default_goal,
            },
            "data_dir": self.data_dir,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaConfig":
        """Create from dictionary."""
        config = cls()

        if "logging" in data:
            log_data = data["logging"]
            config.logging = LoggingConfig(
                enabled=log_data.get("enabled", True),
                log_path=log_data.get("log_path", config.logging.log_path),
                min_quality_to_log=log_data.get("min_quality_to_log", 0.0),
                buffer_size=log_data.get("buffer_size", 100),
                rotate_after_mb=log_data.get("rotate_after_mb", 50.0),
            )

        if "analysis" in data:
            an_data = data["analysis"]
            config.analysis = AnalysisConfig(
                default_window_days=an_data.get("default_window_days", 30),
                min_samples_for_suggestion=an_data.get("min_samples_for_suggestion", 5),
                golden_path_threshold=an_data.get("golden_path_threshold", 0.8),
                min_occurrences_for_pattern=an_data.get("min_occurrences_for_pattern", 3),
                auto_refresh_hours=an_data.get("auto_refresh_hours", 1.0),
            )

        if "suggestions" in data:
            sug_data = data["suggestions"]
            config.suggestions = SuggestionConfig(
                enabled=sug_data.get("enabled", True),
                auto_apply_min_confidence=sug_data.get("auto_apply_min_confidence", 0.9),
                require_user_confirmation=sug_data.get("require_user_confirmation", True),
                max_pending_suggestions=sug_data.get("max_pending_suggestions", 20),
            )

        if "research" in data:
            res_data = data["research"]
            config.research = ResearchConfig(
                agenda_path=res_data.get("agenda_path", config.research.agenda_path),
                default_goal=res_data.get("default_goal", config.research.default_goal),
            )

        config.data_dir = data.get("data_dir", config.data_dir)
        config.verbose = data.get("verbose", False)

        return config


# =============================================================================
# Loading Functions
# =============================================================================

_default_config: Optional[MetaConfig] = None
_config_search_paths: List[Path] = [
    Path.home() / ".ara" / "meta_config.yaml",
    Path.home() / ".config" / "ara" / "meta_config.yaml",
    Path("ara_meta_config.yaml"),
    Path("config/meta_config.yaml"),
]


def load_meta_config(path: Optional[Path] = None) -> MetaConfig:
    """Load meta configuration from file.

    Args:
        path: Explicit config path (optional)

    Returns:
        Loaded configuration
    """
    global _default_config

    # Search for config file
    config_path = path
    if not config_path:
        for search_path in _config_search_paths:
            if search_path.exists():
                config_path = search_path
                break

    if config_path and config_path.exists():
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
            config = MetaConfig.from_dict(data or {})
            logger.info(f"Loaded meta config from {config_path}")
            _default_config = config
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    # Return default config
    config = MetaConfig()
    _default_config = config
    return config


def get_meta_config() -> MetaConfig:
    """Get the current meta configuration.

    Returns:
        Current configuration
    """
    global _default_config
    if _default_config is None:
        _default_config = load_meta_config()
    return _default_config


def save_meta_config(config: MetaConfig, path: Optional[Path] = None) -> None:
    """Save meta configuration to file.

    Args:
        config: Configuration to save
        path: Path to save to (defaults to ~/.ara/meta_config.yaml)
    """
    save_path = path or Path.home() / ".ara" / "meta_config.yaml"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    logger.info(f"Saved meta config to {save_path}")


def create_default_config_file(path: Optional[Path] = None) -> Path:
    """Create a default configuration file.

    Args:
        path: Path to create at

    Returns:
        Path to created file
    """
    config = MetaConfig()
    save_path = path or Path.home() / ".ara" / "meta_config.yaml"
    save_meta_config(config, save_path)
    return save_path
