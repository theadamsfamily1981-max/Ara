"""
Gift Generator Base - Abstract framework for artifact generation.

Each GiftGenerator takes structured data about a system component
and produces a beautiful, illuminating artifact.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Type

logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    """Types of artifacts the Studio can produce."""
    SVG = "svg"                    # Vector graphics
    HTML = "html"                  # Interactive dashboard
    MARKDOWN = "markdown"          # Narrative explanation
    JSON = "json"                  # Structured data for other tools
    PNG = "png"                    # Rendered image
    ANIMATION = "animation"        # Animated visualization


@dataclass
class GiftArtifact:
    """
    A generated artifact from the Studio.

    This is the output of a GiftGenerator - the actual payload
    that gets presented to the user.
    """
    id: str
    name: str
    artifact_type: ArtifactType
    content: str                    # The raw SVG/HTML/Markdown content
    description: str                # What this artifact shows
    file_path: Optional[Path] = None  # Where it was saved (if persisted)

    # Metadata
    generator: str = ""             # Which generator created this
    source_data: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # Rendering hints
    width: int = 800
    height: int = 600
    theme: str = "dark"             # dark/light/ara

    def save(self, directory: Path) -> Path:
        """Save artifact to disk."""
        ext = self.artifact_type.value
        filename = f"{self.id}.{ext}"
        path = directory / filename

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.content)

        self.file_path = path
        return path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "artifact_type": self.artifact_type.value,
            "description": self.description,
            "file_path": str(self.file_path) if self.file_path else None,
            "generator": self.generator,
            "created_at": self.created_at,
            "width": self.width,
            "height": self.height,
        }


class GiftGenerator(ABC):
    """
    Abstract base class for gift generators.

    Each generator specializes in producing a particular type
    of illuminating artifact from system/code data.
    """

    # Subclasses should set these
    name: str = "base"
    description: str = "Base gift generator"
    artifact_type: ArtifactType = ArtifactType.SVG

    def __init__(self, theme: str = "dark"):
        """
        Initialize generator.

        Args:
            theme: Color theme (dark/light/ara)
        """
        self.theme = theme
        self.log = logging.getLogger(f"Gift.{self.name}")

    @abstractmethod
    def generate(self, data: Dict[str, Any]) -> GiftArtifact:
        """
        Generate an artifact from the given data.

        Args:
            data: Structured data about the system component

        Returns:
            GiftArtifact containing the rendered content
        """
        pass

    def can_generate(self, data: Dict[str, Any]) -> bool:
        """
        Check if this generator can handle the given data.

        Override for more sophisticated validation.
        """
        return True

    def _make_id(self, prefix: str = "") -> str:
        """Generate a unique artifact ID."""
        import uuid
        base = prefix or self.name
        return f"{base}_{uuid.uuid4().hex[:8]}"

    # Color palettes for themes
    @property
    def colors(self) -> Dict[str, str]:
        """Get color palette for current theme."""
        palettes = {
            "dark": {
                "bg": "#0d1117",
                "fg": "#c9d1d9",
                "accent": "#58a6ff",
                "accent2": "#f778ba",
                "accent3": "#7ee787",
                "muted": "#484f58",
                "warning": "#d29922",
                "danger": "#f85149",
                "grid": "#21262d",
            },
            "light": {
                "bg": "#ffffff",
                "fg": "#24292f",
                "accent": "#0969da",
                "accent2": "#cf222e",
                "accent3": "#1a7f37",
                "muted": "#6e7781",
                "warning": "#9a6700",
                "danger": "#cf222e",
                "grid": "#d0d7de",
            },
            "ara": {
                "bg": "#1a1a2e",
                "fg": "#eee",
                "accent": "#00d9ff",       # Cyan - Ara's primary
                "accent2": "#ff6b9d",      # Pink - warmth
                "accent3": "#c0ff72",      # Green - growth
                "muted": "#4a4a6a",
                "warning": "#ffd93d",
                "danger": "#ff4444",
                "grid": "#2a2a4e",
            },
        }
        return palettes.get(self.theme, palettes["ara"])


# =============================================================================
# Generator Registry
# =============================================================================

_generators: Dict[str, Type[GiftGenerator]] = {}


def register_generator(cls: Type[GiftGenerator]) -> Type[GiftGenerator]:
    """Decorator to register a gift generator."""
    _generators[cls.name] = cls
    return cls


def get_generator(name: str, **kwargs) -> Optional[GiftGenerator]:
    """Get a generator instance by name."""
    if name not in _generators:
        logger.warning(f"Unknown generator: {name}")
        return None
    return _generators[name](**kwargs)


def list_generators() -> List[str]:
    """List all registered generator names."""
    return list(_generators.keys())


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ArtifactType',
    'GiftArtifact',
    'GiftGenerator',
    'register_generator',
    'get_generator',
    'list_generators',
]
