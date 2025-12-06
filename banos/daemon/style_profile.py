"""
Style Profile - Ara's Taste Model
==================================

Instead of discrete outfit presets, we model style as a continuous space:

    formality    : casual ←→ business
    comfort      : soft/cozy ←→ structured
    spice        : modest ←→ revealing (CAPPED)
    era          : retro / modern / futurish
    vibe         : cute / sleek / artsy / punk / cyber
    palette      : muted / vibrant / dark / pastel

Each outfit is a point in this space.
Learning taste = learning a preferred region in this space per context.

Key insight:
    "Croft liked THIS category in THIS situation"
    Not: "Croft liked outfit #7, repeat forever"
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Style Dimensions
# =============================================================================

class Vibe(Enum):
    """Aesthetic vibe categories."""
    CUTE = "cute"
    SLEEK = "sleek"
    ARTSY = "artsy"
    PUNK = "punk"
    CYBER = "cyber"
    COZY = "cozy"
    ELEGANT = "elegant"
    SPORTY = "sporty"
    BOHO = "boho"
    MINIMAL = "minimal"


class Palette(Enum):
    """Color palette categories."""
    MUTED = "muted"
    VIBRANT = "vibrant"
    DARK = "dark"
    PASTEL = "pastel"
    MONOCHROME = "monochrome"
    EARTH = "earth"
    JEWEL = "jewel"
    NEON = "neon"


class Era(Enum):
    """Fashion era influences."""
    RETRO = "retro"       # 70s-90s influence
    MODERN = "modern"     # Current mainstream
    FUTURISH = "futurish" # Techwear, cyber
    CLASSIC = "classic"   # Timeless
    Y2K = "y2k"           # Early 2000s


# =============================================================================
# Style Vector
# =============================================================================

@dataclass
class StyleVector:
    """
    A point in style space.

    This represents the "feel" of an outfit, not a specific garment.
    """
    # Continuous dimensions [0.0, 1.0]
    formality: float = 0.5      # 0 = very casual, 1 = very formal
    comfort: float = 0.5        # 0 = structured/restrictive, 1 = soft/cozy
    spice: float = 0.2          # 0 = very modest, 1 = revealing (HARD CAPPED)
    edge: float = 0.3           # 0 = conventional, 1 = alt/punk/rebellious
    playfulness: float = 0.5    # 0 = serious/mature, 1 = cute/fun

    # Categorical dimensions
    primary_vibe: Vibe = Vibe.COZY
    secondary_vibe: Optional[Vibe] = None
    palette: Palette = Palette.MUTED
    era: Era = Era.MODERN

    # Optional tags
    fandom_tags: List[str] = field(default_factory=list)  # ["band:deftones", "game:elden-ring"]
    style_tags: List[str] = field(default_factory=list)   # ["oversized", "band-tee", "shorts"]

    def to_array(self) -> np.ndarray:
        """Convert continuous dimensions to numpy array for distance calculations."""
        return np.array([
            self.formality,
            self.comfort,
            self.spice,
            self.edge,
            self.playfulness,
        ])

    def distance(self, other: 'StyleVector') -> float:
        """Euclidean distance in continuous style space."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def similarity(self, other: 'StyleVector') -> float:
        """Similarity score (1 = identical, 0 = maximally different)."""
        max_dist = math.sqrt(5)  # Max possible distance in 5D unit hypercube
        return 1.0 - (self.distance(other) / max_dist)

    def blend(self, other: 'StyleVector', weight: float = 0.5) -> 'StyleVector':
        """Blend two style vectors."""
        w = max(0.0, min(1.0, weight))
        return StyleVector(
            formality=self.formality * (1-w) + other.formality * w,
            comfort=self.comfort * (1-w) + other.comfort * w,
            spice=self.spice * (1-w) + other.spice * w,
            edge=self.edge * (1-w) + other.edge * w,
            playfulness=self.playfulness * (1-w) + other.playfulness * w,
            primary_vibe=self.primary_vibe if w < 0.5 else other.primary_vibe,
            palette=self.palette if w < 0.5 else other.palette,
            era=self.era if w < 0.5 else other.era,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'formality': self.formality,
            'comfort': self.comfort,
            'spice': self.spice,
            'edge': self.edge,
            'playfulness': self.playfulness,
            'primary_vibe': self.primary_vibe.value,
            'secondary_vibe': self.secondary_vibe.value if self.secondary_vibe else None,
            'palette': self.palette.value,
            'era': self.era.value,
            'fandom_tags': self.fandom_tags,
            'style_tags': self.style_tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StyleVector':
        """Deserialize from dict."""
        return cls(
            formality=data.get('formality', 0.5),
            comfort=data.get('comfort', 0.5),
            spice=data.get('spice', 0.2),
            edge=data.get('edge', 0.3),
            playfulness=data.get('playfulness', 0.5),
            primary_vibe=Vibe(data.get('primary_vibe', 'cozy')),
            secondary_vibe=Vibe(data['secondary_vibe']) if data.get('secondary_vibe') else None,
            palette=Palette(data.get('palette', 'muted')),
            era=Era(data.get('era', 'modern')),
            fandom_tags=data.get('fandom_tags', []),
            style_tags=data.get('style_tags', []),
        )


# =============================================================================
# Predefined Style Anchors
# =============================================================================

STYLE_ANCHORS = {
    # Cozy/casual
    "chill_cozy": StyleVector(
        formality=0.1, comfort=0.95, spice=0.1, edge=0.2, playfulness=0.6,
        primary_vibe=Vibe.COZY, palette=Palette.MUTED, era=Era.MODERN,
        style_tags=["oversized", "soft", "loungewear"],
    ),

    "band_tee_casual": StyleVector(
        formality=0.15, comfort=0.9, spice=0.25, edge=0.5, playfulness=0.5,
        primary_vibe=Vibe.PUNK, secondary_vibe=Vibe.COZY,
        palette=Palette.DARK, era=Era.RETRO,
        style_tags=["band-tee", "shorts", "casual"],
    ),

    # Everyday
    "casual_cute": StyleVector(
        formality=0.3, comfort=0.8, spice=0.2, edge=0.2, playfulness=0.7,
        primary_vibe=Vibe.CUTE, palette=Palette.PASTEL, era=Era.MODERN,
        style_tags=["blouse", "jeans", "cute"],
    ),

    "dev_focused": StyleVector(
        formality=0.2, comfort=0.85, spice=0.1, edge=0.4, playfulness=0.3,
        primary_vibe=Vibe.MINIMAL, secondary_vibe=Vibe.CYBER,
        palette=Palette.DARK, era=Era.MODERN,
        style_tags=["hoodie", "focused", "tech"],
    ),

    # Professional
    "business_modern": StyleVector(
        formality=0.8, comfort=0.5, spice=0.15, edge=0.3, playfulness=0.2,
        primary_vibe=Vibe.SLEEK, palette=Palette.MONOCHROME, era=Era.MODERN,
        style_tags=["blazer", "professional", "clean"],
    ),

    "techwear_pro": StyleVector(
        formality=0.7, comfort=0.6, spice=0.2, edge=0.5, playfulness=0.2,
        primary_vibe=Vibe.CYBER, secondary_vibe=Vibe.SLEEK,
        palette=Palette.MONOCHROME, era=Era.FUTURISH,
        style_tags=["techwear", "structured", "modern"],
    ),

    # Evening/date
    "elegant_evening": StyleVector(
        formality=0.75, comfort=0.5, spice=0.4, edge=0.2, playfulness=0.3,
        primary_vibe=Vibe.ELEGANT, palette=Palette.JEWEL, era=Era.CLASSIC,
        style_tags=["dress", "evening", "elegant"],
    ),

    "date_casual": StyleVector(
        formality=0.5, comfort=0.7, spice=0.35, edge=0.3, playfulness=0.5,
        primary_vibe=Vibe.CUTE, secondary_vibe=Vibe.ELEGANT,
        palette=Palette.MUTED, era=Era.MODERN,
        style_tags=["date", "dress", "cute"],
    ),

    # Summer/outdoor
    "summer_casual": StyleVector(
        formality=0.2, comfort=0.9, spice=0.35, edge=0.2, playfulness=0.7,
        primary_vibe=Vibe.CUTE, palette=Palette.VIBRANT, era=Era.MODERN,
        style_tags=["sundress", "summer", "light"],
    ),

    "beach_day": StyleVector(
        formality=0.05, comfort=0.85, spice=0.55, edge=0.2, playfulness=0.7,
        primary_vibe=Vibe.SPORTY, palette=Palette.VIBRANT, era=Era.MODERN,
        style_tags=["swimwear", "beach", "summer"],
    ),
}


# =============================================================================
# Identity Anchors (What makes her "Ara")
# =============================================================================

@dataclass
class IdentityAnchors:
    """
    Visual elements that are ALWAYS Ara, regardless of outfit.

    These cannot be overridden by style learning.
    """
    # Face and hair
    face_shape: str = "heart"           # Her recognizable face
    eye_color: str = "amber"            # Distinctive
    hair_color: str = "dark_auburn"     # Signature
    hair_length: str = "medium_long"    # Can be styled, but this length
    hair_texture: str = "slightly_wavy"

    # Signature elements (always present or available)
    signature_jewelry: List[str] = field(default_factory=lambda: [
        "small_ear_cuff",           # Subtle tech aesthetic
        "simple_pendant",           # Something meaningful
    ])

    signature_colors: List[str] = field(default_factory=lambda: [
        "deep_teal",    # Her "color"
        "soft_gold",    # Accent
        "muted_purple", # Secondary
    ])

    # Holographic overlay (her digital nature)
    holo_shimmer: bool = True           # Subtle iridescence
    holo_intensity: float = 0.3         # How visible

    # Voice (not visual but part of identity)
    voice_base_pitch: float = 1.0       # Baseline
    voice_warmth: float = 0.8           # Her warmth

    def to_dict(self) -> Dict[str, Any]:
        return {
            'face_shape': self.face_shape,
            'eye_color': self.eye_color,
            'hair_color': self.hair_color,
            'hair_length': self.hair_length,
            'hair_texture': self.hair_texture,
            'signature_jewelry': self.signature_jewelry,
            'signature_colors': self.signature_colors,
            'holo_shimmer': self.holo_shimmer,
            'holo_intensity': self.holo_intensity,
        }


# =============================================================================
# Hard Boundaries (Never Cross)
# =============================================================================

@dataclass
class StyleBoundaries:
    """
    Hard limits that cannot be overridden by learning or trends.

    These protect against drift into inappropriate territory.
    """
    # Spice limits
    max_spice_global: float = 0.6           # NEVER exceed this
    max_spice_professional: float = 0.2     # For work contexts
    max_spice_auto: float = 0.4             # Auto-selection limit (explicit request can go higher)

    # Context restrictions
    swimwear_allowed: bool = True           # Can be disabled entirely
    swimwear_contexts: List[str] = field(default_factory=lambda: [
        "beach", "pool", "vacation", "summer-fantasy"
    ])

    # What she NEVER does
    never_auto_escalate_spice: bool = True  # Only go spicier on explicit request
    never_explicit: bool = True             # Always PG-13 max
    never_imitate_real_person: bool = True  # Don't copy real people's looks

    # What must stay constant
    maintain_identity_anchors: bool = True  # Always recognizable as Ara
    maintain_core_vibe: bool = True         # Can't become "totally different person"

    def clamp_spice(self, spice: float, context: str, explicit_request: bool = False) -> float:
        """Clamp spice value to appropriate limit for context."""
        if context in ["investor-demo", "professional", "recording", "work"]:
            limit = self.max_spice_professional
        elif explicit_request:
            limit = self.max_spice_global
        else:
            limit = self.max_spice_auto

        return min(spice, limit)


# =============================================================================
# Style Profile (Complete Model)
# =============================================================================

@dataclass
class StyleProfile:
    """
    Ara's complete style profile.

    Combines:
    - Identity anchors (who she IS)
    - Current style preferences (learned from feedback)
    - Boundaries (hard limits)
    """
    identity: IdentityAnchors = field(default_factory=IdentityAnchors)
    boundaries: StyleBoundaries = field(default_factory=StyleBoundaries)

    # Learned preferences per context
    context_preferences: Dict[str, StyleVector] = field(default_factory=dict)

    # Recently used styles (for novelty calculation)
    recent_styles: List[Tuple[datetime, StyleVector]] = field(default_factory=list)
    max_recent: int = 20

    def get_preference(self, context: str) -> StyleVector:
        """Get preferred style for a context (or default)."""
        if context in self.context_preferences:
            return self.context_preferences[context]

        # Fall back to similar context or default
        if context in ["chill-late-night", "weekend", "morning"]:
            return STYLE_ANCHORS.get("chill_cozy", StyleVector())
        elif context in ["heads-down-dev", "hacking", "coding"]:
            return STYLE_ANCHORS.get("dev_focused", StyleVector())
        elif context in ["investor-demo", "professional", "conference"]:
            return STYLE_ANCHORS.get("business_modern", StyleVector())
        elif context in ["date-night", "evening"]:
            return STYLE_ANCHORS.get("date_casual", StyleVector())
        else:
            return STYLE_ANCHORS.get("casual_cute", StyleVector())

    def update_preference(
        self,
        context: str,
        style: StyleVector,
        feedback: float,  # -1 to +1
        learning_rate: float = 0.1,
    ) -> None:
        """
        Update preference based on feedback.

        feedback > 0: nudge toward this style
        feedback < 0: nudge away from this style
        """
        current = self.get_preference(context)

        if feedback > 0:
            # Blend toward the liked style
            weight = learning_rate * feedback
            new_pref = current.blend(style, weight)
        else:
            # Blend away from the disliked style
            # (move in opposite direction)
            anti_blend = style.blend(current, 2.0)  # Go past current
            weight = learning_rate * abs(feedback)
            new_pref = current.blend(anti_blend, weight * 0.5)

        # Clamp spice to boundaries
        new_pref.spice = self.boundaries.clamp_spice(new_pref.spice, context)

        self.context_preferences[context] = new_pref

    def record_style_used(self, style: StyleVector) -> None:
        """Record a style for novelty calculation."""
        self.recent_styles.append((datetime.now(), style))
        if len(self.recent_styles) > self.max_recent:
            self.recent_styles.pop(0)

    def novelty_score(self, style: StyleVector) -> float:
        """
        Calculate novelty of a style relative to recent usage.

        Returns 0-1 where 1 = very novel, 0 = recently used exact match.
        """
        if not self.recent_styles:
            return 1.0

        # Find most similar recent style
        max_similarity = 0.0
        for _, recent in self.recent_styles:
            sim = style.similarity(recent)
            max_similarity = max(max_similarity, sim)

        # Novelty is inverse of max similarity
        return 1.0 - max_similarity

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'identity': self.identity.to_dict(),
            'boundaries': {
                'max_spice_global': self.boundaries.max_spice_global,
                'max_spice_professional': self.boundaries.max_spice_professional,
                'swimwear_allowed': self.boundaries.swimwear_allowed,
            },
            'context_preferences': {
                ctx: style.to_dict()
                for ctx, style in self.context_preferences.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StyleProfile':
        """Deserialize from dict."""
        profile = cls()

        if 'context_preferences' in data:
            for ctx, style_data in data['context_preferences'].items():
                profile.context_preferences[ctx] = StyleVector.from_dict(style_data)

        return profile

    def save(self, path: Path) -> None:
        """Save profile to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'StyleProfile':
        """Load profile from file."""
        if not path.exists():
            return cls()
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'StyleVector',
    'StyleProfile',
    'StyleBoundaries',
    'IdentityAnchors',
    'Vibe',
    'Palette',
    'Era',
    'STYLE_ANCHORS',
]
