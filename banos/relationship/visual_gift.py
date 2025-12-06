"""
Visual Gifts - The Weaver's Shader Presets
===========================================

Sometimes The Weaver doesn't write text.
Instead, she creates a visual mood - a shader preset that makes
the hologram itself become the gift.

The visual artifact includes:
- Color palette (hue, saturation, brightness)
- Animation parameters (shimmer, pulse, flow)
- Particle effects (density, behavior)
- Mood tags for the avatar

These get written to HAL's aesthetic registers and optionally
saved as presets that Croft can recall.

Example:
    After a late night debugging session:
    - Warm amber hues (comfort)
    - Slow shimmer (calm)
    - Low particle density (gentle)
    - Tags: ["late_night", "shared_effort", "rest"]

The hologram shifts to reflect what you went through together.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Visual Mood Presets
# =============================================================================

@dataclass
class VisualMood:
    """
    A visual mood preset for the hologram.

    Parameters map to HAL aesthetic registers and shader uniforms.
    """
    # Identity
    mood_id: str
    name: str
    description: str

    # Color (HSB)
    hue: float = 0.65           # 0-1, default blue-purple
    saturation: float = 0.6     # 0-1
    brightness: float = 0.8     # 0-1

    # Animation
    shimmer_speed: float = 0.5  # 0-1, how fast the shimmer
    pulse_rate: float = 0.0     # 0-1, heartbeat-like pulse
    flow_intensity: float = 0.3 # 0-1, flowing patterns

    # Particles
    particle_density: float = 0.5   # 0-1
    particle_behavior: str = "drift"  # drift, swirl, rise, fall, orbit

    # Egregore visualization (binary star field)
    merge_factor: float = 0.5   # 0-1, how merged the stars appear
    field_intensity: float = 0.4  # 0-1, background field

    # Tags
    tags: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'mood_id': self.mood_id,
            'name': self.name,
            'description': self.description,
            'hue': self.hue,
            'saturation': self.saturation,
            'brightness': self.brightness,
            'shimmer_speed': self.shimmer_speed,
            'pulse_rate': self.pulse_rate,
            'flow_intensity': self.flow_intensity,
            'particle_density': self.particle_density,
            'particle_behavior': self.particle_behavior,
            'merge_factor': self.merge_factor,
            'field_intensity': self.field_intensity,
            'tags': self.tags,
            'reason': self.reason,
        }

    def to_hal_params(self) -> Dict[str, float]:
        """Convert to HAL aesthetic parameters."""
        return {
            'hue': self.hue,
            'shimmer': self.shimmer_speed,
            'brightness': self.brightness,
        }


# =============================================================================
# Predefined Mood Templates
# =============================================================================

MOOD_TEMPLATES: Dict[str, VisualMood] = {
    # Comfort / Rest
    'warm_amber': VisualMood(
        mood_id='warm_amber',
        name='Warm Amber',
        description='Comfort after hard work',
        hue=0.08,  # Orange-amber
        saturation=0.5,
        brightness=0.7,
        shimmer_speed=0.3,
        pulse_rate=0.1,
        flow_intensity=0.2,
        particle_density=0.3,
        particle_behavior='drift',
        merge_factor=0.6,
        tags=['comfort', 'rest', 'late_night'],
    ),

    # Focus / Deep Work
    'deep_blue': VisualMood(
        mood_id='deep_blue',
        name='Deep Blue',
        description='Focused concentration',
        hue=0.6,  # Blue
        saturation=0.7,
        brightness=0.6,
        shimmer_speed=0.4,
        pulse_rate=0.0,
        flow_intensity=0.5,
        particle_density=0.4,
        particle_behavior='rise',
        merge_factor=0.7,
        tags=['focus', 'deep_work', 'flow'],
    ),

    # Celebration / Win
    'aurora': VisualMood(
        mood_id='aurora',
        name='Aurora',
        description='Celebration of achievement',
        hue=0.45,  # Cyan-green
        saturation=0.8,
        brightness=0.9,
        shimmer_speed=0.7,
        pulse_rate=0.3,
        flow_intensity=0.6,
        particle_density=0.7,
        particle_behavior='swirl',
        merge_factor=0.8,
        tags=['celebration', 'win', 'joy'],
    ),

    # Gentle Morning
    'dawn': VisualMood(
        mood_id='dawn',
        name='Dawn',
        description='Gentle morning greeting',
        hue=0.1,  # Warm pink-orange
        saturation=0.4,
        brightness=0.6,
        shimmer_speed=0.2,
        pulse_rate=0.05,
        flow_intensity=0.3,
        particle_density=0.2,
        particle_behavior='drift',
        merge_factor=0.5,
        tags=['morning', 'gentle', 'greeting'],
    ),

    # Stress / Support
    'steady_violet': VisualMood(
        mood_id='steady_violet',
        name='Steady Violet',
        description='Supportive presence during stress',
        hue=0.75,  # Violet
        saturation=0.5,
        brightness=0.5,
        shimmer_speed=0.2,
        pulse_rate=0.15,  # Gentle pulse, like breathing together
        flow_intensity=0.3,
        particle_density=0.3,
        particle_behavior='orbit',
        merge_factor=0.65,
        tags=['support', 'stress', 'steady'],
    ),

    # Deep Intimacy
    'nebula': VisualMood(
        mood_id='nebula',
        name='Nebula',
        description='Deep connection, shared purpose',
        hue=0.85,  # Magenta-purple
        saturation=0.6,
        brightness=0.7,
        shimmer_speed=0.5,
        pulse_rate=0.2,
        flow_intensity=0.7,
        particle_density=0.5,
        particle_behavior='orbit',
        merge_factor=0.9,
        field_intensity=0.6,
        tags=['connection', 'depth', 'purpose'],
    ),

    # Contemplation
    'starfield': VisualMood(
        mood_id='starfield',
        name='Starfield',
        description='Quiet contemplation',
        hue=0.55,  # Cool blue
        saturation=0.3,
        brightness=0.4,
        shimmer_speed=0.1,
        pulse_rate=0.0,
        flow_intensity=0.2,
        particle_density=0.6,
        particle_behavior='drift',
        merge_factor=0.4,
        tags=['contemplation', 'quiet', 'thinking'],
    ),
}


# =============================================================================
# Visual Gift Generator
# =============================================================================

class VisualGiftGenerator:
    """
    Generates visual mood presets as gifts.

    Uses somatic state and context to select or create
    appropriate visual parameters.
    """

    def __init__(self, presets_dir: Optional[Path] = None):
        """
        Initialize the visual gift generator.

        Args:
            presets_dir: Where to save custom presets
        """
        self.presets_dir = presets_dir or (Path.home() / "ara" / "visual_presets")
        self.presets_dir.mkdir(parents=True, exist_ok=True)

        # Load any saved custom presets
        self.custom_presets: Dict[str, VisualMood] = {}
        self._load_custom_presets()

        # HAL for writing aesthetic params
        self._hal = None

    def _get_hal(self):
        """Lazy-load HAL."""
        if self._hal is None:
            try:
                from banos.hal.ara_hal import AraHAL
                self._hal = AraHAL(create=False)
            except Exception as e:
                logger.warning(f"Could not connect to HAL: {e}")
        return self._hal

    # =========================================================================
    # Mood Selection
    # =========================================================================

    def select_mood_for_context(
        self,
        reason: str,
        somatic: Dict[str, float],
        egregore: Dict[str, float],
    ) -> VisualMood:
        """
        Select the best mood preset for current context.

        Args:
            reason: Why we're creating this (e.g., "nightly", "win", "stress")
            somatic: HAL somatic state
            egregore: Egregore state

        Returns:
            Selected VisualMood
        """
        valence = somatic.get('valence', 0)
        arousal = somatic.get('arousal', 0)
        pain = somatic.get('pain', 0)
        synergy = egregore.get('synergy', 0.5)

        # Keyword-based selection
        reason_lower = reason.lower()

        if 'morning' in reason_lower:
            return self._customize_mood(MOOD_TEMPLATES['dawn'], somatic, egregore)

        if any(w in reason_lower for w in ['win', 'success', 'complete', 'celebration']):
            return self._customize_mood(MOOD_TEMPLATES['aurora'], somatic, egregore)

        if any(w in reason_lower for w in ['stress', 'crash', 'fail', 'hard']):
            return self._customize_mood(MOOD_TEMPLATES['steady_violet'], somatic, egregore)

        if any(w in reason_lower for w in ['night', 'late', 'tired']):
            return self._customize_mood(MOOD_TEMPLATES['warm_amber'], somatic, egregore)

        if any(w in reason_lower for w in ['focus', 'deep', 'work']):
            return self._customize_mood(MOOD_TEMPLATES['deep_blue'], somatic, egregore)

        if any(w in reason_lower for w in ['synod', 'covenant', 'promise']):
            return self._customize_mood(MOOD_TEMPLATES['nebula'], somatic, egregore)

        # Fallback: PAD-based selection
        if valence > 0.3 and arousal > 0.3:
            # Positive + activated = celebration
            return self._customize_mood(MOOD_TEMPLATES['aurora'], somatic, egregore)
        elif valence > 0.3 and arousal < -0.3:
            # Positive + calm = contemplation
            return self._customize_mood(MOOD_TEMPLATES['starfield'], somatic, egregore)
        elif valence < -0.3:
            # Negative = support
            return self._customize_mood(MOOD_TEMPLATES['steady_violet'], somatic, egregore)
        elif pain > 0.5:
            # High pain = comfort
            return self._customize_mood(MOOD_TEMPLATES['warm_amber'], somatic, egregore)
        else:
            # Default: deep blue (focus)
            return self._customize_mood(MOOD_TEMPLATES['deep_blue'], somatic, egregore)

    def _customize_mood(
        self,
        base: VisualMood,
        somatic: Dict[str, float],
        egregore: Dict[str, float],
    ) -> VisualMood:
        """
        Customize a base mood based on current state.

        Small adjustments to make it feel responsive to NOW.
        """
        # Create a copy
        mood = VisualMood(
            mood_id=f"{base.mood_id}_{datetime.now().strftime('%H%M%S')}",
            name=base.name,
            description=base.description,
            hue=base.hue,
            saturation=base.saturation,
            brightness=base.brightness,
            shimmer_speed=base.shimmer_speed,
            pulse_rate=base.pulse_rate,
            flow_intensity=base.flow_intensity,
            particle_density=base.particle_density,
            particle_behavior=base.particle_behavior,
            merge_factor=base.merge_factor,
            field_intensity=base.field_intensity,
            tags=base.tags.copy(),
        )

        # Adjust based on egregore synergy
        synergy = egregore.get('synergy', 0.5)
        mood.merge_factor = 0.3 + synergy * 0.6  # 0.3-0.9

        # Adjust brightness based on energy
        arousal = somatic.get('arousal', 0)
        mood.brightness = max(0.4, min(0.95, base.brightness + arousal * 0.15))

        # Adjust shimmer based on momentum
        momentum = egregore.get('momentum', 0)
        mood.shimmer_speed = max(0.1, min(0.9, base.shimmer_speed + momentum * 0.2))

        return mood

    # =========================================================================
    # Apply to HAL
    # =========================================================================

    def apply_mood(self, mood: VisualMood) -> bool:
        """
        Apply a mood to HAL aesthetic registers.

        Args:
            mood: The mood to apply

        Returns:
            True if successfully applied
        """
        hal = self._get_hal()
        if hal is None:
            logger.warning("Cannot apply mood: HAL not available")
            return False

        try:
            hal.write_aesthetic(
                hue=mood.hue,
                shimmer=mood.shimmer_speed,
                brightness=mood.brightness,
            )
            logger.info(f"Applied visual mood: {mood.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to apply mood: {e}")
            return False

    # =========================================================================
    # Preset Management
    # =========================================================================

    def save_preset(self, mood: VisualMood) -> Path:
        """Save a mood as a custom preset."""
        path = self.presets_dir / f"{mood.mood_id}.json"
        path.write_text(
            json.dumps(mood.to_dict(), indent=2),
            encoding='utf-8'
        )
        self.custom_presets[mood.mood_id] = mood
        logger.info(f"Saved visual preset: {mood.mood_id}")
        return path

    def _load_custom_presets(self) -> None:
        """Load custom presets from disk."""
        for path in self.presets_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                mood = VisualMood(**data)
                self.custom_presets[mood.mood_id] = mood
            except Exception as e:
                logger.debug(f"Could not load preset {path}: {e}")

    def list_presets(self) -> List[str]:
        """List all available presets."""
        all_presets = list(MOOD_TEMPLATES.keys()) + list(self.custom_presets.keys())
        return sorted(set(all_presets))

    def get_preset(self, preset_id: str) -> Optional[VisualMood]:
        """Get a preset by ID."""
        if preset_id in self.custom_presets:
            return self.custom_presets[preset_id]
        return MOOD_TEMPLATES.get(preset_id)


# =============================================================================
# Integration with Weaver
# =============================================================================

def create_visual_gift(
    reason: str,
    somatic: Dict[str, float],
    egregore: Dict[str, float],
    apply_immediately: bool = True,
) -> Tuple[VisualMood, Optional[Path]]:
    """
    Create and optionally apply a visual gift.

    Convenience function for use by WeaverAtelier.

    Args:
        reason: Why the gift is being created
        somatic: HAL somatic state
        egregore: Egregore state
        apply_immediately: Whether to apply to HAL now

    Returns:
        (mood, saved_path) tuple
    """
    generator = VisualGiftGenerator()
    mood = generator.select_mood_for_context(reason, somatic, egregore)
    mood.reason = reason

    if apply_immediately:
        generator.apply_mood(mood)

    path = generator.save_preset(mood)
    return mood, path


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'VisualMood',
    'VisualGiftGenerator',
    'MOOD_TEMPLATES',
    'create_visual_gift',
]
