"""
Ara Palette - Affect to Shader Parameters
==========================================

Maps affect state to GPU shader parameters.

Mappings:
- uncertainty/confusion -> noise/chromatic aberration
- arousal/load -> bloom intensity + particle speed
- valence -> color temperature (warm/cool)
- focus -> depth of field / highlight strength

These are pure functions; actual uniform updates happen in SomaticServer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


@dataclass
class ShaderParams:
    """GPU shader parameters derived from affect."""
    noise_amount: float = 0.0        # Chromatic aberration / noise
    bloom_intensity: float = 0.2     # Glow effect
    color_warmth: float = 0.5        # 0 = cool blue, 1 = warm gold
    focus_strength: float = 0.5      # DOF / highlight intensity
    particle_speed: float = 1.0      # Animation speed multiplier
    saturation: float = 1.0          # Color saturation
    vignette: float = 0.0            # Edge darkening

    def to_dict(self) -> Dict[str, float]:
        return {
            "noise_amount": self.noise_amount,
            "bloom_intensity": self.bloom_intensity,
            "color_warmth": self.color_warmth,
            "focus_strength": self.focus_strength,
            "particle_speed": self.particle_speed,
            "saturation": self.saturation,
            "vignette": self.vignette,
        }


def shader_params_from_affect(affect: Dict[str, float]) -> Dict[str, float]:
    """
    Map affect dimensions to shader parameters.

    Args:
        affect: Dict with valence, arousal, certainty, focus

    Returns:
        Dict of shader uniform values
    """
    valence = affect.get("valence", 0.0)
    arousal = affect.get("arousal", 0.5)
    certainty = affect.get("certainty", 0.5)
    focus = affect.get("focus", 0.5)

    # Noise/aberration: high when uncertain
    noise_amount = clamp((1.0 - certainty) * 0.5, 0.0, 1.0)

    # Bloom: increases with arousal
    bloom_intensity = 0.2 + 0.8 * arousal

    # Color warmth: positive valence = warm, negative = cool
    color_warmth = 0.5 + 0.5 * valence

    # Focus strength: directly from focus dimension
    focus_strength = focus

    # Particle speed: faster when aroused
    particle_speed = 0.5 + 1.5 * arousal

    # Saturation: higher when focused and certain
    saturation = 0.7 + 0.3 * (focus * certainty)

    # Vignette: stronger when aroused and negative valence
    vignette = clamp(arousal * (1.0 - valence) * 0.3, 0.0, 0.5)

    return ShaderParams(
        noise_amount=noise_amount,
        bloom_intensity=bloom_intensity,
        color_warmth=color_warmth,
        focus_strength=focus_strength,
        particle_speed=particle_speed,
        saturation=saturation,
        vignette=vignette,
    ).to_dict()


def glitch_params(severity: float) -> Dict[str, float]:
    """
    Generate shader parameters for glitch effect.

    Args:
        severity: Glitch severity 0-1

    Returns:
        Dict of shader uniform values for glitch overlay
    """
    severity = clamp(severity, 0.0, 1.0)

    return {
        "glitch_intensity": severity,
        "chromatic_offset": severity * 0.05,  # RGB separation
        "scanline_strength": severity * 0.3,
        "noise_overlay": severity * 0.4,
        "color_shift_red": severity * 0.2,    # Shift toward red/danger
        "distortion": severity * 0.02,        # Geometric distortion
    }


def interpolate_params(
    current: Dict[str, float],
    target: Dict[str, float],
    t: float,
) -> Dict[str, float]:
    """
    Linearly interpolate between shader parameter sets.

    Args:
        current: Current parameters
        target: Target parameters
        t: Interpolation factor (0 = current, 1 = target)

    Returns:
        Interpolated parameters
    """
    t = clamp(t, 0.0, 1.0)
    result = {}

    all_keys = set(current.keys()) | set(target.keys())
    for key in all_keys:
        c = current.get(key, 0.0)
        tgt = target.get(key, 0.0)
        result[key] = c * (1 - t) + tgt * t

    return result


__all__ = [
    'ShaderParams',
    'shader_params_from_affect',
    'glitch_params',
    'interpolate_params',
    'clamp',
]
