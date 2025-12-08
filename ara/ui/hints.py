"""
Ara UI Hints - HTC Output to Visual Feedback
============================================

Builds HDOutputHints for the UI layer based on HTC state.

These hints drive:
- Avatar affect (expressions, colors, animation)
- Cockpit focus (which panels/nodes to highlight)
- Theme adjustments (warmth, intensity, blur)

The UI doesn't invent internal state; it surfaces the soul's state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from ara.io.types import (
    HDOutputHint,
    IOChannel,
    AffectHint,
    UIFocusHint,
)


# =============================================================================
# Affect Hint Builder
# =============================================================================

def build_affect_hint(
    valence: float,
    arousal: float,
    certainty: float,
    focus: float,
    ttl_seconds: float = 0.5,
) -> AffectHint:
    """
    Build an affect hint from HTC state.

    Args:
        valence: -1 (negative) to +1 (positive)
        arousal: 0 (calm) to 1 (activated)
        certainty: 0 (confused) to 1 (confident)
        focus: 0 (scattered) to 1 (concentrated)

    Returns:
        AffectHint for avatar/UI theming
    """
    return AffectHint(
        valence=valence,
        arousal=arousal,
        certainty=certainty,
        focus=focus,
        ttl_seconds=ttl_seconds,
    )


def build_affect_hint_from_resonance(
    resonance_profile: Dict[str, float],
    reward_history: List[float],
    attractor_entropy: float = 0.5,
) -> AffectHint:
    """
    Build affect hint from HTC resonance profile.

    This mirrors the affect decoder but returns an HDOutputHint.
    """
    import math

    # Valence from reward history
    if reward_history:
        avg_reward = sum(reward_history[-20:]) / len(reward_history[-20:])
        valence = math.tanh(avg_reward * 2)
    else:
        valence = 0.0

    # Arousal from reward variance
    if len(reward_history) > 1:
        recent = reward_history[-20:]
        mean = sum(recent) / len(recent)
        variance = sum((r - mean) ** 2 for r in recent) / len(recent)
        arousal = math.tanh(math.sqrt(variance) * 3)
    else:
        arousal = 0.3

    # Certainty from entropy
    certainty = 1.0 - attractor_entropy

    # Focus from max resonance
    focus = max(resonance_profile.values()) if resonance_profile else 0.5

    return build_affect_hint(valence, arousal, certainty, focus)


# =============================================================================
# UI Focus Hint Builder
# =============================================================================

def build_ui_focus_hint(
    salient_nodes: List[str],
    salient_flows: List[str],
    preferred_panel: str = "soul_map",
    highlight_intensity: float = 1.0,
    ttl_seconds: float = 1.0,
) -> UIFocusHint:
    """
    Build a UI focus hint for cockpit highlighting.

    Args:
        salient_nodes: Node IDs to highlight
        salient_flows: Flow IDs to highlight
        preferred_panel: Panel to foreground
        highlight_intensity: How strongly to highlight (0-1)

    Returns:
        UIFocusHint for cockpit
    """
    hint = UIFocusHint(
        highlight_nodes=salient_nodes,
        highlight_flows=salient_flows,
        preferred_panel=preferred_panel,
        ttl_seconds=ttl_seconds,
    )
    hint.payload["highlight_intensity"] = highlight_intensity
    return hint


def select_salient_elements(
    resonance_profile: Dict[str, float],
    node_states: List[Dict[str, Any]],
    flow_states: List[Dict[str, Any]],
    top_k: int = 5,
) -> Tuple[List[str], List[str], str]:
    """
    Select salient nodes and flows based on HTC resonance.

    Returns:
        (salient_nodes, salient_flows, preferred_panel)
    """
    # Score nodes by health concern
    node_scores = []
    for node in node_states:
        score = 0.0
        if node.get("max_temp", 0) > 70:
            score += 0.5
        if node.get("error_rate", 0) > 0.01:
            score += 0.3
        if node.get("cpu_load", 0) > 0.9:
            score += 0.2
        node_scores.append((node.get("node_id", ""), score))

    node_scores.sort(key=lambda x: x[1], reverse=True)
    salient_nodes = [n[0] for n in node_scores[:top_k] if n[1] > 0]

    # Score flows by health concern
    flow_scores = []
    for flow in flow_states:
        score = 0.0
        if flow.get("latency_ms", 0) > 100:
            score += 0.4
        if flow.get("error_rate", 0) > 0.01:
            score += 0.4
        flow_scores.append((flow.get("flow_id", ""), score))

    flow_scores.sort(key=lambda x: x[1], reverse=True)
    salient_flows = [f[0] for f in flow_scores[:top_k] if f[1] > 0]

    # Determine preferred panel
    if salient_nodes and any(n.get("max_temp", 0) > 80 for n in node_states):
        preferred_panel = "thermal"
    elif salient_flows:
        preferred_panel = "network"
    else:
        preferred_panel = "soul_map"

    return salient_nodes, salient_flows, preferred_panel


# =============================================================================
# Theme Hint
# =============================================================================

@dataclass
class ThemeHint:
    """UI theme adjustments based on affect."""
    color_temperature: float = 5500  # Kelvin (warm: 3000, cool: 6500)
    saturation: float = 1.0          # 0-1
    brightness: float = 1.0          # 0-1
    blur_amount: float = 0.0         # 0-1 (depth of field)
    particle_speed: float = 1.0      # Animation speed multiplier
    glow_intensity: float = 0.5      # Highlight glow


def build_theme_from_affect(affect: AffectHint) -> ThemeHint:
    """
    Build theme parameters from affect state.

    Mappings (from your design):
    - Valence → Color warmth (negative = cool blue, positive = warm gold)
    - Arousal → Brightness, particle speed
    - Certainty → Blur (uncertain = more blur)
    - Focus → Saturation
    """
    # Color temperature: 3000K (warm) to 6500K (cool)
    # Positive valence = warm, negative = cool
    color_temp = 5500 - (affect.valence * 1500)

    # Brightness: higher when aroused
    brightness = 0.8 + (affect.arousal * 0.2)

    # Saturation: higher when focused
    saturation = 0.7 + (affect.focus * 0.3)

    # Blur: more when uncertain
    blur = (1.0 - affect.certainty) * 0.5

    # Particle speed: faster when aroused
    particle_speed = 0.5 + (affect.arousal * 1.5)

    # Glow: higher when aroused and positive
    glow = 0.3 + (affect.arousal * 0.4) + (max(0, affect.valence) * 0.3)

    return ThemeHint(
        color_temperature=color_temp,
        saturation=saturation,
        brightness=brightness,
        blur_amount=blur,
        particle_speed=particle_speed,
        glow_intensity=min(1.0, glow),
    )


# =============================================================================
# Animation Parameters
# =============================================================================

def affect_to_animation_params(affect: AffectHint) -> Dict[str, float]:
    """
    Convert affect to avatar animation parameters.

    Returns:
        Dictionary of animation parameters for the avatar renderer
    """
    # Eye aperture: more open when aroused, less when uncertain
    eye_aperture = 0.5 + 0.3 * affect.arousal - 0.2 * (1 - affect.certainty)
    eye_aperture = max(0.1, min(1.0, eye_aperture))

    # Blink rate: faster when aroused
    blink_rate = 0.3 + (affect.arousal * 0.4)

    # Breath rate: faster when aroused
    breath_rate = 0.5 + (affect.arousal * 0.5)

    # Micro-movement jitter: high arousal + low certainty
    jitter = max(0.0, affect.arousal - affect.certainty) * 0.5

    # Pupil dilation: larger when aroused or positive
    pupil_size = 0.5 + (affect.arousal * 0.2) + (max(0, affect.valence) * 0.2)

    return {
        "eye_aperture": eye_aperture,
        "blink_rate": blink_rate,
        "breath_rate": breath_rate,
        "jitter": jitter,
        "pupil_size": min(1.0, pupil_size),
        "valence": affect.valence,
        "arousal": affect.arousal,
        "certainty": affect.certainty,
        "focus": affect.focus,
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'build_affect_hint',
    'build_affect_hint_from_resonance',
    'build_ui_focus_hint',
    'select_salient_elements',
    'ThemeHint',
    'build_theme_from_affect',
    'affect_to_animation_params',
]
