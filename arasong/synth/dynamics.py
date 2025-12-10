#!/usr/bin/env python3
"""
AraSong Synthesizer - Emotion-Driven Dynamics
==============================================

Maps emotion_arc to synthesis parameters for expressive, dynamic audio.
Provides velocity curves, vibrato, articulation timing based on emotional state.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EmotionParams:
    """Synthesis parameters derived from emotional state."""
    # Amplitude
    vel_scale: float = 1.0          # Velocity multiplier
    vel_curve: float = 1.0          # Curve shape (>1 = punchy, <1 = soft)

    # Pitch expression
    vibrato_depth_cents: float = 10.0
    vibrato_rate_hz: float = 5.0
    pitch_drift_cents: float = 0.0   # Slow random drift

    # Timing/articulation
    attack_ms: float = 10.0
    decay_ms: float = 50.0
    sustain_level: float = 0.7
    release_ms: float = 100.0

    # Timbral
    brightness: float = 0.5          # Filter cutoff modifier
    breathiness: float = 0.1         # Noise mix
    warmth: float = 0.5              # Low-frequency emphasis


# Emotion presets - map emotion names to parameter sets
EMOTION_MAP: Dict[str, EmotionParams] = {
    # Calm/peaceful states
    "calm": EmotionParams(
        vel_scale=0.75, vel_curve=0.8,
        vibrato_depth_cents=5, vibrato_rate_hz=4.5,
        attack_ms=30, decay_ms=80, sustain_level=0.65, release_ms=200,
        brightness=0.3, breathiness=0.15, warmth=0.7
    ),
    "peaceful": EmotionParams(
        vel_scale=0.7, vel_curve=0.7,
        vibrato_depth_cents=4, vibrato_rate_hz=4.0,
        attack_ms=40, decay_ms=100, sustain_level=0.6, release_ms=250,
        brightness=0.25, breathiness=0.2, warmth=0.8
    ),
    "tender": EmotionParams(
        vel_scale=0.65, vel_curve=0.6,
        vibrato_depth_cents=6, vibrato_rate_hz=4.5,
        attack_ms=35, decay_ms=90, sustain_level=0.55, release_ms=220,
        brightness=0.3, breathiness=0.18, warmth=0.75
    ),
    "intimate": EmotionParams(
        vel_scale=0.6, vel_curve=0.5,
        vibrato_depth_cents=8, vibrato_rate_hz=4.2,
        attack_ms=25, decay_ms=70, sustain_level=0.6, release_ms=180,
        brightness=0.35, breathiness=0.22, warmth=0.7
    ),

    # Warm/positive states
    "warm": EmotionParams(
        vel_scale=0.85, vel_curve=0.9,
        vibrato_depth_cents=8, vibrato_rate_hz=5.0,
        attack_ms=20, decay_ms=60, sustain_level=0.7, release_ms=150,
        brightness=0.45, breathiness=0.12, warmth=0.65
    ),
    "happy": EmotionParams(
        vel_scale=1.0, vel_curve=1.1,
        vibrato_depth_cents=12, vibrato_rate_hz=5.5,
        attack_ms=10, decay_ms=40, sustain_level=0.75, release_ms=100,
        brightness=0.6, breathiness=0.08, warmth=0.5
    ),
    "uplifting": EmotionParams(
        vel_scale=1.1, vel_curve=1.2,
        vibrato_depth_cents=15, vibrato_rate_hz=5.8,
        attack_ms=8, decay_ms=35, sustain_level=0.8, release_ms=90,
        brightness=0.65, breathiness=0.06, warmth=0.45
    ),
    "joyful": EmotionParams(
        vel_scale=1.15, vel_curve=1.3,
        vibrato_depth_cents=18, vibrato_rate_hz=6.0,
        attack_ms=5, decay_ms=30, sustain_level=0.85, release_ms=80,
        brightness=0.7, breathiness=0.05, warmth=0.4
    ),

    # Intense/energetic states
    "excited": EmotionParams(
        vel_scale=1.2, vel_curve=1.4,
        vibrato_depth_cents=20, vibrato_rate_hz=6.5,
        attack_ms=5, decay_ms=25, sustain_level=0.85, release_ms=70,
        brightness=0.75, breathiness=0.04, warmth=0.35
    ),
    "powerful": EmotionParams(
        vel_scale=1.3, vel_curve=1.5,
        vibrato_depth_cents=18, vibrato_rate_hz=6.0,
        attack_ms=3, decay_ms=20, sustain_level=0.9, release_ms=60,
        brightness=0.8, breathiness=0.03, warmth=0.3
    ),
    "confident": EmotionParams(
        vel_scale=1.1, vel_curve=1.2,
        vibrato_depth_cents=12, vibrato_rate_hz=5.5,
        attack_ms=8, decay_ms=35, sustain_level=0.8, release_ms=90,
        brightness=0.6, breathiness=0.08, warmth=0.45
    ),
    "triumphant": EmotionParams(
        vel_scale=1.35, vel_curve=1.6,
        vibrato_depth_cents=22, vibrato_rate_hz=6.2,
        attack_ms=2, decay_ms=15, sustain_level=0.92, release_ms=50,
        brightness=0.85, breathiness=0.02, warmth=0.25
    ),

    # Anticipation/tension states
    "anticipation": EmotionParams(
        vel_scale=0.9, vel_curve=1.0,
        vibrato_depth_cents=10, vibrato_rate_hz=5.2,
        attack_ms=15, decay_ms=50, sustain_level=0.72, release_ms=120,
        brightness=0.5, breathiness=0.1, warmth=0.55
    ),
    "tense": EmotionParams(
        vel_scale=1.05, vel_curve=1.3,
        vibrato_depth_cents=8, vibrato_rate_hz=5.8,
        attack_ms=5, decay_ms=30, sustain_level=0.78, release_ms=80,
        brightness=0.55, breathiness=0.06, warmth=0.4
    ),
    "suspense": EmotionParams(
        vel_scale=0.85, vel_curve=0.9,
        vibrato_depth_cents=6, vibrato_rate_hz=4.8,
        attack_ms=20, decay_ms=70, sustain_level=0.65, release_ms=160,
        brightness=0.4, breathiness=0.14, warmth=0.6
    ),

    # Sad/melancholic states
    "sad": EmotionParams(
        vel_scale=0.7, vel_curve=0.6,
        vibrato_depth_cents=12, vibrato_rate_hz=4.0,
        attack_ms=35, decay_ms=100, sustain_level=0.55, release_ms=250,
        brightness=0.25, breathiness=0.2, warmth=0.75
    ),
    "melancholic": EmotionParams(
        vel_scale=0.75, vel_curve=0.7,
        vibrato_depth_cents=14, vibrato_rate_hz=4.2,
        attack_ms=30, decay_ms=90, sustain_level=0.58, release_ms=220,
        brightness=0.3, breathiness=0.18, warmth=0.7
    ),
    "longing": EmotionParams(
        vel_scale=0.78, vel_curve=0.75,
        vibrato_depth_cents=16, vibrato_rate_hz=4.5,
        attack_ms=28, decay_ms=85, sustain_level=0.6, release_ms=200,
        brightness=0.35, breathiness=0.16, warmth=0.65
    ),

    # Neutral/default
    "neutral": EmotionParams(
        vel_scale=1.0, vel_curve=1.0,
        vibrato_depth_cents=10, vibrato_rate_hz=5.0,
        attack_ms=15, decay_ms=50, sustain_level=0.7, release_ms=120,
        brightness=0.5, breathiness=0.1, warmth=0.5
    ),
    "reassuring": EmotionParams(
        vel_scale=0.85, vel_curve=0.85,
        vibrato_depth_cents=8, vibrato_rate_hz=4.8,
        attack_ms=22, decay_ms=65, sustain_level=0.68, release_ms=160,
        brightness=0.42, breathiness=0.12, warmth=0.62
    ),
}


def get_emotion_params(emotion: str) -> EmotionParams:
    """Get parameters for an emotion, defaulting to neutral."""
    return EMOTION_MAP.get(emotion.lower(), EMOTION_MAP["neutral"])


def interpolate_params(params_a: EmotionParams, params_b: EmotionParams,
                       t: float) -> EmotionParams:
    """Linearly interpolate between two emotion parameter sets."""
    t = np.clip(t, 0.0, 1.0)

    return EmotionParams(
        vel_scale=params_a.vel_scale * (1 - t) + params_b.vel_scale * t,
        vel_curve=params_a.vel_curve * (1 - t) + params_b.vel_curve * t,
        vibrato_depth_cents=params_a.vibrato_depth_cents * (1 - t) + params_b.vibrato_depth_cents * t,
        vibrato_rate_hz=params_a.vibrato_rate_hz * (1 - t) + params_b.vibrato_rate_hz * t,
        pitch_drift_cents=params_a.pitch_drift_cents * (1 - t) + params_b.pitch_drift_cents * t,
        attack_ms=params_a.attack_ms * (1 - t) + params_b.attack_ms * t,
        decay_ms=params_a.decay_ms * (1 - t) + params_b.decay_ms * t,
        sustain_level=params_a.sustain_level * (1 - t) + params_b.sustain_level * t,
        release_ms=params_a.release_ms * (1 - t) + params_b.release_ms * t,
        brightness=params_a.brightness * (1 - t) + params_b.brightness * t,
        breathiness=params_a.breathiness * (1 - t) + params_b.breathiness * t,
        warmth=params_a.warmth * (1 - t) + params_b.warmth * t,
    )


def scale_params_by_energy(params: EmotionParams, energy: float) -> EmotionParams:
    """Scale parameters by energy level (0-1)."""
    energy = np.clip(energy, 0.0, 1.0)

    # Energy affects intensity-related params
    return EmotionParams(
        vel_scale=params.vel_scale * (0.6 + 0.4 * energy),
        vel_curve=params.vel_curve * (0.8 + 0.2 * energy),
        vibrato_depth_cents=params.vibrato_depth_cents * (0.5 + 0.5 * energy),
        vibrato_rate_hz=params.vibrato_rate_hz * (0.9 + 0.1 * energy),
        pitch_drift_cents=params.pitch_drift_cents,
        attack_ms=params.attack_ms * (1.5 - 0.5 * energy),  # Faster attack at high energy
        decay_ms=params.decay_ms * (1.3 - 0.3 * energy),
        sustain_level=params.sustain_level * (0.8 + 0.2 * energy),
        release_ms=params.release_ms * (1.4 - 0.4 * energy),  # Shorter release at high energy
        brightness=params.brightness * (0.7 + 0.3 * energy),
        breathiness=params.breathiness * (1.2 - 0.2 * energy),
        warmth=params.warmth * (1.1 - 0.1 * energy),
    )


@dataclass
class EmotionArcPoint:
    """A point in the emotion arc."""
    time_ratio: float  # 0.0 to 1.0 within section
    emotion: str
    energy: float


@dataclass
class EmotionArc:
    """Emotion arc for a section or song."""
    points: List[EmotionArcPoint] = field(default_factory=list)

    def get_params_at(self, time_ratio: float) -> EmotionParams:
        """Get interpolated parameters at a time position."""
        if not self.points:
            return get_emotion_params("neutral")

        time_ratio = np.clip(time_ratio, 0.0, 1.0)

        # Find surrounding points
        prev_point = self.points[0]
        next_point = self.points[-1]

        for i, point in enumerate(self.points):
            if point.time_ratio >= time_ratio:
                next_point = point
                if i > 0:
                    prev_point = self.points[i - 1]
                else:
                    prev_point = point
                break
            prev_point = point

        # Interpolate between points
        if prev_point.time_ratio == next_point.time_ratio:
            t = 0.0
        else:
            t = (time_ratio - prev_point.time_ratio) / (next_point.time_ratio - prev_point.time_ratio)

        params_a = scale_params_by_energy(get_emotion_params(prev_point.emotion), prev_point.energy)
        params_b = scale_params_by_energy(get_emotion_params(next_point.emotion), next_point.energy)

        return interpolate_params(params_a, params_b, t)


def build_emotion_arc_from_song(song_data: dict) -> Dict[str, EmotionArc]:
    """Build emotion arcs per section from song JSON data."""
    arcs = {}

    for arc_point in song_data.get("emotion_arc", []):
        section = arc_point["section"]
        emotion = arc_point.get("emotion", "neutral")
        energy = arc_point.get("energy", 0.5)

        if section not in arcs:
            arcs[section] = EmotionArc()

        # For now, each section has a single emotion point at start
        # Future: support multiple points within a section
        arcs[section].points.append(EmotionArcPoint(
            time_ratio=0.0,
            emotion=emotion,
            energy=energy
        ))
        # Add end point (same emotion for smooth sustain)
        arcs[section].points.append(EmotionArcPoint(
            time_ratio=1.0,
            emotion=emotion,
            energy=energy
        ))

    return arcs


# =============================================================================
# Velocity Curves
# =============================================================================

def apply_velocity_curve(samples: np.ndarray, vel_scale: float, vel_curve: float,
                        energy_envelope: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply velocity/dynamics to audio samples.

    vel_scale: Overall amplitude multiplier
    vel_curve: Shape of attack (>1 = punchy, <1 = soft)
    energy_envelope: Optional time-varying energy curve
    """
    if energy_envelope is not None:
        # Apply time-varying energy
        shaped = np.power(energy_envelope, vel_curve)
        return samples * vel_scale * shaped
    else:
        return samples * vel_scale


def generate_energy_envelope(num_samples: int, sample_rate: int,
                             params: EmotionParams) -> np.ndarray:
    """Generate an energy envelope based on emotion parameters."""
    # Simple attack/sustain/release envelope scaled by emotion
    attack_samples = int(params.attack_ms * sample_rate / 1000)
    release_samples = int(params.release_ms * sample_rate / 1000)
    sustain_samples = max(0, num_samples - attack_samples - release_samples)

    envelope = np.ones(num_samples, dtype=np.float32)

    # Attack ramp
    if attack_samples > 0:
        t = np.linspace(0, 1, attack_samples, dtype=np.float32)
        envelope[:attack_samples] = np.power(t, 1.0 / params.vel_curve)

    # Release ramp
    if release_samples > 0 and num_samples > release_samples:
        t = np.linspace(1, 0, release_samples, dtype=np.float32)
        envelope[-release_samples:] = np.power(t, params.vel_curve)

    return envelope * params.sustain_level


# =============================================================================
# Pitch Modulation
# =============================================================================

def generate_vibrato(num_samples: int, sample_rate: int,
                     depth_cents: float, rate_hz: float,
                     delay_ratio: float = 0.1) -> np.ndarray:
    """
    Generate vibrato pitch modulation curve.

    Returns array of pitch multipliers (1.0 = no change).
    delay_ratio: portion of note before vibrato starts (natural singing style)
    """
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    delay_samples = int(num_samples * delay_ratio)

    # Vibrato LFO
    lfo = np.sin(2.0 * np.pi * rate_hz * t)

    # Fade in vibrato after delay
    fade_in = np.ones(num_samples, dtype=np.float32)
    if delay_samples > 0:
        fade_in[:delay_samples] = np.linspace(0, 1, delay_samples, dtype=np.float32)

    # Convert cents to pitch multiplier
    cents_mod = depth_cents * lfo * fade_in
    pitch_mult = np.power(2.0, cents_mod / 1200.0)

    return pitch_mult


def generate_pitch_drift(num_samples: int, sample_rate: int,
                         max_drift_cents: float = 5.0,
                         rate_hz: float = 0.3) -> np.ndarray:
    """
    Generate slow random pitch drift for humanization.

    Returns array of pitch multipliers.
    """
    if max_drift_cents == 0:
        return np.ones(num_samples, dtype=np.float32)

    t = np.arange(num_samples, dtype=np.float32) / sample_rate

    # Very slow LFO with slight randomness
    phase = np.random.random() * 2 * np.pi
    drift = max_drift_cents * np.sin(2.0 * np.pi * rate_hz * t + phase)

    # Add some noise for micro-variations
    noise = np.random.randn(num_samples) * (max_drift_cents * 0.1)
    # Smooth the noise
    kernel_size = int(sample_rate * 0.02)  # 20ms smoothing
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
        noise = np.convolve(noise, kernel, mode='same')

    cents_mod = drift + noise
    return np.power(2.0, cents_mod / 1200.0)
