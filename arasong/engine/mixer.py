#!/usr/bin/env python3
"""
AraSong Engine - Mix Automation
================================

Production-grade mixing with ducking, compression, and bus routing.
All operations are numpy-vectorized for speed.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BusConfig:
    """Configuration for an audio bus."""
    name: str
    gain_db: float = 0.0
    pan: float = 0.0  # -1 (left) to 1 (right)
    mute: bool = False
    solo: bool = False


@dataclass
class CompressorConfig:
    """Compressor settings."""
    threshold_db: float = -12.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    makeup_gain_db: float = 0.0
    knee_db: float = 6.0  # Soft knee width


@dataclass
class DuckingConfig:
    """Sidechain ducking settings."""
    amount: float = 0.5        # How much to duck (0-1)
    threshold_db: float = -20.0
    attack_ms: float = 5.0
    release_ms: float = 150.0
    hold_ms: float = 50.0      # Hold time before release


def db_to_linear(db: float) -> float:
    """Convert dB to linear gain."""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear gain to dB."""
    return 20.0 * np.log10(max(linear, 1e-10))


def rms_envelope(samples: np.ndarray, window_ms: float, sample_rate: int) -> np.ndarray:
    """
    Compute RMS envelope of signal.

    Uses a sliding window for smooth envelope following.
    """
    window_samples = max(1, int(window_ms * sample_rate / 1000))

    # Compute squared signal
    squared = samples ** 2

    # Cumulative sum for efficient sliding window
    cumsum = np.cumsum(squared)
    cumsum = np.insert(cumsum, 0, 0)

    # RMS for each position
    rms = np.sqrt((cumsum[window_samples:] - cumsum[:-window_samples]) / window_samples)

    # Pad to original length
    pad_len = len(samples) - len(rms)
    if pad_len > 0:
        rms = np.concatenate([np.zeros(pad_len), rms])

    return rms.astype(np.float32)


def peak_envelope(samples: np.ndarray, window_ms: float, sample_rate: int) -> np.ndarray:
    """Compute peak envelope using max in sliding window."""
    window_samples = max(1, int(window_ms * sample_rate / 1000))

    # Use strided view for efficiency
    from numpy.lib.stride_tricks import sliding_window_view

    if len(samples) < window_samples:
        return np.abs(samples)

    windowed = sliding_window_view(np.abs(samples), window_samples)
    peaks = np.max(windowed, axis=1)

    # Pad to original length
    pad_len = len(samples) - len(peaks)
    if pad_len > 0:
        peaks = np.concatenate([np.zeros(pad_len), peaks])

    return peaks.astype(np.float32)


def smooth_envelope(envelope: np.ndarray, attack_ms: float, release_ms: float,
                    sample_rate: int) -> np.ndarray:
    """
    Apply attack/release smoothing to envelope.

    Attack: how fast envelope rises
    Release: how fast envelope falls
    """
    attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000))
    release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000))

    output = np.zeros_like(envelope)
    current = 0.0

    for i in range(len(envelope)):
        target = envelope[i]
        if target > current:
            # Attack
            current = attack_coeff * current + (1 - attack_coeff) * target
        else:
            # Release
            current = release_coeff * current + (1 - release_coeff) * target
        output[i] = current

    return output


# =============================================================================
# Compressor
# =============================================================================

def compress(samples: np.ndarray, config: CompressorConfig,
             sample_rate: int) -> np.ndarray:
    """
    Apply dynamic range compression.

    Uses soft-knee compression with lookahead-free design.
    """
    # Get envelope
    env = rms_envelope(samples, 10.0, sample_rate)
    env = smooth_envelope(env, config.attack_ms, config.release_ms, sample_rate)

    # Convert to dB
    env_db = 20.0 * np.log10(env + 1e-10)

    # Compute gain reduction
    over_threshold = env_db - config.threshold_db

    # Soft knee
    knee_start = config.threshold_db - config.knee_db / 2
    knee_end = config.threshold_db + config.knee_db / 2

    gain_reduction_db = np.zeros_like(env_db)

    # Below knee: no compression
    # In knee: gradual compression
    # Above knee: full compression

    in_knee = (env_db > knee_start) & (env_db < knee_end)
    above_knee = env_db >= knee_end

    # Soft knee interpolation
    if config.knee_db > 0:
        knee_factor = (env_db[in_knee] - knee_start) / config.knee_db
        gain_reduction_db[in_knee] = knee_factor ** 2 * (1 - 1/config.ratio) * (env_db[in_knee] - knee_start)

    # Full compression above knee
    gain_reduction_db[above_knee] = over_threshold[above_knee] * (1 - 1/config.ratio)

    # Convert to linear gain
    gain = 10.0 ** (-gain_reduction_db / 20.0)

    # Apply makeup gain
    makeup = db_to_linear(config.makeup_gain_db)

    return (samples * gain * makeup).astype(np.float32)


# =============================================================================
# Sidechain Ducking
# =============================================================================

def duck(target: np.ndarray, sidechain: np.ndarray, config: DuckingConfig,
         sample_rate: int) -> np.ndarray:
    """
    Apply sidechain ducking to target signal based on sidechain signal level.

    When sidechain (e.g., vocals) is loud, target (e.g., music) is reduced.
    """
    # Get sidechain envelope
    sc_env = rms_envelope(sidechain, 20.0, sample_rate)
    sc_env = smooth_envelope(sc_env, config.attack_ms, config.release_ms, sample_rate)

    # Convert threshold to linear
    threshold = db_to_linear(config.threshold_db)

    # Compute ducking amount (0 = no duck, 1 = full duck)
    duck_amount = np.clip((sc_env - threshold) / (threshold * 2 + 1e-10), 0, 1)

    # Apply hold time (keep ducking for a bit after signal drops)
    hold_samples = int(config.hold_ms * sample_rate / 1000)
    if hold_samples > 0:
        # Simple hold: use maximum in recent window
        from numpy.lib.stride_tricks import sliding_window_view
        if len(duck_amount) > hold_samples:
            windowed = sliding_window_view(duck_amount, hold_samples)
            held = np.max(windowed, axis=1)
            duck_amount = np.concatenate([duck_amount[:hold_samples-1], held])

    # Compute gain curve
    # Full ducking reduces by config.amount
    gain = 1.0 - duck_amount * config.amount

    # Clamp to reasonable range
    gain = np.clip(gain, 1.0 - config.amount, 1.0)

    return (target * gain).astype(np.float32)


# =============================================================================
# Mix Bus
# =============================================================================

class MixBus:
    """
    Audio mixing bus with effects chain.

    Supports:
    - Multiple input tracks
    - Gain/pan/mute/solo
    - Insert effects (compressor)
    - Send effects (reverb placeholder)
    """

    def __init__(self, name: str, sample_rate: int = 48000):
        self.name = name
        self.sr = sample_rate
        self.tracks: List[np.ndarray] = []
        self.config = BusConfig(name=name)
        self.compressor: Optional[CompressorConfig] = None

    def add_track(self, samples: np.ndarray):
        """Add a track to this bus."""
        self.tracks.append(samples.astype(np.float32))

    def clear(self):
        """Clear all tracks."""
        self.tracks = []

    def set_gain(self, gain_db: float):
        """Set bus gain in dB."""
        self.config.gain_db = gain_db

    def set_compressor(self, config: CompressorConfig):
        """Enable compression on this bus."""
        self.compressor = config

    def render(self) -> np.ndarray:
        """Render all tracks on this bus to a single output."""
        if not self.tracks:
            return np.array([], dtype=np.float32)

        if self.config.mute:
            max_len = max(len(t) for t in self.tracks)
            return np.zeros(max_len, dtype=np.float32)

        # Find max length
        max_len = max(len(t) for t in self.tracks)

        # Sum tracks (pad shorter ones)
        output = np.zeros(max_len, dtype=np.float32)
        for track in self.tracks:
            if len(track) < max_len:
                padded = np.zeros(max_len, dtype=np.float32)
                padded[:len(track)] = track
                output += padded
            else:
                output += track[:max_len]

        # Apply compressor if enabled
        if self.compressor:
            output = compress(output, self.compressor, self.sr)

        # Apply gain
        gain = db_to_linear(self.config.gain_db)
        output *= gain

        return output


class Mixer:
    """
    Main mixing console.

    Buses:
    - vocal_bus: Lead vocals/voice
    - music_bus: Instruments/backing
    - master_bus: Final output

    Features:
    - Automatic vocal ducking of music
    - Master compression
    - Normalization
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate

        # Create buses
        self.vocal_bus = MixBus("vocals", sample_rate)
        self.music_bus = MixBus("music", sample_rate)
        self.master_bus = MixBus("master", sample_rate)

        # Default settings
        self.vocal_bus.set_gain(3.0)  # Vocals slightly louder
        self.music_bus.set_gain(-3.0)  # Music slightly quieter

        # Ducking config
        self.ducking = DuckingConfig(
            amount=0.35,
            threshold_db=-25.0,
            attack_ms=5.0,
            release_ms=200.0,
            hold_ms=80.0
        )

        # Master compression
        self.master_compressor = CompressorConfig(
            threshold_db=-10.0,
            ratio=3.0,
            attack_ms=15.0,
            release_ms=150.0,
            makeup_gain_db=2.0,
            knee_db=8.0
        )

        self.enable_ducking = True
        self.enable_master_comp = True
        self.normalize = True
        self.target_db = -1.0  # Normalization target

    def add_vocals(self, samples: np.ndarray):
        """Add samples to vocal bus."""
        self.vocal_bus.add_track(samples)

    def add_music(self, samples: np.ndarray):
        """Add samples to music bus."""
        self.music_bus.add_track(samples)

    def clear(self):
        """Clear all buses."""
        self.vocal_bus.clear()
        self.music_bus.clear()
        self.master_bus.clear()

    def mixdown(self) -> np.ndarray:
        """
        Mix all buses to final stereo output.

        Returns mono for now (stereo is straightforward extension).
        """
        # Render individual buses
        vocals = self.vocal_bus.render()
        music = self.music_bus.render()

        # Handle empty buses
        if len(vocals) == 0 and len(music) == 0:
            return np.array([], dtype=np.float32)

        # Ensure same length
        max_len = max(len(vocals) if len(vocals) > 0 else 0,
                      len(music) if len(music) > 0 else 0)

        if len(vocals) == 0:
            vocals = np.zeros(max_len, dtype=np.float32)
        elif len(vocals) < max_len:
            vocals = np.pad(vocals, (0, max_len - len(vocals)))

        if len(music) == 0:
            music = np.zeros(max_len, dtype=np.float32)
        elif len(music) < max_len:
            music = np.pad(music, (0, max_len - len(music)))

        # Apply ducking: reduce music when vocals are present
        if self.enable_ducking and len(vocals) > 0:
            music = duck(music, vocals, self.ducking, self.sr)

        # Sum to master
        master = vocals + music

        # Master compression
        if self.enable_master_comp:
            master = compress(master, self.master_compressor, self.sr)

        # Normalize
        if self.normalize:
            peak = np.max(np.abs(master))
            if peak > 0:
                target_linear = db_to_linear(self.target_db)
                master = master * (target_linear / peak)

        return master.astype(np.float32)

    def mixdown_stereo(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mix to stereo (L, R channels)."""
        mono = self.mixdown()
        # For now, return same signal on both channels
        # Future: implement proper stereo imaging
        return mono, mono


def save_wav_numpy(samples: np.ndarray, path: str, sample_rate: int = 48000,
                   channels: int = 2):
    """
    Save numpy array to WAV file.

    Much faster than sample-by-sample struct.pack.
    """
    import wave
    import struct

    # Ensure float32
    samples = samples.astype(np.float32)

    # Clip to [-1, 1]
    samples = np.clip(samples, -1.0, 1.0)

    # Convert to 16-bit integers
    samples_int = (samples * 32767).astype(np.int16)

    # Create stereo if needed
    if channels == 2:
        stereo = np.column_stack([samples_int, samples_int]).flatten()
        data = stereo.tobytes()
    else:
        data = samples_int.tobytes()

    # Write WAV
    with wave.open(path, 'w') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(data)
