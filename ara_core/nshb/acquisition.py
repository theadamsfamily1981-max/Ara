#!/usr/bin/env python3
"""
NSHB Acquisition Layer - Human Signal Acquisition
==================================================

Acquires and preprocesses three signal streams:
    1. Neural: EEG → avalanche-ready binary matrix B(c,τ)
    2. Physiological: HRV, GSR, pupil → φ_phys(t)
    3. Contextual: Task load, focus, errors → φ_ctx(t)

These feed into the Kitten Fabric for λ/Π estimation.

The acquisition layer is hardware-agnostic: real devices provide
callbacks, simulated devices generate synthetic data for testing.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum, auto
from abc import ABC, abstractmethod
import numpy as np


# =============================================================================
# Signal Types
# =============================================================================

class SignalType(Enum):
    """Types of acquired signals."""
    EEG = auto()
    HRV = auto()
    GSR = auto()
    PUPIL = auto()
    RESPIRATION = auto()
    TASK_LOAD = auto()
    FOCUS_SCORE = auto()
    ERROR_RATE = auto()


@dataclass
class SignalMetadata:
    """Metadata for a signal stream."""
    signal_type: SignalType
    sample_rate_hz: float
    n_channels: int = 1
    unit: str = "au"
    device_id: str = "unknown"


# =============================================================================
# Neural Acquisition (EEG)
# =============================================================================

@dataclass
class EEGSample:
    """A single EEG sample across all channels."""
    timestamp: float
    channels: np.ndarray          # (n_channels,) voltage values
    quality: np.ndarray           # (n_channels,) signal quality 0-1


@dataclass
class EEGBuffer:
    """
    Circular buffer for EEG data.

    Stores raw samples and provides windowed access for processing.
    """
    n_channels: int
    sample_rate_hz: float
    buffer_duration_s: float = 10.0

    # Internal storage
    _data: np.ndarray = field(default=None, repr=False)
    _timestamps: np.ndarray = field(default=None, repr=False)
    _quality: np.ndarray = field(default=None, repr=False)
    _write_idx: int = 0
    _samples_written: int = 0

    def __post_init__(self):
        buffer_samples = int(self.buffer_duration_s * self.sample_rate_hz)
        self._data = np.zeros((buffer_samples, self.n_channels), dtype=np.float32)
        self._timestamps = np.zeros(buffer_samples, dtype=np.float64)
        self._quality = np.zeros((buffer_samples, self.n_channels), dtype=np.float32)

    def add_sample(self, sample: EEGSample):
        """Add a sample to the buffer."""
        idx = self._write_idx % len(self._data)
        self._data[idx] = sample.channels
        self._timestamps[idx] = sample.timestamp
        self._quality[idx] = sample.quality
        self._write_idx += 1
        self._samples_written += 1

    def get_window(self, duration_s: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the most recent window of data.

        Returns (data, timestamps) arrays.
        """
        n_samples = min(
            int(duration_s * self.sample_rate_hz),
            self._samples_written,
            len(self._data)
        )

        if n_samples == 0:
            return np.array([]), np.array([])

        end_idx = self._write_idx % len(self._data)
        start_idx = (end_idx - n_samples) % len(self._data)

        if start_idx < end_idx:
            return self._data[start_idx:end_idx], self._timestamps[start_idx:end_idx]
        else:
            # Wrap around
            data = np.concatenate([self._data[start_idx:], self._data[:end_idx]])
            timestamps = np.concatenate([self._timestamps[start_idx:], self._timestamps[:end_idx]])
            return data, timestamps

    def mean_quality(self) -> float:
        """Get mean signal quality across recent samples."""
        if self._samples_written == 0:
            return 0.0
        n = min(self._samples_written, len(self._quality))
        return float(np.mean(self._quality[:n]))


class EEGPreprocessor:
    """
    EEG preprocessing pipeline.

    Applies:
    - Bandpass filtering (1-45 Hz)
    - Notch filter (50/60 Hz)
    - Common average reference
    - Artifact rejection
    """

    def __init__(
        self,
        sample_rate_hz: float,
        lowcut_hz: float = 1.0,
        highcut_hz: float = 45.0,
        notch_hz: float = 60.0,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz
        self.notch_hz = notch_hz

        # Filter coefficients would be computed here
        # For now, simplified processing

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process raw EEG data.

        Args:
            data: (n_samples, n_channels) raw data

        Returns:
            Preprocessed data, same shape
        """
        if data.size == 0:
            return data

        # Simple preprocessing (production would use scipy.signal)
        # 1. Remove DC offset
        data = data - np.mean(data, axis=0, keepdims=True)

        # 2. Common average reference
        car = np.mean(data, axis=1, keepdims=True)
        data = data - car

        # 3. Simple smoothing (stand-in for bandpass)
        if len(data) > 3:
            kernel = np.array([0.25, 0.5, 0.25])
            for ch in range(data.shape[1]):
                data[:, ch] = np.convolve(data[:, ch], kernel, mode='same')

        return data

    def to_binary_events(
        self,
        data: np.ndarray,
        threshold_std: float = 2.0,
    ) -> np.ndarray:
        """
        Convert preprocessed EEG to binary event matrix B(c,τ).

        Events are defined as threshold crossings.

        Returns:
            Binary matrix (n_samples, n_channels)
        """
        if data.size == 0:
            return np.zeros_like(data, dtype=np.int8)

        # Compute per-channel thresholds
        stds = np.std(data, axis=0, keepdims=True)
        stds = np.maximum(stds, 1e-6)  # Avoid division by zero

        # Z-score
        z = np.abs(data) / stds

        # Threshold
        events = (z > threshold_std).astype(np.int8)

        return events


# =============================================================================
# Physiological Acquisition
# =============================================================================

@dataclass
class PhysioSample:
    """A sample of physiological signals."""
    timestamp: float
    hrv_rmssd_ms: float = 50.0      # Heart rate variability
    hr_bpm: float = 70.0            # Heart rate
    gsr_us: float = 2.0             # Galvanic skin response (microsiemens)
    pupil_diameter_mm: float = 4.0  # Pupil diameter
    resp_rate_bpm: float = 15.0     # Respiration rate
    skin_temp_c: float = 33.0       # Skin temperature


@dataclass
class PhysioBuffer:
    """Buffer for physiological signals."""
    buffer_duration_s: float = 60.0
    sample_rate_hz: float = 10.0

    _samples: List[PhysioSample] = field(default_factory=list)

    def add_sample(self, sample: PhysioSample):
        self._samples.append(sample)
        # Trim old samples
        cutoff = time.time() - self.buffer_duration_s
        self._samples = [s for s in self._samples if s.timestamp > cutoff]

    def get_features(self, window_s: float = 30.0) -> np.ndarray:
        """
        Extract feature vector φ_phys(t) from recent samples.

        Returns z-scored, windowed features.
        """
        cutoff = time.time() - window_s
        recent = [s for s in self._samples if s.timestamp > cutoff]

        if len(recent) < 3:
            # Not enough data, return neutral features
            return np.zeros(6, dtype=np.float32)

        # Extract raw values
        hrv = np.array([s.hrv_rmssd_ms for s in recent])
        hr = np.array([s.hr_bpm for s in recent])
        gsr = np.array([s.gsr_us for s in recent])
        pupil = np.array([s.pupil_diameter_mm for s in recent])
        resp = np.array([s.resp_rate_bpm for s in recent])
        temp = np.array([s.skin_temp_c for s in recent])

        # Compute summary statistics
        features = np.array([
            np.mean(hrv) / 50.0,          # Normalized HRV (higher = calmer)
            np.mean(hr) / 80.0,           # Normalized HR
            np.mean(gsr) / 5.0,           # Normalized GSR (higher = aroused)
            np.mean(pupil) / 5.0,         # Normalized pupil
            np.mean(resp) / 15.0,         # Normalized resp
            np.std(hr) / 10.0,            # HR variability indicator
        ], dtype=np.float32)

        return features


# =============================================================================
# Contextual Acquisition
# =============================================================================

@dataclass
class ContextSample:
    """A sample of contextual/behavioral signals."""
    timestamp: float
    task_load: float = 0.5          # 0-1, cognitive load
    focus_score: float = 0.5        # 0-1, attention/focus
    error_rate: float = 0.0         # Errors per minute
    response_time_ms: float = 500.0 # Mean response time
    typing_speed_wpm: float = 40.0  # Typing speed
    mouse_entropy: float = 0.5      # Movement unpredictability


@dataclass
class ContextBuffer:
    """Buffer for contextual signals."""
    buffer_duration_s: float = 300.0  # 5 minutes

    _samples: List[ContextSample] = field(default_factory=list)

    def add_sample(self, sample: ContextSample):
        self._samples.append(sample)
        cutoff = time.time() - self.buffer_duration_s
        self._samples = [s for s in self._samples if s.timestamp > cutoff]

    def get_features(self, window_s: float = 60.0) -> np.ndarray:
        """
        Extract feature vector φ_ctx(t) from recent samples.
        """
        cutoff = time.time() - window_s
        recent = [s for s in self._samples if s.timestamp > cutoff]

        if len(recent) < 2:
            return np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)

        features = np.array([
            np.mean([s.task_load for s in recent]),
            np.mean([s.focus_score for s in recent]),
            np.mean([s.error_rate for s in recent]) / 5.0,  # Normalize
            np.mean([s.response_time_ms for s in recent]) / 1000.0,
        ], dtype=np.float32)

        return features


# =============================================================================
# Unified Acquisition System
# =============================================================================

class AcquisitionSystem:
    """
    Unified acquisition system for NSHB.

    Manages all signal streams and provides synchronized access.
    """

    def __init__(
        self,
        eeg_channels: int = 8,
        eeg_sample_rate: float = 256.0,
        verbose: bool = True,
    ):
        self.verbose = verbose

        # EEG subsystem
        self.eeg_buffer = EEGBuffer(
            n_channels=eeg_channels,
            sample_rate_hz=eeg_sample_rate,
        )
        self.eeg_preprocessor = EEGPreprocessor(eeg_sample_rate)

        # Physio subsystem
        self.physio_buffer = PhysioBuffer()

        # Context subsystem
        self.context_buffer = ContextBuffer()

        # State
        self.running = False
        self.last_update = time.time()

        # Callbacks for real hardware
        self.eeg_callback: Optional[Callable[[], EEGSample]] = None
        self.physio_callback: Optional[Callable[[], PhysioSample]] = None
        self.context_callback: Optional[Callable[[], ContextSample]] = None

        if self.verbose:
            print(f"[Acquisition] Initialized: {eeg_channels}ch EEG @ {eeg_sample_rate}Hz")

    def start(self):
        """Start acquisition."""
        self.running = True
        if self.verbose:
            print("[Acquisition] Started")

    def stop(self):
        """Stop acquisition."""
        self.running = False
        if self.verbose:
            print("[Acquisition] Stopped")

    def update(self):
        """
        Update all buffers with new samples.

        In production, this would be called by hardware callbacks.
        Here we simulate if no callbacks are set.
        """
        now = time.time()

        # EEG
        if self.eeg_callback:
            sample = self.eeg_callback()
            self.eeg_buffer.add_sample(sample)
        else:
            self._simulate_eeg(now)

        # Physio
        if self.physio_callback:
            sample = self.physio_callback()
            self.physio_buffer.add_sample(sample)
        else:
            self._simulate_physio(now)

        # Context
        if self.context_callback:
            sample = self.context_callback()
            self.context_buffer.add_sample(sample)
        else:
            self._simulate_context(now)

        self.last_update = now

    def _simulate_eeg(self, timestamp: float):
        """Generate synthetic EEG sample."""
        n_ch = self.eeg_buffer.n_channels

        # Simulate alpha + noise
        alpha_freq = 10.0
        phase = 2 * math.pi * alpha_freq * timestamp
        alpha = 20 * np.sin(phase + np.random.randn(n_ch) * 0.5)
        noise = np.random.randn(n_ch) * 10

        sample = EEGSample(
            timestamp=timestamp,
            channels=alpha + noise,
            quality=np.ones(n_ch) * 0.9,
        )
        self.eeg_buffer.add_sample(sample)

    def _simulate_physio(self, timestamp: float):
        """Generate synthetic physio sample."""
        sample = PhysioSample(
            timestamp=timestamp,
            hrv_rmssd_ms=50 + np.random.randn() * 10,
            hr_bpm=70 + np.random.randn() * 5,
            gsr_us=2.0 + np.random.randn() * 0.3,
            pupil_diameter_mm=4.0 + np.random.randn() * 0.2,
            resp_rate_bpm=15 + np.random.randn() * 2,
        )
        self.physio_buffer.add_sample(sample)

    def _simulate_context(self, timestamp: float):
        """Generate synthetic context sample."""
        sample = ContextSample(
            timestamp=timestamp,
            task_load=0.5 + np.random.randn() * 0.1,
            focus_score=0.7 + np.random.randn() * 0.1,
            error_rate=np.abs(np.random.randn()) * 0.5,
            response_time_ms=500 + np.random.randn() * 50,
        )
        self.context_buffer.add_sample(sample)

    def get_eeg_window(self, duration_s: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Get preprocessed EEG window."""
        raw, timestamps = self.eeg_buffer.get_window(duration_s)
        if raw.size == 0:
            return raw, timestamps
        processed = self.eeg_preprocessor.process(raw)
        return processed, timestamps

    def get_binary_events(self, duration_s: float = 2.0) -> np.ndarray:
        """Get binary event matrix B(c,τ) for avalanche analysis."""
        raw, _ = self.eeg_buffer.get_window(duration_s)
        if raw.size == 0:
            return np.zeros((0, self.eeg_buffer.n_channels), dtype=np.int8)
        processed = self.eeg_preprocessor.process(raw)
        events = self.eeg_preprocessor.to_binary_events(processed)
        return events

    def get_physio_features(self) -> np.ndarray:
        """Get physiological feature vector φ_phys(t)."""
        return self.physio_buffer.get_features()

    def get_context_features(self) -> np.ndarray:
        """Get contextual feature vector φ_ctx(t)."""
        return self.context_buffer.get_features()

    def get_status(self) -> Dict[str, Any]:
        """Get acquisition status."""
        return {
            "running": self.running,
            "eeg_quality": self.eeg_buffer.mean_quality(),
            "eeg_samples": self.eeg_buffer._samples_written,
            "physio_samples": len(self.physio_buffer._samples),
            "context_samples": len(self.context_buffer._samples),
            "last_update": self.last_update,
        }
