"""
Spike Encoder - HPV to Spike Stream Conversion
===============================================

Convert hyperdimensional state vectors (HPVs) into spike trains
that can drive spiking neural networks.

Encoding strategies:
1. Rate coding: HPV element → spike probability
2. Temporal coding: HPV element → spike timing
3. Population coding: HPV → distributed spike pattern
4. Direct threshold: Bipolar HPV → single spike per +1 element

For the Hebbian policy learner, we need a spike stream that:
- Preserves similarity structure of HPVs
- Produces sparse but informative spike patterns
- Can drive the "pre" input of the three-factor rule

Usage:
    encoder = SpikeEncoder(dim=1024, method="rate")

    # Each timestep:
    spikes = encoder.encode(hpv)  # (1024,) binary

    # Feed to Hebbian learner:
    learner.step(pre=spikes, post=detector_spikes, rho=neuromod)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum, auto


class EncodingMethod(Enum):
    """Spike encoding methods."""
    RATE = auto()        # Element value → spike probability
    TEMPORAL = auto()    # Element value → spike timing
    POPULATION = auto()  # Distributed across population
    THRESHOLD = auto()   # Direct threshold on bipolar


@dataclass
class SpikeEncoderConfig:
    """Configuration for spike encoder."""
    dim: int = 1024                      # Input HPV dimension
    method: EncodingMethod = EncodingMethod.RATE
    rate_scale: float = 1.0              # Scale factor for rate coding
    threshold: float = 0.0               # Threshold for threshold coding
    temporal_steps: int = 10             # Steps for temporal coding
    population_size: int = 4             # Neurons per dimension for population
    noise_std: float = 0.0               # Additive noise


class SpikeEncoder:
    """
    Convert HPVs to spike trains.

    Bridges the gap between HDC (continuous/bipolar vectors)
    and SNN (discrete spike events).
    """

    def __init__(self, config: Optional[SpikeEncoderConfig] = None,
                 dim: int = 1024, method: str = "rate"):
        if config:
            self.cfg = config
        else:
            method_enum = EncodingMethod[method.upper()]
            self.cfg = SpikeEncoderConfig(dim=dim, method=method_enum)

        # State for temporal encoding
        self._temporal_phase = 0
        self._temporal_cache: Optional[np.ndarray] = None

        # Population coding bases (if used)
        if self.cfg.method == EncodingMethod.POPULATION:
            self._init_population_bases()

    def _init_population_bases(self):
        """Initialize population coding basis vectors."""
        rng = np.random.default_rng(42)
        n = self.cfg.population_size
        self._pop_centers = np.linspace(-1, 1, n)
        self._pop_width = 2.0 / n

    @property
    def output_dim(self) -> int:
        """Output dimension (may differ from input for population coding)."""
        if self.cfg.method == EncodingMethod.POPULATION:
            return self.cfg.dim * self.cfg.population_size
        return self.cfg.dim

    def encode(self, hpv: np.ndarray, timestep: int = 0) -> np.ndarray:
        """
        Encode HPV into spike train for current timestep.

        Args:
            hpv: Hypervector, shape (dim,), values in [-1, 1] or {-1, +1}
            timestep: Current timestep (for temporal coding)

        Returns:
            spikes: Binary spike array, shape (output_dim,)
        """
        if self.cfg.noise_std > 0:
            hpv = hpv + np.random.randn(len(hpv)) * self.cfg.noise_std

        if self.cfg.method == EncodingMethod.RATE:
            return self._encode_rate(hpv)
        elif self.cfg.method == EncodingMethod.TEMPORAL:
            return self._encode_temporal(hpv, timestep)
        elif self.cfg.method == EncodingMethod.POPULATION:
            return self._encode_population(hpv)
        elif self.cfg.method == EncodingMethod.THRESHOLD:
            return self._encode_threshold(hpv)
        else:
            raise ValueError(f"Unknown encoding method: {self.cfg.method}")

    def _encode_rate(self, hpv: np.ndarray) -> np.ndarray:
        """
        Rate coding: element value → spike probability.

        Maps [-1, 1] to [0, 1] probability, then samples.
        """
        # Map bipolar [-1, 1] to probability [0, 1]
        prob = (hpv + 1) / 2 * self.cfg.rate_scale
        prob = np.clip(prob, 0, 1)

        # Sample spikes
        spikes = (np.random.rand(len(hpv)) < prob).astype(np.float32)
        return spikes

    def _encode_temporal(self, hpv: np.ndarray, timestep: int) -> np.ndarray:
        """
        Temporal coding: element value → spike timing.

        Higher values spike earlier in the window.
        """
        n_steps = self.cfg.temporal_steps
        phase = timestep % n_steps

        # Map [-1, 1] to spike time [0, n_steps-1]
        # Higher value → earlier spike
        spike_times = ((1 - hpv) / 2 * (n_steps - 1)).astype(int)

        # Spike if current phase matches spike time
        spikes = (spike_times == phase).astype(np.float32)
        return spikes

    def _encode_population(self, hpv: np.ndarray) -> np.ndarray:
        """
        Population coding: value → activation across population.

        Each dimension uses N neurons with Gaussian tuning curves.
        """
        n = self.cfg.population_size
        output = np.zeros(self.cfg.dim * n)

        for i, val in enumerate(hpv):
            # Compute activation for each population neuron
            for j, center in enumerate(self._pop_centers):
                dist = abs(val - center)
                activation = np.exp(-dist**2 / (2 * self._pop_width**2))
                # Sample spike
                if np.random.rand() < activation:
                    output[i * n + j] = 1.0

        return output

    def _encode_threshold(self, hpv: np.ndarray) -> np.ndarray:
        """
        Threshold coding: direct threshold on bipolar.

        Simplest: spike where hpv > threshold.
        """
        spikes = (hpv > self.cfg.threshold).astype(np.float32)
        return spikes

    def encode_sequence(self, hpv: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Encode HPV into a sequence of spike trains.

        Args:
            hpv: Hypervector
            n_steps: Number of timesteps

        Returns:
            spike_trains: Array of shape (n_steps, output_dim)
        """
        trains = []
        for t in range(n_steps):
            trains.append(self.encode(hpv, timestep=t))
        return np.array(trains)


class SpikeDecoder:
    """
    Decode spike trains back to continuous values.

    For reading out detector activations.
    """

    def __init__(self, method: str = "rate", window: int = 10):
        self.method = method
        self.window = window
        self._history: List[np.ndarray] = []

    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """
        Decode spike train to continuous activation.

        Args:
            spikes: Binary spike array

        Returns:
            activations: Continuous values [0, 1]
        """
        self._history.append(spikes)
        if len(self._history) > self.window:
            self._history.pop(0)

        # Rate decode: count spikes in window
        if len(self._history) >= self.window:
            rate = np.mean(self._history, axis=0)
            return rate
        else:
            return spikes.astype(np.float32)

    def reset(self):
        """Reset history."""
        self._history.clear()


# ============================================================================
# HPV-aware Spike Stream
# ============================================================================

class HPVSpikeStream:
    """
    Continuous spike stream from rolling HPV state.

    Integrates with StateStream to produce spike trains
    for the Hebbian learner's "pre" input.
    """

    def __init__(self, encoder: SpikeEncoder):
        self.encoder = encoder
        self._last_hpv: Optional[np.ndarray] = None
        self._timestep = 0

    def update(self, hpv: np.ndarray) -> np.ndarray:
        """
        Update with new HPV and generate spikes.

        Args:
            hpv: Current state hypervector

        Returns:
            spikes: Spike train for this timestep
        """
        self._last_hpv = hpv
        spikes = self.encoder.encode(hpv, self._timestep)
        self._timestep += 1
        return spikes

    def tick(self) -> Optional[np.ndarray]:
        """
        Generate spikes for current HPV (no update).

        Useful for temporal coding where same HPV
        produces different spikes at different times.
        """
        if self._last_hpv is None:
            return None
        spikes = self.encoder.encode(self._last_hpv, self._timestep)
        self._timestep += 1
        return spikes
