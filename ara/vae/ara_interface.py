"""
ara.vae.ara_interface - Ara State Readout Interface

Combines HGF belief dynamics, VAE latent representations, and disentanglement
metrics into a unified "state readout" that Ara can use for self-modulation.

The core idea:
- β-VAE gives us WHERE you are in latent space
- Disentanglement metrics tell us HOW READABLE that space is
- HGF tells us HOW THOSE COORDINATES SHIFT over time/volatility

Ara doesn't read thoughts - she reads geometry + dynamics of your state
and adjusts herself accordingly.

Usage:
    from ara.vae.ara_interface import AraStateReader, StateReadout

    reader = AraStateReader(latent_dim=10)

    # Feed EEG/behavioral data
    readout = reader.update(
        eeg_features=current_eeg,
        hgf_params={'omega_2': -4.0, 'kappa_1': 1.5},
        task_label='reversal_learning',
    )

    # Ara decision logic
    if readout.geometry_health < 0.3:
        # Latent space is tangled - back off, simplify
        ara.reduce_complexity()
    elif readout.volatility > 0.8:
        # High uncertainty - be more supportive
        ara.increase_validation()
    elif readout.disentanglement > 0.7:
        # Clean geometry - can push richer content
        ara.enable_deep_mode()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

# Import metrics
try:
    from ara.vae.metrics import compute_dci_lite
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False


@dataclass
class StateReadout:
    """
    Single snapshot of Ara's state readout.

    Combines latent position, geometry quality, and dynamics.
    """

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    # Latent position (where in the space)
    latent_mean: Optional[np.ndarray] = None
    latent_std: Optional[np.ndarray] = None

    # Geometry quality (how readable is the space)
    disentanglement: float = 0.0  # DCI modularity
    completeness: float = 0.0     # DCI compactness
    informativeness: float = 0.0  # DCI explicitness

    # Derived: overall geometry health (0-1, higher = cleaner)
    geometry_health: float = 0.0

    # HGF dynamics
    omega_2: float = -4.0   # Tonic volatility
    kappa_1: float = 1.0    # Coupling strength
    theta: float = 1.0      # Response temperature

    # Derived volatility estimate (0-1, higher = more volatile beliefs)
    volatility: float = 0.0

    # Prediction errors (if available)
    delta_1: float = 0.0  # Sensory PE
    delta_2: float = 0.0  # Volatility PE

    # Context labels
    task: str = "unknown"
    condition: str = "baseline"

    # Ara recommendations
    recommended_mode: str = "balanced"
    complexity_level: float = 0.5
    support_level: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "disentanglement": self.disentanglement,
            "completeness": self.completeness,
            "informativeness": self.informativeness,
            "geometry_health": self.geometry_health,
            "omega_2": self.omega_2,
            "kappa_1": self.kappa_1,
            "volatility": self.volatility,
            "delta_1": self.delta_1,
            "delta_2": self.delta_2,
            "task": self.task,
            "recommended_mode": self.recommended_mode,
            "complexity_level": self.complexity_level,
            "support_level": self.support_level,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Ara State Readout @ {self.timestamp.strftime('%H:%M:%S')}",
            "=" * 40,
            f"Geometry Health: {self.geometry_health:.2f}",
            f"  Disentanglement: {self.disentanglement:.3f}",
            f"  Completeness:    {self.completeness:.3f}",
            f"  Informativeness: {self.informativeness:.3f}",
            "",
            f"Volatility: {self.volatility:.2f}",
            f"  ω₂={self.omega_2:.2f}, κ₁={self.kappa_1:.2f}",
            f"  δ₁={self.delta_1:.3f}, δ₂={self.delta_2:.3f}",
            "",
            f"Recommendation: {self.recommended_mode}",
            f"  Complexity: {self.complexity_level:.2f}",
            f"  Support: {self.support_level:.2f}",
        ]
        return "\n".join(lines)


class AraStateReader:
    """
    Real-time state reader for Ara.

    Maintains a rolling buffer of latent encodings and computes
    disentanglement metrics on demand.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        buffer_size: int = 100,
        dci_window: int = 50,
    ):
        """
        Initialize state reader.

        Args:
            latent_dim: Dimensionality of latent space
            buffer_size: Rolling buffer size for history
            dci_window: Samples to use for DCI computation
        """
        self.latent_dim = latent_dim
        self.buffer_size = buffer_size
        self.dci_window = dci_window

        # Rolling buffers
        self.latent_buffer: List[np.ndarray] = []
        self.label_buffer: List[Dict] = []
        self.readout_history: List[StateReadout] = []

        # Current HGF state
        self.current_hgf = {
            'omega_2': -4.0,
            'kappa_1': 1.0,
            'theta': 1.0,
            'delta_1': 0.0,
            'delta_2': 0.0,
        }

        # Encoder (set externally or via set_encoder)
        self.encoder = None

    def set_encoder(self, encoder_fn):
        """
        Set the encoder function for raw data → latents.

        Args:
            encoder_fn: Function that takes raw data and returns latent vector
        """
        self.encoder = encoder_fn

    def update(
        self,
        latent: Optional[np.ndarray] = None,
        raw_data: Optional[np.ndarray] = None,
        hgf_params: Optional[Dict] = None,
        task_label: str = "unknown",
        condition: str = "baseline",
    ) -> StateReadout:
        """
        Update state with new observation.

        Args:
            latent: Pre-computed latent vector (if available)
            raw_data: Raw data to encode (if latent not provided)
            hgf_params: Current HGF parameters
            task_label: Current task
            condition: Current condition

        Returns:
            StateReadout with current state assessment
        """
        # Get latent representation
        if latent is not None:
            z = latent
        elif raw_data is not None and self.encoder is not None:
            z = self.encoder(raw_data)
        else:
            z = np.zeros(self.latent_dim)

        # Update HGF state
        if hgf_params:
            self.current_hgf.update(hgf_params)

        # Add to buffer
        self.latent_buffer.append(z)
        self.label_buffer.append({
            'task': task_label,
            'condition': condition,
            **self.current_hgf,
        })

        # Trim buffer
        if len(self.latent_buffer) > self.buffer_size:
            self.latent_buffer = self.latent_buffer[-self.buffer_size:]
            self.label_buffer = self.label_buffer[-self.buffer_size:]

        # Compute readout
        readout = self._compute_readout(task_label, condition)

        # Store in history
        self.readout_history.append(readout)
        if len(self.readout_history) > self.buffer_size:
            self.readout_history = self.readout_history[-self.buffer_size:]

        return readout

    def _compute_readout(self, task: str, condition: str) -> StateReadout:
        """Compute current state readout."""

        readout = StateReadout(task=task, condition=condition)

        # Latent statistics
        if self.latent_buffer:
            z_stack = np.array(self.latent_buffer)
            readout.latent_mean = z_stack.mean(axis=0)
            readout.latent_std = z_stack.std(axis=0)

        # HGF dynamics
        readout.omega_2 = self.current_hgf.get('omega_2', -4.0)
        readout.kappa_1 = self.current_hgf.get('kappa_1', 1.0)
        readout.theta = self.current_hgf.get('theta', 1.0)
        readout.delta_1 = self.current_hgf.get('delta_1', 0.0)
        readout.delta_2 = self.current_hgf.get('delta_2', 0.0)

        # Compute volatility estimate (0-1 scale)
        # Higher omega_2 (less negative) = more volatile
        # Higher kappa_1 = more sensitive to volatility
        omega_norm = (readout.omega_2 + 8) / 7  # Map [-8, -1] to [0, 1]
        kappa_norm = (readout.kappa_1 - 0.2) / 2.8  # Map [0.2, 3.0] to [0, 1]
        readout.volatility = np.clip(0.5 * omega_norm + 0.5 * kappa_norm, 0, 1)

        # Compute disentanglement if enough samples
        if len(self.latent_buffer) >= self.dci_window and HAS_METRICS:
            readout = self._compute_dci(readout)
        else:
            # Default: assume moderate geometry
            readout.disentanglement = 0.5
            readout.completeness = 0.5
            readout.informativeness = 0.5

        # Geometry health: weighted average
        readout.geometry_health = (
            0.4 * readout.disentanglement +
            0.3 * readout.completeness +
            0.3 * readout.informativeness
        )

        # Compute Ara recommendations
        readout = self._compute_recommendations(readout)

        return readout

    def _compute_dci(self, readout: StateReadout) -> StateReadout:
        """Compute DCI-lite from buffer."""

        # Get recent latents and labels
        n = min(len(self.latent_buffer), self.dci_window)
        z = np.array(self.latent_buffer[-n:])

        # Build proxy labels from HGF params
        labels = np.array([
            [
                self.label_buffer[-(n-i)].get('omega_2', -4.0),
                self.label_buffer[-(n-i)].get('kappa_1', 1.0),
                self.label_buffer[-(n-i)].get('theta', 1.0),
            ]
            for i in range(n)
        ])

        try:
            dci = compute_dci_lite(
                z, labels,
                label_names=['omega_2', 'kappa_1', 'theta'],
            )
            readout.disentanglement = dci.get('approximate_modularity', 0.5)
            readout.completeness = 0.5  # Can't compute without factor labels
            readout.informativeness = np.mean(list(dci.get('accuracies', {}).values()))
        except Exception:
            # Fallback on error
            readout.disentanglement = 0.5
            readout.completeness = 0.5
            readout.informativeness = 0.5

        return readout

    def _compute_recommendations(self, readout: StateReadout) -> StateReadout:
        """
        Compute Ara mode recommendations based on state.

        This is where the "math-telepathy" becomes actionable.
        """

        # Decision logic
        gh = readout.geometry_health
        vol = readout.volatility

        # Mode selection
        if gh < 0.3:
            # Tangled geometry - back off
            readout.recommended_mode = "simplified"
            readout.complexity_level = 0.2
            readout.support_level = 0.7
        elif vol > 0.7:
            # High volatility - be supportive
            readout.recommended_mode = "supportive"
            readout.complexity_level = 0.4
            readout.support_level = 0.9
        elif gh > 0.7 and vol < 0.3:
            # Clean geometry, stable - can go deep
            readout.recommended_mode = "deep"
            readout.complexity_level = 0.9
            readout.support_level = 0.4
        else:
            # Balanced
            readout.recommended_mode = "balanced"
            readout.complexity_level = 0.5 + 0.3 * (gh - 0.5)
            readout.support_level = 0.5 + 0.3 * (vol - 0.5)

        return readout

    def get_trend(self, metric: str = 'geometry_health', window: int = 10) -> float:
        """
        Get trend of a metric over recent history.

        Returns:
            Slope estimate (-1 to 1, positive = improving)
        """
        if len(self.readout_history) < 2:
            return 0.0

        n = min(len(self.readout_history), window)
        values = [getattr(self.readout_history[-(n-i)], metric, 0.5) for i in range(n)]

        # Simple linear trend
        x = np.arange(n)
        slope = np.polyfit(x, values, 1)[0]

        # Normalize to [-1, 1]
        return np.clip(slope * 10, -1, 1)

    def should_alert(self) -> Optional[str]:
        """
        Check if Ara should raise an alert.

        Returns:
            Alert message or None
        """
        if not self.readout_history:
            return None

        current = self.readout_history[-1]

        # Check for concerning patterns
        if current.geometry_health < 0.2:
            return "geometry_collapsed"

        if current.volatility > 0.9:
            return "high_volatility"

        # Check for rapid deterioration
        gh_trend = self.get_trend('geometry_health')
        if gh_trend < -0.5:
            return "geometry_declining"

        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ara_reader_from_vae(
    vae_params: Dict,
    config: Any,
    latent_dim: int = 10,
) -> AraStateReader:
    """
    Create AraStateReader with VAE encoder.

    Args:
        vae_params: Trained VAE parameters
        config: VAE config
        latent_dim: Latent dimensionality

    Returns:
        Configured AraStateReader
    """
    from ara.vae.jax_spmd import encoder

    def encode_fn(x):
        x_flat = x.reshape(1, -1)
        mu, _ = encoder(vae_params, x_flat)
        return np.array(mu[0])

    reader = AraStateReader(latent_dim=latent_dim)
    reader.set_encoder(encode_fn)

    return reader


def log_readout_to_wandb(readout: StateReadout, step: Optional[int] = None):
    """
    Log state readout to Weights & Biases.

    Args:
        readout: Current state readout
        step: Optional step number
    """
    try:
        import wandb
        wandb.log(readout.to_dict(), step=step)
    except ImportError:
        pass
