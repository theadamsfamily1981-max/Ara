"""Thermodynamic Regulation - The Soul of Physics.

Thoughts have a physical cost. This module implements thermodynamic
monitoring using Entropy Production (Π_q) and Variational Free Energy (VFE)
to penalize inefficient thinking and force recovery when the system
"overheats" cognitively.

Key Concepts:

    Entropy Production (Π_q): Rate of entropy generation during processing
        - Measures cognitive "heat" generated
        - High Π_q = inefficient thinking, likely hallucination
        - Bounded by biological limits (~10 kT per bit at room temperature)

    Variational Free Energy (VFE): Prediction error + complexity
        - Measures surprise + model complexity
        - Minimizing VFE = accurate, parsimonious models
        - High VFE = poor predictions, overfitting

    Energy Budget: Total computational energy available
        - Depletes with processing
        - Regenerates during recovery
        - Forces rest when exhausted

Physical Constraints:
    - Landauer's Principle: kT ln(2) per bit erased
    - Second Law: Entropy can only increase
    - Free Energy Principle: Systems minimize VFE

This implements thermodynamic monitoring from grok_tgsfn/thermodynamics.py.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import warnings
import math
import sys
from pathlib import Path

# Add TFAN to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Try to import TFAN thermodynamics
_TFAN_THERMO_AVAILABLE = False
try:
    from grok_tgsfn.thermodynamics import ThermodynamicMonitor as TFANThermodynamicMonitor
    _TFAN_THERMO_AVAILABLE = True
except ImportError:
    pass

# Physical constants
BOLTZMANN_K = 1.380649e-23  # J/K
ROOM_TEMPERATURE = 300.0    # K
KT = BOLTZMANN_K * ROOM_TEMPERATURE  # ~4.14e-21 J
LANDAUER_LIMIT = KT * math.log(2)  # ~2.87e-21 J per bit erased


class ThermalState(Enum):
    """Thermal state of the cognitive system."""
    COOL = auto()       # Normal operation, low entropy production
    WARM = auto()       # Elevated entropy, caution advised
    HOT = auto()        # High entropy, efficiency degraded
    OVERHEATING = auto() # Critical, force recovery


@dataclass
class ThermodynamicStats:
    """Thermodynamic statistics from processing."""
    Pi_q: float                    # Entropy production rate
    vfe: float                     # Variational free energy
    energy_consumed: float         # Energy consumed (normalized)
    efficiency: float              # Processing efficiency [0, 1]
    dissipation: float             # Heat dissipation
    correlation_entropy: float     # Entropy from correlations
    thermal_state: ThermalState
    bits_processed: int
    energy_per_bit: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Pi_q": self.Pi_q,
            "vfe": self.vfe,
            "energy_consumed": self.energy_consumed,
            "efficiency": self.efficiency,
            "dissipation": self.dissipation,
            "correlation_entropy": self.correlation_entropy,
            "thermal_state": self.thermal_state.name,
            "bits_processed": self.bits_processed,
            "energy_per_bit": self.energy_per_bit,
        }


@dataclass
class EnergyBudget:
    """Energy budget for cognitive processing."""
    total_capacity: float      # Total energy capacity
    current_level: float       # Current energy level
    consumption_rate: float    # Energy per cognitive step
    recovery_rate: float       # Energy recovery per second
    last_update: float         # Timestamp of last update

    def consume(self, amount: float) -> bool:
        """Consume energy. Returns False if insufficient."""
        if self.current_level < amount:
            return False
        self.current_level -= amount
        return True

    def recover(self, elapsed_seconds: float):
        """Recover energy over time."""
        recovery = elapsed_seconds * self.recovery_rate
        self.current_level = min(self.total_capacity, self.current_level + recovery)
        self.last_update = time.time()

    @property
    def percentage(self) -> float:
        return self.current_level / self.total_capacity if self.total_capacity > 0 else 0.0


class ThermodynamicMonitor:
    """
    Thermodynamic Monitor - Tracks cognitive "heat" and efficiency.

    Monitors entropy production during processing and can trigger
    recovery mode when the system "overheats" cognitively.

    Args:
        max_entropy_threshold: Maximum allowed entropy production
        energy_capacity: Total energy budget
        tau_membrane: Membrane time constant for neural dynamics
        device: Compute device
    """

    def __init__(
        self,
        max_entropy_threshold: float = 2.0,
        energy_capacity: float = 100.0,
        consumption_rate: float = 0.1,
        recovery_rate: float = 0.05,
        tau_membrane: float = 10.0,
        device: str = "cpu",
    ):
        self.max_entropy_threshold = max_entropy_threshold
        self.tau_membrane = tau_membrane
        self.device = device

        # TFAN monitor if available
        self.tfan_monitor = None
        if _TFAN_THERMO_AVAILABLE:
            try:
                self.tfan_monitor = TFANThermodynamicMonitor()
            except Exception as e:
                warnings.warn(f"Failed to init TFAN thermodynamic monitor: {e}")

        # Energy budget
        self.energy_budget = EnergyBudget(
            total_capacity=energy_capacity,
            current_level=energy_capacity,
            consumption_rate=consumption_rate,
            recovery_rate=recovery_rate,
            last_update=time.time(),
        )

        # History
        self._stats_history: List[ThermodynamicStats] = []
        self._Pi_q_history: List[float] = []

        # Running averages
        self._avg_Pi_q = 0.0
        self._avg_vfe = 0.0
        self._total_bits = 0

    def compute_entropy_production(
        self,
        membrane_potentials: Optional[torch.Tensor] = None,
        spikes: Optional[torch.Tensor] = None,
        activations: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> ThermodynamicStats:
        """
        Compute entropy production during processing.

        Π_q = dissipation + correlation_entropy

        Args:
            membrane_potentials: Neural membrane potentials (spiking models)
            spikes: Spike trains (spiking models)
            activations: Layer activations (transformer models)
            predictions: Model predictions
            targets: Target values (for VFE computation)

        Returns:
            ThermodynamicStats with all metrics
        """
        if self.tfan_monitor is not None:
            return self._convert_tfan_stats(
                self.tfan_monitor.compute_entropy_production(
                    membrane_potentials, spikes, self.tau_membrane
                )
            )

        # Fallback implementation

        # Compute from activations if provided
        if activations is not None:
            return self._compute_from_activations(activations, predictions, targets)

        # Compute from spikes if provided
        if membrane_potentials is not None and spikes is not None:
            return self._compute_from_spikes(membrane_potentials, spikes)

        # Default minimal computation
        return ThermodynamicStats(
            Pi_q=0.0,
            vfe=0.0,
            energy_consumed=0.0,
            efficiency=1.0,
            dissipation=0.0,
            correlation_entropy=0.0,
            thermal_state=ThermalState.COOL,
            bits_processed=0,
            energy_per_bit=0.0,
        )

    def _compute_from_activations(
        self,
        activations: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> ThermodynamicStats:
        """Compute thermodynamic stats from transformer activations."""
        # Flatten activations
        flat = activations.flatten().float()

        # Estimate bits processed (information content)
        # Using entropy of activation distribution
        probs = torch.softmax(flat / flat.std(), dim=0)
        info_entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
        bits_processed = max(1, int(info_entropy * len(flat) / 1000))

        # Compute dissipation (variance-based approximation)
        # High variance = more "heat" generated
        variance = flat.var().item()
        dissipation = math.log1p(variance)

        # Compute correlation entropy (mutual information approximation)
        # Reshape to look at correlations
        if activations.dim() >= 2:
            reshaped = activations.view(-1, activations.shape[-1])
            # Correlation matrix
            corr = torch.corrcoef(reshaped.T)
            # Entropy of correlations
            corr_flat = corr.flatten()
            corr_probs = torch.softmax(corr_flat.abs(), dim=0)
            correlation_entropy = -torch.sum(
                corr_probs * torch.log2(corr_probs + 1e-10)
            ).item() / 100.0  # Normalize
        else:
            correlation_entropy = 0.0

        # Entropy production rate
        Pi_q = dissipation + correlation_entropy

        # Variational Free Energy (if predictions available)
        vfe = 0.0
        if predictions is not None and targets is not None:
            # Prediction error
            prediction_error = torch.nn.functional.mse_loss(
                predictions.float(), targets.float()
            ).item()
            # Complexity penalty (L2 of activations)
            complexity = (flat ** 2).mean().item()
            vfe = prediction_error + 0.1 * complexity

        # Energy consumption
        energy_consumed = self.energy_budget.consumption_rate * (1 + Pi_q / 10)

        # Compute efficiency
        # Ideal: Landauer limit per bit
        ideal_energy = bits_processed * LANDAUER_LIMIT
        actual_energy = energy_consumed * self.energy_budget.total_capacity / 100
        efficiency = ideal_energy / (actual_energy + 1e-10) if actual_energy > 0 else 1.0
        efficiency = min(1.0, efficiency)  # Cap at 1

        # Energy per bit
        energy_per_bit = actual_energy / bits_processed if bits_processed > 0 else 0.0

        # Determine thermal state
        thermal_state = self._determine_thermal_state(Pi_q)

        # Update energy budget
        self.energy_budget.consume(energy_consumed)

        # Update history
        self._Pi_q_history.append(Pi_q)
        if len(self._Pi_q_history) > 100:
            self._Pi_q_history.pop(0)

        # Update running averages
        alpha = 0.1
        self._avg_Pi_q = alpha * Pi_q + (1 - alpha) * self._avg_Pi_q
        self._avg_vfe = alpha * vfe + (1 - alpha) * self._avg_vfe
        self._total_bits += bits_processed

        stats = ThermodynamicStats(
            Pi_q=Pi_q,
            vfe=vfe,
            energy_consumed=energy_consumed,
            efficiency=efficiency,
            dissipation=dissipation,
            correlation_entropy=correlation_entropy,
            thermal_state=thermal_state,
            bits_processed=bits_processed,
            energy_per_bit=energy_per_bit,
        )

        self._stats_history.append(stats)
        if len(self._stats_history) > 100:
            self._stats_history.pop(0)

        return stats

    def _compute_from_spikes(
        self,
        membrane_potentials: torch.Tensor,
        spikes: torch.Tensor,
    ) -> ThermodynamicStats:
        """Compute thermodynamic stats from spiking neural network."""
        # Dissipation from membrane potential decay
        # dV/dt = -(V - V_rest) / tau_m
        # Dissipation ∝ (V - V_rest)^2 / tau_m
        V_rest = 0.0
        dissipation = ((membrane_potentials - V_rest) ** 2).mean().item() / self.tau_membrane

        # Spike train entropy
        spike_rate = spikes.float().mean().item()
        if 0 < spike_rate < 1:
            spike_entropy = -(
                spike_rate * math.log2(spike_rate + 1e-10) +
                (1 - spike_rate) * math.log2(1 - spike_rate + 1e-10)
            )
        else:
            spike_entropy = 0.0

        # Bits from spikes
        bits_processed = int(spikes.sum().item())

        # Entropy production
        Pi_q = dissipation + spike_entropy

        # Energy per spike (Landauer limit baseline)
        energy_per_bit = LANDAUER_LIMIT * 1e18  # Scale to reasonable units

        return ThermodynamicStats(
            Pi_q=Pi_q,
            vfe=0.0,  # Would need predictions
            energy_consumed=bits_processed * self.energy_budget.consumption_rate / 1000,
            efficiency=1.0 / (1 + Pi_q),
            dissipation=dissipation,
            correlation_entropy=spike_entropy,
            thermal_state=self._determine_thermal_state(Pi_q),
            bits_processed=bits_processed,
            energy_per_bit=energy_per_bit,
        )

    def _determine_thermal_state(self, Pi_q: float) -> ThermalState:
        """Determine thermal state from entropy production."""
        if Pi_q < self.max_entropy_threshold * 0.3:
            return ThermalState.COOL
        elif Pi_q < self.max_entropy_threshold * 0.6:
            return ThermalState.WARM
        elif Pi_q < self.max_entropy_threshold:
            return ThermalState.HOT
        else:
            return ThermalState.OVERHEATING

    def _convert_tfan_stats(self, tfan_stats: Any) -> ThermodynamicStats:
        """Convert TFAN stats to our format."""
        return ThermodynamicStats(
            Pi_q=getattr(tfan_stats, 'Pi_q', 0.0),
            vfe=getattr(tfan_stats, 'vfe', 0.0),
            energy_consumed=getattr(tfan_stats, 'energy', 0.0),
            efficiency=getattr(tfan_stats, 'efficiency', 1.0),
            dissipation=getattr(tfan_stats, 'dissipation', 0.0),
            correlation_entropy=getattr(tfan_stats, 'correlation_entropy', 0.0),
            thermal_state=ThermalState.COOL,
            bits_processed=getattr(tfan_stats, 'bits', 0),
            energy_per_bit=0.0,
        )

    def is_overheating(self) -> bool:
        """Check if system is overheating."""
        if not self._stats_history:
            return False
        return self._stats_history[-1].thermal_state == ThermalState.OVERHEATING

    def should_force_recovery(self) -> bool:
        """Check if recovery should be forced."""
        # Overheating
        if self.is_overheating():
            return True

        # Energy exhausted
        if self.energy_budget.percentage < 0.1:
            return True

        # Sustained high entropy
        if len(self._Pi_q_history) >= 5:
            recent_avg = np.mean(self._Pi_q_history[-5:])
            if recent_avg > self.max_entropy_threshold * 0.8:
                return True

        return False

    def recover(self, duration_seconds: float = 10.0):
        """Perform recovery (rest)."""
        self.energy_budget.recover(duration_seconds)
        # Reset thermal state by clearing recent history
        if len(self._Pi_q_history) > 10:
            self._Pi_q_history = self._Pi_q_history[-5:]

    def get_cost_report(self) -> Dict[str, Any]:
        """Get comprehensive cost report."""
        return {
            "current_thermal_state": (
                self._stats_history[-1].thermal_state.name
                if self._stats_history else "UNKNOWN"
            ),
            "average_Pi_q": self._avg_Pi_q,
            "average_vfe": self._avg_vfe,
            "total_bits_processed": self._total_bits,
            "energy_remaining_pct": self.energy_budget.percentage * 100,
            "should_recover": self.should_force_recovery(),
            "max_entropy_threshold": self.max_entropy_threshold,
            "recent_efficiency": (
                np.mean([s.efficiency for s in self._stats_history[-10:]])
                if self._stats_history else 1.0
            ),
        }

    def reset(self):
        """Reset thermodynamic state."""
        self.energy_budget.current_level = self.energy_budget.total_capacity
        self.energy_budget.last_update = time.time()
        self._stats_history.clear()
        self._Pi_q_history.clear()
        self._avg_Pi_q = 0.0
        self._avg_vfe = 0.0
        self._total_bits = 0


# Convenience factory
def create_thermodynamic_monitor(
    max_entropy: float = 2.0,
    energy_capacity: float = 100.0,
) -> ThermodynamicMonitor:
    """Create a ThermodynamicMonitor instance."""
    return ThermodynamicMonitor(
        max_entropy_threshold=max_entropy,
        energy_capacity=energy_capacity,
    )


__all__ = [
    "ThermodynamicMonitor",
    "ThermodynamicStats",
    "ThermalState",
    "EnergyBudget",
    "create_thermodynamic_monitor",
    "BOLTZMANN_K",
    "ROOM_TEMPERATURE",
    "KT",
    "LANDAUER_LIMIT",
]
