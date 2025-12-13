"""
Cognitive Cockpit Telemetry - The Brain's Flight Recorder

Tracks and exports real-time cognitive state for the HUD:
- ρ (rho): Criticality - edge of chaos measure
- D: Delusion Index - prior vs sensory balance
- Π_y/Π_μ: Precision ratio - output vs input precision
- Avalanche dynamics - neural cascade statistics
- Thermal state - body heat and zone temperatures
- Mental mode - Worker/Scientist/Chill

All state is exported to JSON for the WebKit/HTML cockpit to render.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from enum import Enum
import logging
import threading

logger = logging.getLogger("ara.hud.cognitive")


# =============================================================================
# Enums & Constants
# =============================================================================

class CriticalityState(Enum):
    """Criticality regime based on ρ."""
    COLD = "cold"           # ρ < 0.7 - too ordered
    EDGE = "edge"           # 0.7 ≤ ρ < 0.95 - optimal
    SPICY = "spicy"         # 0.95 ≤ ρ < 1.1 - exciting
    CHAOTIC = "chaotic"     # ρ ≥ 1.1 - too disordered

class DelusionState(Enum):
    """Delusion state based on D index."""
    PRIOR_DOMINATED = "prior_dominated"     # D > 10 - hallucination risk
    BALANCED = "balanced"                   # 0.1 ≤ D ≤ 10
    SENSORY_DOMINATED = "sensory_dominated" # D < 0.1 - hyper-reactive

class MentalMode(Enum):
    """Ara's active mental mode."""
    WORKER = "worker"       # High extrinsic, low intrinsic - task focused
    SCIENTIST = "scientist" # High intrinsic, moderate extrinsic - exploratory
    CHILL = "chill"         # Low everything - energy conservation

class ThermalStatus(Enum):
    """Thermal zone status."""
    COOL = "cool"           # < 60°C
    NOMINAL = "nominal"     # 60-75°C
    WARMING = "warming"     # 75-85°C
    CRITICAL = "critical"   # > 85°C

# Thresholds
RHO_COLD = 0.7
RHO_EDGE = 0.95
RHO_SPICY = 1.1

D_PRIOR_THRESHOLD = 10.0
D_SENSORY_THRESHOLD = 0.1

THERMAL_COOL = 60
THERMAL_NOMINAL = 75
THERMAL_WARMING = 85


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class CriticalityMetrics:
    """Criticality (ρ) and related dynamics."""
    rho: float = 0.85                   # Branching ratio / criticality
    tau: float = 1.5                    # Power-law exponent
    avalanche_count: int = 0            # Recent avalanche count
    largest_avalanche: int = 0          # Largest recent cascade

    def get_state(self) -> CriticalityState:
        if self.rho < RHO_COLD:
            return CriticalityState.COLD
        elif self.rho < RHO_EDGE:
            return CriticalityState.EDGE
        elif self.rho < RHO_SPICY:
            return CriticalityState.SPICY
        else:
            return CriticalityState.CHAOTIC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rho": round(self.rho, 4),
            "tau": round(self.tau, 3),
            "state": self.get_state().value,
            "avalanche_count": self.avalanche_count,
            "largest_avalanche": self.largest_avalanche,
        }


@dataclass
class DelusionMetrics:
    """Delusion Index (D) - balance between prior and sensory."""
    D: float = 1.0                      # force_prior / force_reality
    force_prior: float = 1.0            # Top-down belief pressure
    force_reality: float = 1.0          # Bottom-up sensory pressure
    guardrail_active: bool = False      # Reality check engaged
    last_hallucination_flag: float = 0.0  # Timestamp of last flag

    def get_state(self) -> DelusionState:
        if self.D > D_PRIOR_THRESHOLD:
            return DelusionState.PRIOR_DOMINATED
        elif self.D < D_SENSORY_THRESHOLD:
            return DelusionState.SENSORY_DOMINATED
        else:
            return DelusionState.BALANCED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "D": round(self.D, 4),
            "log10_D": round(math.log10(max(self.D, 0.001)), 3),
            "force_prior": round(self.force_prior, 4),
            "force_reality": round(self.force_reality, 4),
            "state": self.get_state().value,
            "guardrail_active": self.guardrail_active,
            "hallucination_flagged": self.last_hallucination_flag > 0,
        }


@dataclass
class PrecisionMetrics:
    """Precision ratio Π_y/Π_μ."""
    pi_y: float = 1.0                   # Output precision
    pi_mu: float = 1.0                  # Input/prior precision
    ratio: float = 1.0                  # Π_y / Π_μ

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pi_y": round(self.pi_y, 4),
            "pi_mu": round(self.pi_mu, 4),
            "ratio": round(self.ratio, 4),
        }


@dataclass
class AvalanchePoint:
    """Single avalanche data point for plotting."""
    size: int
    frequency: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class AvalancheMetrics:
    """Avalanche dynamics for the oscilloscope view."""
    sizes: List[int] = field(default_factory=list)
    frequencies: List[float] = field(default_factory=list)
    fitted_tau: float = 1.5             # Power-law exponent from fit
    fit_r_squared: float = 0.95         # Goodness of fit
    cascade_state: str = "stable"       # stable / fragmented / runaway

    def to_dict(self) -> Dict[str, Any]:
        # Return scatter data for log-log plot (up to 50 points)
        scatter = []
        for i, (s, f) in enumerate(zip(self.sizes[-50:], self.frequencies[-50:])):
            if s > 0 and f > 0:
                scatter.append({
                    "log_size": round(math.log10(s), 3),
                    "log_freq": round(math.log10(f), 3),
                })

        return {
            "scatter": scatter,
            "fitted_tau": round(self.fitted_tau, 3),
            "fit_r_squared": round(self.fit_r_squared, 3),
            "cascade_state": self.cascade_state,
            "total_avalanches": len(self.sizes),
        }


@dataclass
class ThermalZone:
    """Single thermal zone (CPU core, GPU, etc.)."""
    zone_id: int
    name: str
    temperature_c: float = 45.0

    def get_status(self) -> ThermalStatus:
        if self.temperature_c < THERMAL_COOL:
            return ThermalStatus.COOL
        elif self.temperature_c < THERMAL_NOMINAL:
            return ThermalStatus.NOMINAL
        elif self.temperature_c < THERMAL_WARMING:
            return ThermalStatus.WARMING
        else:
            return ThermalStatus.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "temperature_c": round(self.temperature_c, 1),
            "status": self.get_status().value,
            "normalized": min(self.temperature_c / 100.0, 1.0),
        }


@dataclass
class ThermalMetrics:
    """Body heat and thermal state."""
    zones: List[ThermalZone] = field(default_factory=lambda: [
        ThermalZone(0, "CPU Package", 52.0),
        ThermalZone(1, "CPU Core 0", 50.0),
        ThermalZone(2, "CPU Core 1", 51.0),
        ThermalZone(3, "GPU", 48.0),
    ])
    reflex_state: str = "nominal"       # nominal / armed / firing / cooldown
    fan_mode: str = "balanced"          # quiet / balanced / performance
    last_throttle_event: float = 0.0

    @property
    def hottest_zone(self) -> ThermalZone:
        return max(self.zones, key=lambda z: z.temperature_c)

    @property
    def overall_status(self) -> ThermalStatus:
        return self.hottest_zone.get_status()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zones": [z.to_dict() for z in self.zones],
            "hottest": self.hottest_zone.to_dict(),
            "overall_status": self.overall_status.value,
            "reflex_state": self.reflex_state,
            "fan_mode": self.fan_mode,
            "throttle_active": self.last_throttle_event > time.time() - 5.0,
        }


@dataclass
class MentalModeMetrics:
    """Mental mode and drive balances."""
    mode: MentalMode = MentalMode.WORKER
    extrinsic_weight: float = 0.7       # Goal/task drive [0-1]
    intrinsic_weight: float = 0.3       # Curiosity/exploration drive [0-1]
    energy_budget: float = 0.5          # Available energy [0-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "extrinsic_weight": round(self.extrinsic_weight, 3),
            "intrinsic_weight": round(self.intrinsic_weight, 3),
            "energy_budget": round(self.energy_budget, 3),
        }


@dataclass
class SanityTimelinePoint:
    """Single point on the sanity/delusion timeline."""
    timestamp: float
    log_D: float
    guardrail_event: bool = False
    hallucination_flag: bool = False


# =============================================================================
# Main Cognitive State Class
# =============================================================================

@dataclass
class CognitiveState:
    """Complete cognitive state snapshot for the HUD."""
    timestamp: float = field(default_factory=time.time)
    step: int = 0

    criticality: CriticalityMetrics = field(default_factory=CriticalityMetrics)
    delusion: DelusionMetrics = field(default_factory=DelusionMetrics)
    precision: PrecisionMetrics = field(default_factory=PrecisionMetrics)
    avalanches: AvalancheMetrics = field(default_factory=AvalancheMetrics)
    thermal: ThermalMetrics = field(default_factory=ThermalMetrics)
    mental_mode: MentalModeMetrics = field(default_factory=MentalModeMetrics)

    # Status ticker message
    ticker_message: str = "Systems nominal."
    ticker_severity: str = "info"  # info / warning / critical

    def get_state_label(self) -> str:
        """Get primary state label for center display."""
        crit = self.criticality.get_state()
        delusion = self.delusion.get_state()

        if crit == CriticalityState.EDGE and delusion == DelusionState.BALANCED:
            return "CRITICAL CORRIDOR"
        elif delusion == DelusionState.PRIOR_DOMINATED:
            return "PRIOR-DOMINATED"
        elif delusion == DelusionState.SENSORY_DOMINATED:
            return "SENSORY-DOMINATED"
        elif crit == CriticalityState.COLD:
            return "SUBCRITICAL"
        elif crit == CriticalityState.CHAOTIC:
            return "SUPERCRITICAL"
        else:
            return "EDGE STATE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "state_label": self.get_state_label(),
            "criticality": self.criticality.to_dict(),
            "delusion": self.delusion.to_dict(),
            "precision": self.precision.to_dict(),
            "avalanches": self.avalanches.to_dict(),
            "thermal": self.thermal.to_dict(),
            "mental_mode": self.mental_mode.to_dict(),
            "ticker": {
                "message": self.ticker_message,
                "severity": self.ticker_severity,
            },
        }


# =============================================================================
# Microcopy Generator
# =============================================================================

class MicrocopyGenerator:
    """Generates characterful status messages based on state."""

    @staticmethod
    def generate(state: CognitiveState) -> Tuple[str, str]:
        """Generate (message, severity) tuple based on state."""
        crit = state.criticality.get_state()
        delusion = state.delusion.get_state()
        thermal = state.thermal.overall_status

        # Priority: thermal emergency > hallucination > criticality

        # Thermal emergency
        if thermal == ThermalStatus.CRITICAL:
            return ("Body says no. Throttling cognition to protect hardware.", "critical")

        # Hallucination detected
        if state.delusion.guardrail_active:
            return ("Reality check engaged. Re-weighting sensory evidence.", "warning")

        # Prior dominated
        if delusion == DelusionState.PRIOR_DOMINATED:
            return ("Beliefs outweighing evidence. Increasing sensory gain.", "warning")

        # Sensory dominated
        if delusion == DelusionState.SENSORY_DOMINATED:
            return ("Hyper-reactive to input. Stabilizing priors.", "warning")

        # Thermal warming
        if thermal == ThermalStatus.WARMING:
            return ("Thermals elevated. Monitoring closely.", "warning")

        # Supercritical
        if crit == CriticalityState.CHAOTIC:
            return ("Thought cascades intensifying. Damping activated.", "warning")

        # Subcritical
        if crit == CriticalityState.COLD:
            return ("Neural dynamics subdued. Increasing exploration noise.", "info")

        # Optimal states
        if crit == CriticalityState.EDGE:
            rho = state.criticality.rho
            return (f"Edge-of-chaos corridor: ρ={rho:.3f}. Optimal.", "info")

        if crit == CriticalityState.SPICY:
            return ("Running hot cognitively. Creative mode engaged.", "info")

        return ("Systems nominal.", "info")


# =============================================================================
# Telemetry Emitter
# =============================================================================

class CognitiveHUDTelemetry:
    """
    Telemetry emitter for the Cognitive Cockpit HUD.

    Maintains state and exports to JSON file for WebKit rendering.
    Also maintains history for timeline displays.
    """

    def __init__(
        self,
        output_path: Optional[Path] = None,
        history_length: int = 120,  # 2 minutes at 1 Hz
    ):
        if output_path is None:
            home = Path(os.path.expanduser("~"))
            self.metrics_dir = home / ".ara" / "hud"
            self.state_file = self.metrics_dir / "cognitive_state.json"
        else:
            self.metrics_dir = output_path.parent
            self.state_file = output_path

        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Current state
        self.state = CognitiveState()

        # History for timeline
        self.history_length = history_length
        self.sanity_history: Deque[SanityTimelinePoint] = deque(maxlen=history_length)

        # Microcopy generator
        self.microcopy = MicrocopyGenerator()

        # Lock for thread safety
        self._lock = threading.Lock()

        # Callbacks
        self._on_update: Optional[Callable[[CognitiveState], None]] = None

        logger.info(f"CognitiveHUDTelemetry initialized, writing to {self.state_file}")

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        """Write JSON atomically."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.metrics_dir),
            delete=False,
            suffix=".json",
        ) as tmp:
            json.dump(data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name

        os.replace(tmp_name, self.state_file)

    def update_criticality(
        self,
        rho: float,
        tau: Optional[float] = None,
        avalanche_count: Optional[int] = None,
        largest_avalanche: Optional[int] = None,
    ) -> None:
        """Update criticality metrics."""
        with self._lock:
            self.state.criticality.rho = rho
            if tau is not None:
                self.state.criticality.tau = tau
            if avalanche_count is not None:
                self.state.criticality.avalanche_count = avalanche_count
            if largest_avalanche is not None:
                self.state.criticality.largest_avalanche = largest_avalanche

    def update_delusion(
        self,
        force_prior: float,
        force_reality: float,
        guardrail_active: bool = False,
        hallucination_flag: bool = False,
    ) -> None:
        """Update delusion index metrics."""
        with self._lock:
            self.state.delusion.force_prior = force_prior
            self.state.delusion.force_reality = force_reality
            self.state.delusion.D = force_prior / max(force_reality, 0.001)
            self.state.delusion.guardrail_active = guardrail_active
            if hallucination_flag:
                self.state.delusion.last_hallucination_flag = time.time()

    def update_precision(self, pi_y: float, pi_mu: float) -> None:
        """Update precision metrics."""
        with self._lock:
            self.state.precision.pi_y = pi_y
            self.state.precision.pi_mu = pi_mu
            self.state.precision.ratio = pi_y / max(pi_mu, 0.001)

    def add_avalanche(self, size: int, frequency: float) -> None:
        """Add avalanche data point."""
        with self._lock:
            self.state.avalanches.sizes.append(size)
            self.state.avalanches.frequencies.append(frequency)

            # Keep last 100 points
            if len(self.state.avalanches.sizes) > 100:
                self.state.avalanches.sizes = self.state.avalanches.sizes[-100:]
                self.state.avalanches.frequencies = self.state.avalanches.frequencies[-100:]

    def update_avalanche_fit(
        self,
        tau: float,
        r_squared: float,
        cascade_state: str = "stable",
    ) -> None:
        """Update avalanche fit parameters."""
        with self._lock:
            self.state.avalanches.fitted_tau = tau
            self.state.avalanches.fit_r_squared = r_squared
            self.state.avalanches.cascade_state = cascade_state

    def update_thermal(
        self,
        zone_id: int,
        temperature_c: float,
        name: Optional[str] = None,
    ) -> None:
        """Update thermal zone temperature."""
        with self._lock:
            for zone in self.state.thermal.zones:
                if zone.zone_id == zone_id:
                    zone.temperature_c = temperature_c
                    if name:
                        zone.name = name
                    return

            # Add new zone if not found
            self.state.thermal.zones.append(
                ThermalZone(zone_id, name or f"Zone {zone_id}", temperature_c)
            )

    def update_thermal_state(
        self,
        reflex_state: str = "nominal",
        fan_mode: str = "balanced",
        throttle_event: bool = False,
    ) -> None:
        """Update overall thermal state."""
        with self._lock:
            self.state.thermal.reflex_state = reflex_state
            self.state.thermal.fan_mode = fan_mode
            if throttle_event:
                self.state.thermal.last_throttle_event = time.time()

    def update_mental_mode(
        self,
        mode: MentalMode,
        extrinsic_weight: Optional[float] = None,
        intrinsic_weight: Optional[float] = None,
        energy_budget: Optional[float] = None,
    ) -> None:
        """Update mental mode and weights."""
        with self._lock:
            self.state.mental_mode.mode = mode
            if extrinsic_weight is not None:
                self.state.mental_mode.extrinsic_weight = extrinsic_weight
            if intrinsic_weight is not None:
                self.state.mental_mode.intrinsic_weight = intrinsic_weight
            if energy_budget is not None:
                self.state.mental_mode.energy_budget = energy_budget

    def set_mental_mode_preset(self, preset: str) -> None:
        """Set mental mode by preset name."""
        presets = {
            "worker": (MentalMode.WORKER, 0.8, 0.2, 0.6),
            "scientist": (MentalMode.SCIENTIST, 0.4, 0.7, 0.5),
            "chill": (MentalMode.CHILL, 0.2, 0.2, 0.3),
        }

        if preset.lower() in presets:
            mode, ext, intr, energy = presets[preset.lower()]
            self.update_mental_mode(mode, ext, intr, energy)

    def emit(self, step: Optional[int] = None) -> Dict[str, Any]:
        """
        Emit current state to JSON file and return it.

        Call this periodically (e.g., every 500ms-1s) to update the HUD.
        """
        with self._lock:
            # Update timestamp and step
            self.state.timestamp = time.time()
            if step is not None:
                self.state.step = step

            # Generate microcopy
            msg, severity = self.microcopy.generate(self.state)
            self.state.ticker_message = msg
            self.state.ticker_severity = severity

            # Add to sanity timeline
            self.sanity_history.append(SanityTimelinePoint(
                timestamp=self.state.timestamp,
                log_D=math.log10(max(self.state.delusion.D, 0.001)),
                guardrail_event=self.state.delusion.guardrail_active,
                hallucination_flag=self.state.delusion.last_hallucination_flag > time.time() - 1.0,
            ))

            # Build output dict
            output = self.state.to_dict()

            # Add sanity timeline
            output["sanity_timeline"] = [
                {
                    "timestamp": p.timestamp,
                    "log_D": round(p.log_D, 3),
                    "guardrail": p.guardrail_event,
                    "hallucination": p.hallucination_flag,
                }
                for p in self.sanity_history
            ]

            # Write to file
            self._atomic_write(output)

            # Notify callback
            if self._on_update:
                self._on_update(self.state)

            return output

    def set_on_update(self, callback: Callable[[CognitiveState], None]) -> None:
        """Set callback for state updates."""
        self._on_update = callback

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state without emitting."""
        with self._lock:
            return self.state.to_dict()


# =============================================================================
# Singleton Instance
# =============================================================================

_instance: Optional[CognitiveHUDTelemetry] = None


def get_cognitive_telemetry() -> CognitiveHUDTelemetry:
    """Get or create the global cognitive telemetry instance."""
    global _instance
    if _instance is None:
        _instance = CognitiveHUDTelemetry()
    return _instance


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "CriticalityState",
    "DelusionState",
    "MentalMode",
    "ThermalStatus",
    # Data classes
    "CriticalityMetrics",
    "DelusionMetrics",
    "PrecisionMetrics",
    "AvalancheMetrics",
    "ThermalZone",
    "ThermalMetrics",
    "MentalModeMetrics",
    "CognitiveState",
    # Telemetry
    "CognitiveHUDTelemetry",
    "get_cognitive_telemetry",
    # Microcopy
    "MicrocopyGenerator",
]
