"""
HSF Counterfactual Dynamics
============================

The physics engine for "what-if" simulation.

Key insight: The HSF already captures how load → field dynamics work.
We can use that same machinery to simulate alternate configurations.

Given:
- Historical load traces (real jobs, traffic, events)
- A candidate ConfigScenario

We simulate:
- F₀(t) under C₀ (what actually happened)
- F₁(t) under C₁ (what would have happened with the change)

Then compare: stability, immune load, throughput, etc.

This is not guessing - it's replaying real days under alternate physics.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from enum import Enum, auto
import hashlib

from .lanes import ItemMemory
from .zones import Zone, ZoneState


class ChangeType(Enum):
    """Types of configuration changes."""
    ADD_NODE = auto()        # Add a new machine/lane
    REMOVE_NODE = auto()     # Remove a machine/lane
    MODIFY_NODE = auto()     # Change node capabilities
    ADD_POLICY = auto()      # Add reflex/immune rule
    REMOVE_POLICY = auto()   # Remove reflex/immune rule
    MODIFY_POLICY = auto()   # Tune policy thresholds
    REROUTE = auto()         # Change job/traffic routing
    PROMOTE = auto()         # Promote node role (intern→worker)
    DEMOTE = auto()          # Demote node role


@dataclass
class ConfigDelta:
    """
    A single atomic change to configuration.

    Changes are composable - a scenario may have multiple deltas.
    """
    change_type: ChangeType
    target: str              # What's being changed
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_hv_seed(self) -> str:
        """Generate consistent seed for HV encoding."""
        parts = [self.change_type.name, self.target]
        parts.extend(f"{k}={v}" for k, v in sorted(self.params.items()))
        return ":".join(parts)


@dataclass
class ConfigScenario:
    """
    A complete configuration scenario to simulate.

    Represents a possible future: "what if we made these changes?"
    """
    id: str
    name: str
    description: str
    deltas: List[ConfigDelta]

    # Cost estimates
    hardware_cost: float = 0.0    # $ for new hardware
    human_hours: float = 0.0      # Hours of human work
    complexity_score: float = 0.0  # 0-1, how complex/risky

    # Metadata
    author: str = "dreamforge"
    created_at: float = 0.0

    def total_cost(self, hourly_rate: float = 50.0) -> float:
        """Total cost including human time."""
        return self.hardware_cost + (self.human_hours * hourly_rate)


@dataclass
class ConfigEncoder:
    """
    Encodes configuration scenarios as hypervectors.

    Each scenario gets a unique HV based on its deltas.
    Similar scenarios → similar HVs (for clustering).
    """
    dim: int = 8192
    item_memory: ItemMemory = field(default_factory=lambda: ItemMemory())

    def __post_init__(self):
        self.item_memory = ItemMemory(dim=self.dim)

    def encode_delta(self, delta: ConfigDelta) -> np.ndarray:
        """Encode a single delta as HV."""
        # Get base HV for change type
        type_hv = self.item_memory[f"change:{delta.change_type.name}"]

        # Get target HV
        target_hv = self.item_memory[f"target:{delta.target}"]

        # Bind type with target
        bound = self.item_memory.bind(type_hv, target_hv)

        # Add parameter encoding if present
        if delta.params:
            param_hvs = []
            for k, v in delta.params.items():
                key_hv = self.item_memory[f"param:{k}"]
                # Quantize numeric values
                if isinstance(v, (int, float)):
                    level = int(min(31, max(0, v * 10)))  # Rough quantization
                    val_hv = self.item_memory.permute(key_hv, level)
                else:
                    val_hv = self.item_memory[f"val:{v}"]
                param_hvs.append(self.item_memory.bind(key_hv, val_hv))

            if param_hvs:
                param_bundle = self.item_memory.bundle(param_hvs)
                bound = self.item_memory.bind(bound, param_bundle)

        return bound

    def encode_scenario(self, scenario: ConfigScenario) -> np.ndarray:
        """Encode complete scenario as HV."""
        if not scenario.deltas:
            return self.item_memory[f"scenario:{scenario.id}"]

        delta_hvs = [self.encode_delta(d) for d in scenario.deltas]
        return self.item_memory.bundle(delta_hvs)

    def similarity(self, s1: ConfigScenario, s2: ConfigScenario) -> float:
        """Compute similarity between two scenarios."""
        hv1 = self.encode_scenario(s1)
        hv2 = self.encode_scenario(s2)
        return self.item_memory.similarity(hv1, hv2)


@dataclass
class LoadTrace:
    """
    A recorded trace of system load over time.

    Used to replay real workloads under alternate configs.
    """
    trace_id: str
    description: str
    duration_ticks: int
    samples: List[Dict[str, Dict[str, float]]]  # tick → lane → values

    # Metadata about this period
    had_anomalies: bool = False
    worst_zone: Zone = Zone.GOOD
    interesting_events: List[str] = field(default_factory=list)

    @classmethod
    def from_field_history(cls, trace_id: str, samples: List[Dict],
                           description: str = "") -> "LoadTrace":
        """Create trace from recorded field samples."""
        worst = Zone.GOOD
        anomalies = False
        for sample in samples:
            if 'zone' in sample and sample['zone'] > worst:
                worst = sample['zone']
                if worst >= Zone.WEIRD:
                    anomalies = True

        return cls(
            trace_id=trace_id,
            description=description,
            duration_ticks=len(samples),
            samples=[s.get('telemetry', {}) for s in samples],
            had_anomalies=anomalies,
            worst_zone=worst,
        )


@dataclass
class FieldDynamics:
    """
    Learned dynamics model: F(t+1) ≈ G(F(t), load(t), config)

    This is a simplified linear model for fast simulation.
    Real implementation would learn from historical data.
    """
    dim: int = 8192

    # Learned parameters (simplified)
    _decay_rate: float = 0.95
    _load_sensitivity: float = 0.3
    _config_effect: Dict[ChangeType, float] = field(default_factory=dict)

    def __post_init__(self):
        # Default effects for each change type
        self._config_effect = {
            ChangeType.ADD_NODE: -0.15,      # Reduces load pressure
            ChangeType.REMOVE_NODE: 0.2,      # Increases load pressure
            ChangeType.MODIFY_NODE: -0.05,    # Small improvement
            ChangeType.ADD_POLICY: -0.1,      # Better control
            ChangeType.REMOVE_POLICY: 0.1,    # Less control
            ChangeType.MODIFY_POLICY: -0.02,  # Fine tuning
            ChangeType.REROUTE: -0.08,        # Better distribution
            ChangeType.PROMOTE: -0.1,         # More capable
            ChangeType.DEMOTE: 0.05,          # Less capable
        }

    def estimate_config_effect(self, scenario: ConfigScenario) -> float:
        """
        Estimate the cumulative effect of a config change.

        Negative = improvement (less stress)
        Positive = degradation (more stress)
        """
        total_effect = 0.0
        for delta in scenario.deltas:
            base_effect = self._config_effect.get(delta.change_type, 0.0)

            # Scale by target importance (simplified heuristic)
            if 'gpu' in delta.target.lower():
                base_effect *= 1.5  # GPU changes have bigger impact
            elif 'network' in delta.target.lower():
                base_effect *= 1.3
            elif 'print' in delta.target.lower():
                base_effect *= 0.8  # Print farm less critical

            total_effect += base_effect

        return total_effect

    def simulate_step(self, current_field: np.ndarray,
                      load_sample: Dict[str, float],
                      config_effect: float,
                      item_memory: ItemMemory) -> np.ndarray:
        """
        Simulate one step of field evolution.

        Returns new field state.
        """
        # Decay current state
        decayed = current_field.astype(np.float32) * self._decay_rate

        # Add load influence
        # (In real implementation, this would use the lane encoders)
        load_magnitude = sum(load_sample.values()) / max(len(load_sample), 1)
        load_noise = np.random.randn(self.dim) * self._load_sensitivity * load_magnitude

        # Apply config effect (shifts the "basin of attraction")
        config_shift = config_effect * 0.1

        # Combine
        new_field = decayed + load_noise + config_shift

        # Binarize for HSF
        return (new_field > 0).astype(np.uint8)


@dataclass
class GhostReplay:
    """
    Replays a load trace under different configurations.

    The "ghost" field runs in parallel to reality, showing what
    would have happened under alternate physics.
    """
    dynamics: FieldDynamics
    item_memory: ItemMemory
    dim: int = 8192

    def __post_init__(self):
        if self.item_memory is None:
            self.item_memory = ItemMemory(dim=self.dim)
        if self.dynamics is None:
            self.dynamics = FieldDynamics(dim=self.dim)

    def replay(self, trace: LoadTrace, scenario: ConfigScenario,
               baseline_hv: Optional[np.ndarray] = None) -> "ReplayResult":
        """
        Replay a load trace under a configuration scenario.

        Returns metrics about how the field would have behaved.
        """
        config_effect = self.dynamics.estimate_config_effect(scenario)

        # Initialize field
        if baseline_hv is not None:
            field_state = baseline_hv.copy()
        else:
            field_state = np.random.randint(0, 2, self.dim, dtype=np.uint8)

        # Track metrics
        zone_history = []
        deviation_history = []
        critical_ticks = 0
        weird_ticks = 0

        baseline_mean = np.mean(baseline_hv) if baseline_hv is not None else 0.5

        for tick, sample in enumerate(trace.samples):
            # Flatten sample to single dict
            flat_sample = {}
            for lane, values in sample.items():
                for k, v in values.items():
                    flat_sample[f"{lane}:{k}"] = v

            # Simulate step
            field_state = self.dynamics.simulate_step(
                field_state, flat_sample, config_effect, self.item_memory
            )

            # Compute deviation from baseline
            if baseline_hv is not None:
                deviation = 1.0 - self.item_memory.similarity(field_state, baseline_hv)
            else:
                deviation = abs(np.mean(field_state) - 0.5) * 2

            deviation_history.append(deviation)

            # Estimate zone from deviation
            if deviation < 0.15:
                zone = Zone.GOOD
            elif deviation < 0.30:
                zone = Zone.WARM
            elif deviation < 0.45:
                zone = Zone.WEIRD
                weird_ticks += 1
            else:
                zone = Zone.CRITICAL
                critical_ticks += 1

            zone_history.append(zone)

        return ReplayResult(
            scenario_id=scenario.id,
            trace_id=trace.trace_id,
            duration_ticks=len(trace.samples),
            zone_history=zone_history,
            deviation_history=deviation_history,
            critical_ticks=critical_ticks,
            weird_ticks=weird_ticks,
            avg_deviation=np.mean(deviation_history),
            max_deviation=max(deviation_history) if deviation_history else 0.0,
        )


@dataclass
class ReplayResult:
    """Results from a ghost replay."""
    scenario_id: str
    trace_id: str
    duration_ticks: int
    zone_history: List[Zone]
    deviation_history: List[float]
    critical_ticks: int
    weird_ticks: int
    avg_deviation: float
    max_deviation: float

    @property
    def stability_score(self) -> float:
        """
        Stability score: 0 (always critical) to 1 (always good).
        """
        if self.duration_ticks == 0:
            return 1.0
        bad_ticks = self.critical_ticks + self.weird_ticks
        return 1.0 - (bad_ticks / self.duration_ticks)

    @property
    def worst_zone(self) -> Zone:
        """Worst zone reached during replay."""
        if not self.zone_history:
            return Zone.GOOD
        return max(self.zone_history)
