"""
The Cathedral - Phase-Gated Multi-Channel Brain
=================================================

A multi-channel bit-serial brain that layers:
- Phase coding (SENSE/SYMBOL/META channels)
- Sparsity (population as importance code)
- Topology (path through network as interpretation)
- Noise (controlled stochasticity per channel)

Same wires, same clock, but three interleaved logical fabrics:
SNN (sensory), HDC (symbolic), and meta-control.

Architecture:
    Physical Lane → Phase Demux → [SENSE, SYMBOL, META] streams
                                       ↓
    SENSE  → SNN integrate-and-fire path (fast reactions)
    SYMBOL → HDC XOR-binding path (concept composition)
    META   → Control register updates (gating, policy)

This is the "Cathedral brain" - high-dimensional thought
flowing through phase-coded channels.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
import time
import hashlib

from .phase import (
    Phase, PhaseConfig, MacroFrame, ChannelStream,
    PhaseMultiplexer, PhaseCounter, LaneTopology,
    create_default_phase_config,
)


@dataclass
class NeuronState:
    """
    State of a single phase-gated neuron.

    Each neuron maintains separate state for each phase/channel:
    - SENSE: membrane potential (SNN mode)
    - SYMBOL: accumulated hypervector bits (HDC mode)
    - META: control registers (gating, thresholds)
    """
    neuron_id: int
    dim: int = 256  # Bits of state per channel

    # Per-channel state
    sense_potential: float = 0.0
    sense_threshold: float = 1.0
    sense_spike_count: int = 0

    symbol_accumulator: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.int32))
    symbol_contributions: int = 0

    meta_gates: int = 0xFF  # 8 gate bits, all open by default
    meta_mode: int = 0      # Operating mode

    # Phase counter for this neuron
    _phase_counter: PhaseCounter = field(default_factory=PhaseCounter)

    def __post_init__(self):
        if self.symbol_accumulator.shape[0] != self.dim:
            self.symbol_accumulator = np.zeros(self.dim, dtype=np.int32)

    def reset(self):
        """Reset neuron state."""
        self.sense_potential = 0.0
        self.sense_spike_count = 0
        self.symbol_accumulator = np.zeros(self.dim, dtype=np.int32)
        self.symbol_contributions = 0
        self._phase_counter.reset()

    def process_bit(self, bit: int) -> Optional[Dict[str, Any]]:
        """
        Process one incoming bit based on current phase.

        Returns event dict if something notable happened (spike, gate change).
        """
        phase = self._phase_counter.tick()
        event = None

        if phase == Phase.SENSE:
            # SNN integrate-and-fire
            self.sense_potential += bit * 0.25  # Weight
            self.sense_potential *= 0.95        # Leak

            if self.sense_potential >= self.sense_threshold:
                self.sense_spike_count += 1
                self.sense_potential = 0.0
                event = {"type": "spike", "neuron": self.neuron_id}

        elif phase == Phase.SYMBOL:
            # HDC accumulation (for later majority vote)
            bit_idx = self._phase_counter.frame % self.dim
            if bit:
                self.symbol_accumulator[bit_idx] += 1
            self.symbol_contributions += 1

        elif phase == Phase.META:
            # Control register update
            # Use bit as part of gate configuration
            bit_idx = self._phase_counter.frame % 8
            if bit:
                self.meta_gates |= (1 << bit_idx)
            else:
                self.meta_gates &= ~(1 << bit_idx)

        return event

    def get_symbol_hv(self) -> np.ndarray:
        """Get accumulated symbol hypervector via majority vote."""
        if self.symbol_contributions == 0:
            return np.zeros(self.dim, dtype=np.uint8)

        threshold = self.symbol_contributions / (2 * self.dim)
        return (self.symbol_accumulator > threshold).astype(np.uint8)

    def is_gate_open(self, gate_idx: int) -> bool:
        """Check if a specific gate is open."""
        return bool(self.meta_gates & (1 << gate_idx))


@dataclass
class PhaseGatedCluster:
    """
    A cluster of phase-gated neurons.

    Clusters represent different "corridors" of the Cathedral:
    - Sensorimotor corridor (SENSE-biased)
    - Conceptual corridor (SYMBOL-biased)
    - Control corridor (META-biased)
    """
    cluster_id: str
    bias: Phase  # Primary phase bias
    neurons: List[NeuronState] = field(default_factory=list)
    dim: int = 256

    # Cluster-level hypervector (bundled from neurons)
    _cluster_hv: Optional[np.ndarray] = None

    def __post_init__(self):
        if not self.neurons:
            # Create default neurons
            for i in range(16):
                self.neurons.append(NeuronState(
                    neuron_id=i,
                    dim=self.dim,
                ))

    def process_frame(self, frame: MacroFrame) -> List[Dict[str, Any]]:
        """Process a macro-frame across all neurons."""
        events = []

        # Get the bit for our biased phase
        primary_bit = frame.get_bit(self.bias)

        for neuron in self.neurons:
            # All neurons get the primary bit for this cluster's bias
            event = neuron.process_bit(primary_bit)
            if event:
                event["cluster"] = self.cluster_id
                events.append(event)

        return events

    def get_cluster_symbol_hv(self) -> np.ndarray:
        """Bundle symbol HVs from all neurons."""
        hvs = [n.get_symbol_hv() for n in self.neurons]
        if not hvs:
            return np.zeros(self.dim, dtype=np.uint8)

        total = np.sum(hvs, axis=0)
        threshold = len(hvs) / 2
        self._cluster_hv = (total > threshold).astype(np.uint8)
        return self._cluster_hv

    def get_spike_rate(self) -> float:
        """Get average spike rate across neurons."""
        total_spikes = sum(n.sense_spike_count for n in self.neurons)
        return total_spikes / len(self.neurons) if self.neurons else 0.0

    def reset(self):
        """Reset all neurons."""
        for neuron in self.neurons:
            neuron.reset()
        self._cluster_hv = None


@dataclass
class CathedralState:
    """
    Global state of the Cathedral brain.

    Tracks:
    - Which clusters are active
    - Current operating mode (Steward/Scientist/Architect)
    - Policy constraints
    - Global hypervector context
    """
    mode: str = "steward"  # steward, scientist, architect
    active_clusters: set = field(default_factory=set)
    policy_bits: int = 0xFF  # Safety constraints

    # Global context hypervectors
    context_hv: Optional[np.ndarray] = None
    risk_hv: Optional[np.ndarray] = None
    role_hv: Optional[np.ndarray] = None

    # Metrics
    total_spikes: int = 0
    total_frames: int = 0

    def update_from_meta(self, meta_bits: List[int]):
        """Update state from META channel bits."""
        if len(meta_bits) >= 8:
            # First 3 bits: mode
            mode_bits = meta_bits[0] * 4 + meta_bits[1] * 2 + meta_bits[2]
            if mode_bits == 0:
                self.mode = "steward"
            elif mode_bits == 1:
                self.mode = "scientist"
            elif mode_bits == 2:
                self.mode = "architect"

            # Bits 3-7: policy
            self.policy_bits = sum(b << i for i, b in enumerate(meta_bits[3:8]))


@dataclass
class Cathedral:
    """
    The Cathedral Brain - Phase-Gated Multi-Channel Architecture.

    Combines:
    - Phase multiplexing (SENSE/SYMBOL/META on same wires)
    - Cluster topology (sensorimotor vs conceptual corridors)
    - Population coding (importance via lane count)
    - Controlled noise (per-channel stochasticity)
    """
    config: PhaseConfig = field(default_factory=create_default_phase_config)
    topology: LaneTopology = field(default_factory=LaneTopology)
    dim: int = 256

    # Clusters organized by bias
    clusters: Dict[str, PhaseGatedCluster] = field(default_factory=dict)

    # Global state
    state: CathedralState = field(default_factory=CathedralState)

    # Multiplexer for stream handling
    mux: PhaseMultiplexer = field(default_factory=PhaseMultiplexer)

    # Event log
    _events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.clusters:
            # Create default clusters
            self.clusters["sensorimotor"] = PhaseGatedCluster(
                cluster_id="sensorimotor",
                bias=Phase.SENSE,
                dim=self.dim,
            )
            self.clusters["conceptual"] = PhaseGatedCluster(
                cluster_id="conceptual",
                bias=Phase.SYMBOL,
                dim=self.dim,
            )
            self.clusters["control"] = PhaseGatedCluster(
                cluster_id="control",
                bias=Phase.META,
                dim=self.dim,
            )
            self.clusters["background"] = PhaseGatedCluster(
                cluster_id="background",
                bias=Phase.SPARE,
                dim=self.dim,
            )

    def reset(self):
        """Reset all state."""
        for cluster in self.clusters.values():
            cluster.reset()
        self.state = CathedralState()
        self._events = []

    def stream_hypervectors(
        self,
        sense_hv: Optional[np.ndarray] = None,
        symbol_hv: Optional[np.ndarray] = None,
        meta_hv: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Stream hypervectors through the Cathedral.

        Each HV is assigned to its corresponding phase channel,
        then multiplexed and processed through clusters.
        """
        # Multiplex HVs into frames
        frames = self.mux.multiplex_hypervectors(sense_hv, symbol_hv, meta_hv)

        all_events = []
        for frame in frames:
            self.state.total_frames += 1

            # Process through each cluster
            for cluster in self.clusters.values():
                events = cluster.process_frame(frame)
                all_events.extend(events)
                self.state.total_spikes += len([e for e in events if e.get("type") == "spike"])

        self._events.extend(all_events)

        # Collect results
        return {
            "frames_processed": len(frames),
            "events": all_events,
            "spike_count": len([e for e in all_events if e.get("type") == "spike"]),
        }

    def query_symbol_state(self) -> Dict[str, np.ndarray]:
        """Get accumulated symbol HVs from each cluster."""
        return {
            cluster_id: cluster.get_cluster_symbol_hv()
            for cluster_id, cluster in self.clusters.items()
        }

    def get_spike_rates(self) -> Dict[str, float]:
        """Get spike rates per cluster."""
        return {
            cluster_id: cluster.get_spike_rate()
            for cluster_id, cluster in self.clusters.items()
        }

    def inject_sense_spikes(self, spike_train: List[int]) -> List[Dict[str, Any]]:
        """
        Inject a spike train into the SENSE channel.

        Creates frames with only SENSE bits set.
        """
        events = []
        for spike_bit in spike_train:
            frame = MacroFrame(frame_id=self.state.total_frames)
            frame.set_bit(Phase.SENSE, spike_bit)
            self.state.total_frames += 1

            # Process through sensorimotor cluster primarily
            cluster = self.clusters.get("sensorimotor")
            if cluster:
                cluster_events = cluster.process_frame(frame)
                events.extend(cluster_events)

        self._events.extend(events)
        return events

    def update_meta(self, meta_bits: List[int]):
        """Update META channel state."""
        self.state.update_from_meta(meta_bits)

    def similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute similarity between two hypervectors."""
        if hv1.shape != hv2.shape:
            return 0.0
        matches = np.sum(hv1 == hv2)
        return (2 * matches - len(hv1)) / len(hv1)

    def get_combined_state_hv(self) -> np.ndarray:
        """
        Get combined state as a single hypervector.

        Bundles symbol HVs from all clusters.
        """
        cluster_hvs = list(self.query_symbol_state().values())
        if not cluster_hvs:
            return np.zeros(self.dim, dtype=np.uint8)

        total = np.sum(cluster_hvs, axis=0)
        threshold = len(cluster_hvs) / 2
        return (total > threshold).astype(np.uint8)


@dataclass
class CathedralEncoder:
    """
    Encodes semantic content for the Cathedral.

    Creates hypervectors for:
    - Sensory events (GPU thermal spike, disk IO, etc.)
    - Symbolic context (project, mood, machine, risk level)
    - Meta control (mode, policy, authorization)
    """
    dim: int = 256
    _item_memory: Dict[str, np.ndarray] = field(default_factory=dict)

    def _get_base_hv(self, name: str) -> np.ndarray:
        """Get or create a base hypervector for a name."""
        if name not in self._item_memory:
            seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            self._item_memory[name] = rng.integers(0, 2, size=self.dim, dtype=np.uint8)
        return self._item_memory[name]

    def encode_sensory_event(
        self,
        event_type: str,
        source: str,
        severity: float,
    ) -> np.ndarray:
        """
        Encode a sensory event for SENSE channel.

        Example: GPU thermal spike from worker-1 at severity 0.8
        """
        type_hv = self._get_base_hv(f"sense:type:{event_type}")
        source_hv = self._get_base_hv(f"sense:source:{source}")

        # Bind type and source
        event_hv = np.bitwise_xor(type_hv, source_hv)

        # Modulate by severity (flip some bits based on severity)
        if severity > 0:
            flip_count = int(self.dim * (1 - severity) * 0.3)
            flip_indices = np.random.choice(self.dim, flip_count, replace=False)
            event_hv[flip_indices] = 1 - event_hv[flip_indices]

        return event_hv

    def encode_symbolic_context(
        self,
        project: Optional[str] = None,
        machine: Optional[str] = None,
        role: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode symbolic context for SYMBOL channel.

        Example: project=ara, machine=gpu-worker-1, role=worker, risk=high
        """
        hvs = []

        if project:
            hvs.append(self._get_base_hv(f"symbol:project:{project}"))
        if machine:
            hvs.append(self._get_base_hv(f"symbol:machine:{machine}"))
        if role:
            hvs.append(self._get_base_hv(f"symbol:role:{role}"))
        if risk_level:
            hvs.append(self._get_base_hv(f"symbol:risk:{risk_level}"))

        if not hvs:
            return np.zeros(self.dim, dtype=np.uint8)

        # Bundle all context elements
        total = np.sum(hvs, axis=0)
        threshold = len(hvs) / 2
        return (total > threshold).astype(np.uint8)

    def encode_meta_control(
        self,
        mode: str = "steward",
        auth_level: int = 0,
        safety_override: bool = False,
    ) -> np.ndarray:
        """
        Encode meta control state for META channel.

        Example: mode=scientist, auth_level=2, safety_override=False
        """
        mode_hv = self._get_base_hv(f"meta:mode:{mode}")
        auth_hv = self._get_base_hv(f"meta:auth:{auth_level}")

        # Bind mode and auth
        meta_hv = np.bitwise_xor(mode_hv, auth_hv)

        # Apply safety bit
        if safety_override:
            # Flip specific "safety" positions
            safety_positions = [0, 16, 32, 48, 64]
            for pos in safety_positions:
                if pos < self.dim:
                    meta_hv[pos] = 1 - meta_hv[pos]

        return meta_hv


def create_cathedral(dim: int = 256, config: Optional[PhaseConfig] = None) -> Cathedral:
    """Create a Cathedral brain with default configuration."""
    if config is None:
        config = create_default_phase_config()

    return Cathedral(
        config=config,
        dim=dim,
    )
