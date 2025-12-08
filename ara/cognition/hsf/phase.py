"""
Phase-Coded Channels - Multi-Channel Bit-Serial Brain
======================================================

Upgrade from bit[t] to bit[t, φ] where φ = phase index.

On a single physical lane, we interleave multiple logical channels:
    φ = 0 → SENSE   (raw spikes, sensory SNN)
    φ = 1 → SYMBOL  (HDC binding/bundling)
    φ = 2 → META    (control, gating, tags)
    φ = 3 → SPARE   (future expansion / debug)

Wire view:
    t:    0   1   2   3   4   5   6   7   ...
    φ:    0   1   2   3   0   1   2   3   ...
    bit:  S   Y   M   -   S   Y   M   -   ...

This is "4D in 3D slices" - folding extra dimensions into phase/time
without increasing pin count.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import IntEnum
import time


class Phase(IntEnum):
    """Phase indices for channel multiplexing."""
    SENSE = 0    # Raw spikes, sensory SNN, fast reactions
    SYMBOL = 1   # HDC binding/bundling, concepts, context
    META = 2     # Control, gating, tags, policy
    SPARE = 3    # Reserved for future / debug


@dataclass
class PhaseConfig:
    """Configuration for phase-coded channel system."""
    num_phases: int = 4          # P - macro-frame length
    clock_hz: int = 100_000_000  # System clock (100 MHz default)

    # Channel assignments
    sense_phase: int = 0
    symbol_phase: int = 1
    meta_phase: int = 2
    spare_phase: int = 3

    # Per-channel noise levels (0.0 = deterministic, 1.0 = max noise)
    sense_noise: float = 0.01    # Low noise for safety
    symbol_noise: float = 0.05   # Some exploration in HDC
    meta_noise: float = 0.0      # Strict control channel
    spare_noise: float = 0.1     # Can be noisy

    @property
    def macro_frame_cycles(self) -> int:
        """Cycles per macro-frame."""
        return self.num_phases

    @property
    def effective_channel_rate(self) -> float:
        """Effective rate per channel (Hz)."""
        return self.clock_hz / self.num_phases

    def get_noise_level(self, phase: int) -> float:
        """Get noise level for a phase."""
        if phase == self.sense_phase:
            return self.sense_noise
        elif phase == self.symbol_phase:
            return self.symbol_noise
        elif phase == self.meta_phase:
            return self.meta_noise
        else:
            return self.spare_noise


@dataclass
class MacroFrame:
    """
    A single macro-frame containing one bit per phase.

    Represents P cycles of a physical lane, containing
    interleaved bits from each logical channel.
    """
    frame_id: int
    bits: Dict[Phase, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        # Initialize all phases to 0 if not set
        for phase in Phase:
            if phase not in self.bits:
                self.bits[phase] = 0

    def set_bit(self, phase: Phase, value: int):
        """Set a single phase bit."""
        self.bits[phase] = value & 1

    def get_bit(self, phase: Phase) -> int:
        """Get a single phase bit."""
        return self.bits.get(phase, 0)

    def to_wire_bits(self) -> List[int]:
        """Serialize to wire format (phase 0, 1, 2, 3, ...)."""
        return [self.bits.get(Phase(i), 0) for i in range(len(Phase))]

    @classmethod
    def from_wire_bits(cls, frame_id: int, wire_bits: List[int]) -> "MacroFrame":
        """Deserialize from wire format."""
        frame = cls(frame_id=frame_id)
        for i, bit in enumerate(wire_bits):
            if i < len(Phase):
                frame.bits[Phase(i)] = bit & 1
        return frame


@dataclass
class ChannelStream:
    """
    A stream of bits for a single logical channel.

    Accumulates bits over many macro-frames for one phase.
    """
    phase: Phase
    bits: List[int] = field(default_factory=list)

    def append(self, bit: int):
        """Add a bit to the stream."""
        self.bits.append(bit & 1)

    def to_bytes(self) -> bytes:
        """Pack bits into bytes."""
        result = []
        for i in range(0, len(self.bits), 8):
            byte_bits = self.bits[i:i+8]
            # Pad with zeros if needed
            while len(byte_bits) < 8:
                byte_bits.append(0)
            byte_val = sum(b << j for j, b in enumerate(byte_bits))
            result.append(byte_val)
        return bytes(result)

    def to_hypervector(self, dim: int) -> np.ndarray:
        """Convert to hypervector (taking first dim bits)."""
        hv = np.zeros(dim, dtype=np.uint8)
        for i, bit in enumerate(self.bits[:dim]):
            hv[i] = bit
        return hv

    @classmethod
    def from_hypervector(cls, phase: Phase, hv: np.ndarray) -> "ChannelStream":
        """Create stream from hypervector."""
        stream = cls(phase=phase)
        stream.bits = [int(b) for b in hv]
        return stream


@dataclass
class PhaseMultiplexer:
    """
    Multiplexes multiple channel streams onto a single physical lane.

    Takes separate SENSE/SYMBOL/META streams and interleaves them
    into a sequence of macro-frames.
    """
    config: PhaseConfig = field(default_factory=PhaseConfig)

    # Internal state
    _frame_counter: int = 0
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def multiplex(self, streams: Dict[Phase, ChannelStream]) -> List[MacroFrame]:
        """
        Multiplex multiple channel streams into macro-frames.

        All streams should have the same length.
        """
        # Find max length
        max_len = max(len(s.bits) for s in streams.values()) if streams else 0

        frames = []
        for i in range(max_len):
            frame = MacroFrame(frame_id=self._frame_counter)
            self._frame_counter += 1

            for phase, stream in streams.items():
                if i < len(stream.bits):
                    bit = stream.bits[i]
                    # Apply noise based on phase
                    noise_level = self.config.get_noise_level(phase.value)
                    if noise_level > 0 and self._rng.random() < noise_level:
                        bit = 1 - bit  # Flip bit
                    frame.set_bit(phase, bit)

            frames.append(frame)

        return frames

    def demultiplex(self, frames: List[MacroFrame]) -> Dict[Phase, ChannelStream]:
        """
        Demultiplex macro-frames back into separate channel streams.
        """
        streams = {phase: ChannelStream(phase=phase) for phase in Phase}

        for frame in frames:
            for phase in Phase:
                streams[phase].append(frame.get_bit(phase))

        return streams

    def multiplex_hypervectors(
        self,
        sense_hv: Optional[np.ndarray] = None,
        symbol_hv: Optional[np.ndarray] = None,
        meta_hv: Optional[np.ndarray] = None,
    ) -> List[MacroFrame]:
        """
        Convenience method to multiplex hypervectors directly.
        """
        streams = {}

        if sense_hv is not None:
            streams[Phase.SENSE] = ChannelStream.from_hypervector(Phase.SENSE, sense_hv)
        if symbol_hv is not None:
            streams[Phase.SYMBOL] = ChannelStream.from_hypervector(Phase.SYMBOL, symbol_hv)
        if meta_hv is not None:
            streams[Phase.META] = ChannelStream.from_hypervector(Phase.META, meta_hv)

        return self.multiplex(streams)


@dataclass
class PhaseCounter:
    """
    Hardware-style phase counter for RTL mapping.

    Tracks current phase within macro-frame.
    Conceptual RTL:
        reg [$clog2(P)-1:0] phase;
        always @(posedge clk) begin
            if (rst) phase <= 0;
            else     phase <= (phase == P-1) ? 0 : phase + 1;
        end
    """
    num_phases: int = 4
    _phase: int = 0
    _cycle: int = 0

    def reset(self):
        """Reset counter."""
        self._phase = 0
        self._cycle = 0

    def tick(self) -> int:
        """Advance one clock cycle, return current phase."""
        current = self._phase
        self._phase = (self._phase + 1) % self.num_phases
        self._cycle += 1
        return current

    @property
    def phase(self) -> int:
        """Current phase."""
        return self._phase

    @property
    def cycle(self) -> int:
        """Total cycles elapsed."""
        return self._cycle

    @property
    def frame(self) -> int:
        """Current macro-frame number."""
        return self._cycle // self.num_phases

    def is_sense(self) -> bool:
        return self._phase == Phase.SENSE

    def is_symbol(self) -> bool:
        return self._phase == Phase.SYMBOL

    def is_meta(self) -> bool:
        return self._phase == Phase.META


@dataclass
class LaneTopology:
    """
    Lane assignment topology for the cathedral.

    Different lanes can be biased toward different channels:
    - Lanes 0-255: Heavy SENSE (sensorimotor corridor)
    - Lanes 256-511: Heavy SYMBOL (conceptual corridor)
    - Lanes 512+: Reserved for background/stream
    """
    total_lanes: int = 1024
    sense_lanes: Tuple[int, int] = (0, 256)      # [start, end)
    symbol_lanes: Tuple[int, int] = (256, 512)
    meta_lanes: Tuple[int, int] = (512, 768)
    background_lanes: Tuple[int, int] = (768, 1024)

    def get_lane_bias(self, lane_id: int) -> Phase:
        """Get the primary phase bias for a lane."""
        if self.sense_lanes[0] <= lane_id < self.sense_lanes[1]:
            return Phase.SENSE
        elif self.symbol_lanes[0] <= lane_id < self.symbol_lanes[1]:
            return Phase.SYMBOL
        elif self.meta_lanes[0] <= lane_id < self.meta_lanes[1]:
            return Phase.META
        else:
            return Phase.SPARE

    def get_lanes_for_phase(self, phase: Phase) -> Tuple[int, int]:
        """Get the lane range biased toward a phase."""
        if phase == Phase.SENSE:
            return self.sense_lanes
        elif phase == Phase.SYMBOL:
            return self.symbol_lanes
        elif phase == Phase.META:
            return self.meta_lanes
        else:
            return self.background_lanes

    def population_code(self, phase: Phase, importance: float) -> List[int]:
        """
        Generate lane indices based on importance.

        importance ≈ population count across lanes.
        Higher importance → more lanes active.
        """
        start, end = self.get_lanes_for_phase(phase)
        num_lanes = end - start
        active_count = max(1, int(num_lanes * importance))
        return list(range(start, start + active_count))


# Convenience functions

def create_default_phase_config() -> PhaseConfig:
    """Create default 4-phase configuration."""
    return PhaseConfig(
        num_phases=4,
        sense_phase=0,
        symbol_phase=1,
        meta_phase=2,
        spare_phase=3,
    )


def create_high_precision_config() -> PhaseConfig:
    """Create 8-phase config with finer granularity."""
    return PhaseConfig(
        num_phases=8,
        sense_noise=0.005,
        symbol_noise=0.02,
        meta_noise=0.0,
    )
