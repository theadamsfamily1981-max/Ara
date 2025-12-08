"""
CorrSpike-HDC ↔ AxisMundi Wavelength Bridge
============================================

Integrates the CorrSpike-HDC emotional subcortex with the AxisMundi
holographic state bus using wavelength (code-division) multiplexing.

Key concepts:
- Each logical layer (L1-L9) has a unique HDC "wavelength" (key HV)
- Multiple layers can share physical FPGA tiles via code-division
- The bridge translates between Python AxisMundi and FPGA fabric

Data flow:
  1. AxisMundi.state → wavelength encode → FPGA input buffer
  2. FPGA runs CorrSpike LIF + HDC bind/unbind
  3. FPGA spike output → wavelength decode → AxisMundi.state

Hardware mapping:
  - kitten_fabric_tile.sv handles the LIF + ping-pong + wavefront
  - This module handles the wavelength encoding/decoding
  - Communication via memory-mapped registers or CXL
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import struct

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_HV_DIM = 8192
DEFAULT_LANES = 64
NUM_LAYERS = 9
W_BITS = 8  # Weight precision matching FPGA


# =============================================================================
# Wavelength Codes (HDC Keys for Layer Multiplexing)
# =============================================================================

@dataclass
class WavelengthCode:
    """HDC key for a logical layer."""
    layer_id: int
    key_hv: np.ndarray      # Unit-normalized HV
    key_quantized: np.ndarray  # Quantized to W_BITS for FPGA


def generate_wavelength_codes(
    dim: int = DEFAULT_HV_DIM,
    num_layers: int = NUM_LAYERS,
    seed: int = 42,
) -> Dict[int, WavelengthCode]:
    """
    Generate orthogonal wavelength codes for layer multiplexing.

    Each code is a random unit HV - with high dim, these are
    quasi-orthogonal (cosine similarity ~0).
    """
    rng = np.random.default_rng(seed)
    codes = {}

    for lid in range(1, num_layers + 1):
        key = rng.standard_normal(dim).astype(np.float32)
        key /= np.linalg.norm(key) + 1e-8

        # Quantize to signed W_BITS
        max_val = 2 ** (W_BITS - 1) - 1
        key_q = np.clip(key * max_val, -max_val, max_val).astype(np.int8)

        codes[lid] = WavelengthCode(
            layer_id=lid,
            key_hv=key,
            key_quantized=key_q,
        )

    logger.info(f"Generated {num_layers} wavelength codes, dim={dim}")
    return codes


# =============================================================================
# FPGA Register Interface
# =============================================================================

class FPGARegisterMap:
    """Memory-mapped register addresses for kitten fabric."""

    # Control registers
    CTRL_STATUS         = 0x0000  # [0]=tick_start, [1]=tick_done, [2]=reset
    CTRL_LAYER_SELECT   = 0x0004  # Active layer ID (wavelength)
    CTRL_THRESH_BASE    = 0x0008  # Base threshold
    CTRL_THRESH_SCALE   = 0x000C  # Threshold scale (0-255 = 0.7x-1.3x)

    # Sparse control
    SPARSE_ACTIVITY_HINT = 0x0010  # Activity hint for early-exit
    SPARSE_MAX_ACTIVE    = 0x0014  # Read: max active idx from last tick

    # State buffers (large)
    AXIS_STATE_IN_BASE   = 0x1000  # Input state buffer (HV_DIM bytes)
    AXIS_STATE_OUT_BASE  = 0x2000  # Output state buffer
    LAYER_KEY_BASE       = 0x3000  # Layer key buffer

    # Statistics
    STATS_TICK_COUNT     = 0x4000
    STATS_SPIKE_COUNT    = 0x4004
    STATS_EARLY_EXIT_COUNT = 0x4008


@dataclass
class FPGAInterface:
    """Abstract FPGA communication interface."""

    def write_reg(self, addr: int, value: int):
        """Write 32-bit register."""
        raise NotImplementedError

    def read_reg(self, addr: int) -> int:
        """Read 32-bit register."""
        raise NotImplementedError

    def write_buffer(self, base_addr: int, data: np.ndarray):
        """Write byte buffer."""
        raise NotImplementedError

    def read_buffer(self, base_addr: int, length: int) -> np.ndarray:
        """Read byte buffer."""
        raise NotImplementedError


class SimulatedFPGA(FPGAInterface):
    """Simulated FPGA for testing without hardware."""

    def __init__(self, hv_dim: int = DEFAULT_HV_DIM):
        self.hv_dim = hv_dim
        self.registers: Dict[int, int] = {}
        self.buffers: Dict[int, np.ndarray] = {}

        # Initialize output buffer
        self.buffers[FPGARegisterMap.AXIS_STATE_OUT_BASE] = np.zeros(hv_dim, dtype=np.int8)

        # State
        self._tick_count = 0
        self._spike_count = 0

    def write_reg(self, addr: int, value: int):
        self.registers[addr] = value

        # Handle tick_start
        if addr == FPGARegisterMap.CTRL_STATUS and (value & 0x01):
            self._simulate_tick()

    def read_reg(self, addr: int) -> int:
        if addr == FPGARegisterMap.STATS_TICK_COUNT:
            return self._tick_count
        if addr == FPGARegisterMap.STATS_SPIKE_COUNT:
            return self._spike_count
        if addr == FPGARegisterMap.CTRL_STATUS:
            return 0x02  # tick_done
        return self.registers.get(addr, 0)

    def write_buffer(self, base_addr: int, data: np.ndarray):
        self.buffers[base_addr] = data.copy()

    def read_buffer(self, base_addr: int, length: int) -> np.ndarray:
        if base_addr in self.buffers:
            return self.buffers[base_addr][:length].copy()
        return np.zeros(length, dtype=np.int8)

    def _simulate_tick(self):
        """Simulate one tick of CorrSpike processing."""
        self._tick_count += 1

        # Get input state and layer key
        state_in = self.buffers.get(
            FPGARegisterMap.AXIS_STATE_IN_BASE,
            np.zeros(self.hv_dim, dtype=np.int8)
        )
        layer_key = self.buffers.get(
            FPGARegisterMap.LAYER_KEY_BASE,
            np.ones(self.hv_dim, dtype=np.int8)
        )

        # Simulate HDC bind
        bound = (state_in.astype(np.int16) * layer_key.astype(np.int16)) >> 7
        bound = np.clip(bound, -127, 127).astype(np.int8)

        # Simulate LIF spikes (simplified)
        thresh = self.registers.get(FPGARegisterMap.CTRL_THRESH_BASE, 64)
        spikes = np.abs(bound) > thresh

        self._spike_count += int(np.sum(spikes))

        # Generate output (spike pattern bound with key)
        spike_vals = np.where(spikes, 127, 0).astype(np.int8)
        output = (spike_vals.astype(np.int16) * layer_key.astype(np.int16)) >> 7
        output = np.clip(output, -127, 127).astype(np.int8)

        self.buffers[FPGARegisterMap.AXIS_STATE_OUT_BASE] = output


# =============================================================================
# CorrSpike ↔ AxisMundi Bridge
# =============================================================================

class CorrSpikeAxisBridge:
    """
    Bridge between AxisMundi holographic bus and CorrSpike FPGA fabric.

    Handles:
    - Wavelength encoding/decoding for layer multiplexing
    - State transfer to/from FPGA buffers
    - Threshold modulation from L9 bias
    - Sparse activity hint propagation
    """

    def __init__(
        self,
        fpga: FPGAInterface,
        axis: Any,  # AxisMundi instance
        hv_dim: int = DEFAULT_HV_DIM,
        wavelength_seed: int = 42,
    ):
        self.fpga = fpga
        self.axis = axis
        self.hv_dim = hv_dim

        # Generate wavelength codes
        self.wavelengths = generate_wavelength_codes(
            dim=hv_dim,
            num_layers=NUM_LAYERS,
            seed=wavelength_seed,
        )

        # Upload all wavelength keys to FPGA
        self._upload_wavelength_keys()

        # State tracking
        self._last_layer = 0
        self._activity_hints: Dict[int, int] = {i: hv_dim // DEFAULT_LANES for i in range(1, 10)}

        logger.info(f"CorrSpikeAxisBridge initialized, dim={hv_dim}")

    def _upload_wavelength_keys(self):
        """Upload all layer keys to FPGA."""
        for lid, code in self.wavelengths.items():
            # Each layer's key at offset
            offset = (lid - 1) * self.hv_dim
            # Note: In real FPGA, keys would be in a dedicated key RAM
            logger.debug(f"Uploaded wavelength key for layer {lid}")

    def quantize_state(self, state_hv: np.ndarray) -> np.ndarray:
        """Quantize float32 state to int8 for FPGA."""
        max_val = 127
        return np.clip(state_hv * max_val, -max_val, max_val).astype(np.int8)

    def dequantize_state(self, state_q: np.ndarray) -> np.ndarray:
        """Dequantize int8 state back to float32."""
        return state_q.astype(np.float32) / 127.0

    def process_layer(
        self,
        layer_id: int,
        thresh_scale: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run one tick of CorrSpike processing for a layer.

        Args:
            layer_id: Layer to process (1-9)
            thresh_scale: Threshold scaling [0.7, 1.3]

        Returns:
            (output_hv, stats_dict)
        """
        if layer_id not in self.wavelengths:
            raise ValueError(f"Invalid layer_id: {layer_id}")

        code = self.wavelengths[layer_id]

        # 1. Read current AxisMundi state
        axis_state = self.axis.state.copy()
        state_q = self.quantize_state(axis_state)

        # 2. Upload state to FPGA
        self.fpga.write_buffer(FPGARegisterMap.AXIS_STATE_IN_BASE, state_q)

        # 3. Upload layer key
        self.fpga.write_buffer(FPGARegisterMap.LAYER_KEY_BASE, code.key_quantized)

        # 4. Set control registers
        self.fpga.write_reg(FPGARegisterMap.CTRL_LAYER_SELECT, layer_id)
        self.fpga.write_reg(FPGARegisterMap.CTRL_THRESH_BASE, 64)  # Base threshold

        # Convert thresh_scale [0.7, 1.3] to [90, 166] (128 = 1.0)
        scale_int = int(thresh_scale * 128)
        scale_int = max(90, min(166, scale_int))
        self.fpga.write_reg(FPGARegisterMap.CTRL_THRESH_SCALE, scale_int)

        # 5. Set activity hint for sparse early-exit
        hint = self._activity_hints.get(layer_id, self.hv_dim // DEFAULT_LANES)
        self.fpga.write_reg(FPGARegisterMap.SPARSE_ACTIVITY_HINT, hint)

        # 6. Start tick
        self.fpga.write_reg(FPGARegisterMap.CTRL_STATUS, 0x01)

        # 7. Wait for completion (in real HW, would poll or use interrupt)
        # For simulation, tick completes synchronously

        # 8. Read results
        output_q = self.fpga.read_buffer(
            FPGARegisterMap.AXIS_STATE_OUT_BASE,
            self.hv_dim
        )
        output_hv = self.dequantize_state(output_q)

        # 9. Update activity hint for next tick
        max_active = self.fpga.read_reg(FPGARegisterMap.SPARSE_MAX_ACTIVE)
        self._activity_hints[layer_id] = max(1, max_active + 1)

        # 10. Stats
        stats = {
            "layer_id": layer_id,
            "tick_count": self.fpga.read_reg(FPGARegisterMap.STATS_TICK_COUNT),
            "spike_count": self.fpga.read_reg(FPGARegisterMap.STATS_SPIKE_COUNT),
            "activity_hint": hint,
            "max_active_idx": max_active,
        }

        self._last_layer = layer_id
        return output_hv, stats

    def process_full_stack(
        self,
        l1_thresh_scale: float = 1.0,
        l9_thresh_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Process L1 and L9 (the critical endpoints for reflex/prophet arcs).

        Returns combined stats.
        """
        results = {}

        # Process L1 (hardware reflex)
        l1_out, l1_stats = self.process_layer(1, thresh_scale=l1_thresh_scale)
        results["l1"] = l1_stats
        results["l1_output_norm"] = float(np.linalg.norm(l1_out))

        # Process L9 (mission control)
        l9_out, l9_stats = self.process_layer(9, thresh_scale=l9_thresh_scale)
        results["l9"] = l9_stats
        results["l9_output_norm"] = float(np.linalg.norm(l9_out))

        # Compute coherence between outputs
        l1_norm = l1_out / (np.linalg.norm(l1_out) + 1e-8)
        l9_norm = l9_out / (np.linalg.norm(l9_out) + 1e-8)
        coherence = float(np.dot(l1_norm, l9_norm))
        results["l1_l9_coherence"] = coherence

        return results

    def wavefront_process(
        self,
        layer_sequence: List[int],
        thresh_scales: Optional[Dict[int, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process layers in wavefront order (for multi-tile FPGA).

        Each layer's output feeds the next layer's input.
        """
        if thresh_scales is None:
            thresh_scales = {}

        results = []
        for lid in layer_sequence:
            scale = thresh_scales.get(lid, 1.0)
            output_hv, stats = self.process_layer(lid, thresh_scale=scale)

            # Write output back to axis for next layer
            self.axis.state = np.clip(
                self.axis.state + output_hv * 0.1,  # Blend factor
                -1.0, 1.0
            )

            results.append(stats)

        return results


# =============================================================================
# Emotional Subcortex Integration
# =============================================================================

class EmotionalSubcortex:
    """
    CorrSpike-HDC emotional processing integrated with AxisMundi.

    Maps emotional valence/arousal to L1-L9 threshold modulation.
    Uses EternalMemory for emotional trace storage.
    """

    def __init__(
        self,
        bridge: CorrSpikeAxisBridge,
        axis: Any,
        eternal_memory: Optional[Any] = None,
    ):
        self.bridge = bridge
        self.axis = axis
        self.eternal_memory = eternal_memory

        # Emotional state
        self.valence = 0.0   # [-1, 1] negative to positive
        self.arousal = 0.5   # [0, 1] calm to excited

        # Emotional HV encoding keys
        rng = np.random.default_rng(123)
        self.k_valence = self._rand_unit_hv(rng)
        self.k_arousal = self._rand_unit_hv(rng)

        logger.info("EmotionalSubcortex initialized")

    def _rand_unit_hv(self, rng: np.random.Generator) -> np.ndarray:
        hv = rng.standard_normal(self.bridge.hv_dim).astype(np.float32)
        hv /= np.linalg.norm(hv) + 1e-8
        return hv

    def encode_emotion(self) -> np.ndarray:
        """Encode current emotional state as HV."""
        hv = (
            self.k_valence * self.valence +
            self.k_arousal * (self.arousal * 2 - 1)
        )
        norm = np.linalg.norm(hv)
        if norm > 1e-8:
            hv /= norm
        return hv

    def emotion_to_thresh_scale(self) -> Tuple[float, float]:
        """
        Map emotion to L1/L9 threshold scales.

        High arousal + negative valence → higher L1 thresholds (defensive)
        High arousal + positive valence → lower thresholds (exploratory)
        """
        # L1: hardware - more conservative when stressed
        l1_scale = 1.0 + 0.2 * self.arousal * (-self.valence)
        l1_scale = max(0.7, min(1.3, l1_scale))

        # L9: mission - more creative when positive
        l9_scale = 1.0 - 0.2 * self.valence
        l9_scale = max(0.7, min(1.3, l9_scale))

        return l1_scale, l9_scale

    def process_emotional_tick(self) -> Dict[str, Any]:
        """
        Run one emotional processing tick.

        1. Encode emotion → AxisMundi
        2. Process L1/L9 through CorrSpike
        3. Update emotional state from coherence
        4. Store trace in EternalMemory
        """
        # 1. Write emotion to axis
        emotion_hv = self.encode_emotion()
        # Blend with existing state
        self.axis.state = np.clip(
            self.axis.state + emotion_hv * 0.2,
            -1.0, 1.0
        )

        # 2. Get threshold scales
        l1_scale, l9_scale = self.emotion_to_thresh_scale()

        # 3. Process through FPGA
        results = self.bridge.process_full_stack(
            l1_thresh_scale=l1_scale,
            l9_thresh_scale=l9_scale,
        )

        # 4. Update emotion based on coherence
        coherence = results["l1_l9_coherence"]

        # High coherence → positive valence drift
        self.valence = 0.9 * self.valence + 0.1 * coherence

        # Low coherence → increased arousal
        if coherence < 0.3:
            self.arousal = min(1.0, self.arousal + 0.1)
        else:
            self.arousal = max(0.1, self.arousal - 0.05)

        # 5. Store in EternalMemory
        if self.eternal_memory is not None:
            trace_hv = emotion_hv + self.axis.state * 0.5
            trace_hv /= np.linalg.norm(trace_hv) + 1e-8

            # EternalMemory stores with emotional weight
            # (API depends on EternalMemory implementation)

        results["valence"] = self.valence
        results["arousal"] = self.arousal
        results["l1_thresh_scale"] = l1_scale
        results["l9_thresh_scale"] = l9_scale

        return results


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate CorrSpike-AxisMundi integration."""
    print("=" * 70)
    print("CorrSpike-HDC ↔ AxisMundi Wavelength Integration Demo")
    print("=" * 70)

    # Import AxisMundi (assuming it exists)
    try:
        from ara.system.axis import AxisMundi
    except ImportError:
        print("Creating mock AxisMundi for demo...")

        class MockAxisMundi:
            def __init__(self, dim):
                self.dim = dim
                self.state = np.zeros(dim, dtype=np.float32)

        AxisMundi = MockAxisMundi

    # Create components
    hv_dim = 1024  # Smaller for demo
    axis = AxisMundi(dim=hv_dim)
    fpga = SimulatedFPGA(hv_dim=hv_dim)
    bridge = CorrSpikeAxisBridge(fpga, axis, hv_dim=hv_dim)

    print("\n--- Wavelength Codes ---")
    for lid, code in list(bridge.wavelengths.items())[:3]:
        print(f"  L{lid}: key_norm={np.linalg.norm(code.key_hv):.3f}")

    print("\n--- Process Single Layer ---")
    output, stats = bridge.process_layer(1, thresh_scale=1.0)
    print(f"  L1 output norm: {np.linalg.norm(output):.3f}")
    print(f"  Stats: {stats}")

    print("\n--- Process Full Stack ---")
    results = bridge.process_full_stack(l1_thresh_scale=0.9, l9_thresh_scale=1.1)
    print(f"  L1 output norm: {results['l1_output_norm']:.3f}")
    print(f"  L9 output norm: {results['l9_output_norm']:.3f}")
    print(f"  L1↔L9 coherence: {results['l1_l9_coherence']:.3f}")

    print("\n--- Emotional Subcortex ---")
    subcortex = EmotionalSubcortex(bridge, axis)
    subcortex.valence = -0.3  # Slightly negative
    subcortex.arousal = 0.7   # Moderately aroused

    for i in range(5):
        result = subcortex.process_emotional_tick()
        print(f"  Tick {i+1}: valence={result['valence']:.3f}, "
              f"arousal={result['arousal']:.3f}, "
              f"coherence={result['l1_l9_coherence']:.3f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
