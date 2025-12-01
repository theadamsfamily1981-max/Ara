"""
CXL/FPGA Real-Time Antifragile Control Plane

Software scaffold for hyper-low-latency control plane that can be
offloaded to CXL-attached FPGA for real-time execution.

Target: p95 < 200ms for PGU check, p95 < 50ms for L1 Homeostatic loop

Architecture:
1. Control Loop Bypassing PCIe - LIF rules, L1 Homeostat, TurboCache on FPGA
2. CXL Memory Paging - KV cache and SNN weights in FPGA local memory
3. Real-Time PGU - Cache check on FPGA fabric

This module provides:
- Software emulation of the control plane
- HLS-exportable function signatures
- Latency monitoring for hardware offload decisions
- CXL memory management abstractions

Usage:
    from ara.cxl_control import ControlPlane, FPGAEmulator

    plane = ControlPlane()
    result = plane.fast_control_step(pad_state, current_temp)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
from datetime import datetime
from enum import Enum
import logging
import time
import math

# Add paths
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("ara.cxl_control")

# Try numpy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class ControlPlaneMode(str, Enum):
    """Control plane execution mode."""
    SOFTWARE = "software"       # Pure Python emulation
    EMULATED_HLS = "emulated"  # HLS-compatible emulation
    FPGA_OFFLOAD = "fpga"      # Real FPGA (future)
    CXL_DIRECT = "cxl"         # CXL memory-mapped (future)


@dataclass
class LatencyMetrics:
    """Latency tracking for control plane operations."""
    operation: str
    start_ns: int
    end_ns: int
    target_us: float  # Target latency in microseconds

    @property
    def latency_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1000

    @property
    def within_target(self) -> bool:
        return self.latency_us <= self.target_us

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "latency_us": self.latency_us,
            "target_us": self.target_us,
            "within_target": self.within_target,
        }


@dataclass
class ControlState:
    """
    State for real-time control loop.

    This structure is HLS-exportable and designed for FPGA implementation.
    All fields are fixed-width for hardware synthesis.
    """
    # Input PAD (fixed-point Q15 in hardware)
    valence: float = 0.0       # [-1, 1]
    arousal: float = 0.5       # [0, 1]
    dominance: float = 0.5     # [0, 1]

    # L1 Homeostatic state
    goal_valence: float = 0.0
    goal_arousal: float = 0.5
    error_integral: float = 0.0

    # L3 Metacontrol outputs
    temperature_mult: float = 1.0
    memory_mult: float = 1.0
    attention_gain: float = 1.0

    # PGU cache state
    cache_hit: bool = False
    cache_key_hash: int = 0

    # Timing
    last_update_ns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "temperature_mult": self.temperature_mult,
            "memory_mult": self.memory_mult,
            "attention_gain": self.attention_gain,
            "cache_hit": self.cache_hit,
        }


def hls_lif_step(
    v: float,
    current: float,
    tau: float,
    v_th: float,
    dt: float = 1.0,
) -> Tuple[float, int]:
    """
    HLS-exportable LIF neuron step.

    This function signature is compatible with Vitis HLS synthesis.

    Args:
        v: Current membrane potential
        current: Input current
        tau: Time constant
        v_th: Threshold
        dt: Time step (ms)

    Returns:
        (new_v, spike) where spike is 0 or 1
    """
    # LIF dynamics: dv/dt = -(v - v_rest)/tau + I
    dv = (-v / tau + current) * dt
    new_v = v + dv

    # Spike check
    spike = 1 if new_v >= v_th else 0
    if spike:
        new_v = 0.0  # Reset

    return new_v, spike


def hls_l1_homeostat(
    current_valence: float,
    current_arousal: float,
    goal_valence: float,
    goal_arousal: float,
    error_integral: float,
    kp: float = 0.1,
    ki: float = 0.01,
) -> Tuple[float, float, float]:
    """
    HLS-exportable L1 Homeostatic controller (PI control).

    Args:
        current_*: Current PAD values
        goal_*: Homeostatic goal
        error_integral: Accumulated error
        kp: Proportional gain
        ki: Integral gain

    Returns:
        (valence_correction, arousal_correction, new_error_integral)
    """
    # Compute errors
    valence_error = goal_valence - current_valence
    arousal_error = goal_arousal - current_arousal

    # PI control
    valence_correction = kp * valence_error + ki * error_integral
    arousal_correction = kp * arousal_error + ki * error_integral

    # Update integral (with anti-windup)
    new_integral = error_integral + (valence_error + arousal_error)
    new_integral = max(-10.0, min(10.0, new_integral))

    return valence_correction, arousal_correction, new_integral


def hls_l3_metacontrol(
    valence: float,
    arousal: float,
    dominance: float,
) -> Tuple[float, float, float]:
    """
    HLS-exportable L3 Metacontrol computation.

    Control Law:
    - Arousal → Temperature [0.8, 1.3]
    - Valence → Memory P [0.7, 1.2]
    - Dominance → Attention gain [0.8, 1.2]

    Returns:
        (temperature_mult, memory_mult, attention_gain)
    """
    # Temperature: higher arousal → higher temp (more exploration)
    temp_mult = 0.8 + 0.5 * arousal

    # Memory: lower valence → more conservative
    mem_mult = 0.95 + 0.25 * (valence + 1) / 2

    # Attention: higher dominance → higher gain
    attn_gain = 0.8 + 0.4 * dominance

    return temp_mult, mem_mult, attn_gain


def hls_pgu_cache_lookup(
    key_hash: int,
    cache_table: List[Tuple[int, Any]],
    table_size: int = 1024,
) -> Tuple[bool, Any]:
    """
    HLS-exportable PGU cache lookup (hash table).

    Uses simple open addressing for FPGA implementation.

    Args:
        key_hash: Hash of the query key
        cache_table: List of (hash, value) pairs
        table_size: Size of hash table

    Returns:
        (hit, cached_value)
    """
    idx = key_hash % table_size

    # Linear probe (limited for HLS)
    for probe in range(4):  # Max 4 probes
        check_idx = (idx + probe) % table_size
        if check_idx < len(cache_table):
            stored_hash, value = cache_table[check_idx]
            if stored_hash == key_hash:
                return True, value

    return False, None


class CXLMemoryManager:
    """
    CXL Memory Management abstraction.

    Manages memory movement between host and FPGA local memory.
    In software mode, this is a simple cache simulation.
    """

    def __init__(
        self,
        local_size_mb: int = 64,
        page_size_kb: int = 4,
    ):
        self.local_size = local_size_mb * 1024 * 1024
        self.page_size = page_size_kb * 1024

        # Simulated local memory
        self.local_pages: Dict[int, bytes] = {}
        self.page_table: Dict[int, int] = {}  # virtual → local

        # Statistics
        self.page_faults = 0
        self.page_hits = 0

        logger.info(f"CXL Memory Manager: {local_size_mb}MB local, {page_size_kb}KB pages")

    def access(self, virtual_addr: int) -> bool:
        """
        Access a virtual address, returning hit/miss.

        In real CXL, this would trigger memory-mapped access.
        """
        page_num = virtual_addr // self.page_size

        if page_num in self.page_table:
            self.page_hits += 1
            return True
        else:
            self.page_faults += 1
            # Simulate page-in
            self.page_table[page_num] = len(self.page_table)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get memory access statistics."""
        total = self.page_hits + self.page_faults
        return {
            "page_hits": self.page_hits,
            "page_faults": self.page_faults,
            "hit_rate": self.page_hits / total if total > 0 else 0,
            "pages_resident": len(self.page_table),
        }


class FPGAEmulator:
    """
    FPGA Control Plane Emulator.

    Runs HLS-compatible functions in software with accurate timing.
    """

    def __init__(
        self,
        target_latency_us: float = 100.0,
    ):
        self.target_latency_us = target_latency_us
        self.latency_history: List[LatencyMetrics] = []

        # LIF state (would be registers in FPGA)
        self.lif_v: List[float] = [0.0] * 64
        self.lif_tau: List[float] = [10.0] * 64
        self.lif_th: List[float] = [1.0] * 64

        # PGU cache (would be BRAM in FPGA)
        self.pgu_cache: List[Tuple[int, Any]] = []
        self.cache_size = 1024

        logger.info(f"FPGA Emulator initialized, target latency: {target_latency_us}us")

    def run_control_cycle(
        self,
        state: ControlState,
        input_current: float = 0.0,
    ) -> ControlState:
        """
        Run one control cycle (emulating FPGA execution).

        This combines L1 Homeostat, L3 Metacontrol, and LIF update.
        """
        start_ns = time.time_ns()

        # L1 Homeostatic correction
        val_corr, aro_corr, new_integral = hls_l1_homeostat(
            state.valence, state.arousal,
            state.goal_valence, state.goal_arousal,
            state.error_integral,
        )

        # Apply correction
        state.valence += val_corr * 0.1
        state.arousal += aro_corr * 0.1
        state.valence = max(-1, min(1, state.valence))
        state.arousal = max(0, min(1, state.arousal))
        state.error_integral = new_integral

        # L3 Metacontrol
        temp, mem, attn = hls_l3_metacontrol(
            state.valence, state.arousal, state.dominance
        )
        state.temperature_mult = temp
        state.memory_mult = mem
        state.attention_gain = attn

        # LIF update (first neuron as example)
        if self.lif_v:
            self.lif_v[0], spike = hls_lif_step(
                self.lif_v[0], input_current,
                self.lif_tau[0], self.lif_th[0]
            )

        # PGU cache check
        key_hash = hash((state.valence, state.arousal)) & 0xFFFFFFFF
        hit, _ = hls_pgu_cache_lookup(key_hash, self.pgu_cache, self.cache_size)
        state.cache_hit = hit
        state.cache_key_hash = key_hash

        end_ns = time.time_ns()
        state.last_update_ns = end_ns

        # Record latency
        metrics = LatencyMetrics(
            operation="control_cycle",
            start_ns=start_ns,
            end_ns=end_ns,
            target_us=self.target_latency_us,
        )
        self.latency_history.append(metrics)

        return state

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latency_history:
            return {"p50_us": 0, "p95_us": 0, "p99_us": 0}

        latencies = [m.latency_us for m in self.latency_history[-1000:]]

        if NUMPY_AVAILABLE:
            return {
                "p50_us": float(np.percentile(latencies, 50)),
                "p95_us": float(np.percentile(latencies, 95)),
                "p99_us": float(np.percentile(latencies, 99)),
                "mean_us": float(np.mean(latencies)),
                "within_target_pct": sum(1 for l in latencies if l <= self.target_latency_us) / len(latencies) * 100,
            }
        else:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            return {
                "p50_us": sorted_lat[n // 2],
                "p95_us": sorted_lat[int(n * 0.95)],
                "p99_us": sorted_lat[int(n * 0.99)],
                "mean_us": sum(latencies) / n,
                "within_target_pct": sum(1 for l in latencies if l <= self.target_latency_us) / n * 100,
            }


class ControlPlane:
    """
    Real-Time Antifragile Control Plane.

    Coordinates L1 Homeostatic loop, L3 Metacontrol, and PGU cache
    with target p95 < 200ms latency.

    In production, this would offload to CXL-attached FPGA.
    """

    def __init__(
        self,
        mode: ControlPlaneMode = ControlPlaneMode.EMULATED_HLS,
        target_latency_us: float = 200.0,
    ):
        self.mode = mode
        self.target_latency = target_latency_us

        # Components
        self.emulator = FPGAEmulator(target_latency_us)
        self.memory_mgr = CXLMemoryManager()

        # State
        self.state = ControlState()

        # Metrics for Prometheus
        self.metrics_history: List[Dict] = []

        logger.info(f"ControlPlane initialized in {mode.value} mode")

    def fast_control_step(
        self,
        valence: float,
        arousal: float,
        dominance: float = 0.5,
        input_current: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute one fast control step.

        This is the main entry point for real-time control.
        Target: p95 < 200us

        Args:
            valence: Current PAD valence [-1, 1]
            arousal: Current PAD arousal [0, 1]
            dominance: Current PAD dominance [0, 1]
            input_current: External input current for LIF

        Returns:
            Control outputs (temperature_mult, memory_mult, etc.)
        """
        # Update state with inputs
        self.state.valence = valence
        self.state.arousal = arousal
        self.state.dominance = dominance

        # Run control cycle
        self.state = self.emulator.run_control_cycle(self.state, input_current)

        # Record metrics
        result = {
            "temperature_mult": self.state.temperature_mult,
            "memory_mult": self.state.memory_mult,
            "attention_gain": self.state.attention_gain,
            "cache_hit": self.state.cache_hit,
            "latency_us": self.emulator.latency_history[-1].latency_us if self.emulator.latency_history else 0,
        }

        self.metrics_history.append(result)

        return result

    def set_homeostatic_goal(self, valence: float, arousal: float):
        """Set L1 homeostatic goal state."""
        self.state.goal_valence = valence
        self.state.goal_arousal = arousal

    def get_status(self) -> Dict[str, Any]:
        """Get control plane status."""
        return {
            "mode": self.mode.value,
            "state": self.state.to_dict(),
            "latency_stats": self.emulator.get_latency_stats(),
            "memory_stats": self.memory_mgr.get_stats(),
        }

    def export_hls_pragma(self) -> str:
        """
        Generate Vitis HLS pragma hints for synthesis.

        Returns C++ pragma annotations for FPGA implementation.
        """
        return """
// Vitis HLS Pragmas for Control Plane
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=valence
#pragma HLS INTERFACE s_axilite port=arousal
#pragma HLS INTERFACE s_axilite port=dominance

#pragma HLS PIPELINE II=1
#pragma HLS LATENCY max=100

// LIF state in BRAM
#pragma HLS RESOURCE variable=lif_v core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=lif_v cyclic factor=8

// PGU cache in BRAM
#pragma HLS RESOURCE variable=pgu_cache core=RAM_2P_BRAM
"""


# Singleton
_control_plane: Optional[ControlPlane] = None


def get_control_plane() -> ControlPlane:
    """Get or create global control plane."""
    global _control_plane
    if _control_plane is None:
        _control_plane = ControlPlane()
    return _control_plane


def fast_control(
    valence: float,
    arousal: float,
    dominance: float = 0.5,
) -> Dict[str, Any]:
    """
    Convenience function for fast control step.

    Returns control outputs with low latency.
    """
    plane = get_control_plane()
    return plane.fast_control_step(valence, arousal, dominance)


__all__ = [
    "ControlPlaneMode",
    "LatencyMetrics",
    "ControlState",
    "hls_lif_step",
    "hls_l1_homeostat",
    "hls_l3_metacontrol",
    "hls_pgu_cache_lookup",
    "CXLMemoryManager",
    "FPGAEmulator",
    "ControlPlane",
    "get_control_plane",
    "fast_control",
]
