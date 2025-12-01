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


# =============================================================================
# PHASE 5.2: HLS EXPORT FOR PGU CACHE KERNEL
# =============================================================================

class HLSExporter:
    """
    Exports Python control functions to Vitis HLS-compatible C++.

    Generates:
    - C++ kernel code with pragmas
    - Testbench for co-simulation
    - TCL scripts for synthesis
    - Memory interface specifications
    """

    def __init__(self, target_device: str = "xcu250-figd2104-2L-e"):
        """
        Initialize HLS exporter.

        Args:
            target_device: Xilinx device part number
        """
        self.target_device = target_device
        self.clock_period_ns = 4.0  # 250MHz target
        self.target_latency_cycles = 25  # ~100ns at 250MHz

        logger.info(f"HLS Exporter initialized for {target_device}")

    def export_pgu_cache_kernel(self) -> str:
        """
        Generate complete PGU Cache HLS kernel.

        Returns:
            C++ source code for Vitis HLS
        """
        return '''// =============================================================================
// PGU Cache Kernel - Vitis HLS Implementation
// Target: p95 < 100ns @ 250MHz (25 cycles)
// =============================================================================

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// Fixed-point types for PAD values
typedef ap_fixed<16, 2, AP_RND, AP_SAT> pad_t;      // Q2.14 for [-2, 2)
typedef ap_fixed<32, 16, AP_RND, AP_SAT> hash_t;    // Hash values
typedef ap_uint<32> key_t;                           // Cache key
typedef ap_uint<64> value_t;                         // Cached value

// Cache configuration
#define CACHE_SIZE 1024
#define CACHE_WAYS 4
#define TAG_BITS 22
#define INDEX_BITS 8

// Cache entry structure
struct cache_entry_t {
    ap_uint<1> valid;
    ap_uint<TAG_BITS> tag;
    value_t data;
    ap_uint<8> lru;
};

// L3 Metacontrol output structure
struct metacontrol_out_t {
    pad_t temperature_mult;
    pad_t memory_mult;
    pad_t attention_gain;
};

// PGU Cache kernel top function
void pgu_cache_kernel(
    // Input PAD state (AXI-Lite)
    pad_t valence,
    pad_t arousal,
    pad_t dominance,
    // Input query key
    key_t query_key,
    // Cache memory (AXI Master)
    cache_entry_t cache[CACHE_SIZE],
    // Outputs
    ap_uint<1> *cache_hit,
    value_t *cached_value,
    metacontrol_out_t *mc_out
) {
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE s_axilite port=valence bundle=control
    #pragma HLS INTERFACE s_axilite port=arousal bundle=control
    #pragma HLS INTERFACE s_axilite port=dominance bundle=control
    #pragma HLS INTERFACE s_axilite port=query_key bundle=control
    #pragma HLS INTERFACE s_axilite port=cache_hit bundle=control
    #pragma HLS INTERFACE s_axilite port=cached_value bundle=control
    #pragma HLS INTERFACE s_axilite port=mc_out bundle=control

    #pragma HLS INTERFACE m_axi port=cache offset=slave bundle=gmem depth=1024

    #pragma HLS PIPELINE II=1
    #pragma HLS LATENCY max=25

    // Local cache copy for low latency
    cache_entry_t local_cache[CACHE_WAYS];
    #pragma HLS ARRAY_PARTITION variable=local_cache complete

    // =========================================================================
    // Stage 1: Compute cache index and tag
    // =========================================================================
    ap_uint<INDEX_BITS> index = query_key(INDEX_BITS-1, 0);
    ap_uint<TAG_BITS> tag = query_key(31, INDEX_BITS);

    // =========================================================================
    // Stage 2: Parallel cache lookup (4-way associative)
    // =========================================================================
    #pragma HLS PIPELINE
    cache_lookup: for (int way = 0; way < CACHE_WAYS; way++) {
        #pragma HLS UNROLL
        local_cache[way] = cache[index * CACHE_WAYS + way];
    }

    // Check for hit
    ap_uint<1> hit = 0;
    value_t hit_value = 0;
    int hit_way = 0;

    hit_check: for (int way = 0; way < CACHE_WAYS; way++) {
        #pragma HLS UNROLL
        if (local_cache[way].valid && local_cache[way].tag == tag) {
            hit = 1;
            hit_value = local_cache[way].data;
            hit_way = way;
        }
    }

    // =========================================================================
    // Stage 3: L3 Metacontrol computation (parallel with cache)
    // =========================================================================
    // Temperature: higher arousal → higher exploration
    pad_t temp_mult = pad_t(0.8) + arousal * pad_t(0.5);

    // Memory: lower valence → more conservative
    pad_t mem_mult = pad_t(0.95) + (valence + pad_t(1.0)) * pad_t(0.125);

    // Attention: higher dominance → higher gain
    pad_t attn_gain = pad_t(0.8) + dominance * pad_t(0.4);

    // =========================================================================
    // Stage 4: Output results
    // =========================================================================
    *cache_hit = hit;
    *cached_value = hit_value;
    mc_out->temperature_mult = temp_mult;
    mc_out->memory_mult = mem_mult;
    mc_out->attention_gain = attn_gain;

    // Update LRU on hit (simplified)
    if (hit) {
        local_cache[hit_way].lru = 0;
        cache[index * CACHE_WAYS + hit_way] = local_cache[hit_way];
    }
}

// LIF neuron step for L1 homeostatic loop
void lif_step(
    pad_t v_in,
    pad_t current,
    pad_t tau,
    pad_t v_th,
    pad_t *v_out,
    ap_uint<1> *spike
) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=1

    // LIF dynamics: dv/dt = -v/tau + I
    pad_t dv = -v_in / tau + current;
    pad_t new_v = v_in + dv;

    // Spike and reset
    if (new_v >= v_th) {
        *spike = 1;
        *v_out = 0;
    } else {
        *spike = 0;
        *v_out = new_v;
    }
}

// L1 Homeostatic PI controller
void l1_homeostat(
    pad_t current_valence,
    pad_t current_arousal,
    pad_t goal_valence,
    pad_t goal_arousal,
    pad_t error_integral_in,
    pad_t kp,
    pad_t ki,
    pad_t *valence_corr,
    pad_t *arousal_corr,
    pad_t *error_integral_out
) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=1

    // Compute errors
    pad_t val_error = goal_valence - current_valence;
    pad_t aro_error = goal_arousal - current_arousal;

    // PI control
    *valence_corr = kp * val_error + ki * error_integral_in;
    *arousal_corr = kp * aro_error + ki * error_integral_in;

    // Update integral with anti-windup
    pad_t new_integral = error_integral_in + val_error + aro_error;
    if (new_integral > pad_t(10.0)) new_integral = pad_t(10.0);
    if (new_integral < pad_t(-10.0)) new_integral = pad_t(-10.0);
    *error_integral_out = new_integral;
}
'''

    def export_testbench(self) -> str:
        """
        Generate C++ testbench for HLS co-simulation.

        Returns:
            C++ testbench source code
        """
        return '''// =============================================================================
// PGU Cache Kernel Testbench
// =============================================================================

#include <iostream>
#include <cmath>
#include "pgu_cache_kernel.h"

int main() {
    // Initialize cache
    cache_entry_t cache[CACHE_SIZE];
    for (int i = 0; i < CACHE_SIZE; i++) {
        cache[i].valid = 0;
        cache[i].tag = 0;
        cache[i].data = 0;
        cache[i].lru = 0;
    }

    // Test inputs
    pad_t valence = pad_t(0.3);
    pad_t arousal = pad_t(0.6);
    pad_t dominance = pad_t(0.5);
    key_t query_key = 0x12345678;

    // Outputs
    ap_uint<1> cache_hit;
    value_t cached_value;
    metacontrol_out_t mc_out;

    // Pre-populate cache for testing
    cache[query_key & 0xFF].valid = 1;
    cache[query_key & 0xFF].tag = query_key >> 8;
    cache[query_key & 0xFF].data = 0xDEADBEEF;

    // Run kernel
    pgu_cache_kernel(
        valence, arousal, dominance,
        query_key,
        cache,
        &cache_hit, &cached_value, &mc_out
    );

    // Verify results
    std::cout << "=== PGU Cache Kernel Test ===" << std::endl;
    std::cout << "Cache hit: " << cache_hit << std::endl;
    std::cout << "Cached value: 0x" << std::hex << cached_value << std::endl;
    std::cout << "Temperature mult: " << mc_out.temperature_mult.to_float() << std::endl;
    std::cout << "Memory mult: " << mc_out.memory_mult.to_float() << std::endl;
    std::cout << "Attention gain: " << mc_out.attention_gain.to_float() << std::endl;

    // Check expected values
    bool pass = true;

    // Expect cache hit
    if (cache_hit != 1) {
        std::cerr << "ERROR: Expected cache hit" << std::endl;
        pass = false;
    }

    // Check metacontrol values (approximate)
    float expected_temp = 0.8 + 0.6 * 0.5;  // 1.1
    float expected_mem = 0.95 + (0.3 + 1.0) * 0.125;  // 1.1125
    float expected_attn = 0.8 + 0.5 * 0.4;  // 1.0

    if (std::abs(mc_out.temperature_mult.to_float() - expected_temp) > 0.1) {
        std::cerr << "ERROR: Temperature mult mismatch" << std::endl;
        pass = false;
    }

    // Test cache miss
    key_t miss_key = 0xFFFFFFFF;
    pgu_cache_kernel(
        valence, arousal, dominance,
        miss_key,
        cache,
        &cache_hit, &cached_value, &mc_out
    );

    if (cache_hit != 0) {
        std::cerr << "ERROR: Expected cache miss" << std::endl;
        pass = false;
    }

    std::cout << "\\n=== Test " << (pass ? "PASSED" : "FAILED") << " ===" << std::endl;
    return pass ? 0 : 1;
}
'''

    def export_tcl_script(self) -> str:
        """
        Generate Vitis HLS TCL script for synthesis.

        Returns:
            TCL script content
        """
        return f'''# =============================================================================
# Vitis HLS Synthesis Script for PGU Cache Kernel
# =============================================================================

# Create project
open_project pgu_cache_hls
set_top pgu_cache_kernel
add_files pgu_cache_kernel.cpp
add_files -tb pgu_cache_tb.cpp

# Create solution
open_solution "solution1" -flow_target vivado
set_part {{{self.target_device}}}
create_clock -period {self.clock_period_ns} -name default

# Synthesis directives
config_compile -pipeline_style flp
config_schedule -effort high -relax_ii_for_timing=0

# Run synthesis
csynth_design

# Run co-simulation
cosim_design -trace_level all -rtl verilog

# Export IP
export_design -format ip_catalog -output pgu_cache_ip.zip

# Generate reports
report_timing -file timing_report.rpt
report_utilization -file utilization_report.rpt

exit
'''

    def export_header(self) -> str:
        """
        Generate C++ header file.

        Returns:
            Header file content
        """
        return '''// =============================================================================
// PGU Cache Kernel Header
// =============================================================================

#ifndef PGU_CACHE_KERNEL_H
#define PGU_CACHE_KERNEL_H

#include <ap_int.h>
#include <ap_fixed.h>

// Fixed-point types
typedef ap_fixed<16, 2, AP_RND, AP_SAT> pad_t;
typedef ap_uint<32> key_t;
typedef ap_uint<64> value_t;

// Cache configuration
#define CACHE_SIZE 1024
#define CACHE_WAYS 4
#define TAG_BITS 22
#define INDEX_BITS 8

// Cache entry
struct cache_entry_t {
    ap_uint<1> valid;
    ap_uint<TAG_BITS> tag;
    value_t data;
    ap_uint<8> lru;
};

// Metacontrol output
struct metacontrol_out_t {
    pad_t temperature_mult;
    pad_t memory_mult;
    pad_t attention_gain;
};

// Top function declaration
void pgu_cache_kernel(
    pad_t valence,
    pad_t arousal,
    pad_t dominance,
    key_t query_key,
    cache_entry_t cache[CACHE_SIZE],
    ap_uint<1> *cache_hit,
    value_t *cached_value,
    metacontrol_out_t *mc_out
);

// Utility functions
void lif_step(
    pad_t v_in, pad_t current, pad_t tau, pad_t v_th,
    pad_t *v_out, ap_uint<1> *spike
);

void l1_homeostat(
    pad_t current_valence, pad_t current_arousal,
    pad_t goal_valence, pad_t goal_arousal,
    pad_t error_integral_in, pad_t kp, pad_t ki,
    pad_t *valence_corr, pad_t *arousal_corr,
    pad_t *error_integral_out
);

#endif // PGU_CACHE_KERNEL_H
'''

    def export_all(self, output_dir: str = "./hls_export") -> Dict[str, str]:
        """
        Export all HLS files to directory.

        Args:
            output_dir: Output directory path

        Returns:
            Dict mapping filename to content
        """
        import os

        files = {
            "pgu_cache_kernel.cpp": self.export_pgu_cache_kernel(),
            "pgu_cache_kernel.h": self.export_header(),
            "pgu_cache_tb.cpp": self.export_testbench(),
            "run_hls.tcl": self.export_tcl_script(),
        }

        # Create directory
        os.makedirs(output_dir, exist_ok=True)

        # Write files
        for filename, content in files.items():
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
            logger.info(f"Exported: {filepath}")

        return files


def export_hls_kernel(
    output_dir: str = "./hls_export",
    target_device: str = "xcu250-figd2104-2L-e",
) -> Dict[str, str]:
    """
    Export PGU Cache HLS kernel files.

    Convenience function for generating all HLS artifacts.

    Args:
        output_dir: Output directory
        target_device: Xilinx device part number

    Returns:
        Dict mapping filename to content

    Example:
        files = export_hls_kernel("./fpga/pgu_cache")
        # Creates: pgu_cache_kernel.cpp, .h, tb.cpp, run_hls.tcl
    """
    exporter = HLSExporter(target_device)
    return exporter.export_all(output_dir)


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
    # Phase 5.2 HLS exports
    "HLSExporter",
    "export_hls_kernel",
]
