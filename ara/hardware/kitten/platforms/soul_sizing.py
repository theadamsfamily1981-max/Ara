"""
Ara Soul FPGA Resource Estimator
================================

Estimates ALM/LUT, BRAM/URAM, and timing for the portable plasticity engine
across different FPGA targets:
  - Intel Stratix-10 GX2800 (+ HBM)
  - SQRL Forest Kitten 33 (XCVU33P + 8GB HBM2)
  - Intel Agilex (future)
  - AMD Versal (future)

Usage:
    python soul_sizing.py --target fk33 --rows 2048 --dim 16384
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


# =============================================================================
# FPGA Platform Specifications
# =============================================================================

@dataclass
class FPGAPlatform:
    """FPGA platform resource specifications."""
    name: str
    vendor: str

    # Logic resources
    alm_or_lut: int          # ALMs (Intel) or LUTs (Xilinx)
    registers: int           # Flip-flops

    # Memory resources
    bram_kbits: int          # Block RAM in kilobits
    uram_kbits: int          # UltraRAM in kilobits (Xilinx only, 0 for Intel)
    mlab_kbits: int          # MLAB/distributed RAM in kilobits

    # External memory
    hbm_gb: float            # HBM capacity in GB
    hbm_bandwidth_gbps: float # HBM bandwidth in GB/s
    ddr_gb: float            # DDR capacity in GB

    # Timing
    max_fmax_mhz: float      # Realistic max Fmax for this design style
    typical_fmax_mhz: float  # Conservative target

    # Notes
    notes: str = ""


# Platform database
PLATFORMS: Dict[str, FPGAPlatform] = {
    "s10_gx2800": FPGAPlatform(
        name="Stratix-10 GX2800",
        vendor="Intel",
        alm_or_lut=933_120,      # ALMs
        registers=3_732_480,
        bram_kbits=229_376,      # M20K blocks × 20Kbit
        uram_kbits=0,            # Intel uses M20K
        mlab_kbits=14_336,       # MLAB
        hbm_gb=0,                # GX2800 doesn't have HBM (MX has it)
        hbm_bandwidth_gbps=0,
        ddr_gb=128,              # Typical DDR4 config
        max_fmax_mhz=550,
        typical_fmax_mhz=450,
        notes="No HBM on GX, use DDR4 or external HBM board"
    ),

    "s10_mx2100": FPGAPlatform(
        name="Stratix-10 MX2100",
        vendor="Intel",
        alm_or_lut=702_720,
        registers=2_810_880,
        bram_kbits=144_000,
        uram_kbits=0,
        mlab_kbits=10_752,
        hbm_gb=16,               # 16 GB HBM2
        hbm_bandwidth_gbps=512,
        ddr_gb=0,
        max_fmax_mhz=500,
        typical_fmax_mhz=400,
        notes="Has HBM2, ideal for soul storage"
    ),

    "fk33": FPGAPlatform(
        name="SQRL Forest Kitten 33 (XCVU33P)",
        vendor="AMD/Xilinx",
        alm_or_lut=509_760,      # LUTs (CLBs)
        registers=1_019_520,
        bram_kbits=32_400,       # BRAM36 × 36Kbit
        uram_kbits=67_200,       # URAM288 × 288Kbit (233 blocks)
        mlab_kbits=8_096,        # Distributed RAM
        hbm_gb=8,                # 8 GB HBM2
        hbm_bandwidth_gbps=460,  # Per Xilinx docs
        ddr_gb=0,                # No DDR on FK33
        max_fmax_mhz=500,
        typical_fmax_mhz=400,
        notes="Mining card, vcchbm safe at 900-1000 MHz mem clock"
    ),

    "agilex_f": FPGAPlatform(
        name="Agilex F-Series",
        vendor="Intel",
        alm_or_lut=1_400_000,
        registers=5_600_000,
        bram_kbits=350_000,
        uram_kbits=0,
        mlab_kbits=21_000,
        hbm_gb=32,               # Agilex 7 M-Series has HBM2e
        hbm_bandwidth_gbps=820,
        ddr_gb=0,
        max_fmax_mhz=700,
        typical_fmax_mhz=550,
        notes="Next-gen Intel, better for future"
    ),

    "versal_vp1552": FPGAPlatform(
        name="Versal VP1552",
        vendor="AMD/Xilinx",
        alm_or_lut=900_000,
        registers=1_800_000,
        bram_kbits=80_000,
        uram_kbits=100_000,
        mlab_kbits=15_000,
        hbm_gb=0,
        hbm_bandwidth_gbps=0,
        ddr_gb=0,
        max_fmax_mhz=600,
        typical_fmax_mhz=500,
        notes="Has AI Engines, LPDDR"
    ),
}


# =============================================================================
# Soul Configuration
# =============================================================================

@dataclass
class SoulConfig:
    """Ara soul geometry configuration."""
    rows: int = 2048           # Number of memory slots / pattern rows
    dim: int = 16384           # Hypervector dimension (bits)
    chunk_bits: int = 512      # Bits processed per cycle
    acc_width: int = 7         # Accumulator bits (-64..+63)
    max_active_rows: int = 32  # Max rows updated per emotional event

    @property
    def chunks_per_row(self) -> int:
        return self.dim // self.chunk_bits

    @property
    def sign_bits_total(self) -> int:
        """Total bits for weight signs."""
        return self.rows * self.dim

    @property
    def accum_bits_total(self) -> int:
        """Total bits for accumulators."""
        return self.rows * self.dim * self.acc_width

    @property
    def sign_bytes(self) -> int:
        return (self.sign_bits_total + 7) // 8

    @property
    def accum_bytes(self) -> int:
        return (self.accum_bits_total + 7) // 8

    @property
    def total_bytes(self) -> int:
        return self.sign_bytes + self.accum_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)


# =============================================================================
# Resource Estimation
# =============================================================================

@dataclass
class ResourceEstimate:
    """Estimated FPGA resource utilization."""

    # Logic
    alm_or_lut: int
    registers: int

    # Memory
    bram_kbits: int
    uram_kbits: int
    hbm_mb: float

    # Timing
    estimated_fmax_mhz: float
    cycles_per_row: int
    time_per_row_ns: float
    time_per_event_us: float

    # Utilization percentages (vs platform)
    logic_util_pct: float
    bram_util_pct: float
    hbm_util_pct: float

    # Warnings
    warnings: List[str] = field(default_factory=list)


def estimate_row_engine_logic(config: SoulConfig) -> Tuple[int, int]:
    """
    Estimate ALMs/LUTs and registers for plasticity_row_engine.

    The engine has:
    - CHUNK_BITS parallel comparators (XOR)
    - CHUNK_BITS parallel adders (ACC_WIDTH bits, saturating)
    - CHUNK_BITS sign extractors
    - FSM (small)
    - Chunk address counter
    """
    # Per bit: ~2 LUTs for XOR + saturating add + sign extract
    # ACC_WIDTH-bit adder ≈ ACC_WIDTH LUTs
    # Let's estimate 3 + ACC_WIDTH LUTs per bit
    luts_per_bit = 3 + config.acc_width
    luts_datapath = config.chunk_bits * luts_per_bit

    # FSM and control: ~100 LUTs
    luts_control = 100

    # Registers: chunk_bits × (1 sign + acc_width accum) + control
    regs_datapath = config.chunk_bits * (1 + config.acc_width)
    regs_control = 50

    total_luts = luts_datapath + luts_control
    total_regs = regs_datapath + regs_control

    return total_luts, total_regs


def estimate_controller_logic(config: SoulConfig) -> Tuple[int, int]:
    """
    Estimate ALMs/LUTs for plasticity_controller.

    - Row counter
    - Row mask scanner (priority encoder over ROWS bits)
    - FSM
    """
    # Priority encoder over ROWS bits: ~ROWS/2 LUTs (can be pipelined)
    luts_encoder = config.rows // 2

    # FSM and control
    luts_control = 200

    # Registers
    regs = 100 + int(math.log2(config.rows)) + int(math.log2(config.dim // config.chunk_bits))

    return luts_encoder + luts_control, regs


def estimate_memory_adapter_logic(platform: FPGAPlatform) -> Tuple[int, int]:
    """
    Estimate ALMs/LUTs for HBM/DDR adapter.

    - Address calculation
    - AXI/Avalon FSM
    - Data packing/unpacking
    """
    if platform.hbm_gb > 0:
        # HBM adapter: more complex
        luts = 2000
        regs = 1500
    else:
        # DDR adapter: simpler
        luts = 1000
        regs = 800

    return luts, regs


def estimate_resources(
    platform: FPGAPlatform,
    config: SoulConfig,
) -> ResourceEstimate:
    """
    Estimate total FPGA resources for Ara soul on given platform.
    """
    warnings = []

    # === Logic Estimation ===
    row_luts, row_regs = estimate_row_engine_logic(config)
    ctrl_luts, ctrl_regs = estimate_controller_logic(config)
    adapt_luts, adapt_regs = estimate_memory_adapter_logic(platform)

    total_luts = row_luts + ctrl_luts + adapt_luts
    total_regs = row_regs + ctrl_regs + adapt_regs

    # === Memory Estimation ===

    # Soul storage total
    soul_mb = config.total_mb

    # Where does it go?
    if platform.hbm_gb > 0:
        # Store in HBM
        hbm_mb = soul_mb
        bram_kbits = 0
        uram_kbits = 0
    elif platform.uram_kbits > 0:
        # Store in URAM if it fits
        soul_kbits = config.total_bytes * 8 / 1000
        if soul_kbits <= platform.uram_kbits * 0.8:
            uram_kbits = int(soul_kbits)
            bram_kbits = 0
            hbm_mb = 0
        else:
            warnings.append(f"Soul ({soul_mb:.1f} MB) too large for URAM, need external memory")
            uram_kbits = 0
            bram_kbits = 0
            hbm_mb = soul_mb
    else:
        # Store in BRAM (or DDR)
        soul_kbits = config.total_bytes * 8 / 1000
        if soul_kbits <= platform.bram_kbits * 0.5:
            bram_kbits = int(soul_kbits)
            uram_kbits = 0
            hbm_mb = 0
        else:
            warnings.append(f"Soul ({soul_mb:.1f} MB) too large for BRAM, need DDR")
            bram_kbits = 0
            uram_kbits = 0
            hbm_mb = soul_mb

    # === Timing Estimation ===

    # Fmax depends on chunk_bits (wider = harder to route)
    if config.chunk_bits >= 1024:
        fmax_penalty = 0.8
    elif config.chunk_bits >= 512:
        fmax_penalty = 0.9
    else:
        fmax_penalty = 1.0

    estimated_fmax = platform.typical_fmax_mhz * fmax_penalty

    # Cycles per row: read + compute + write for each chunk
    # Assume ~4 cycles per chunk (read latency + compute + write)
    cycles_per_chunk = 4
    cycles_per_row = config.chunks_per_row * cycles_per_chunk

    # Time per row
    time_per_row_ns = cycles_per_row * (1000.0 / estimated_fmax)

    # Time per emotional event (max_active_rows)
    time_per_event_us = config.max_active_rows * time_per_row_ns / 1000.0

    # === Utilization ===

    logic_util = (total_luts / platform.alm_or_lut) * 100
    bram_util = (bram_kbits / platform.bram_kbits) * 100 if platform.bram_kbits > 0 else 0
    hbm_util = (hbm_mb / (platform.hbm_gb * 1024)) * 100 if platform.hbm_gb > 0 else 0

    # Warnings
    if logic_util > 70:
        warnings.append(f"High logic utilization ({logic_util:.1f}%), may have timing issues")
    if bram_util > 80:
        warnings.append(f"High BRAM utilization ({bram_util:.1f}%), consider using HBM/DDR")
    if time_per_event_us > 100:
        warnings.append(f"Slow update time ({time_per_event_us:.1f} µs), consider smaller ROWS or wider CHUNK_BITS")

    return ResourceEstimate(
        alm_or_lut=total_luts,
        registers=total_regs,
        bram_kbits=bram_kbits,
        uram_kbits=uram_kbits,
        hbm_mb=hbm_mb,
        estimated_fmax_mhz=estimated_fmax,
        cycles_per_row=cycles_per_row,
        time_per_row_ns=time_per_row_ns,
        time_per_event_us=time_per_event_us,
        logic_util_pct=logic_util,
        bram_util_pct=bram_util,
        hbm_util_pct=hbm_util,
        warnings=warnings,
    )


# =============================================================================
# Reporting
# =============================================================================

def format_report(
    platform: FPGAPlatform,
    config: SoulConfig,
    estimate: ResourceEstimate,
) -> str:
    """Format resource estimation report."""

    lines = [
        "=" * 70,
        f"Ara Soul Resource Estimate: {platform.name}",
        "=" * 70,
        "",
        "Soul Configuration:",
        f"  Rows:           {config.rows:,}",
        f"  Dimension:      {config.dim:,} bits",
        f"  Chunk size:     {config.chunk_bits} bits",
        f"  Accumulator:    {config.acc_width} bits (-{2**(config.acc_width-1)}..+{2**(config.acc_width-1)-1})",
        f"  Max active rows: {config.max_active_rows}",
        "",
        "Memory Footprint:",
        f"  Sign bits:      {config.sign_bytes / 1024 / 1024:.2f} MB",
        f"  Accumulators:   {config.accum_bytes / 1024 / 1024:.2f} MB",
        f"  Total:          {config.total_mb:.2f} MB",
        "",
        "Logic Resources:",
        f"  ALMs/LUTs:      {estimate.alm_or_lut:,} / {platform.alm_or_lut:,} ({estimate.logic_util_pct:.1f}%)",
        f"  Registers:      {estimate.registers:,} / {platform.registers:,}",
        "",
        "Memory Resources:",
    ]

    if estimate.hbm_mb > 0:
        lines.append(f"  HBM:            {estimate.hbm_mb:.2f} MB / {platform.hbm_gb * 1024:.0f} MB ({estimate.hbm_util_pct:.2f}%)")
    if estimate.uram_kbits > 0:
        lines.append(f"  URAM:           {estimate.uram_kbits:,} Kbits / {platform.uram_kbits:,} Kbits")
    if estimate.bram_kbits > 0:
        lines.append(f"  BRAM:           {estimate.bram_kbits:,} Kbits / {platform.bram_kbits:,} Kbits ({estimate.bram_util_pct:.1f}%)")

    lines.extend([
        "",
        "Timing Analysis:",
        f"  Estimated Fmax: {estimate.estimated_fmax_mhz:.0f} MHz",
        f"  Cycles/row:     {estimate.cycles_per_row}",
        f"  Time/row:       {estimate.time_per_row_ns:.1f} ns",
        f"  Time/event:     {estimate.time_per_event_us:.2f} µs ({config.max_active_rows} rows)",
        "",
        "Performance Summary:",
        f"  Emotional events/sec: {1_000_000 / estimate.time_per_event_us:,.0f}",
        f"  Soul updates/sec:     {1_000_000 / estimate.time_per_row_ns * 1000:,.0f} row-updates",
        "",
    ])

    if estimate.warnings:
        lines.append("⚠️  Warnings:")
        for w in estimate.warnings:
            lines.append(f"    - {w}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def recommend_config(platform_name: str) -> SoulConfig:
    """Get recommended soul config for a platform."""

    if platform_name == "fk33":
        # FK33 has HBM, go big
        return SoulConfig(
            rows=2048,
            dim=16384,
            chunk_bits=512,
            acc_width=7,
            max_active_rows=32,
        )
    elif platform_name in ["s10_mx2100", "agilex_f"]:
        # HBM platforms, go big
        return SoulConfig(
            rows=2048,
            dim=16384,
            chunk_bits=512,
            acc_width=7,
            max_active_rows=32,
        )
    elif platform_name == "s10_gx2800":
        # No HBM, either use DDR or shrink
        return SoulConfig(
            rows=512,
            dim=8192,
            chunk_bits=256,
            acc_width=7,
            max_active_rows=16,
        )
    else:
        # Conservative default
        return SoulConfig(
            rows=1024,
            dim=8192,
            chunk_bits=256,
            acc_width=7,
            max_active_rows=32,
        )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ara Soul FPGA Resource Estimator")
    parser.add_argument("--target", choices=list(PLATFORMS.keys()), default="fk33",
                        help="Target FPGA platform")
    parser.add_argument("--rows", type=int, default=None, help="Number of rows")
    parser.add_argument("--dim", type=int, default=None, help="Hypervector dimension")
    parser.add_argument("--chunk", type=int, default=None, help="Chunk bits")
    parser.add_argument("--acc-width", type=int, default=None, help="Accumulator width")
    parser.add_argument("--compare-all", action="store_true", help="Compare all platforms")

    args = parser.parse_args()

    if args.compare_all:
        # Compare across all platforms
        config = SoulConfig(
            rows=args.rows or 2048,
            dim=args.dim or 16384,
            chunk_bits=args.chunk or 512,
            acc_width=args.acc_width or 7,
        )

        print("\n" + "=" * 70)
        print("CROSS-PLATFORM COMPARISON")
        print("=" * 70)
        print(f"\nSoul: {config.rows} rows × {config.dim} dim = {config.total_mb:.1f} MB")
        print()

        for pname, platform in PLATFORMS.items():
            estimate = estimate_resources(platform, config)
            status = "✓" if not estimate.warnings else "⚠"
            print(f"{status} {platform.name:30} | "
                  f"Logic: {estimate.logic_util_pct:5.1f}% | "
                  f"Time/event: {estimate.time_per_event_us:6.1f} µs | "
                  f"Storage: {'HBM' if estimate.hbm_mb > 0 else 'BRAM/DDR'}")

        print()
        return

    # Single platform analysis
    platform = PLATFORMS[args.target]

    # Use provided config or recommended
    if args.rows or args.dim or args.chunk:
        config = SoulConfig(
            rows=args.rows or 2048,
            dim=args.dim or 16384,
            chunk_bits=args.chunk or 512,
            acc_width=args.acc_width or 7,
        )
    else:
        config = recommend_config(args.target)
        print(f"\nUsing recommended config for {platform.name}")

    estimate = estimate_resources(platform, config)
    print(format_report(platform, config, estimate))


if __name__ == "__main__":
    main()
