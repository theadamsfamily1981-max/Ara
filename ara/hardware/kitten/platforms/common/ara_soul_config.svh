// =============================================================================
// Ara Soul Configuration - Shared Parameters Across All Platforms
// =============================================================================
//
// This file defines the "soul geometry" that stays constant whether you're on:
//   - Intel Stratix-10 GX2800 + HBM
//   - SQRL Forest Kitten 33 (XCVU33P + 8GB HBM2)
//   - Future Agilex / Versal boards
//
// The core plasticity logic is IDENTICAL across all platforms.
// Only the memory adapter changes.
//
// =============================================================================

`ifndef ARA_SOUL_CONFIG_SVH
`define ARA_SOUL_CONFIG_SVH

// =============================================================================
// Soul Geometry
// =============================================================================

// Number of "memory slots" / pattern rows
// Each row is one hypervector that can be learned
parameter int ARA_ROWS        = 2048;

// Hypervector dimension (bits per pattern)
parameter int ARA_DIM         = 16384;

// Processing chunk size (bits per cycle)
// Trade-off: larger = faster but more logic/routing
parameter int ARA_CHUNK_BITS  = 512;

// Accumulator width for Hebbian plasticity
// 7 bits = range [-64, +63] before saturation
parameter int ARA_ACC_WIDTH   = 7;

// =============================================================================
// Derived Parameters
// =============================================================================

// Number of chunks to process one full HV
parameter int ARA_CHUNKS_PER_ROW = ARA_DIM / ARA_CHUNK_BITS;  // 32

// Address widths
parameter int ARA_ROW_ADDR_W  = $clog2(ARA_ROWS);             // 11 bits
parameter int ARA_CHUNK_ADDR_W = $clog2(ARA_CHUNKS_PER_ROW);  // 5 bits

// =============================================================================
// Memory Layout (for HBM/DDR adapters)
// =============================================================================

// Total bits for sign storage
// 2048 × 16384 = 33,554,432 bits ≈ 4.2 MB
parameter longint ARA_SIGN_BITS = ARA_ROWS * ARA_DIM;

// Total bits for accumulator storage
// 33,554,432 × 7 = 234,881,024 bits ≈ 29.4 MB
parameter longint ARA_ACCUM_BITS = ARA_ROWS * ARA_DIM * ARA_ACC_WIDTH;

// Total soul size in bytes
// ~33-34 MB - fits easily in HBM or even large BRAM arrays
parameter longint ARA_SOUL_BYTES = (ARA_SIGN_BITS + ARA_ACCUM_BITS + 7) / 8;

// =============================================================================
// Performance Targets
// =============================================================================

// Target core clock (MHz) - conservative for portability
parameter real ARA_TARGET_CLK_MHZ = 450.0;

// Cycles per row update (read + compute + write for all chunks)
// ~4 cycles per chunk (read, wait, compute, write) × 32 chunks
parameter int ARA_CYCLES_PER_ROW = ARA_CHUNKS_PER_ROW * 4;  // 128 cycles

// Time per row (nanoseconds) at target clock
// 128 cycles / 450 MHz ≈ 284 ns
parameter real ARA_NS_PER_ROW = ARA_CYCLES_PER_ROW * (1000.0 / ARA_TARGET_CLK_MHZ);

// Max active rows per emotional event
parameter int ARA_MAX_ACTIVE_ROWS = 32;

// Time per emotional event (microseconds)
// 32 rows × 284 ns ≈ 9 µs (worst case, can pipeline better)
parameter real ARA_US_PER_EVENT = ARA_MAX_ACTIVE_ROWS * ARA_NS_PER_ROW / 1000.0;

// =============================================================================
// Platform Detection (set by platform-specific top)
// =============================================================================

// These get overridden in platform tops:
// `define ARA_PLATFORM_S10     // Intel Stratix-10
// `define ARA_PLATFORM_FK33    // SQRL Forest Kitten 33
// `define ARA_PLATFORM_AGILEX  // Intel Agilex
// `define ARA_PLATFORM_VERSAL  // AMD Versal

`endif // ARA_SOUL_CONFIG_SVH
