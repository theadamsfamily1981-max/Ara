# Hardware Bring-Up Checklist (Phase 2)

Pre-flight validation for FPGA deployment. All software gates have passed.

## Software Validation Status

| Check | Status | Result |
|-------|--------|--------|
| HLS Export | ✅ PASS | 4 files generated, all pragmas present |
| Module Imports | ✅ PASS | ara.cxl_control, ara.metacontrol, tfan.system |
| Certification | ✅ PASS | AF Score: 1.81×, Δp99: +16.74ms |
| Closed-Loop Demo | ✅ PASS | All 5 stages completed in 986ms |
| A-Cert Pipeline | ✅ PASS | 4 jobs, all scripts present |

## Generated HLS Artifacts

```
build/hls/
├── pgu_cache_kernel.cpp   (5784 bytes) - Main HLS kernel
├── pgu_cache_kernel.h     (1439 bytes) - Header file
├── pgu_cache_tb.cpp       (2573 bytes) - C++ testbench
└── run_hls.tcl            (867 bytes)  - Synthesis script
```

## Hardware Requirements

### Target FPGA
- **Device:** Xilinx Alveo U250 (or CXL-capable equivalent)
- **Part Number:** `xcu250-figd2104-2L-e`
- **Clock Target:** 250MHz (4ns period)

### Software Tools
- [ ] Vitis HLS 2023.1+ installed
- [ ] Vivado 2023.1+ installed
- [ ] XRT (Xilinx Runtime) installed
- [ ] FPGA drivers configured

### Host System
- [ ] PCIe Gen3/Gen4 x16 slot available
- [ ] CXL support (if using CXL memory path)
- [ ] 64GB+ system RAM recommended

## Hardware Bring-Up Steps

### Step 1: Generate HLS IP

```bash
# Export HLS files
cd /home/user/Ara
python3 -c "from ara.cxl_control import export_hls_kernel; export_hls_kernel('build/hls')"

# Run Vitis HLS synthesis
cd build/hls
vitis_hls -f run_hls.tcl

# Expected outputs:
# - Latency: ≤25 cycles
# - II (Initiation Interval): 1
# - BRAM: ~10% utilization
# - LUT: ~5% utilization
```

### Step 2: Verify Synthesis Results

Check `solution1/syn/report/pgu_cache_kernel_csynth.rpt`:

| Metric | Target | Acceptable |
|--------|--------|------------|
| Latency (cycles) | ≤25 | ≤50 |
| Initiation Interval | 1 | ≤2 |
| Clock Period Met | Yes | Yes |
| BRAM_18K | <100 | <200 |

### Step 3: Create Vivado Project

```bash
# Generate IP catalog
cd build/hls
vitis_hls -f run_hls.tcl  # exports pgu_cache_ip.zip

# Import into Vivado
vivado -mode batch -source create_project.tcl
```

### Step 4: Build Bitstream

```bash
# Run implementation
cd vivado_project
vivado -mode batch -source build_bitstream.tcl

# Expected: ~30-60 minutes for U250
```

### Step 5: Program FPGA

```bash
# Load bitstream
xbutil program -d 0000:03:00.1 -u pgu_cache.xclbin

# Verify
xbutil examine -d 0000:03:00.1
```

### Step 6: Run Hardware Validation

```bash
# Hardware latency test
python3 scripts/test_fpga_latency.py --device 0000:03:00.1

# Expected results:
# - PGU Cache Lookup: <100ns
# - L1 Homeostat: <50ns
# - L3 Metacontrol: <100ns
```

## Validation Checklist

### Pre-Synthesis
- [ ] HLS export completes without errors
- [ ] All 4 files generated (cpp, h, tb, tcl)
- [ ] Pragmas verified in kernel code
- [ ] Testbench compiles and passes

### Post-Synthesis
- [ ] Timing constraints met
- [ ] Resource utilization within bounds
- [ ] Co-simulation passes
- [ ] No critical warnings

### Post-Implementation
- [ ] Bitstream generated successfully
- [ ] FPGA programmed without errors
- [ ] Device detected by xbutil
- [ ] Kernel responds to test queries

### Performance Validation
- [ ] p95 latency < target (200μs for PGU)
- [ ] Cache hit rate > 50%
- [ ] No memory errors
- [ ] Sustained throughput achieved

## Fallback Plan

If hardware issues occur:

1. **Synthesis Failure:** Check clock constraints, reduce target frequency
2. **Timing Failure:** Increase pipeline stages, relax II constraint
3. **Resource Overflow:** Reduce cache size, simplify datapath
4. **Runtime Errors:** Verify memory alignment, check AXI transactions

## Contact Points

- HLS Issues: Check Vitis HLS User Guide (UG1399)
- Vivado Issues: Check Vivado Design Suite User Guide (UG910)
- XRT Issues: Check XRT Documentation on GitHub

## Quick Commands Reference

```bash
# Generate HLS
python3 -c "from ara.cxl_control import export_hls_kernel; export_hls_kernel('build/hls')"

# Run synthesis
cd build/hls && vitis_hls -f run_hls.tcl

# Check FPGA
xbutil examine

# Run certification
python3 scripts/certify_antifragility_delta.py --burst-factor 2.0 --duration 15

# Run closed-loop demo
python3 scripts/demo_closed_loop_antifragility.py --stress-level high
```

---

**Status:** Ready for Hardware Bring-Up
**Last Validated:** 2025-12-01
**Software Baseline:** AF Score 1.81×, Δp99 +16.74ms
