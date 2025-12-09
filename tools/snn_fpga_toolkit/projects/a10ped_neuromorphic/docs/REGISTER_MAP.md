# AI Tile Register Map

**Auto-generated documentation**

- **Generated**: 2025-11-24 15:07:51
- **Source**: `ai_tile_registers.yaml` v1.0.0
- **Description**: Register map for neuromorphic AI tile on Arria 10 FPGA
- **License**: BSD-3-Clause

---

## Overview

The AI tile exposes a **4096-byte** register space via PCIe BAR0.
All registers are **32-bit** and **word-aligned** (4-byte boundaries).

## Register Summary

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x0000 | CTRL | RW | Control register for command execution and tile configuration |
| 0x0004 | STATUS | RO | Status register indicating tile state and error conditions |
| 0x0008 | CMD_SRC_LO | RW | Source address in DDR4 memory [31:0] |
| 0x000C | CMD_SRC_HI | RW | Source address in DDR4 memory [63:32] |
| 0x0010 | CMD_DST_LO | RW | Destination address in DDR4 memory [31:0] |
| 0x0014 | CMD_DST_HI | RW | Destination address in DDR4 memory [63:32] |
| 0x0018 | CMD_LEN | RW | Transfer length in bytes (must be 64-byte aligned) |
| 0x001C | CMD_CFG | RW | Command mode and parameter configuration |
| 0x0020 | VERSION | RO | Hardware and firmware version information |
| 0x0024 | CAPABILITIES | RO | Hardware feature flags |
| 0x0028 | SNN_THRESHOLD | RW | LIF neuron spike threshold (16.16 fixed-point) |
| 0x002C | SNN_LEAK | RW | LIF neuron membrane leak rate (16.16 fixed-point) |
| 0x0030 | SNN_REFRACT | RW | Refractory period in clock cycles |
| 0x0034 | ERROR_CODE | RO | Detailed error code from last failed operation |
| 0x0038 | PERF_CYCLES | RO | Clock cycles for last command execution |
| 0x003C | DDR_BANDWIDTH | RO | DDR bandwidth utilization (bytes per 1000 cycles) |
| 0x0040 | TEMPERATURE | RO | FPGA junction temperature from on-die sensor |
| 0x0044 | SCRATCH | RW | Scratch register for testing (read/write test pattern) |

## Detailed Register Descriptions

### CTRL (0x0000)

**Control register for command execution and tile configuration**

- **Access**: RW
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [0] | START | RW | 0 | Start command execution (self-clearing) |
| [1] | RESET | RW | 0 | Soft reset of AI core (self-clearing) |
| [2] | IRQ_EN | RW | 0 | Enable interrupt on command completion |
| [3] | ABORT | RW | 0 | Abort current command (self-clearing) |
| [1864] | RESERVED | RO | 0 | Reserved for future use |

### STATUS (0x0004)

**Status register indicating tile state and error conditions**

- **Access**: RO
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [0] | BUSY | RO | 0 | Command in progress (1 = busy, 0 = idle) |
| [1] | DONE | RO | 0 | Command completed successfully |
| [2] | ERROR | RO | 0 | Error occurred during execution |
| [3] | IRQ_PENDING | RO | 0 | Interrupt pending (cleared by reading this register) |
| [4] | DDR_READY | RO | 0 | DDR4 controller calibration complete |
| [5] | THERMAL_WARNING | RO | 0 | Temperature exceeds warning threshold |
| [1866] | RESERVED | RO | 0 | Reserved for future use |

### CMD_SRC_LO (0x0008)

**Source address in DDR4 memory [31:0]**

- **Access**: RW
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [1860] | ADDR_LO | RW | 0 | Lower 32 bits of 64-bit source address |

### CMD_SRC_HI (0x000C)

**Source address in DDR4 memory [63:32]**

- **Access**: RW
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [1860] | ADDR_HI | RW | 0 | Upper 32 bits of 64-bit source address |

### CMD_DST_LO (0x0010)

**Destination address in DDR4 memory [31:0]**

- **Access**: RW
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [1860] | ADDR_LO | RW | 0 | Lower 32 bits of 64-bit destination address |

### CMD_DST_HI (0x0014)

**Destination address in DDR4 memory [63:32]**

- **Access**: RW
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [1860] | ADDR_HI | RW | 0 | Upper 32 bits of 64-bit destination address |

### CMD_LEN (0x0018)

**Transfer length in bytes (must be 64-byte aligned)**

- **Access**: RW
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [1860] | LENGTH | RW | 0 | Number of bytes to transfer (max 16MB) |

### CMD_CFG (0x001C)

**Command mode and parameter configuration**

- **Access**: RW
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [180] | MODE | RW | 0 | Command mode:
0x0 = MEMCOPY (simple memory copy)
0x1 = SNN_INFER (LIF neuron inference)
0x2 = TOPOLOGICAL_FIELD (topological field network)
0x3 = RESERVED
0x4-0xF = Reserved for future algorithms
 |
| [304] | PRECISION | RW | 0 | Data precision for SNN mode:
0x0 = INT8
0x1 = INT16
0x2 = FP16
0x3 = FP32
 |
| [906] | NEURON_COUNT | RW | 256 | Number of neurons (for SNN mode, 1-1024) |
| [1396] | TIME_STEPS | RW | 100 | Number of time steps for SNN simulation |
| [1884] | RESERVED | RW | 0 | Reserved for future use |

### VERSION (0x0020)

**Hardware and firmware version information**

- **Access**: RO
- **Reset Value**: 0x01000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [420] | PATCH | RO | 0 | Patch version number |
| [908] | MINOR | RO | 0 | Minor version number |
| [1396] | MAJOR | RO | 1 | Major version number |
| [1884] | RESERVED | RO | 0 | Reserved |

### CAPABILITIES (0x0024)

**Hardware feature flags**

- **Access**: RO
- **Reset Value**: 0x00000003

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [0] | HAS_MEMCOPY | RO | 1 | Memory copy support |
| [1] | HAS_SNN | RO | 1 | SNN inference support |
| [2] | HAS_TOPOLOGICAL | RO | 0 | Topological field network support |
| [3] | HAS_IRQ | RO | 0 | Interrupt support |
| [4] | HAS_MULTI_PRECISION | RO | 0 | Multiple precision modes (INT8/16, FP16/32) |
| [1865] | RESERVED | RO | 0 | Reserved for future capabilities |

### SNN_THRESHOLD (0x0028)

**LIF neuron spike threshold (16.16 fixed-point)**

- **Access**: RW
- **Reset Value**: 0x00010000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [1860] | THRESHOLD | RW | 65536 | Spike threshold in Q16.16 format (default: 1.0) |

### SNN_LEAK (0x002C)

**LIF neuron membrane leak rate (16.16 fixed-point)**

- **Access**: RW
- **Reset Value**: 0x00000100

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [1860] | LEAK_RATE | RW | 256 | Leak rate in Q16.16 format (default: 0.00390625) |

### SNN_REFRACT (0x0030)

**Refractory period in clock cycles**

- **Access**: RW
- **Reset Value**: 0x00000008

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [900] | REFRACT_CYCLES | RW | 8 | Number of cycles neuron cannot spike after firing |
| [1876] | RESERVED | RO | 0 | Reserved |

### ERROR_CODE (0x0034)

**Detailed error code from last failed operation**

- **Access**: RO
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [420] | CODE | RO | 0 | Error codes:
0x00 = No error
0x01 = Invalid address (out of DDR range)
0x02 = Alignment error (not 64-byte aligned)
0x03 = DDR not ready
0x04 = Invalid command mode
0x05 = Timeout
0x06 = DMA error
0x07-0xFF = Reserved
 |
| [1868] | RESERVED | RO | 0 | Reserved |

### PERF_CYCLES (0x0038)

**Clock cycles for last command execution**

- **Access**: RO
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [1860] | CYCLES | RO | 0 | Cycle count (rolls over at 2^32) |

### DDR_BANDWIDTH (0x003C)

**DDR bandwidth utilization (bytes per 1000 cycles)**

- **Access**: RO
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [1860] | BW_UTIL | RO | 0 | Average bytes transferred per 1000 clock cycles |

### TEMPERATURE (0x0040)

**FPGA junction temperature from on-die sensor**

- **Access**: RO
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [900] | TEMP_C | RO | 0 | Temperature in Celsius * 256 (Q8.8 fixed-point) |
| [1876] | RESERVED | RO | 0 | Reserved |

### SCRATCH (0x0044)

**Scratch register for testing (read/write test pattern)**

- **Access**: RW
- **Reset Value**: 0x00000000

| Bits | Field | Access | Reset | Description |
|------|-------|--------|-------|-------------|
| [1860] | DATA | RW | 0 | User-defined test data |

## Command Sequences

### Memory Copy

Simple memory copy from SRC to DST

1. Write CMD_SRC_LO/HI with source address
2. Write CMD_DST_LO/HI with destination address
3. Write CMD_LEN with byte count (64-byte aligned)
4. Write CMD_CFG with MODE=0x0 (MEMCOPY)
5. Write CTRL.START=1 to execute
6. Poll STATUS.BUSY until 0
7. Check STATUS.DONE (1=success) or STATUS.ERROR (1=fail)
8. Read ERROR_CODE if STATUS.ERROR=1

### Snn Inference

LIF neuron inference

1. Write SNN_THRESHOLD, SNN_LEAK, SNN_REFRACT to configure neurons
2. Write CMD_SRC_LO/HI with input spike train address
3. Write CMD_DST_LO/HI with output spike buffer address
4. Write CMD_CFG with MODE=0x1, NEURON_COUNT, TIME_STEPS
5. Write CTRL.START=1 to execute
6. Poll STATUS.BUSY until 0
7. Read PERF_CYCLES to get inference latency
8. Process output spikes from DST buffer

## Validation Rules

- CMD_SRC_LO/HI must be within DDR4 address range (0 to 8GB-1)
- CMD_DST_LO/HI must be within DDR4 address range
- CMD_LEN must be 64-byte aligned (lower 6 bits = 0)
- CMD_LEN must not exceed 16MB (0x01000000)
- SNN NEURON_COUNT must be 1-1024
- Command must not be started while STATUS.BUSY=1

---

*This document is auto-generated. Do not edit manually.*
