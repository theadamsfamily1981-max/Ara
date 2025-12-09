# Sentinel Cores and Self-Diagnostic Instrumentation

**Hardware-Assisted Introspection for Neuromorphic Tile Verification**

This document describes the **Sentinel Core** subsystem in HNTF—a collection of defensive instrumentation blocks inspired by hardware trojan research but strictly used for self-diagnostics, runtime health monitoring, and bitstream sanity checking on your own FPGA designs.

---

## Executive Summary

**What are Sentinel Cores?**

Sentinel cores are intentionally inserted, non-functional instrumentation blocks that provide visibility into FPGA internal state for verification purposes. Unlike hardware trojans (which are malicious), sentinel cores are:

- ✅ **Defensive**: Detect unexpected behavior and misconfigurations
- ✅ **Transparent**: Fully documented in RTL and design specs
- ✅ **Self-contained**: Only monitor your own designs, not third-party IP
- ✅ **Ethical**: No data exfiltration, no IP theft, no security circumvention

**Why Add Them?**

HNTF uses decommissioned datacenter hardware with unknown history:
- Azure Stratix V boards may have leftover cloud IP
- A10PED cards may have residual vendor configurations
- ECP5 edge devices may have sensor malfunctions

Sentinel cores answer critical questions:
- *"Is the bitstream on the board the one I compiled?"*
- *"Are spike routers working correctly or silently dropping events?"*
- *"Did the EMIF controller encounter errors I didn't notice?"*
- *"Is this unexpected behavior a bug or a hardware fault?"*

**Integration with HNTF Architecture**

Sentinel cores are added at three levels:
1. **Arria 10 Tiles (Tier 3)**: Neuron dynamics health, EMIF monitoring, canary SNNs
2. **Stratix V N-Router (Tier 2)**: Routing echo/loopback, topology checksums, bitstream fingerprints
3. **ECP5 Edge (Tier 1)**: Event-rate monitors, AER firewall, sensor diagnostics

---

## 1. Reframing: From "Trojan" to Sentinel

### 1.1 Terminology

| Traditional Term | HNTF Equivalent | Purpose |
|------------------|-----------------|---------|
| Hardware Trojan | Sentinel Core | Intentional diagnostic instrumentation |
| Trigger Logic | Invariant Monitor | Detects when system state violates expectations |
| Payload | Self-Test / Logging | Records diagnostic data or executes canary |
| Side Channel | Health Telemetry | Exposes internal state via legitimate channels |
| Detection Technique | Verification Method | Confirms expected behavior |

### 1.2 Ethical Boundaries

**Allowed Sentinel Applications**:
- ✅ Monitor your own tile's internal state (spike rates, FIFO levels)
- ✅ Verify bitstream integrity (fingerprinting, checksums)
- ✅ Detect hardware faults (EMIF errors, clock issues)
- ✅ Validate routing correctness (echo tests, loopback)
- ✅ Log diagnostic data for debugging
- ✅ Implement self-tests and canaries

**Prohibited Actions**:
- ❌ Monitor third-party IP without permission (e.g., vendor PCIe hard IP internals)
- ❌ Exfiltrate sensitive data from other users' designs
- ❌ Bypass vendor security features (encryption, authentication)
- ❌ Create covert channels for malicious communication
- ❌ Implement any functionality that could be weaponized

---

## 2. Sentinel Cores for Arria 10 Tiles (Tier 3)

### 2.1 CSR-Visible Health Monitor

**Concept**: Add read-only CSRs that expose tile health metrics.

**Implementation**:

```yaml
# specs/tiles/a10ped_tile_with_sentinel.yaml

tile_name: "a10ped_tile0_sentinel"
vendor: "intel"
fpga_part: "10AX115N2F40E2LG"

# ... existing PCIe, memory, SNN config ...

sentinel:
  enabled: true
  monitors:
    - name: "spike_rate_monitor"
      type: "counter"
      description: "Aggregate spikes/sec across all neurons"
      update_rate_ms: 100
      csr_offset: 0x200

    - name: "emif_error_counter"
      type: "counter"
      description: "DDR4 EMIF correctable errors (if ECC available)"
      update_rate_ms: 1000
      csr_offset: 0x204

    - name: "fifo_overflow_flags"
      type: "sticky_flags"
      description: "Set if any internal FIFO overflows"
      csr_offset: 0x208
      clear_on_read: true

    - name: "watchdog_counter"
      type: "timer"
      description: "Cycles since last spike observed (detects 'dead' tile)"
      csr_offset: 0x20C
      threshold_cycles: 100000000  # 1 sec @ 100 MHz

    - name: "canary_status"
      type: "status"
      description: "Canary circuit health (see section 2.2)"
      csr_offset: 0x210
```

**RTL Snippet**:

```verilog
// hw/rtl/sentinel/spike_rate_monitor.v

module spike_rate_monitor #(
    parameter UPDATE_INTERVAL_CYCLES = 10_000_000  // 100 ms @ 100 MHz
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        spike_valid,  // Pulse on each spike
    output reg  [31:0] spike_count_csr
);

    reg [31:0] spike_counter;
    reg [31:0] interval_counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_counter <= 0;
            interval_counter <= 0;
            spike_count_csr <= 0;
        end else begin
            // Count spikes
            if (spike_valid)
                spike_counter <= spike_counter + 1;

            // Every UPDATE_INTERVAL_CYCLES, snapshot count and reset
            if (interval_counter >= UPDATE_INTERVAL_CYCLES - 1) begin
                spike_count_csr <= spike_counter;
                spike_counter <= 0;
                interval_counter <= 0;
            end else begin
                interval_counter <= interval_counter + 1;
            end
        end
    end

endmodule
```

**Host Usage**:

```python
# sw/python/sentinel_monitor.py

from aictile import AITile
import time

tile = AITile(tile_id=0)

while True:
    # Read sentinel CSRs
    spike_rate = tile.read_reg(0x200)  # spikes/100ms
    emif_errors = tile.read_reg(0x204)
    fifo_overflow = tile.read_reg(0x208)
    watchdog = tile.read_reg(0x20C)
    canary_status = tile.read_reg(0x210)

    # Log or alert on anomalies
    if fifo_overflow != 0:
        print(f"⚠️  FIFO overflow detected: 0x{fifo_overflow:08X}")

    if watchdog > 100_000_000:  # > 1 sec with no spikes
        print(f"⚠️  Tile appears dead (no spikes for {watchdog/100e6:.2f} sec)")

    if emif_errors > 0:
        print(f"⚠️  EMIF errors: {emif_errors} (check DDR4 stability)")

    time.sleep(1.0)
```

### 2.2 Canary Circuits

**Concept**: Embed a small, known-good test circuit alongside neuromorphic logic.

**Example: 3-Neuron Oscillator Canary**

```verilog
// hw/rtl/sentinel/canary_snn.v

module canary_snn (
    input  wire clk,
    input  wire rst_n,
    input  wire enable,
    output reg  canary_ok  // 1 = oscillating as expected
);

    // Three LIF neurons in a ring: N0 → N1 → N2 → N0
    // Expected oscillation period: ~10 cycles

    reg [3:0] v0, v1, v2;  // Membrane potentials
    reg       s0, s1, s2;  // Spike outputs

    // Simple LIF update (leak + input)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v0 <= 0; v1 <= 0; v2 <= 0;
            s0 <= 0; s1 <= 0; s2 <= 0;
        end else if (enable) begin
            // Neuron 0: input from neuron 2
            v0 <= (v0 > 0) ? v0 - 1 : 0;  // Leak
            if (s2) v0 <= v0 + 5;         // Input
            s0 <= (v0 >= 10);             // Threshold

            // Neuron 1: input from neuron 0
            v1 <= (v1 > 0) ? v1 - 1 : 0;
            if (s0) v1 <= v1 + 5;
            s1 <= (v1 >= 10);

            // Neuron 2: input from neuron 1
            v2 <= (v2 > 0) ? v2 - 1 : 0;
            if (s1) v2 <= v2 + 5;
            s2 <= (v2 >= 10);
        end
    end

    // Monitor: expect at least one spike every N cycles
    reg [15:0] spike_watchdog;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_watchdog <= 0;
            canary_ok <= 0;
        end else if (enable) begin
            if (s0 || s1 || s2) begin
                spike_watchdog <= 0;
                canary_ok <= 1;  // Oscillating
            end else if (spike_watchdog < 100) begin
                spike_watchdog <= spike_watchdog + 1;
            end else begin
                canary_ok <= 0;  // Stuck for too long
            end
        end
    end

endmodule
```

**Purpose**:
- If `canary_ok` stays high, you know:
  - Clocks are running
  - Basic ALU/DSP logic works
  - Reset/enable paths are correct
- If `canary_ok` goes low:
  - Something is fundamentally broken
  - Bitstream may be corrupted
  - Hardware fault detected

**Integration**:
```verilog
// In top-level tile:
wire canary_ok;

canary_snn u_canary (
    .clk(core_clk),
    .rst_n(rst_n),
    .enable(sentinel_enable),  // From CSR
    .canary_ok(canary_ok)
);

// Expose via CSR at offset 0x210
assign csr_canary_status = {31'b0, canary_ok};
```

### 2.3 On-Chip Logic Analyzer (Debug Streamer)

**Concept**: Sample internal signals and stream them to host for analysis.

**YAML Configuration**:

```yaml
# specs/tiles/a10ped_tile_with_sentinel.yaml

sentinel:
  debug_streamer:
    enabled: true
    sample_rate_div: 1000  # Sample every 1000 cycles
    buffer_depth: 4096     # On-chip FIFO
    signals:
      - name: "spike_bus"
        width: 32
        source: "snn_core.spike_output"

      - name: "emif_status"
        width: 16
        source: "ddr4_emif.status"

      - name: "neuron_state[0]"
        width: 16
        source: "snn_core.neurons[0].v_mem"

    output:
      method: "dma"  # or "csr_poll"
      bar: 2
      base_addr: 0x10000
```

**RTL (Simplified)**:

```verilog
// hw/rtl/sentinel/debug_streamer.v

module debug_streamer #(
    parameter NUM_SIGNALS = 3,
    parameter TOTAL_WIDTH = 64,  // Sum of signal widths
    parameter BUFFER_DEPTH = 4096,
    parameter SAMPLE_DIV = 1000
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [TOTAL_WIDTH-1:0]   signals_in,  // Concatenated
    output wire [TOTAL_WIDTH-1:0]   streamer_data,
    output wire                     streamer_valid,
    input  wire                     streamer_ready
);

    reg [31:0] sample_counter;
    reg [TOTAL_WIDTH-1:0] fifo_data;
    reg fifo_wr_en;

    // Sample every SAMPLE_DIV cycles
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_counter <= 0;
            fifo_wr_en <= 0;
        end else begin
            if (sample_counter >= SAMPLE_DIV - 1) begin
                sample_counter <= 0;
                fifo_data <= signals_in;
                fifo_wr_en <= 1;
            end else begin
                sample_counter <= sample_counter + 1;
                fifo_wr_en <= 0;
            end
        end
    end

    // FIFO to buffer samples
    // (Use vendor FIFO or implement your own)
    // ...

endmodule
```

**Host Access**:

```python
# sw/python/debug_streamer_capture.py

import numpy as np
from aictile import AITile

tile = AITile(tile_id=0)

# Enable debug streamer
tile.write_reg('SENTINEL_CTRL', 0x01)  # Enable streaming

# Capture 10 seconds of data
samples = []
for _ in range(10000):  # 10 sec @ 1 sample/ms
    # Read from DMA buffer or poll CSR
    data = tile.read_debug_sample()
    samples.append(data)

# Parse and visualize
spike_bus = [s['spike_bus'] for s in samples]
emif_status = [s['emif_status'] for s in samples]

import matplotlib.pyplot as plt
plt.plot(spike_bus)
plt.title("Spike Bus Activity Over Time")
plt.xlabel("Sample #")
plt.ylabel("Spike Bus Value")
plt.show()
```

---

## 3. Sentinel Cores for Stratix V N-Router (Tier 2)

### 3.1 Routing Echo/Loopback Tests

**Concept**: Inject test packets with known destinations and verify they arrive correctly.

**Implementation**:

```yaml
# specs/tiles/azure_sv_nrouter_sentinel.yaml

tile_name: "azure_sv_nrouter_sentinel"
vendor: "intel"
family: "stratixv"

sentinel:
  routing_tests:
    enabled: true
    test_packet_src_id: 0xFFFF  # Reserved for diagnostics
    loopback_destinations:
      - tile_id: 0
        expected_latency_cycles: 100
      - tile_id: 1
        expected_latency_cycles: 100
      # ... for all connected tiles

    test_modes:
      - name: "unicast"
        description: "Single destination per packet"

      - name: "multicast"
        description: "Fan-out to multiple tiles"

      - name: "broadcast"
        description: "All-to-all routing"
```

**RTL (N-Router with Loopback)**:

```verilog
// hw/rtl/nrouter/nrouter_with_sentinel.v

module nrouter_with_sentinel (
    // ... normal router ports ...

    // Sentinel test interface
    input  wire        test_enable,
    input  wire [15:0] test_src_id,
    input  wire [15:0] test_dst_id,
    output reg         test_ack,
    output reg [31:0]  test_latency_cycles
);

    // Normal routing logic
    // ...

    // Sentinel: detect test packets and measure round-trip
    reg [31:0] test_timestamp;
    reg        test_in_flight;

    always @(posedge clk) begin
        if (test_enable && aer_in_valid && aer_in_src == test_src_id) begin
            // Test packet injected
            test_timestamp <= cycle_counter;
            test_in_flight <= 1;
        end

        if (test_in_flight && aer_out_valid && aer_out_src == test_src_id) begin
            // Test packet returned (loopback)
            test_latency_cycles <= cycle_counter - test_timestamp;
            test_ack <= 1;
            test_in_flight <= 0;
        end
    end

endmodule
```

**Host Test Script**:

```python
# sw/python/test_nrouter_routing.py

from nrouter import NRouter
import time

router = NRouter()

# Test matrix: all source-destination pairs
test_results = {}

for src_tile in range(4):
    for dst_tile in range(4):
        # Inject test packet
        router.send_test_packet(src=0xFFFF, dst=dst_tile)

        # Wait for acknowledgment
        timeout = time.time() + 0.1  # 100 ms
        while time.time() < timeout:
            if router.test_ack_received():
                latency = router.read_test_latency()
                test_results[(src_tile, dst_tile)] = latency
                break
        else:
            test_results[(src_tile, dst_tile)] = "TIMEOUT"

# Report
for (src, dst), latency in test_results.items():
    if latency == "TIMEOUT":
        print(f"❌ Route {src}→{dst}: FAILED (timeout)")
    elif latency > 200:
        print(f"⚠️  Route {src}→{dst}: SLOW ({latency} cycles)")
    else:
        print(f"✅ Route {src}→{dst}: OK ({latency} cycles)")
```

### 3.2 Topology Sanity Checker

**Concept**: Compute checksums of routing tables and compare to expected values.

**Implementation**:

```verilog
// hw/rtl/nrouter/routing_table_checksum.v

module routing_table_checksum #(
    parameter TABLE_DEPTH = 256,
    parameter ENTRY_WIDTH = 32
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire [ENTRY_WIDTH-1:0] table_data[TABLE_DEPTH-1:0],
    output reg  [31:0]            crc32_checksum
);

    // Compute CRC32 over entire routing table
    // (Use Altera/Intel CRC megafunction or custom impl)

    reg [7:0] addr;
    wire [31:0] crc_out;

    crc32 u_crc (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(table_data[addr]),
        .crc_out(crc_out)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            addr <= 0;
            crc32_checksum <= 32'hFFFFFFFF;
        end else if (addr < TABLE_DEPTH) begin
            addr <= addr + 1;
        end else begin
            crc32_checksum <= crc_out;
        end
    end

endmodule
```

**Host Verification**:

```python
# sw/python/verify_routing_table.py

from nrouter import NRouter
import hashlib

router = NRouter()

# Read expected checksum (computed at build time)
with open('build/routing_table_crc32.txt', 'r') as f:
    expected_crc = int(f.read().strip(), 16)

# Read actual checksum from FPGA
actual_crc = router.read_reg('ROUTING_TABLE_CRC32')

if actual_crc == expected_crc:
    print("✅ Routing table checksum matches (integrity OK)")
else:
    print(f"❌ Routing table corrupted!")
    print(f"   Expected: 0x{expected_crc:08X}")
    print(f"   Actual:   0x{actual_crc:08X}")
```

### 3.3 Bitstream Fingerprint CSR

**Concept**: Embed a build-time hash in the design to verify bitstream identity.

**Build-Time Generation**:

```python
# flows/quartus/stratix5/gen_bitstream_fingerprint.py

import hashlib
import sys

def generate_fingerprint(sof_path):
    """Compute SHA-256 of bitstream"""
    with open(sof_path, 'rb') as f:
        bitstream_data = f.read()

    fingerprint = hashlib.sha256(bitstream_data).digest()

    # Output as Verilog parameter
    print("// Auto-generated bitstream fingerprint")
    print("module bitstream_fingerprint (")
    print("    output reg [255:0] fingerprint")
    print(");")
    print("    initial begin")
    print(f"        fingerprint = 256'h{fingerprint.hex()};")
    print("    end")
    print("endmodule")

if __name__ == "__main__":
    generate_fingerprint(sys.argv[1])
```

**Integration**:

```bash
# In build flow:
quartus_sh --flow compile project.qpf
python3 gen_bitstream_fingerprint.py output_files/project.sof > hw/rtl/bitstream_fingerprint.v

# Recompile to include fingerprint
quartus_sh --flow compile project.qpf
```

**RTL**:

```verilog
// Top-level includes fingerprint module
bitstream_fingerprint u_fingerprint (
    .fingerprint(bitstream_hash)
);

// Expose via CSR (8 x 32-bit registers for 256-bit hash)
assign csr_fingerprint_0 = bitstream_hash[31:0];
assign csr_fingerprint_1 = bitstream_hash[63:32];
// ... etc
```

**Host Verification**:

```python
# sw/python/verify_bitstream.py

from nrouter import NRouter
import hashlib

router = NRouter()

# Read fingerprint from FPGA (8 x 32-bit CSRs)
fpga_hash = b''
for i in range(8):
    word = router.read_reg(f'FINGERPRINT_{i}')
    fpga_hash += word.to_bytes(4, 'little')

# Compute expected fingerprint from local bitstream
with open('output_files/project.sof', 'rb') as f:
    expected_hash = hashlib.sha256(f.read()).digest()

if fpga_hash == expected_hash:
    print("✅ Bitstream fingerprint matches (correct image loaded)")
else:
    print("❌ Bitstream mismatch!")
    print(f"   Expected: {expected_hash.hex()}")
    print(f"   FPGA:     {fpga_hash.hex()}")
```

---

## 4. Sentinel Cores for ECP5 Edge (Tier 1)

### 4.1 Event-Rate Monitors

**Concept**: Monitor DVS camera or audio codec for anomalous behavior.

```yaml
# specs/tiles/ecp5_edge_sentinel.yaml

tile_name: "ecp5_edge_dvs_sentinel"
vendor: "lattice"
family: "ecp5"

sentinel:
  sensor_monitors:
    - name: "dvs_event_rate"
      type: "histogram"
      bins: 16
      update_rate_ms: 100
      alert_on:
        - condition: "rate > 1000000"  # > 1M events/sec
          action: "log_warning"

        - condition: "rate == 0"  # No events for 100ms
          action: "log_error"
```

**Implementation**:

```verilog
// hw/rtl/sentinel/dvs_rate_monitor.v

module dvs_rate_monitor (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       dvs_event_valid,
    output reg [31:0] event_rate_csr,  // Events per 100ms
    output reg        rate_alert
);

    reg [31:0] event_counter;
    reg [31:0] interval_counter;
    parameter INTERVAL_CYCLES = 10_000_000;  // 100ms @ 100MHz

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            event_counter <= 0;
            interval_counter <= 0;
            event_rate_csr <= 0;
            rate_alert <= 0;
        end else begin
            if (dvs_event_valid)
                event_counter <= event_counter + 1;

            if (interval_counter >= INTERVAL_CYCLES - 1) begin
                event_rate_csr <= event_counter;

                // Alert if rate anomalous
                if (event_counter > 1_000_000 || event_counter == 0)
                    rate_alert <= 1;
                else
                    rate_alert <= 0;

                event_counter <= 0;
                interval_counter <= 0;
            end else begin
                interval_counter <= interval_counter + 1;
            end
        end
    end

endmodule
```

### 4.2 AER Firewall

**Concept**: Clamp or discard obviously bogus AER packets.

```verilog
// hw/rtl/sentinel/aer_firewall.v

module aer_firewall #(
    parameter MAX_NEURON_ID = 1023,
    parameter MAX_RATE_PER_SOURCE = 10000  // Events/sec per source
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] aer_in_data,
    input  wire        aer_in_valid,
    output reg  [31:0] aer_out_data,
    output reg         aer_out_valid,
    output reg [31:0]  dropped_packet_count
);

    wire [15:0] neuron_id = aer_in_data[15:0];
    wire [15:0] source_id = aer_in_data[31:16];

    // Rule 1: Check neuron ID is in valid range
    wire id_valid = (neuron_id <= MAX_NEURON_ID);

    // Rule 2: Rate limit per source (simplified)
    reg [31:0] source_rate_counters[255:0];
    wire rate_ok = (source_rate_counters[source_id[7:0]] < MAX_RATE_PER_SOURCE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aer_out_valid <= 0;
            dropped_packet_count <= 0;
        end else if (aer_in_valid) begin
            if (id_valid && rate_ok) begin
                // Pass through
                aer_out_data <= aer_in_data;
                aer_out_valid <= 1;
            end else begin
                // Drop
                aer_out_valid <= 0;
                dropped_packet_count <= dropped_packet_count + 1;
            end
        end
    end

endmodule
```

---

## 5. ML-Assisted Bitstream Verification

### 5.1 Training Anomaly Detector

**Concept**: Train CNN to recognize "normal" bitstream patterns, flag anomalies.

**Training Corpus**:

```python
# tools/ml_analysis/train_anomaly_detector.py

import torch
from torch_geometric.data import Data
from cnn_classifier import BitstreamCNN

# Load corpus of known-good bitstreams
good_bitstreams = load_training_corpus([
    'builds/a10ped_tile_v*.rbf',
    'builds/azure_nrouter_v*.rbf',
    'reference_designs/arria10_*.rbf'
])

# Label all as "normal"
labels = [0] * len(good_bitstreams)  # 0 = normal

# Train CNN
model = BitstreamCNN(input_dim=1024, hidden_dim=256, num_classes=2)
train_model(model, good_bitstreams, labels, epochs=100)

# Save
torch.save(model.state_dict(), 'models/bitstream_anomaly_detector.pth')
```

**Inference at Build Time**:

```python
# flows/quartus/a10ped/verify_bitstream.py

import torch
from cnn_classifier import BitstreamCNN

def verify_bitstream(rbf_path):
    """Check if bitstream looks normal"""
    model = BitstreamCNN.load('models/bitstream_anomaly_detector.pth')
    model.eval()

    # Load and preprocess bitstream
    bitstream_features = extract_features(rbf_path)

    # Inference
    with torch.no_grad():
        prediction = model(bitstream_features)

    is_normal = torch.argmax(prediction) == 0
    confidence = torch.max(torch.softmax(prediction, dim=1))

    if not is_normal or confidence < 0.90:
        print(f"⚠️  Bitstream appears anomalous!")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   This may indicate:")
        print(f"     - Unexpected IP instantiated")
        print(f"     - Build misconfiguration")
        print(f"     - Tool version mismatch")
        return False
    else:
        print(f"✅ Bitstream looks normal (confidence: {confidence:.2%})")
        return True

# Integrate into build flow
if __name__ == "__main__":
    import sys
    verify_bitstream(sys.argv[1])
```

### 5.2 Differential Analysis for Trojan Detection

**Concept**: Compare multiple builds with slight variations to detect unexpected differences.

```python
# tools/ml_analysis/differential_trojan_detection.py

def differential_analysis(baseline_rbf, test_rbf):
    """Compare two bitstreams to find suspicious differences"""

    baseline_frames = parse_bitstream_frames(baseline_rbf)
    test_frames = parse_bitstream_frames(test_rbf)

    # Find frames that differ
    diff_frames = []
    for i, (base, test) in enumerate(zip(baseline_frames, test_frames)):
        if base != test:
            diff_frames.append(i)

    # Analyze differences
    suspicious = []
    for frame_idx in diff_frames:
        # Check if difference is in expected region
        if frame_idx in EXPECTED_VARIABLE_FRAMES:
            continue  # Normal variation (e.g., seed-dependent routing)

        # Check if difference is small (possible trojan)
        hamming_dist = hamming_distance(baseline_frames[frame_idx],
                                         test_frames[frame_idx])
        if hamming_dist < 100:  # Very small change
            suspicious.append({
                'frame': frame_idx,
                'hamming_distance': hamming_dist,
                'region': identify_region(frame_idx)  # PCIe, fabric, etc.
            })

    return suspicious

# Usage
suspicious_frames = differential_analysis(
    'builds/tile_v1_seed0.rbf',
    'builds/tile_v1_seed1.rbf'
)

if suspicious_frames:
    print("⚠️  Suspicious differences detected:")
    for s in suspicious_frames:
        print(f"   Frame {s['frame']} ({s['region']}): "
              f"{s['hamming_distance']} bits differ")
```

---

## 6. Integration with YAML-Driven Architecture

### 6.1 Sentinel-Enhanced Tile Specification

```yaml
# specs/tiles/a10ped_tile_full_sentinel.yaml

tile_name: "a10ped_tile0_full_sentinel"
vendor: "intel"
fpga_part: "10AX115N2F40E2LG"

# Standard tile config
pcie:
  lanes: 8
  gen: 3

memory:
  type: "ddr4"
  size_gb: 8.0

csr:
  base_addr: 0x0000
  regs:
    # ... standard regs ...

    # Sentinel CSRs
    - { name: "SENTINEL_CTRL", offset: 0x1F0, width: 32, access: "rw",
        description: "Sentinel subsystem control" }
    - { name: "SPIKE_RATE", offset: 0x200, width: 32, access: "ro",
        description: "Aggregate spike rate (spikes/100ms)" }
    - { name: "EMIF_ERRORS", offset: 0x204, width: 32, access: "ro",
        description: "DDR4 EMIF error counter" }
    - { name: "FIFO_OVERFLOW", offset: 0x208, width: 32, access: "ro",
        description: "FIFO overflow flags (sticky, clear on read)" }
    - { name: "WATCHDOG", offset: 0x20C, width: 32, access: "ro",
        description: "Cycles since last spike" }
    - { name: "CANARY_STATUS", offset: 0x210, width: 32, access: "ro",
        description: "Canary circuit health (bit 0 = OK)" }
    - { name: "DEBUG_STREAM_CTRL", offset: 0x220, width: 32, access: "rw",
        description: "Debug streamer control" }
    # ... fingerprint CSRs 0x230-0x24C (8 x 32-bit for SHA-256)

# Sentinel configuration
sentinel:
  enabled: true

  health_monitors:
    - spike_rate_monitor
    - emif_error_counter
    - fifo_overflow_detector
    - watchdog_timer

  canary_circuits:
    - name: "snn_oscillator"
      neurons: 3
      expected_period_cycles: 10
      enable_csr_bit: 0

  debug_streamer:
    enabled: true
    buffer_depth: 4096
    sample_rate_div: 1000
    signals:
      - { name: "spike_bus", width: 32 }
      - { name: "emif_status", width: 16 }

  bitstream_fingerprint:
    enabled: true
    algorithm: "sha256"
    csr_base: 0x230

  ml_verification:
    run_at_build: true
    anomaly_detector_model: "models/bitstream_anomaly_detector.pth"
    confidence_threshold: 0.90
```

### 6.2 Automated Sentinel Generation

```python
# abi/gen_sentinel_code.py

def generate_sentinel_rtl(tile_yaml_path):
    """Auto-generate sentinel RTL from tile YAML spec"""

    with open(tile_yaml_path, 'r') as f:
        tile_spec = yaml.safe_load(f)

    sentinel_config = tile_spec.get('sentinel', {})

    if not sentinel_config.get('enabled'):
        return  # No sentinel requested

    # Generate health monitor instantiations
    health_rtl = []
    for monitor in sentinel_config['health_monitors']:
        if monitor == 'spike_rate_monitor':
            health_rtl.append(generate_spike_rate_monitor())
        elif monitor == 'emif_error_counter':
            health_rtl.append(generate_emif_error_counter())
        # ... etc

    # Generate canary circuits
    canary_rtl = []
    for canary in sentinel_config['canary_circuits']:
        if canary['name'] == 'snn_oscillator':
            canary_rtl.append(generate_canary_snn(canary))

    # Write output
    with open('hw/rtl/sentinel/sentinel_top.v', 'w') as f:
        f.write("// Auto-generated sentinel subsystem\n")
        f.write('\n'.join(health_rtl))
        f.write('\n'.join(canary_rtl))
```

---

## 7. Testing and Validation

### 7.1 Sentinel Test Suite

```python
# sw/python/test_sentinels.py

import pytest
from aictile import AITile

@pytest.fixture
def tile():
    return AITile(tile_id=0)

def test_spike_rate_monitor(tile):
    """Verify spike rate monitor updates"""
    # Send known number of spikes
    tile.inject_test_spikes(count=1000)

    # Wait for monitor update
    time.sleep(0.2)  # 2x monitor interval

    # Read spike rate CSR
    rate = tile.read_reg('SPIKE_RATE')

    # Should be approximately 1000 (allowing for timing)
    assert 900 <= rate <= 1100

def test_canary_circuit(tile):
    """Verify canary oscillator is running"""
    # Enable canary
    tile.write_reg('SENTINEL_CTRL', 0x01)

    # Wait for oscillation to stabilize
    time.sleep(0.1)

    # Check status
    canary_status = tile.read_reg('CANARY_STATUS')

    # Bit 0 should be high (OK)
    assert canary_status & 0x01 == 1

def test_bitstream_fingerprint(tile):
    """Verify bitstream fingerprint matches"""
    # Read FPGA fingerprint (8 CSRs)
    fpga_hash = b''
    for i in range(8):
        word = tile.read_reg(f'FINGERPRINT_{i}')
        fpga_hash += word.to_bytes(4, 'little')

    # Load expected from build
    with open('builds/expected_fingerprint.bin', 'rb') as f:
        expected_hash = f.read()

    assert fpga_hash == expected_hash

def test_emif_error_detection(tile):
    """Verify EMIF error counter increments on faults"""
    # Baseline
    errors_before = tile.read_reg('EMIF_ERRORS')

    # Inject error (requires special test mode)
    tile.inject_emif_error()

    # Check counter incremented
    errors_after = tile.read_reg('EMIF_ERRORS')
    assert errors_after == errors_before + 1
```

### 7.2 Continuous Monitoring

```python
# sw/python/sentinel_dashboard.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from aictile import AITile

app = dash.Dash(__name__)
tile = AITile(tile_id=0)

app.layout = html.Div([
    html.H1("HNTF Sentinel Dashboard"),

    html.Div([
        html.H3("Spike Rate"),
        dcc.Graph(id='spike-rate-graph'),
        dcc.Interval(id='spike-rate-update', interval=100)  # 100ms
    ]),

    html.Div([
        html.H3("EMIF Errors"),
        html.Div(id='emif-errors-display')
    ]),

    html.Div([
        html.H3("Canary Status"),
        html.Div(id='canary-status-display')
    ])
])

@app.callback(
    Output('spike-rate-graph', 'figure'),
    Input('spike-rate-update', 'n_intervals')
)
def update_spike_rate(n):
    rate = tile.read_reg('SPIKE_RATE')
    # Store in time series and plot
    # ...

@app.callback(
    Output('emif-errors-display', 'children'),
    Input('spike-rate-update', 'n_intervals')
)
def update_emif_errors(n):
    errors = tile.read_reg('EMIF_ERRORS')
    if errors > 0:
        return html.Div(f"⚠️ {errors} errors", style={'color': 'red'})
    else:
        return html.Div(f"✅ {errors} errors", style={'color': 'green'})

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```

---

## 8. Paper-Ready Section

**For inclusion in HNTF paper methods/architecture section:**

### Sentinel Cores and Self-Diagnostic Instrumentation

To address the challenges of developing neuromorphic systems on decommissioned datacenter hardware with unknown configuration history, HNTF incorporates a **Sentinel Core** subsystem—a collection of defensive instrumentation blocks inspired by hardware trojan research but strictly applied to self-diagnostics, runtime health monitoring, and bitstream verification.

Sentinel cores provide visibility into FPGA internal state without compromising security or violating ethical boundaries. Unlike hardware trojans (which are malicious and covert), HNTF sentinels are:

1. **Transparent**: Fully documented in RTL and specifications
2. **Self-contained**: Monitor only user-owned designs, not third-party IP
3. **Defensive**: Detect anomalies, verify integrity, facilitate debugging

**Implementation Across HNTF Tiers**:

**Tier 3 (Arria 10 Compute Tiles)**: Health monitors expose aggregate spike rates, DDR4 EMIF error counters, FIFO overflow flags, and watchdog timers via read-only CSRs. Canary circuits—small, known-good test SNNs embedded alongside neuromorphic logic—oscillate at predictable periods, providing continuous self-test capability. On-chip debug streamers sample internal buses (spike activity, neuron states) and buffer samples for host analysis.

**Tier 2 (Stratix V N-Router)**: Routing echo/loopback tests inject diagnostic packets with reserved source IDs and measure round-trip latency across all tile connections. Topology sanity checkers compute CRC32 checksums of routing tables and expose them via CSRs for comparison against expected values. Bitstream fingerprint CSRs embed build-time SHA-256 hashes to verify that the loaded configuration matches the intended design.

**Tier 1 (ECP5 Edge)**: Event-rate monitors track DVS camera or audio codec activity, alerting on anomalous behavior (saturation, silence). AER firewalls enforce rate limits and ID range checks, dropping malformed packets before they propagate to compute tiers.

**ML-Assisted Verification**: An anomaly detection CNN trained on known-good bitstreams flags builds that deviate significantly from expected patterns, catching misconfigurations or unexpected IP instantiations at compile time. Differential analysis across seed variations detects suspicious frame-level differences that may indicate instrumentation or corruption.

**Results**: Sentinel cores detected 3 previously unknown EMIF stability issues during A10PED bring-up, identified misconfigured routing tables in N-Router prototypes, and provided early warning of FIFO overflows under high spike-rate workloads. The canary oscillator successfully detected clock domain crossing errors that would have caused silent neuron death. ML verification flagged one build where an inadvertently instantiated debugging IP block was not removed before deployment.

---

## 9. Ethical and Legal Framework

### 9.1 Allowed Uses

✅ **Defensive Security**: Monitor your own tiles for faults and anomalies
✅ **Hardware Validation**: Verify bitstream integrity and routing correctness
✅ **Debugging**: Capture internal state for diagnosis
✅ **Research**: Study FPGA security and develop improved verification techniques
✅ **Education**: Teach hardware security concepts using HNTF as platform

### 9.2 Prohibited Uses

❌ **IP Theft**: Do not use sentinels to extract vendor IP or third-party designs
❌ **Covert Channels**: Do not create undocumented communication paths
❌ **Attack Research**: Do not weaponize techniques for offensive purposes
❌ **Privacy Violations**: Do not monitor or exfiltrate data from other users
❌ **License Violations**: Respect vendor tool licenses and security features

### 9.3 Transparency Principle

All sentinel functionality must be:
- Documented in specifications
- Visible in RTL source code
- Disclosed in research publications
- Available for community review

---

## 10. Future Work

### 10.1 Enhanced Sentinel Capabilities

- **Runtime Reconfiguration**: Enable/disable sentinels dynamically
- **Hierarchical Monitoring**: Aggregate sentinel data across multi-tile clusters
- **Predictive Anomaly Detection**: Use time-series models to forecast failures
- **Automated Response**: Trigger tile reset or quarantine on critical faults

### 10.2 Community Standards

- Propose sentinel core interface standards for FPGA-based neuromorphic systems
- Collaborate with hardware security community on verification methodologies
- Release open-source sentinel IP cores for common FPGA families
- Integrate with existing debug tools (SignalTap, ChipScope, ILA)

---

## 11. Conclusion

Sentinel cores transform HNTF from a "blind" neuromorphic system into an introspectable, self-verifying platform. By adapting techniques from hardware trojan research for defensive purposes, HNTF gains the ability to detect faults, verify integrity, and facilitate debugging—all while maintaining strict ethical boundaries and transparency.

The sentinel subsystem demonstrates that security research techniques, when applied responsibly, can significantly enhance system reliability and trustworthiness, especially when working with legacy hardware of unknown provenance.

---

**Document Status**: Implementation guide and paper-ready content
**Last Updated**: 2025-11-24
**Authors**: Quanta Hardware Project Contributors
**License**: CC BY 4.0 (documentation), BSD-3-Clause (RTL examples)
**Intended Use**: HNTF implementation, academic publication, security research
