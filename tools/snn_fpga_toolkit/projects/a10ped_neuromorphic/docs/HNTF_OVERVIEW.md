# Heterogeneous Neuromorphic Tile Fabric (HNTF) – Overview

**One-Page Summary for README / Slides / Grant Pitches**

---

## Concept

The **Heterogeneous Neuromorphic Tile Fabric (HNTF)** is a modular architecture for building low-cost, high-performance neuromorphic systems from **decommissioned datacenter hardware**. Instead of relying on a single monolithic accelerator, HNTF composes multiple specialized FPGA-based "tiles" over a high-speed network fabric, each optimized for a distinct role in the neuromorphic pipeline:

* Low-power edge pre-processing,
* High-bandwidth spike routing and control,
* Dense synaptic computation and neuron dynamics.

The fabric targets **spiking neural networks (SNNs)** and event-driven workloads, but the tile abstraction is general enough to support other dataflows.

---

## Architectural Tiers

### Tier 1 – Edge Pre-Processing (Open ECP5 Tier)

* **Hardware**: Lattice **ECP5** FPGA boards with fully open-source tools (Yosys + nextpnr + Trellis).
* **Role**: Interface with sensors (cameras, audio, etc.), perform basic filtering and thresholding, and convert continuous data streams into **Address-Event Representation (AER)** spikes.
* **Goal**: Offload low-level I/O and pre-processing from the main compute tiles, and provide a reproducible, vendor-independent edge tier.

### Tier 2 – Routing and Control (Stratix V N-Router)

* **Hardware**: Datacenter-grade **Stratix V** card (e.g., Azure X930613-001).
* **Role**: Serve as a centralized **Neuromorphic Router (N-Router)**:

  * Terminates a PCIe link to the host,
  * Maintains routing tables and fan-out logic for spikes,
  * Bridges between the host and the compute tiles via QSFP-based links into an InfiniBan/Ethernet switch fabric.
* **Goal**: Separate global spike routing and cluster control from local synaptic computation, enabling flexible topologies and dynamic reconfiguration.

### Tier 3 – Computational Core (Arria 10 Neuromorphic Tiles)

* **Hardware**: **BittWare A10PED** (dual Arria 10 GX) and similar Arria 10–based boards (e.g., Gidel HawkEye).
* **Role**: Implement **Neuromorphic Tile Shells** that:

  * Expose a common CSR/streaming interface,
  * Use on-board DDR4/HMC for synaptic weights and neuron state,
  * Run SNN dynamics (LIF, Izhikevich, etc.) on Arria 10 DSP blocks.
* **Goal**: Concentrate compute and memory bandwidth where it matters—synaptic updates and neuron integration—while leaving routing concerns to Tier 2.

All tiles are connected via **commodity Mellanox NICs and a used InfiniBand switch**, forming a high-speed, low-latency fabric that can be scaled by adding more tiles or boards.

---

## Key Ideas and Contributions

### 1. Tile Abstraction Across Heterogeneous FPGAs

A uniform tile interface (CSRs + streaming AER links + memory mapping) is defined across very different devices:

* ECP5 (open tools, edge I/O),
* Stratix V (routing, host connectivity),
* Arria 10 (compute and memory).

This allows incremental addition or replacement of boards without rewriting the entire stack.

### 2. Reuse of Decommissioned Datacenter Hardware

The design explicitly targets low-cost, secondary-market boards (A10PED, Azure Stratix V, etc.), turning "discarded" accelerators into a neuromorphic cluster rather than relying on expensive, bespoke neuromorphic ASICs.

### 3. Hybrid Open/Proprietary Toolchain Strategy

* Intel-only tasks (PCIe hard IP, EMIF) are handled via Quartus.
* Neuromorphic logic, tile interfaces, and edge pre-processing are developed in **open tools** (Yosys, Verilator, nextpnr).

Quartus is treated as a thin backend, reducing vendor lock-in and improving portability.

### 4. ML-Assisted Bitstream Analysis as a Supporting Tool

ML models (CNN-based region classifiers, clustering over configuration frames) are used to:

* Identify likely PCIe/EMIF/fabric regions on Arria 10 and Stratix V nets,
* Guide placement and shell design on partially documented boards,
* Provide an additional sanity check alongside conventional RE and timing analysis.

ML does not replace vendor tools but accelerates understanding of legacy architectures.

### 5. End-to-End Neuromorphic Path

The HNTF defines a complete pipeline from sensors → events → routing → synaptic computation → returned events/outputs, with clear separation of roles between tiers. This structure supports ongoing evolution:

* New neuron models or learning rules implemented as tile variants,
* Alternative routers or fabrics (e.g., Ethernet-only) swapped into Tier 2,
* Different edge boards or sensors plugged into Tier 1.

---

## Why It Matters

### Cost

Leverages inexpensive, decommissioned FPGAs instead of bespoke neuromorphic ASICs.

**Price Comparison**:
- **BittWare A10PED** (dual Arria 10 GX1150, 32GB DDR4, HMC): ~$200-500 (used)
- **Azure Stratix V board** (X930613-001): ~$100-300 (decommissioned)
- **Lattice ECP5 boards**: ~$50-150 (new, open tools)
- **Mellanox ConnectX-3** NICs + InfiniBand switch: ~$200-400 (used)
- **Total system cost**: $1,000-2,000 for 4-8 tiles

Compare to:
- **Intel Loihi 2**: Not commercially available, research-only
- **IBM TrueNorth**: Discontinued
- **BrainScaleS-2**: Academic platform, not for sale
- **SpiNNaker 2**: Limited availability, high cost
- **GPU-based SNN**: $1,000+ for capable GPU, high power consumption

### Flexibility

Heterogeneous tiles can be reconfigured and repurposed as algorithms and workloads evolve.

**Examples**:
- Swap neuron models (LIF → Izhikevich → custom) without changing routing tier
- Add plasticity (STDP) to specific tiles while others remain feedforward
- Scale compute by adding more Tier 3 tiles
- Upgrade edge tier (ECP5 → Artix-7) without affecting compute

### Reproducibility

Open tools on the edge and unified tile abstractions make the system easier to study, extend, and share.

**Reproducibility Features**:
- YAML specifications version-controlled in Git
- Automated build scripts (no manual Quartus GUI)
- Report parsers generate structured JSON
- Docker containers for consistent environments
- Complete documentation (12+ technical documents)

### Research Value

Provides a practical platform for experimenting with neuromorphic computation, routing policies, and hybrid ML/RE techniques at meaningful scales.

**Research Applications**:
- Large-scale SNN simulations (10K+ neurons per tile, multi-tile networks)
- Novel neuron models and plasticity rules
- Neuromorphic algorithm development
- Energy efficiency comparisons (FPGA vs. GPU vs. ASIC)
- Hybrid learning (combining SNNs with traditional ML)
- Real-time sensory processing (vision, audio)

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Host System (x86)                         │
│  • SNN model specification and training                         │
│  • Monitoring, logging, control plane                           │
│  • PCIe Gen3 x8 link to Tier 2 router                          │
└────────────────────────┬────────────────────────────────────────┘
                         │ PCIe Gen3 x8
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Tier 2: Neuromorphic Router (Stratix V)            │
│  • Azure X930613-001 or similar                                 │
│  • PCIe endpoint + QSFP+ links                                  │
│  • Spike routing tables and fan-out                             │
│  • Cluster coordination and control                             │
└────────────────────────┬────────────────────────────────────────┘
                         │ QSFP+ @ 40 Gbps
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            InfiniBand/Ethernet Switch (Mellanox)                │
│  • Low-latency spike fabric (< 1 μs)                           │
│  • Connects all Tier 3 compute tiles                            │
│  • Scalable: add more tiles as needed                           │
└────┬────────────┬────────────┬────────────┬─────────────────────┘
     │            │            │            │
     ▼            ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Tile 0  │  │ Tile 1  │  │ Tile 2  │  │ Tile N  │  Tier 3: Compute
│ A10 GX  │  │ A10 GX  │  │ A10 GX  │  │ A10 GX  │  (Arria 10)
│ 8GB DDR4│  │ 8GB DDR4│  │ 8GB DDR4│  │ 8GB DDR4│
│ 512 LIF │  │ 1K Izh  │  │ Custom  │  │ Learning│
│ neurons │  │ neurons │  │ neurons │  │ + STDP  │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
     ▲            ▲            ▲            ▲
     │            │            │            │
     └────────────┴────────────┴────────────┘
              AER event streams from Tier 1 (ECP5 edge)
```

---

## Technical Specifications

### Tier 1: ECP5 Edge Tier

| Specification | Value |
|---------------|-------|
| **FPGA** | Lattice ECP5-85F |
| **Logic Elements** | 85K |
| **Memory** | 3.7 Mbit BRAM |
| **I/O** | GPIO, LVDS, HDMI |
| **Tools** | Yosys + nextpnr + Trellis (fully open) |
| **Typical Use** | DVS camera interface, audio codec, sensor pre-processing |

### Tier 2: Stratix V Router

| Specification | Value |
|---------------|-------|
| **FPGA** | Intel Stratix V GS (5SGSED8) |
| **Logic Elements** | ~550K ALMs (~1.1M LEs) |
| **Memory** | 77 Mbit M20K blocks |
| **Transceivers** | Dual 10G SFP+, QSFP+ |
| **PCIe** | Gen3 x8 endpoint |
| **Tools** | Quartus Prime Pro (proprietary) |
| **Typical Use** | Host interface, spike routing, cluster control |

### Tier 3: Arria 10 Compute Tiles

| Specification | Value (per FPGA) |
|---------------|------------------|
| **FPGA** | Intel Arria 10 GX1150 |
| **Logic Elements** | 1.15M ALMs (~2.3M LEs) |
| **DSP Blocks** | 1,518 (IEEE 754 FP capable) |
| **Memory** | 32 GB DDR4-2400 + 2 GB HMC |
| **Bandwidth** | ~60 GB/s combined |
| **Transceivers** | 96 @ 28.3 Gbps, QSFP28 |
| **Tools** | Quartus Prime Pro (proprietary) |
| **Typical Use** | SNN neuron dynamics, synaptic updates, state management |

### Network Fabric

| Specification | Value |
|---------------|-------|
| **Interconnect** | InfiniBand FDR or 40G Ethernet |
| **Switch** | Mellanox SX6036 (36-port) or similar |
| **Latency** | < 1 μs hop latency |
| **Bandwidth** | 40 Gbps per link |
| **Topology** | Star (switch-centric) or mesh (future) |

---

## Performance Characteristics

### Compute Capacity (per A10 tile)

- **Neurons**: 512-2048 depending on model complexity
- **Synapses**: 100K-1M depending on connectivity and precision
- **Spike Rate**: Up to 100M events/sec aggregate throughput
- **Latency**: < 10 μs neuron update latency
- **Power**: 25-40W per FPGA (typical)

### Scaling

- **Single Tile**: ~1K neurons, suitable for proof-of-concept
- **4-Tile System**: ~4-8K neurons, small-scale experiments
- **16-Tile System**: ~16-32K neurons, medium-scale networks
- **64-Tile System**: ~64-128K neurons, approaching cortical column scale

Compare to biological scale:
- **Mouse cortex**: ~16M neurons, ~10^11 synapses
- **Human cortex**: ~16B neurons, ~10^14 synapses
- **HNTF 64-tile**: ~100K neurons, ~10M synapses (0.0006% of mouse cortex)

### Energy Efficiency

**Preliminary estimates** (to be validated with real hardware):
- **Energy per spike**: ~10-100 pJ (FPGA overhead)
- **System power**: 100-500W for 16-tile system
- **Comparison**:
  - Biological neurons: ~1-10 pJ/spike
  - Loihi 2: ~10-50 pJ/spike (estimated)
  - GPU (RTX 3090): ~500-1000 pJ/spike equivalent (high power, parallel processing)

**Note**: FPGA-based neuromorphic computing trades energy efficiency for flexibility and reconfigurability.

---

## Current Status and Roadmap

### Completed ✅

- [x] HNTF architecture definition
- [x] YAML-driven specification framework
- [x] ML-assisted bitstream analysis methodology
- [x] GNN logic optimization strategy
- [x] Complete documentation suite (12+ documents)
- [x] Build automation (Yosys, Quartus scripts)
- [x] Report parsing tools (timing, utilization)
- [x] Tile and board YAML schemas

### In Progress 🔄

- [ ] A10PED tile implementation (RTL + bring-up)
- [ ] Azure Stratix V router implementation
- [ ] ECP5 edge tier reference design
- [ ] Software stack (drivers, API, runtime)
- [ ] ML model training (CNN for RE, GNN for optimization)

### Planned 📋

- [ ] Hardware validation on real boards
- [ ] SNN benchmark suite
- [ ] Multi-tile network demonstrations
- [ ] Energy efficiency measurements
- [ ] Open-source release (models, datasets, tools)

---

## Getting Started

### For Researchers

**Explore the architecture**:
```bash
git clone https://github.com/user/quanta-hw
cd quanta-hw
cat docs/HNTF_OVERVIEW.md  # This document
cat docs/ML_ECOSYSTEM_OVERVIEW.md  # Full technical details
```

**Read the papers**:
- `docs/ML_ASSISTED_BITSTREAM_ANALYSIS.md` - ML for legacy boards
- `docs/GNN_LOGIC_OPTIMIZATION.md` - GNN for synthesis
- `docs/LIMITATIONS_AND_FUTURE_WORK.md` - Roadmap and challenges

### For Hardware Engineers

**Build a tile**:
```bash
# Validate your board specification
python3 tools/validate/check_tile_spec.py specs/tiles/a10ped_tile.yaml

# Generate constraints
python3 flows/quartus/a10ped/gen_qsf.py \
  specs/boards/a10ped_board.yaml \
  out/a10ped/build/project.qsf

# Build with Quartus
./flows/quartus/a10ped/build_tile.sh
```

### For Software Developers

**Use the API** (coming soon):
```python
from hntf import Cluster, LIFNeuron

# Initialize cluster
cluster = Cluster.discover()  # Find all tiles

# Create SNN model
model = cluster.create_network()
layer1 = model.add_layer(LIFNeuron, count=512, tile=cluster.tiles[0])
layer2 = model.add_layer(LIFNeuron, count=256, tile=cluster.tiles[1])
model.connect(layer1, layer2, weights=...)

# Run inference
spikes_in = load_dvs_events("input.aedat")
spikes_out = model.run(spikes_in, duration=1.0)
```

---

## Community and Support

### Resources

- **Documentation**: https://github.com/user/quanta-hw/tree/main/docs
- **Issue Tracker**: https://github.com/user/quanta-hw/issues
- **Discussions**: https://github.com/user/quanta-hw/discussions
- **Papers**: `docs/` directory contains paper-ready sections

### Contributing

We welcome contributions in all forms:

- **Hardware**: Validate tile designs, add new board support
- **Software**: Implement drivers, APIs, tools
- **ML**: Train models, improve datasets
- **Documentation**: Tutorials, examples, case studies
- **Testing**: Report bugs, benchmark performance

See `CONTRIBUTING.md` for guidelines.

### Citation

If you use HNTF in your research, please cite:

```bibtex
@misc{hntf_2025,
  title={Heterogeneous Neuromorphic Tile Fabric:
         Building SNN Systems from Decommissioned Datacenter Hardware},
  author={Quanta Hardware Project Contributors},
  year={2025},
  howpublished={\url{https://github.com/user/quanta-hw}},
  note={Modular architecture for low-cost neuromorphic computing
        using Intel Arria 10, Stratix V, and Lattice ECP5 FPGAs}
}
```

---

## Contact

**Project Lead**: [Your Name/Organization]
**Email**: [contact@example.com]
**Website**: [https://quanta-hw.example.com]

---

**Document Version**: 1.0
**Last Updated**: 2025-11-24
**License**: CC BY 4.0 (documentation)
**Intended Use**: README, slides, grant pitches, project overviews
