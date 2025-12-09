# Limitations and Future Work

**Paper-Ready Section for Heterogeneous Neuromorphic Tile Fabric (HNTF)**

This document outlines the current limitations of the HNTF architecture and proposes concrete directions for future development. This content can be integrated directly into research papers as a concluding section.

---

## 5.x+1 Limitations and Future Work

Despite its practicality and low cost, the Heterogeneous Neuromorphic Tile Fabric (HNTF) has several important limitations that shape both its current scope and future evolution.

### Toolchain and Vendor Dependence

A primary limitation is the continued reliance on **proprietary compilation toolchains** for Intel devices. Arria 10 and Stratix V–based boards (A10PED, Azure card, HawkEye) still require Intel Quartus for synthesis, place-and-route, and bitstream generation, particularly when instantiating PCIe hard IPs, EMIF/DDR controllers, and high-speed transceivers. Even when front-end logic is developed with open tools (Yosys, Verilator), the final realization remains tied to a closed backend.

This dependence constrains long-term reproducibility, limits portability to new Intel families, and complicates deployment in environments where proprietary tools cannot be installed. It also means that any major change in licensing or support from the vendor could impact the platform.

**Future work.** Extend open-source flows beyond Cyclone V (e.g., Mistral + nextpnr) toward Arria 10–like architectures, and investigate the feasibility of an intermediate architecture representation that can target multiple vendor backends (Intel, Xilinx, Lattice) with minimal changes to neuromorphic tile RTL.

### Incomplete Architectural Knowledge and ML-RE Limits

The HNTF intentionally leverages **ML-assisted bitstream analysis** only as a supporting tool. While CNN-based region classification and clustering accelerate the identification of configuration regions (e.g., PCIe/EMIF vs. fabric), they do not provide a complete architectural database for Arria 10 or Stratix V, nor do they offer guarantees about coverage or correctness.

Furthermore, bitstream encryption and authentication features, if enabled, close off the configuration space entirely. Even when bitstreams are visible, ML-based inferences are probabilistic and must be validated by conventional methods.

**Future work.** Combine ML-based region proposals with more systematic reverse-engineering campaigns on supported dev boards to gradually refine an architecture-level model of Arria 10/Stratix V fabrics, and evaluate how far such models can be pushed without violating security or licensing constraints.

### Hardware Constraints and Practical Integration

The datacenter-class boards used in HNTF are **thermally and electrically non-trivial**: they were originally designed for tightly controlled server environments. When repurposed in desktop or lab settings, issues such as power delivery, cooling, PCIe slot topology, and mechanical clearance become significant.

The Mellanox-based fabric and QSFP interconnects also introduce complexity in cabling, switch configuration, and link bring-up, especially when mixing legacy and consumer hardware.

**Future work.** Develop a set of reference "integration recipes" (power, cooling, PCIe layout, switch configuration) for common host configurations, and explore small-form-factor or lower-power FPGA alternatives that can serve as drop-in neuromorphic tiles with fewer environmental constraints.

### Software Stack Maturity and Usability

While the HNTF design emphasizes open drivers (e.g., altera-pcie–derived kernels) and a uniform CSR/streaming ABI, the overall software stack—runtime, orchestration, monitoring—is still at a **research prototype** level. Tooling for:

* Tile discovery and enumeration,
* Dynamic mapping of SNN models to tiles,
* Monitoring, logging, and debugging across tiers,

is minimal, and the learning curve for new users remains high.

**Future work.** Build a higher-level runtime and control plane that exposes HNTF as:

* A logical "neuromorphic cluster" with a clean API (e.g., gRPC/REST),
* A target backend for existing SNN frameworks or simulators,
* A set of reproducible Docker/Ansible configurations for deployment.

### Algorithmic Scope and Neuromorphic Diversity

Current design focuses on **feedforward and moderately recurrent SNNs** using relatively simple neuron models (e.g., LIF/Izhikevich). Plasticity mechanisms (STDP, metaplasticity, neuromodulation) and more exotic topologies are not yet first-class citizens in the tile shells.

**Future work.** Extend tile microarchitectures to support:

* On-chip plasticity engines with programmable learning rules,
* Hybrid analog–digital or mixed-precision schemes,
* Topology-aware routing policies in the N-Router (e.g., small-world or hierarchical connectivity patterns).

Together, these limitations and directions define a clear roadmap: reduce vendor lock-in, deepen architectural understanding, harden the physical and software integration, and enrich the neuromorphic capabilities of each tile.

---

## Detailed Future Work Breakdown

### Short-Term (6-12 months)

**Toolchain**:
- [ ] Implement Quartus-as-a-service wrapper with Docker containers
- [ ] Create YAML templates for common tile configurations
- [ ] Develop automated regression testing for bitstream generation

**ML-RE**:
- [ ] Build training corpus of 100+ Arria 10 reference designs
- [ ] Train baseline CNN classifiers for region classification
- [ ] Validate ML predictions on A10PED and HawkEye boards

**Hardware Integration**:
- [ ] Document power/cooling requirements for common motherboards
- [ ] Create reference PCIe topology diagrams
- [ ] Test board combinations in various chassis configurations

**Software Stack**:
- [ ] Implement tile discovery service (PCIe enumeration + identification)
- [ ] Create Python API for multi-tile coordination
- [ ] Develop monitoring dashboard (web-based)

**Neuromorphic Algorithms**:
- [ ] Implement basic STDP learning rule in tile RTL
- [ ] Benchmark LIF vs. Izhikevich neuron performance
- [ ] Create SNN model zoo with example networks

### Medium-Term (1-2 years)

**Toolchain**:
- [ ] Contribute Arria 10 support to Mistral/nextpnr project
- [ ] Develop intermediate representation for cross-vendor tiles
- [ ] Create fully open alternative for simple tile configurations

**ML-RE**:
- [ ] Extend coverage to Azure Stratix V boards
- [ ] Implement graph-based routing inference (GNN)
- [ ] Release open dataset of paired HDL/bitstream/reports

**Hardware Integration**:
- [ ] Design custom carrier boards for salvaged FPGA modules
- [ ] Investigate low-power alternatives (Artix-7, Cyclone 10)
- [ ] Create reference designs for small form factor builds

**Software Stack**:
- [ ] Implement gRPC/REST API for cluster management
- [ ] Integrate with existing SNN frameworks (Brian2, NEST, etc.)
- [ ] Create Docker/Ansible deployment automation

**Neuromorphic Algorithms**:
- [ ] Add support for complex plasticity (metaplasticity, neuromodulation)
- [ ] Implement topology-aware routing in N-Router
- [ ] Explore hybrid analog-digital neuron implementations

### Long-Term (2-5 years)

**Toolchain**:
- [ ] Achieve full open-source flow for at least one tile type
- [ ] Establish HNTF as target backend for neuromorphic compilers
- [ ] Contribute to standardization efforts (if community emerges)

**ML-RE**:
- [ ] Develop complete architectural database for Arria 10/Stratix V
- [ ] Implement automated bitstream-to-HDL decompilation (research challenge)
- [ ] Federated learning across multiple research groups

**Hardware Integration**:
- [ ] Design HNTF-specific boards optimized for neuromorphic workloads
- [ ] Explore alternative interconnects (PCIe 4.0/5.0, CXL)
- [ ] Investigate chiplet-based neuromorphic designs

**Software Stack**:
- [ ] Full-featured neuromorphic cluster orchestration platform
- [ ] Integration with cloud deployment (Kubernetes, etc.)
- [ ] Commercial-grade monitoring and debugging tools

**Neuromorphic Algorithms**:
- [ ] Support for large-scale cortical simulations
- [ ] Hybrid learning (SNN + traditional deep learning)
- [ ] Exploration of novel brain-inspired architectures

---

## Research Questions and Open Problems

### Fundamental Questions

1. **What is the optimal granularity for heterogeneous tile specialization?**
   - Should each tile implement a single neuron model, or support multiple?
   - How does tile specialization affect load balancing and resource utilization?

2. **How far can open-source toolchains be pushed for Intel FPGAs?**
   - Is it feasible to create a Mistral-like database for Arria 10?
   - What are the fundamental barriers (technical vs. legal)?

3. **What role should ML play in FPGA development workflows?**
   - Is CNN-based region classification sufficient, or do we need more sophisticated models?
   - Can GNNs truly replace traditional heuristics, or are hybrid approaches optimal?

4. **How does HNTF compare to neuromorphic ASICs in practical applications?**
   - What are the true energy/performance/cost trade-offs?
   - Where does FPGA-based neuromorphic computing have advantages?

### Technical Challenges

1. **Timing closure for complex SNN tiles**
   - High fan-out from spike routers stresses FPGA routing
   - DDR4 memory interface timing constraints are tight
   - How to achieve >200 MHz operation reliably?

2. **Memory bandwidth bottlenecks**
   - Synaptic weight access dominates memory traffic
   - DDR4 provides ~20 GB/s per channel, but random access patterns hurt efficiency
   - Explore HBM2, HMC, or on-chip SRAM hierarchies?

3. **Inter-tile communication latency**
   - QSFP links have sub-microsecond latency, but protocol overhead matters
   - How to minimize spike transmission delays?
   - Explore custom lightweight protocols vs. standard Ethernet?

4. **Power management across heterogeneous tiers**
   - Edge tiles need low power, compute tiles need high performance
   - How to coordinate DVFS across tiles?
   - Monitor and control power at cluster level?

5. **Debugging distributed SNN execution**
   - Traditional debuggers don't work for distributed spiking systems
   - How to visualize spike activity across multiple tiles?
   - Develop neuromorphic-specific debugging tools?

### Algorithmic Questions

1. **What SNN models benefit most from FPGA implementation?**
   - Deep convolutional SNNs vs. recurrent networks?
   - Rate-coded vs. temporal-coded networks?
   - Online learning vs. offline-trained networks?

2. **How to efficiently implement plasticity on FPGAs?**
   - STDP requires spike-timing history (memory-intensive)
   - Exploration vs. exploitation in learning (non-deterministic)
   - Can approximations make on-chip learning practical?

3. **Optimal routing strategies for neuromorphic fabrics**
   - Broadcast vs. multicast vs. unicast for spike delivery?
   - Static routing tables vs. dynamic routing?
   - How to handle topology changes during learning?

---

## Community and Ecosystem Development

### Open-Source Contributions

**Immediate Goals**:
- Release YAML specifications under permissive license (MIT/Apache)
- Publish validated tile designs (RTL + constraints)
- Share trained ML models (CNNs, GNNs) with datasets

**Long-Term Goals**:
- Establish HNTF as a community standard for heterogeneous neuromorphic systems
- Contribute to F4PGA, SymbiFlow, and other open FPGA projects
- Create neuromorphic computing special interest group

### Educational Resources

**Documentation**:
- Step-by-step tutorials for building HNTF systems
- Video walkthroughs of tile bring-up
- Case studies of real-world applications

**Courses and Workshops**:
- FPGA-based neuromorphic computing course materials
- Hands-on workshops at conferences (FPL, FPT, ISFPGA)
- Online bootcamp for beginners

**Student Projects**:
- Curated list of project ideas suitable for undergrad/grad students
- Mentorship program for contributors
- Integration with university courses

### Standardization Efforts

**Interfaces**:
- Propose standard tile CSR layout
- Define common AER streaming protocol
- Create reference implementations

**Benchmarks**:
- Develop neuromorphic benchmark suite for FPGAs
- Establish performance metrics (spikes/sec, energy/spike, etc.)
- Compare HNTF against other neuromorphic platforms

**Collaboration**:
- Engage with neuromorphic computing community (NICE, CapoCaccia, etc.)
- Partner with FPGA vendors (Intel, Xilinx, Lattice)
- Work with SNN framework developers (Brian2, NEST, Lava)

---

## Ethical and Legal Considerations

### Intellectual Property

**Current Position**:
- Only analyze unencrypted bitstreams or self-compiled designs
- Respect vendor security features (no circumvention)
- Educational/research use of decommissioned hardware only

**Future Considerations**:
- As HNTF matures, clarify licensing for commercial use
- Engage with vendors on open FPGA initiatives
- Ensure community contributions remain unencumbered

### Environmental Impact

**Hardware Reuse**:
- HNTF extends life of datacenter hardware (reduces e-waste)
- But: decommissioned boards may have lower energy efficiency than modern devices
- Future: quantify environmental trade-offs (reuse vs. new hardware)

**Energy Consumption**:
- Neuromorphic computing promises energy efficiency, but FPGA implementation overhead matters
- Measure actual power consumption vs. GPU/CPU/ASIC alternatives
- Optimize for both performance and energy

### Accessibility and Equity

**Cost Barriers**:
- While "low-cost," HNTF still requires significant investment ($1-5K for basic system)
- Proprietary tools (Quartus) have licensing costs
- Future: explore cloud-hosted HNTF for broader access

**Technical Expertise**:
- Current system requires FPGA, networking, and neuromorphic computing knowledge
- High barrier to entry limits adoption
- Future: develop turn-key solutions and educational pathways

---

## Conclusion

The Heterogeneous Neuromorphic Tile Fabric represents a pragmatic approach to building neuromorphic computing systems from decommissioned datacenter hardware. While significant limitations remain—from vendor tool dependence to incomplete architectural knowledge to software stack immaturity—the architecture provides a clear roadmap for addressing these challenges.

Future work spans technical (open toolchains, improved ML-RE), practical (integration recipes, software stack), and algorithmic (plasticity, topology-aware routing) dimensions. Success requires not only technical innovation but also community building, standardization, and careful consideration of ethical implications.

By reducing vendor lock-in, deepening architectural understanding, hardening physical and software integration, and enriching neuromorphic capabilities, HNTF can evolve from a research prototype into a robust platform for neuromorphic computing experimentation and deployment.

---

**Document Status**: Paper-ready limitations and future work section
**Last Updated**: 2025-11-24
**Authors**: Quanta Hardware Project Contributors
**License**: CC BY 4.0 (documentation)
**Intended Use**: Direct integration into research papers as concluding section
