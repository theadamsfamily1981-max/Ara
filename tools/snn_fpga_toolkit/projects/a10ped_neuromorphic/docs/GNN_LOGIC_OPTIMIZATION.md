# Graph Neural Networks for Logic Optimization in Neuromorphic FPGA Tiles

**Leveraging GNNs to Optimize SNN Kernels on BittWare A10PED and Azure Stratix V**

This document explores how Graph Neural Networks (GNNs) can enhance logic optimization within the Quanta Hardware Project's YAML-driven EDA flow, specifically targeting neuromorphic AI tile development on legacy Intel accelerator cards.

---

## Executive Summary

Graph Neural Networks (GNNs) have emerged as a transformative tool in electronic design automation (EDA), addressing the growing complexity of modern circuit designs by leveraging graph-based representations to model netlists and predict optimization outcomes. For the Quanta Hardware Project's neuromorphic tiles on the **BittWare A10PED** (dual Arria 10 GX1150) and **Azure Stratix V**, GNNs offer promising opportunities to:

- **Optimize SNN kernel placement** by treating neurons as graph nodes and synapses as edges
- **Predict resource utilization** for DSP blocks (1,518 per A10 FPGA) and transceivers (96 at 28.3 Gbps)
- **Reduce synthesis iterations** in Quartus Prime by 20-40% through learned heuristics
- **Enable cross-platform optimization** by learning from reference designs and transferring to legacy boards

Research from conferences like DAC and ICCAD indicates GNNs can reduce optimization time by 20-40% and improve metrics like power by 10-25%, though integration with proprietary tools like Quartus remains challenging.

---

## 1. Background: Logic Optimization in FPGA Synthesis

### 1.1 Traditional EDA Flows

Logic optimization is a core stage in FPGA synthesis, transforming RTL descriptions into efficient gate-level netlists through:

1. **Technology mapping**: Match logic functions to library cells (ALMs, DSPs, memories)
2. **Gate sizing**: Select optimal drive strengths for timing/power trade-offs
3. **Logic rewriting**: Simplify Boolean expressions using algebraic techniques
4. **Placement and routing**: Physical optimization considering wire delays

For the **Arria 10 GX1150** on the A10PED:
- **1.15 million logic elements** (ALMs with fracturable LUTs)
- **1,518 variable-precision DSP blocks** (floating-point capable)
- **32 GB DDR4** + **2 GB HMC** per FPGA
- **PCIe Gen3 x8** endpoints (8 GT/s per lane)
- **96 transceivers** at up to 28.3 Gbps

Traditional heuristic-based methods in **Quartus Prime** (Intel's FPGA compiler) struggle with:
- **Scalability**: 1M+ LEs create massive search spaces
- **Multi-objective optimization**: Area vs. delay vs. power trade-offs
- **SNN-specific constraints**: Event-driven logic patterns differ from CNN/general compute

### 1.2 Why GNNs for FPGAs?

GNNs treat circuits as graphs:
- **Nodes**: Logic gates, DSP blocks, memory controllers, SNN neurons
- **Edges**: Wires, routing channels, synaptic connections
- **Features**: Gate types, fan-in/out, timing slack, resource types

This enables:
- **Message passing**: Propagate timing/power information through the netlist
- **Attention mechanisms**: Focus on critical paths or high-fan-out nets
- **End-to-end learning**: Predict optimization outcomes without hand-crafted rules

---

## 2. GNN Architectures for EDA

### 2.1 Model Comparison

| Model Type | Mechanism | Best For | Complexity | Example Use |
|------------|-----------|----------|------------|-------------|
| **GCN** (Graph Convolutional Network) | Aggregates neighbor features via convolution | Technology mapping | Low | Fast area/delay prediction |
| **GAT** (Graph Attention Network) | Learns edge weights via attention | Gate sizing, critical paths | Medium | Timing-aware optimization |
| **GraphSAGE** | Samples neighbors for scalability | Large netlists (>1M nodes) | Medium | A10PED-scale designs |
| **HGNN** (Heterogeneous GNN) | Handles multiple node/edge types | Multi-IP designs (PCIe + DDR + SNN) | High | Hybrid tile optimization |

### 2.2 Training Pipeline

```python
# Conceptual GNN training for A10PED tile optimization

import torch
from torch_geometric.nn import GCNConv, GATConv

class TileOptimizationGNN(torch.nn.Module):
    """
    GNN for predicting resource utilization and timing metrics
    for neuromorphic AI tiles on Arria 10 FPGAs
    """
    def __init__(self, num_node_features, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 64)

        # Predict multiple outputs
        self.predict_alms = torch.nn.Linear(64, 1)  # ALM count
        self.predict_dsps = torch.nn.Linear(64, 1)  # DSP utilization
        self.predict_fmax = torch.nn.Linear(64, 1)  # Max frequency

    def forward(self, x, edge_index):
        # x: Node features [N, num_features]
        # edge_index: Graph connectivity [2, E]

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))

        # Global pooling
        x_global = torch.mean(x, dim=0, keepdim=True)

        return {
            'alms': self.predict_alms(x_global),
            'dsps': self.predict_dsps(x_global),
            'fmax_mhz': self.predict_fmax(x_global)
        }

# Training on Quartus-generated netlists
def train_on_a10_designs(model, design_dataset):
    """
    Train GNN on paired (netlist, Quartus reports) data

    Dataset structure:
    - Input: Graph from .v netlist (nodes=gates, edges=wires)
    - Labels: Parsed from Quartus .fit.summary and .sta.rpt
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        for netlist_graph, quartus_reports in design_dataset:
            optimizer.zero_grad()

            predictions = model(netlist_graph.x, netlist_graph.edge_index)

            # Multi-task loss
            loss = (
                F.mse_loss(predictions['alms'], quartus_reports['alm_usage']) +
                F.mse_loss(predictions['dsps'], quartus_reports['dsp_usage']) +
                F.mse_loss(predictions['fmax_mhz'], quartus_reports['fmax'])
            )

            loss.backward()
            optimizer.step()
```

---

## 3. Applications to A10PED Neuromorphic Tiles

### 3.1 SNN Kernel Optimization

**Challenge**: Spiking Neural Networks have unique characteristics:
- **Event-driven computation**: Sparse activations, irregular timing
- **Synaptic connectivity**: High fan-out from neurons to many targets
- **Leaky integration**: Requires DSP blocks or custom logic

**GNN Solution**: Model SNN as a graph where:
- **Nodes**: Individual LIF neurons, input buffers, spike routers
- **Edges**: Synaptic connections with weights and delays
- **Features**: Neuron parameters (threshold, leak rate), connection density

```yaml
# Example: SNN core specification for GNN optimization
# specs/tiles/a10ped_snn_optimized.yaml

snn_core:
  top_module: "snn_core_gnn_opt"
  parameters:
    neuron_count: 512
    precision: "int16"
    topology: "small_world"  # Graph structure
    avg_fan_out: 64

  optimization:
    method: "gnn_assisted"
    model: "models/a10_snn_optimizer.pth"
    objectives:
      - name: "minimize_dsp_usage"
        weight: 0.4
      - name: "maximize_fmax"
        weight: 0.4
      - name: "minimize_routing_congestion"
        weight: 0.2

  constraints:
    max_dsp_blocks: 500  # Out of 1,518 available
    target_fmax_mhz: 250
    max_alm_usage_percent: 30
```

**Workflow Integration**:
```bash
# Step 1: Generate SNN netlist from YAML
python3 tools/snn/gen_snn_netlist.py \
  specs/tiles/a10ped_snn_optimized.yaml \
  --output hw/rtl/snn_core_gnn_opt.v

# Step 2: Convert netlist to graph for GNN
python3 tools/gnn/netlist_to_graph.py \
  hw/rtl/snn_core_gnn_opt.v \
  --output data/graphs/snn_core.pt

# Step 3: Predict optimization targets
python3 tools/gnn/predict_optimization.py \
  --graph data/graphs/snn_core.pt \
  --model models/a10_snn_optimizer.pth \
  --output predictions/snn_core_targets.json

# Step 4: Apply predictions as Quartus constraints
python3 tools/gnn/apply_gnn_constraints.py \
  --predictions predictions/snn_core_targets.json \
  --qsf-template flows/quartus/a10ped/project.qsf \
  --output flows/quartus/a10ped/project_gnn_opt.qsf

# Step 5: Run Quartus with GNN-informed constraints
cd flows/quartus/a10ped
./build_tile.sh --qsf project_gnn_opt.qsf
```

### 3.2 PCIe + DDR + SNN Co-Optimization

**Challenge**: The A10PED tile integrates:
- **PCIe Gen3 x8 Hard IP**: Fixed location, dedicated transceivers
- **DDR4 EMIF**: Requires specific I/O banks
- **SNN fabric**: Flexible placement but timing-sensitive

**GNN Solution**: Heterogeneous graph with multiple node types:

```python
# Node types in hybrid tile graph
NODE_TYPES = {
    'pcie_endpoint': {
        'features': ['lane_count', 'gen', 'bar_sizes'],
        'fixed_location': True
    },
    'ddr_controller': {
        'features': ['channels', 'width_bits', 'frequency_mhz'],
        'bank_constraints': ['3A', '3B']
    },
    'snn_neuron': {
        'features': ['threshold', 'leak_rate', 'fan_out'],
        'placement': 'flexible'
    },
    'csr_register': {
        'features': ['width', 'access_mode', 'reset_value'],
        'timing_critical': False
    }
}

# Edge types represent different interaction patterns
EDGE_TYPES = {
    'avalon_mm': {'features': ['data_width', 'pipeline_depth']},
    'axi_stream': {'features': ['tdata_width', 'backpressure']},
    'spike_route': {'features': ['weight', 'delay_cycles']},
    'clock_domain_crossing': {'features': ['async', 'fifo_depth']}
}
```

**Expected Results** (based on literature benchmarks):
- **15-25% reduction** in ALM usage via better logic clustering
- **10-20% Fmax improvement** through timing-aware placement
- **20-40% faster** synthesis time (fewer Quartus iterations)

### 3.3 Cross-Platform Transfer Learning

**Challenge**: Limited training data for legacy boards (A10PED, Azure Stratix V).

**GNN Solution**: Train on well-supported reference boards, transfer to target platforms:

```python
# Transfer learning workflow
def transfer_a10_dev_kit_to_a10ped(source_model, target_designs):
    """
    Fine-tune GNN trained on Arria 10 dev kit for A10PED-specific constraints

    Strategy:
    1. Pre-train on large corpus of dev kit designs
    2. Freeze early layers (capture generic FPGA patterns)
    3. Fine-tune later layers on A10PED-specific data (PCIe/HMC configs)
    """

    # Freeze GCN layers that learned generic patterns
    for param in source_model.conv1.parameters():
        param.requires_grad = False
    for param in source_model.conv2.parameters():
        param.requires_grad = False

    # Fine-tune prediction heads on A10PED data
    optimizer = torch.optim.Adam([
        {'params': source_model.conv3.parameters()},
        {'params': source_model.predict_alms.parameters()},
        {'params': source_model.predict_dsps.parameters()},
        {'params': source_model.predict_fmax.parameters()}
    ], lr=0.0001)

    # Train on small A10PED dataset (10-50 designs)
    for epoch in range(50):
        for a10ped_graph, a10ped_reports in target_designs:
            # ... training loop
            pass

    return source_model
```

**Data Sources**:
- **Reference designs**: Arria 10 dev kit tutorials (100+ designs)
- **EPFL benchmarks**: Arithmetic, control logic (open netlists)
- **ISPD contests**: Placement/routing datasets
- **Our own YAML-generated designs**: From `specs/tiles/*.yaml`

---

## 4. Integration with YAML-Driven Architecture

### 4.1 Extended Build Flow

The GNN layer sits between YAML specification and Quartus compilation:

```
┌──────────────┐
│ YAML Spec    │  specs/tiles/a10ped_tile.yaml
│ (User Input) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Gen Netlist  │  abi/gen_code.py, tools/snn/gen_snn_netlist.py
│              │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Netlist→Graph│  tools/gnn/netlist_to_graph.py
│              │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ GNN Prediction       │  tools/gnn/predict_optimization.py
│ • ALM/DSP allocation │
│ • Timing estimates   │
│ • Placement hints    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Apply Constraints    │  tools/gnn/apply_gnn_constraints.py
│ • Update .qsf        │
│ • Set LOC directives │
│ • Adjust seed        │
└──────┬───────────────┘
       │
       ▼
┌──────────────┐
│ Quartus      │  flows/quartus/a10ped/build_tile.sh
│ Compile      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Parse Reports│  tools/parse_reports/*.py
│              │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Update GNN   │  Feedback loop: actual vs. predicted
│ Training Set │
└──────────────┘
```

### 4.2 YAML Extension for GNN Configuration

```yaml
# specs/tiles/a10ped_tile_gnn.yaml

tile_name: "a10ped_tile0_gnn_optimized"
vendor: "intel"
fpga_part: "10AX115N2F40E2LG"

# ... existing PCIe, memory, CSR config ...

# New: GNN optimization section
optimization:
  enabled: true
  method: "gnn_assisted"

  models:
    - name: "snn_optimizer"
      path: "models/a10_snn_optimizer.pth"
      type: "GraphSAGE"
      trained_on: ["epfl_benchmarks", "a10_dev_kit_designs"]

    - name: "timing_predictor"
      path: "models/a10_timing_gat.pth"
      type: "GAT"
      trained_on: ["ispd_2015", "quartus_generated_corpus"]

  objectives:
    - metric: "alm_usage"
      target: "minimize"
      weight: 0.3
      constraint: "< 30% of total"

    - metric: "dsp_usage"
      target: "minimize"
      weight: 0.2
      constraint: "< 500 blocks"

    - metric: "fmax"
      target: "maximize"
      weight: 0.4
      constraint: "> 250 MHz"

    - metric: "power"
      target: "minimize"
      weight: 0.1
      constraint: "< 25W per FPGA"

  validation:
    cross_check_with_quartus: true
    confidence_threshold: 0.85  # Only apply if prediction confidence > 85%
    fallback_to_heuristics: true  # Use Quartus defaults if GNN uncertain
```

### 4.3 Tool Implementation

**New tools to add**:

```bash
# tools/gnn/netlist_to_graph.py
# Convert Verilog/VHDL to PyTorch Geometric graph

# tools/gnn/predict_optimization.py
# Run GNN inference on netlist graph

# tools/gnn/apply_gnn_constraints.py
# Convert GNN predictions to Quartus .qsf directives

# tools/gnn/train_models.py
# Train/fine-tune GNN models on design corpus

# tools/gnn/validate_predictions.py
# Compare GNN predictions to actual Quartus results
```

---

## 5. Performance Expectations (Literature-Based)

### 5.1 Benchmark Results from Research

| Benchmark | GNN Improvement vs. Traditional | Area Reduction | Delay Reduction | Power Savings | Source |
|-----------|--------------------------------|----------------|-----------------|---------------|--------|
| **EPFL Suite** (arithmetic/control) | 15-25% | 10% | 20% | 12% | "Multi-Task Learning for Logic Synthesis" (arXiv 2024) |
| **ISPD 2015** (placement) | 20-30% | 8% | 15% | 18% | "Advancing Physical Design using GNNs" (ICCAD 2022) |
| **ITC'99** (mixed logic/IP) | 25-35% | 12% | 22% | 15% | "GNNs for EDA Problems" (ACM 2022) |
| **A10PED-like** (simulated dual FPGA) | 10-20% | 5% | 10% | 8% | Extrapolated from above |

### 5.2 Expected A10PED Results

For a **512-neuron SNN tile** on Arria 10 GX1150:

**Baseline (Quartus heuristics)**:
- ALM usage: 45,000 / 427,200 (10.5%)
- DSP blocks: 480 / 1,518 (31.6%)
- Fmax: 220 MHz
- Compile time: 35 minutes

**GNN-optimized (predicted)**:
- ALM usage: 38,000 / 427,200 (8.9%) — **15% reduction**
- DSP blocks: 420 / 1,518 (27.7%) — **12% reduction**
- Fmax: 250 MHz — **14% improvement**
- Compile time: 25 minutes — **29% faster** (fewer iterations)

---

## 6. Challenges and Limitations

### 6.1 Technical Challenges

**Data Scarcity**:
- **Problem**: Limited public Arria 10 netlist datasets
- **Mitigation**: Transfer learning from dev kits, generate synthetic designs from YAML specs
- **Hedging**: Hybrid GNN + heuristic approach with confidence thresholds

**Computational Cost**:
- **Problem**: Training on 1M+ node graphs requires GPU resources
- **Mitigation**: Graph sampling (GraphSAGE), distributed training
- **Hedging**: Pre-train on smaller designs, fine-tune on target scale

**Tool Integration**:
- **Problem**: Quartus is proprietary, limited scriptability for GNN outputs
- **Mitigation**: Use .qsf assignments, Tcl scripts, seed exploration
- **Hedging**: Validate GNN predictions against actual runs, fall back if confidence low

**Encrypted IP**:
- **Problem**: Can't extract graphs from vendor hard IP blocks (PCIe HIP, EMIF)
- **Mitigation**: Treat as black boxes with known interfaces
- **Hedging**: Train GNN on fabric logic only, use vendor timing models for IP

### 6.2 Ethical and Legal Considerations

**IP Protection**:
- **Concern**: Could GNN-based RE enable IP theft?
- **Mitigation**: Train only on:
  - Self-generated designs from YAML
  - Open-source benchmarks (EPFL, ISPD)
  - Legacy designs with clear educational use rights
- **Policy**: Never train on proprietary third-party bitstreams without permission

**Supply Chain Security**:
- **Concern**: GNNs for hardware trojan detection vs. insertion
- **Mitigation**: Focus on optimization, not obfuscation/attack
- **Transparency**: Open-source models and training data for community review

### 6.3 Controversies in the Field

**Overfitting vs. Generalization**:
- **Debate**: Do GNNs overfit to benchmark characteristics?
- **Evidence**: Transfer learning shows 10-15% degradation on new designs
- **Hedging**: Use ensemble models, validate across diverse test sets

**Hype vs. Reality**:
- **Claim**: "GNNs will replace traditional EDA"
- **Reality**: Hybrid approaches outperform pure GNN or pure heuristic
- **Stance**: Position GNNs as **augmentation**, not replacement

---

## 7. Future Directions

### 7.1 Short-Term (6-12 months)

1. **Build training corpus**:
   - Generate 100+ designs from YAML specs with varying parameters
   - Collect Quartus reports for all builds
   - Create PyTorch Geometric dataset

2. **Train baseline models**:
   - GCN for area/delay prediction
   - GAT for timing-critical path analysis
   - GraphSAGE for large SNN netlists

3. **Integrate with build flow**:
   - Implement `tools/gnn/` scripts
   - Validate predictions on 10 test designs
   - Measure actual speedup in synthesis time

### 7.2 Medium-Term (1-2 years)

1. **Transfer to Azure Stratix V**:
   - Fine-tune models for Stratix V architecture differences
   - Handle HBM2 instead of DDR4
   - Optimize for dual-FPGA configurations

2. **Multi-objective optimization**:
   - Train reinforcement learning agent on top of GNN
   - Explore Pareto frontiers (area vs. speed vs. power)
   - Interactive optimization with designer preferences

3. **Open-source release**:
   - Publish trained models and datasets
   - Contribute to F4PGA/SymbiFlow GNN efforts
   - Collaborate with academic research groups

### 7.3 Long-Term (2-5 years)

1. **Federated learning for EDA**:
   - Aggregate knowledge across multiple organizations
   - Preserve IP confidentiality while improving models
   - Create industry-wide GNN optimization standard

2. **GNN-native synthesis**:
   - Replace Quartus front-end with GNN-based flow
   - Direct graph-to-bitstream compilation (research challenge)
   - Integration with open FPGA architectures

3. **Neuromorphic-specific GNN architectures**:
   - Design GNN layers that mirror spiking dynamics
   - Co-optimize SNN algorithm and FPGA mapping
   - Enable "spiking GNNs for spiking hardware"

---

## 8. Related Work and Key Citations

### 8.1 Foundational Papers

1. **"Why are Graph Neural Networks Effective for EDA Problems?"** (ACM 2022)
   - Analysis of GNN inductive biases for circuits
   - Shows 75-90% accuracy on various EDA tasks
   - https://dl.acm.org/doi/10.1145/3508352.3561093

2. **"A Multi-Task Learning Approach for Logic Synthesis Optimization"** (arXiv 2024)
   - Simultaneously optimizes area and delay
   - 15% better than ABC on EPFL benchmarks
   - https://arxiv.org/abs/2409.06077

3. **"On Advancing Physical Design using Graph Neural Networks"** (ICCAD 2022)
   - Macro placement with GNNs
   - 10-30% better utilization than commercial tools
   - https://www.gtcad.gatech.edu/www/papers/Yi-Chen-ICCAD22.pdf

### 8.2 Survey and Overview Papers

4. **"A Survey of Graph Neural Networks for Electronic Design Automation"** (arXiv 2021)
   - Comprehensive overview of GNN applications in EDA
   - Covers synthesis, placement, routing, verification
   - https://arxiv.org/abs/2103.11708

5. **"Versatile Multi-stage Graph Neural Network for Circuit Representation"** (NeurIPS 2022)
   - Hierarchical GNN for large-scale designs
   - Handles heterogeneous IP blocks
   - https://papers.neurips.cc/paper/2022/file/7fa548155f40c014372146be387c4f6a-Paper-Conference.pdf

### 8.3 Industry and Open-Source Efforts

6. **Google Research: Circuit Training**
   - Uses GNNs for chip floorplanning
   - Achieved super-human performance on Google TPU
   - Code: https://github.com/google-research/circuit_training

7. **Stanford CS224W: Circuit Quest**
   - Student project using GraphSAGE for logic synthesis
   - Detailed methodology and results
   - https://medium.com/stanford-cs224w/circuit-quest-embarking-on-a-gnn-adventure-in-logic-synthesis-1b92643d3dc9

8. **DfX-NYUAD/GNN4IC GitHub**
   - Curated list of must-read GNN for IC design papers
   - Organized by application area
   - https://github.com/DfX-NYUAD/GNN4IC

### 8.4 Vendor Resources

9. **Intel Quartus Prime EDA Resources**
   - Official documentation for Arria 10 synthesis
   - Scripting APIs for automation
   - https://www.intel.com/content/www/www/us/en/products/details/fpga/development-tools/quartus-prime/resource.html

10. **Cadence: Graph Neural Networks for EDA**
    - Industrial perspective on GNN adoption
    - Integration with existing tools
    - https://www.cadence.com/en_US/home/tools/system-design-and-verification/ml-ai/graph-neural-networks-for-eda.html

---

## 9. Conclusion

Graph Neural Networks offer a promising path to enhance logic optimization for the Quanta Hardware Project's neuromorphic AI tiles on legacy Intel accelerator cards. By integrating GNN-assisted prediction into the YAML-driven build flow, we can:

- **Reduce synthesis time** by 20-40% through smarter heuristics
- **Improve resource utilization** by 10-25% via learned placement
- **Enable cross-platform optimization** through transfer learning
- **Maintain ethical standards** by training only on authorized datasets

The approach complements existing ML-assisted bitstream analysis (for reverse engineering) with forward optimization (for design), creating a complete bidirectional workflow for legacy FPGA repurposing.

**Key Recommendations**:
1. Start with **simple GCN models** for area/delay prediction
2. Build training corpus from **YAML-generated designs**
3. Implement **hybrid GNN + Quartus** flow with confidence thresholds
4. Validate on **A10PED hardware** before scaling to Azure Stratix V
5. Release **open-source models** to benefit the community

**Next Steps**:
- Implement `tools/gnn/` scripts per section 4.3
- Generate initial training dataset (50-100 designs)
- Train baseline GCN model on Arria 10 dev kit corpus
- Integrate with `build_tile.sh` for end-to-end validation

---

## Appendix A: Quick Start Guide

### A.1 Installing Dependencies

```bash
# PyTorch Geometric for GNN models
pip install torch torchvision
pip install torch-geometric

# Graph manipulation
pip install networkx
pip install pydot

# Verilog parsing (for netlist → graph)
pip install pyverilog

# YAML for specs
pip install pyyaml
```

### A.2 Generating Training Data

```bash
# Create 100 design variants from YAML template
python3 tools/gnn/generate_training_corpus.py \
  --template specs/tiles/a10ped_tile.yaml \
  --variations 100 \
  --output data/training_corpus/

# Build all variants with Quartus
for design in data/training_corpus/*.yaml; do
  python3 flows/quartus/a10ped/build_from_yaml.py --spec $design
done

# Parse all Quartus reports
python3 tools/gnn/collect_quartus_results.py \
  --corpus data/training_corpus/ \
  --output data/training_labels.json
```

### A.3 Training Your First GNN

```bash
# Convert netlists to graphs
python3 tools/gnn/build_graph_dataset.py \
  --corpus data/training_corpus/ \
  --output data/graph_dataset.pt

# Train GCN model
python3 tools/gnn/train_models.py \
  --dataset data/graph_dataset.pt \
  --model-type GCN \
  --epochs 100 \
  --output models/a10_gcn_v1.pth

# Evaluate on test set
python3 tools/gnn/evaluate_model.py \
  --model models/a10_gcn_v1.pth \
  --test-set data/test_graphs.pt
```

### A.4 Using GNN in Build Flow

```bash
# Build with GNN optimization enabled
python3 flows/quartus/a10ped/build_tile.sh \
  --spec specs/tiles/a10ped_snn_optimized.yaml \
  --gnn-model models/a10_gcn_v1.pth \
  --gnn-confidence-threshold 0.85

# Compare GNN vs. baseline
python3 tools/gnn/compare_results.py \
  --gnn-build out/a10ped/gnn_opt/ \
  --baseline-build out/a10ped/baseline/ \
  --metrics alm dsp fmax power compile_time
```

---

## Appendix B: GNN Model Architecture Details

### B.1 Node Feature Engineering

For Arria 10 netlists, each node (gate/module) has features:

```python
NODE_FEATURES = {
    # Basic attributes
    'gate_type': categorical([
        'alm_lut', 'alm_reg', 'dsp_block', 'm20k_ram',
        'io_buffer', 'pcie_hip', 'emif_ctrl'
    ]),
    'fan_in': int,      # Number of input connections
    'fan_out': int,     # Number of output connections

    # Timing
    'input_delay_ns': float,
    'output_delay_ns': float,
    'combinational_delay_ns': float,

    # Resource
    'alm_count': int,
    'dsp_count': int,
    'mem_bits': int,

    # Placement (if available from prior run)
    'x_coord': int or None,
    'y_coord': int or None,
    'logic_array_block': int or None,

    # Criticality
    'on_critical_path': bool,
    'timing_slack_ns': float,

    # Custom (for SNN)
    'is_neuron': bool,
    'is_synapse': bool,
    'spike_rate_hz': float or None
}
```

### B.2 Edge Feature Engineering

For connections between nodes:

```python
EDGE_FEATURES = {
    # Basic
    'connection_type': categorical([
        'wire', 'clock', 'reset', 'avalon_mm',
        'axi_stream', 'spike_channel'
    ]),
    'width_bits': int,

    # Timing
    'wire_delay_ns': float,
    'setup_time_ns': float,
    'hold_time_ns': float,

    # Routing
    'manhattan_distance': int,  # If placement known
    'routing_congestion': float,  # From Quartus router

    # Custom
    'synaptic_weight': float or None,  # For SNN edges
    'transmission_delay_cycles': int or None
}
```

### B.3 Graph Construction from Verilog

```python
import pyverilog
from torch_geometric.data import Data

def verilog_to_graph(verilog_path):
    """Convert Verilog netlist to PyTorch Geometric graph"""

    # Parse Verilog
    ast, _ = pyverilog.parse([verilog_path])

    nodes = []
    edges = []
    node_features = []
    edge_features = []

    # Walk AST and extract modules/instances
    for module in ast.descriptions:
        for item in module.items:
            if isinstance(item, pyverilog.Instance):
                # Create node for this instance
                node_id = len(nodes)
                nodes.append(item.name)
                node_features.append(extract_node_features(item))

                # Create edges for connections
                for port in item.portlist:
                    if port.argname:  # Connected port
                        target_node = find_node_by_net(port.argname)
                        edges.append([node_id, target_node])
                        edge_features.append(extract_edge_features(port))

    # Convert to PyTorch Geometric format
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
```

---

**Document Status**: Research and planning phase
**Last Updated**: 2025-11-24
**Authors**: Quanta Hardware Project Contributors
**License**: MIT (documentation); trained models subject to dataset licenses
