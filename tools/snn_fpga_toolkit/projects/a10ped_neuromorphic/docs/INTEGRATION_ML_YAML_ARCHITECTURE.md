# Integration: ML-Assisted Analysis ↔ YAML-Driven Architecture

**Connecting Legacy Hardware Understanding with Modern Development Workflows**

This document explains how ML-assisted bitstream analysis (documented in `ML_ASSISTED_BITSTREAM_ANALYSIS.md`) integrates with the YAML-driven "Quartus-as-a-service" architecture (documented in `README_ARCH.md`) to enable systematic repurposing of legacy Intel accelerator cards.

---

## The Two-Way Bridge

### Forward Path: YAML → Hardware
**Specification-driven development for known platforms**

```yaml
# specs/tiles/a10ped_tile.yaml
tile_name: "a10ped_tile0"
vendor: "intel"
fpga_part: "10AX115N2F40E2LG"

pcie:
  lanes: 8
  gen: 3
  endpoint_name: "pcie_a10ped_ep"
  bars:
    - { number: 0, size_kb: 1024, purpose: "csr" }
    - { number: 2, size_kb: 262144, purpose: "dma" }

memory:
  type: "ddr4"
  size_gb: 8.0
  interface: "emif"
  channels: 1
```

The YAML specification flows through automated tools:
1. `gen_qsf.py` → Quartus pin constraints
2. `project.tcl` → Automated project setup
3. `build_tile.sh` → Complete batch compilation
4. `parse_*.py` → Structured validation reports

**Result**: Reproducible bitstream generation with full visibility into the design process.

### Reverse Path: Hardware → Understanding
**ML-assisted exploration of legacy configurations**

When we encounter a **decommissioned board with unknown configuration**:

1. **Extract bitstream** (via JTAG, flash readout, or decommissioning process)
2. **Apply ML models** trained on reference designs:
   - CNN classifier → region heatmaps
   - Clustering → frame grouping by function
3. **Generate hypotheses** about PCIe/memory/fabric layout
4. **Validate with minimal test designs** (compiled with our YAML flow)
5. **Refine YAML specifications** based on observed constraints

**Result**: Informed YAML configurations that respect board-specific quirks discovered through ML analysis.

---

## Concrete Example: A10PED Bring-Up

### Stage 1: Initial YAML Specification (Vendor Datasheet)

From BittWare documentation, we know:
- Dual Arria 10 GX1150 (10AX115N2F40E2LG)
- PCIe Gen3 x8 per FPGA
- 8GB DDR4 + HMC per FPGA
- QSFP28, SFP+

Create `specs/boards/a10ped_board.yaml` with known pins.

### Stage 2: ML Analysis of Existing Images (If Available)

If we have access to bitstreams from previous A10PED users:

```python
# Apply trained models to extracted bitstream
python3 tools/ml_analysis/classify_regions.py \
  --bitstream a10ped_legacy.rbf \
  --model models/a10_cnn_classifier.pth \
  --output a10ped_regions.json

# Output: region_proposals.json
{
  "pcie_regions": [
    {"frame_range": [0, 120], "confidence": 0.89},
    {"frame_range": [1200, 1320], "confidence": 0.82}
  ],
  "emif_regions": [
    {"frame_range": [500, 700], "confidence": 0.91}
  ],
  "fabric_regions": [
    {"frame_range": [121, 499], "confidence": 0.95},
    {"frame_range": [701, 1199], "confidence": 0.93}
  ]
}
```

### Stage 3: Hypothesis Testing with YAML-Driven Builds

Create minimal test designs to validate ML predictions:

**Test 1: PCIe-only design**
```yaml
# specs/tests/a10ped_pcie_test.yaml
tile_name: "pcie_test"
pcie:
  lanes: 8
  gen: 3
# No DDR, no fabric logic
```

Compile → compare bitstream → check if PCIe regions match ML predictions.

**Test 2: DDR-only design**
```yaml
# specs/tests/a10ped_ddr_test.yaml
tile_name: "ddr_test"
memory:
  type: "ddr4"
  size_gb: 8.0
# No PCIe endpoint, minimal fabric
```

Compile → compare bitstream → validate EMIF region predictions.

### Stage 4: Refine YAML Based on Findings

If ML analysis + testing reveals:
- PCIe HIP must use specific transceiver channels
- DDR4 EMIF has preferred bank groups
- HMC requires particular I/O planning

Update the board YAML:
```yaml
# specs/boards/a10ped_board.yaml (refined)
pcie:
  refclk_pin: "PIN_AR37"  # ← Validated via testing
  lane_pins:
    - { lane: 0, tx_p: "PIN_AV31", ... }  # ← From ML + datasheet

memory:
  ddr4:
    controllers:
      - name: "ddr4_ch0"
        bank_group: "3A"  # ← ML suggested, testing confirmed
        pins_file: "constraints/a10ped/ddr4_ch0_pins.csv"

constraints:
  ml_informed:
    - "Avoid placing SNN cores in frames 0-120 (PCIe conflict)"
    - "HMC requires bank 2B I/O resources"
```

### Stage 5: Full Neuromorphic Tile Deployment

Now build the complete tile with confidence:

```bash
# Validate spec
python3 tools/validate/check_tile_spec.py specs/tiles/a10ped_tile.yaml

# Generate constraints (informed by ML + testing)
python3 flows/quartus/a10ped/gen_qsf.py \
  specs/boards/a10ped_board.yaml \
  out/a10ped/build/project.qsf

# Run full build
cd flows/quartus/a10ped
./build_tile.sh

# Verify timing meets SNN requirements
python3 tools/parse_reports/parse_quartus_timing.py \
  out/a10ped/build/output_files/a10ped_tile0.sta.rpt
```

---

## Bidirectional Feedback Loop

### ML Models Improve from YAML Designs

Each validated build adds to the training corpus:

```python
# Record successful configuration
def log_validated_build(yaml_spec, bitstream_path, timing_report):
    """Add verified design to ML training dataset"""
    entry = {
        "source": "yaml_flow",
        "tile": yaml_spec["tile_name"],
        "features": extract_features(yaml_spec),  # PCIe lanes, DDR, etc.
        "bitstream": bitstream_path,
        "timing_met": timing_report["timing_met"],
        "utilization": extract_utilization(bitstream_path)
    }
    append_to_corpus("ml_training/validated_designs.jsonl", entry)

# After successful A10PED build:
log_validated_build(
    load_yaml("specs/tiles/a10ped_tile.yaml"),
    "out/a10ped/build/a10ped_tile0.rbf",
    parse_timing_report("out/a10ped/build/timing_summary.json")
)
```

**Result**: ML models learn from our successful YAML-driven designs, improving predictions for future platforms.

### YAML Specs Improve from ML Insights

ML analysis reveals hidden constraints:

```python
# Extract design rules from ML clustering
def suggest_yaml_constraints(ml_analysis_results):
    """Convert ML insights into YAML constraint hints"""
    constraints = []

    # If PCIe and EMIF regions show strong anti-correlation
    if ml_analysis_results["pcie_emif_overlap"] < 0.05:
        constraints.append({
            "rule": "spatial_separation",
            "subsystems": ["pcie", "emif"],
            "reason": "ML clustering shows frame conflict"
        })

    # If certain frame ranges correlate with timing violations
    for violation in ml_analysis_results["timing_violations"]:
        constraints.append({
            "rule": "avoid_region",
            "frame_range": violation["frames"],
            "reason": f"High Fmax sensitivity: {violation['description']}"
        })

    return constraints

# Apply to board YAML
constraints = suggest_yaml_constraints(a10ped_ml_results)
update_yaml_with_constraints("specs/boards/a10ped_board.yaml", constraints)
```

---

## Practical Workflow Integration

### Tool Chain Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Developer Workflow                        │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Write YAML   │   │ Analyze      │   │ Validate     │
│ Spec         │   │ Legacy       │   │ on Hardware  │
│              │   │ Bitstream    │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        │ gen_qsf.py        │ classify_regions  │ parse_reports
        │ gen_code.py       │ cluster_frames    │ check_timing
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Auto-gen     │   │ Region       │   │ Structured   │
│ Constraints  │   │ Proposals    │   │ Reports      │
│ + RTL        │   │ + Clusters   │   │ (JSON)       │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                   ┌──────────────┐
                   │ Quartus      │
                   │ Batch Build  │
                   └──────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │ Bitstream    │
                   │ + Reports    │
                   └──────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ JTAG Program │   │ Add to ML    │   │ Update       │
│ FPGA         │   │ Corpus       │   │ YAML Spec    │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                      (iterate)
```

### Command Reference

**1. Start with Legacy Hardware Analysis**
```bash
# Extract bitstream from board
quartus_cpf --convert old_design.sof old_design.rbf

# Analyze with ML
python3 tools/ml_analysis/classify_regions.py \
  --bitstream old_design.rbf \
  --output region_analysis.json

# Review proposals
cat region_analysis.json | jq '.pcie_regions'
```

**2. Create YAML Specification**
```bash
# Start from template
cp specs/tiles/template_tile.yaml specs/tiles/my_board_tile.yaml

# Edit based on datasheet + ML insights
vim specs/tiles/my_board_tile.yaml

# Validate
python3 tools/validate/check_tile_spec.py specs/tiles/my_board_tile.yaml
```

**3. Generate Build Artifacts**
```bash
# Generate constraints
python3 flows/quartus/a10ped/gen_qsf.py \
  specs/boards/my_board.yaml \
  out/my_board/build/project.qsf

# Generate RTL
python3 abi/gen_code.py \
  specs/tiles/my_board_tile.yaml

# Build
cd flows/quartus/my_board
./build_tile.sh
```

**4. Validate on Hardware**
```bash
# Program FPGA
quartus_pgm -c 1 -m jtag \
  -o "p;out/my_board/build/output_files/my_tile.sof@1"

# Test via driver
cd sw/driver
make install
python3 ../python/test_tile.py
```

**5. Update ML Corpus**
```bash
# Log successful build
python3 tools/ml_analysis/log_build.py \
  --spec specs/tiles/my_board_tile.yaml \
  --bitstream out/my_board/build/my_tile.rbf \
  --reports out/my_board/build/*.json \
  --status success

# Retrain models (periodic)
python3 tools/ml_analysis/train_models.py \
  --corpus ml_training/validated_designs.jsonl \
  --output models/updated_classifier.pth
```

---

## Azure Stratix V Case Study (Planned)

### Challenge
The Azure "cat board" (X930613-001) uses Stratix V GS with:
- Unknown pin assignments (no public schematics)
- Possible custom PCIe configurations
- Legacy UEFI/firmware dependencies

### ML-Assisted Approach

**Phase 1: Data Collection**
```bash
# Obtain decommissioned Azure board
# Extract factory bitstream (if available)
python3 tools/hardware/extract_flash.py --board azure_sv --output azure_factory.rbf

# Compile minimal test designs on Stratix V dev kit
for test in pcie_only ddr3_only fabric_only; do
  python3 flows/quartus/stratix5/build_test.py --test $test
done
```

**Phase 2: ML Analysis**
```bash
# Train Stratix V-specific models
python3 tools/ml_analysis/train_models.py \
  --device stratix5 \
  --corpus stratix5_dev_kit_designs.jsonl \
  --output models/sv_classifier.pth

# Analyze Azure board bitstream
python3 tools/ml_analysis/classify_regions.py \
  --bitstream azure_factory.rbf \
  --model models/sv_classifier.pth \
  --output azure_analysis.json
```

**Phase 3: Hypothesis Testing**
```bash
# Compile test design with ML-informed constraints
python3 flows/quartus/azure_sv/gen_qsf.py \
  --board-spec specs/boards/azure_sv_hypothetical.yaml \
  --ml-constraints azure_analysis.json \
  --output out/azure/test.qsf

# Build and test
./flows/quartus/azure_sv/build_tile.sh
```

**Phase 4: YAML Refinement**
```bash
# Update board spec based on validation
python3 tools/ml_analysis/refine_yaml.py \
  --input specs/boards/azure_sv_hypothetical.yaml \
  --test-results out/azure/validation_results.json \
  --output specs/boards/azure_sv_validated.yaml
```

---

## Benefits of Integrated Approach

### 1. Accelerated Bring-Up
- **Traditional approach**: Weeks of manual trial-and-error
- **ML-YAML approach**: Days with guided exploration

### 2. Reproducible Designs
- All configurations tracked in version-controlled YAML
- ML insights documented as structured constraints
- Build process fully automated

### 3. Portable Knowledge
- ML models transfer across device families (A10 → SV)
- YAML templates adapt to new boards
- Report parsers work across projects

### 4. Continuous Improvement
- Each successful build improves ML models
- Failed experiments still contribute data
- Community can share validated YAMLs

### 5. Ethical Transparency
- ML analysis bounded to unencrypted bitstreams
- YAML specs openly document design intent
- No bypassing of vendor security features

---

## Future Work

### ML-Enhanced YAML Generation
```python
# Proposed: AI-assisted YAML authoring
python3 tools/ml_analysis/suggest_tile_yaml.py \
  --board a10ped \
  --application neuromorphic \
  --constraints ml_analysis/a10ped_regions.json \
  --output specs/tiles/a10ped_tile_v2.yaml
```

### Cross-Platform Optimization
```python
# Proposed: Transfer learning for new platforms
python3 tools/ml_analysis/transfer_model.py \
  --source-device arria10 \
  --target-device stratix5 \
  --corpus stratix5_minimal_corpus.jsonl \
  --output models/sv_transfer_classifier.pth
```

### Automated Validation Loop
```bash
# Proposed: CI/CD integration
python3 tools/ci/validate_all_boards.py \
  --specs specs/tiles/*.yaml \
  --ml-check true \
  --hardware-test $(detect_available_boards)
```

---

## Conclusion

The integration of **ML-assisted bitstream analysis** and **YAML-driven development** creates a powerful, ethical, and reproducible workflow for repurposing legacy FPGA accelerator cards:

- **ML provides insights** where documentation is missing
- **YAML captures knowledge** in human- and machine-readable form
- **Automated tools** bridge the gap between understanding and implementation
- **Feedback loops** ensure continuous improvement

This approach enables the Quanta Hardware Project to systematically expand neuromorphic tile support across diverse platforms (A10PED, Azure Stratix V, FK33) while maintaining full transparency and respecting vendor security boundaries.

---

## References

- `README_ARCH.md` - YAML-driven architecture overview
- `ML_ASSISTED_BITSTREAM_ANALYSIS.md` - ML techniques and ethical scope
- `specs/` - Tile and board YAML specifications
- `flows/` - Automated build scripts
- `tools/` - Validation and parsing utilities
