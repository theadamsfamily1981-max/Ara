# Complete Workflow Guide

**FPGA Bitstream ML Analysis Toolkit - From Raw Bitstream to Trojan Detection**

This guide walks through the complete end-to-end workflow for training and deploying the Hardware Trojan detection system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Workflow Overview](#workflow-overview)
3. [Phase 1: Dataset Generation](#phase-1-dataset-generation)
4. [Phase 2: Preprocessing](#phase-2-preprocessing)
5. [Phase 3: Model Training](#phase-3-model-training)
6. [Phase 4: Inference](#phase-4-inference)
7. [Phase 5: Integration](#phase-5-integration)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Software Requirements

```bash
# Python dependencies
pip install -r env/requirements.txt

# Quartus Prime Pro (for dataset generation)
# Version 23.4 or later recommended
which quartus_sh quartus_cpf
```

### Hardware Requirements

- **For Training**: NVIDIA GPU recommended (CUDA support), minimum 8GB VRAM
- **For Inference**: CPU sufficient (GPU optional for speed)
- **Storage**: ~50GB for full training dataset (100 clean + 100 infected samples)

### Golden Design

You need at least one "golden" FPGA design (your trusted reference) compiled with Quartus. This should be:
- A real FPGA project (.qpf file)
- Successfully compilable
- Representative of your actual hardware designs

---

## Workflow Overview

```
┌─────────────────┐
│ Golden Design   │ (Your trusted FPGA project)
│ (.qpf project)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Phase 1: Dataset Generation             │
│ ─────────────────────────────────────── │
│ • Compile with multiple seeds (clean)   │
│ • Insert Trojans via ECO (infected)     │
│ • Output: .rbf bitstreams               │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Phase 2: Preprocessing                  │
│ ─────────────────────────────────────── │
│ • Detect frame width (autocorrelation)  │
│ • Compute entropy maps                  │
│ • Output: .png images                   │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Phase 3: Model Training                 │
│ ─────────────────────────────────────── │
│ • Train ResNet-50 CNN                   │
│ • Validate on held-out data             │
│ • Output: arria10_cnn.pt                │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Phase 4: Inference                      │
│ ─────────────────────────────────────── │
│ • Analyze new bitstreams                │
│ • Detect Trojans                        │
│ • Output: Classification results        │
└─────────────────────────────────────────┘
```

---

## Phase 1: Dataset Generation

### Step 1.1: Prepare Golden Design

```bash
# Navigate to your Quartus project directory
cd /path/to/your/quartus/project

# Verify project compiles
quartus_sh --flow compile your_project.qpf
```

### Step 1.2: Generate Clean Samples

Generate topological diversity through seed variation:

```bash
# Generate 50 clean samples with different fitter seeds
quartus_sh -t /path/to/fpga-bitstream-ml/factory/tcl/generate_clean.tcl \
    your_project \
    /path/to/fpga-bitstream-ml/data/raw/arria10/clean \
    50

# This will take 5-30 minutes per seed (device dependent)
# Total time: ~4-25 hours for 50 samples
```

**Output**:
```
data/raw/arria10/clean/
  seed_001/
    design.rbf
    metadata.json
  seed_002/...
  ...
  seed_050/...
```

### Step 1.3: Generate Infected Samples

**IMPORTANT**: The current ECO Trojan injection script (`inject_trojan.tcl`) is a **placeholder/template**. It demonstrates the workflow but requires device-specific cell instantiation to be production-ready.

For actual Trojan insertion, you have two options:

**Option A: Implement Full ECO (Advanced)**

Edit `factory/tcl/inject_trojan.tcl` and replace placeholder sections with actual Quartus ECO commands:

```tcl
# Example for Arria 10:
create_cell -type dff -count 32 -name trojan_counter
place_cell -cell trojan_counter -location LAB_X47_Y102
make_connection -from [get_nets clk_100mhz] -to trojan_counter CLK
route_design -incremental
```

See [Quartus ECO User Guide](https://www.intel.com/content/www/us/en/docs/programmable/683230/current/engineering-change-orders.html) for device-specific syntax.

**Option B: Use Simulated Infections (Quick Start)**

For initial testing/development, manually create "infected" variants:

```bash
# Copy some clean samples to infected directory
mkdir -p data/raw/arria10/infected
cp -r data/raw/arria10/clean/seed_001 data/raw/arria10/infected/simulated_001
cp -r data/raw/arria10/clean/seed_002 data/raw/arria10/infected/simulated_002

# Edit .rbf files to simulate Trojans (for testing only):
# This creates synthetic anomalies for proof-of-concept
python tools/inject_synthetic_noise.py data/raw/arria10/infected/
```

### Step 1.4: Batch Generation (Automated)

For automated dataset generation with balanced classes:

```bash
quartus_sh -t factory/tcl/batch_run.tcl \
    your_project \
    data/raw/arria10 \
    50 \  # Number of clean samples
    50    # Number of infected samples
```

**Expected Duration**: 8-50 hours total (highly parallelizable if you have multiple build servers)

---

## Phase 2: Preprocessing

Convert raw .rbf bitstreams to entropy map images for CNN input.

### Step 2.1: Single File (Test)

Test preprocessing on one file first:

```bash
python preprocessing/bitstream_to_image.py \
    --rbf data/raw/arria10/clean/seed_001/design.rbf \
    --out test_entropy.png \
    --visualize

# Check outputs:
# - test_entropy.png (grayscale entropy map)
# - test_entropy.json (metadata)
# - test_entropy_comparison.png (visualization)
```

Verify the output looks reasonable:
- Black regions = empty/unused
- Gray = regular logic
- White = dense routing

### Step 2.2: Batch Processing

Process entire dataset:

```bash
# Sequential (slow but reliable)
python cli/preprocess_dataset.py \
    --input data/raw/arria10 \
    --output data/images/arria10

# Parallel (fast, uses 8 CPU cores)
python cli/preprocess_dataset.py \
    --input data/raw/arria10 \
    --output data/images/arria10 \
    --parallel 8 \
    --summary preprocessing_summary.json
```

**Expected Duration**:
- Sequential: 15-40 seconds/bitstream × 100 samples = 25-67 minutes
- Parallel (8 cores): ~4-9 minutes

**Output**:
```
data/images/arria10/
  clean/
    seed_001_design_entropy.png
    seed_001_design_entropy.json
    ...
  infected/
    timebomb_001_design_entropy.png
    ...
```

---

## Phase 3: Model Training

Train the ResNet-50 CNN on preprocessed images.

### Step 3.1: Verify Dataset

```bash
# Check dataset is properly loaded
python models/arria10_cnn/dataset.py data/images/arria10

# Expected output:
# Total samples: 100
# Class distribution:
#   clean: 50
#   infected: 50
#   balance_ratio: 1.0
```

### Step 3.2: Train CNN

```bash
# Basic training (50 epochs, pretrained ResNet-50)
python cli/train_model.py \
    --data data/images/arria10 \
    --output data/models/arria10_cnn.pt \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4

# Advanced: longer training with frozen backbone
python cli/train_model.py \
    --data data/images/arria10 \
    --output data/models/arria10_cnn.pt \
    --epochs 100 \
    --freeze-backbone \
    --device cuda
```

**Training Output** (live):
```
Epoch 1/50 [Train]: 100%|███████| 25/25 [00:15<00:00, loss: 0.4231, acc: 78.12%]
Epoch 1/50 [Val]:   100%|███████| 7/7 [00:03<00:00, loss: 0.3124, acc: 85.00%]

Epoch 1/50 Summary:
  Train Loss: 0.4231 | Train Acc: 78.12%
  Val Loss:   0.3124 | Val Acc:   85.00%
  Val Precision: 87.50% | Recall: 82.35% | F1: 84.85%
  ✓ Best model saved (val_acc: 85.00%)
======================================================================
...
```

**Expected Duration**: 2-5 hours on GPU, 8-20 hours on CPU

**Expected Performance** (from research):
- Validation accuracy: 85-95%
- False positive rate: <5%
- F1 score: >90%

**Outputs**:
```
data/models/
  arria10_cnn.pt         # Trained model checkpoint
  arria10_cnn.json       # Training history
```

### Step 3.3: Evaluate Model

```bash
# Test on held-out samples
python models/arria10_cnn/infer_cnn.py \
    --model data/models/arria10_cnn.pt \
    --image data/images/arria10/clean/seed_050_design_entropy.png

# Expected: "Prediction: CLEAN"
```

---

## Phase 4: Inference

### Step 4.1: Single Bitstream Analysis

Analyze a new (unknown) bitstream:

```bash
# End-to-end analysis (.rbf → entropy → CNN → verdict)
python cli/run_inference.py \
    --rbf /path/to/suspicious_design.rbf \
    --model data/models/arria10_cnn.pt \
    --threshold 0.9 \
    --save-entropy

# Exit code: 0 = clean, 1 = infected
```

**Output**:
```
======================================================================
FPGA BITSTREAM TROJAN ANALYSIS
======================================================================

[1/2] Preprocessing bitstream to entropy map...
      Input: suspicious_design.rbf
      Format: rbf
      Detected width: 1024 bits
      Entropy stats: mean=0.6234, anomalies=15

[2/2] Running CNN Trojan detector...
      Model: arria10_cnn.pt
      Trojan probability: 0.9523

======================================================================
FINAL VERDICT
======================================================================

  Classification:  INFECTED
  Probability:     0.9523 (95.23%)
  Confidence:      0.9523 (95.23%)

  ⚠ WARNING: Hardware Trojan Detected!

  RISK ASSESSMENT:
    Severity: HIGH (>90% confidence)
    Recommendation: DO NOT DEPLOY

  SUGGESTED ACTIONS:
    1. DO NOT flash this bitstream to hardware
    2. Review build logs for anomalies
    3. Verify HDL source integrity
    4. Check for unauthorized ECO changes
    5. Re-compile with different seed
    6. Compare with known-good bitstreams

======================================================================
```

### Step 4.2: Batch Analysis

Scan multiple bitstreams:

```bash
#!/bin/bash
# batch_scan.sh

MODEL="data/models/arria10_cnn.pt"

for rbf in bitstreams/*.rbf; do
    echo "Analyzing: $rbf"
    python cli/run_inference.py \
        --rbf "$rbf" \
        --model "$MODEL" \
        --threshold 0.9 \
        --json "results/$(basename $rbf .rbf).json" \
        --quiet

    if [ $? -eq 1 ]; then
        echo "  ⚠ INFECTED detected!"
    else
        echo "  ✓ Clean"
    fi
done
```

---

## Phase 5: Integration

### Integration with HNTF Build System

Add pre-deployment verification to your FPGA build flow:

**File**: `flows/quartus/a10ped/build_tile.sh`

```bash
#!/bin/bash
# ... (existing build script) ...

# After Quartus compilation
echo "Compiling FPGA design..."
quartus_sh --flow compile a10ped_tile0

# ML-based Trojan verification (add this section)
echo "Running ML bitstream verification..."
python3 /path/to/fpga-bitstream-ml/cli/run_inference.py \
    --rbf output_files/a10ped_tile0.rbf \
    --model /path/to/arria10_cnn.pt \
    --threshold 0.90 \
    --quiet

VERIFICATION_RESULT=$?

if [ $VERIFICATION_RESULT -eq 0 ]; then
    echo "✓ Bitstream verification passed"

    # Safe to flash
    echo "Programming FPGA..."
    quartus_pgm -m jtag -o "p;output_files/a10ped_tile0.sof"

else
    echo "✗ Bitstream verification FAILED"
    echo "ERROR: Trojan detected! Aborting flash."
    exit 1
fi
```

### Continuous Integration

For automated builds (GitHub Actions, Jenkins, etc.):

```yaml
# .github/workflows/fpga_verify.yml
name: FPGA Bitstream Verification

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r fpga-bitstream-ml/env/requirements.txt

      - name: Download trained model
        run: wget https://your-server/models/arria10_cnn.pt

      - name: Verify bitstream
        run: |
          python fpga-bitstream-ml/cli/run_inference.py \
            --rbf artifacts/bitstream.rbf \
            --model arria10_cnn.pt \
            --threshold 0.95
```

---

## Troubleshooting

### Issue: "No module named 'numpy'"

```bash
# Install dependencies
pip install -r env/requirements.txt

# Or individually
pip install numpy scipy torch torchvision scikit-image matplotlib
```

### Issue: "Bitstream has high entropy (compressed)"

For Arria 10, disable compression:

```tcl
# In your Quartus project settings or Tcl script:
set_global_assignment -name GENERATE_COMPRESSED_SOF OFF
```

Then recompile.

### Issue: "Width detection gives inconsistent results"

This can happen with very small bitstreams. Solutions:
1. Use manual width specification: `--width 1024`
2. Ensure bitstream is uncompressed
3. Try multiple candidate widths and visually inspect

### Issue: "Training accuracy stuck at ~50%"

This indicates the model is not learning. Possible causes:
1. **Insufficient data**: Need at least 50+ samples per class
2. **Invalid Trojan insertion**: Check that infected samples are actually different
3. **Data imbalance**: Ensure roughly equal clean/infected samples
4. **Corrupted images**: Verify entropy maps look reasonable

### Issue: "Quartus ECO commands fail"

The `inject_trojan.tcl` script is a template. For production:
1. Consult Quartus ECO documentation for your device family
2. Test on a simple design first
3. Use Trust-Hub benchmarks as reference implementations
4. Consider simulated infections for initial development

---

## Performance Benchmarks

Based on validation with reference hardware:

| Operation | Duration (typical) | Notes |
|-----------|-------------------|-------|
| Quartus compile (seed) | 5-30 min | Device dependent |
| ECO Trojan injection | 2-10 min | When implemented |
| RBF → entropy (single) | 15-40 sec | CPU dependent |
| RBF → entropy (batch, 8 cores) | 2-5 sec/file | Parallelized |
| CNN training (50 epochs) | 2-5 hours | GPU recommended |
| CNN inference (single) | ~200ms | CPU sufficient |

**Total Pipeline (100 clean + 100 infected)**:
- Data generation: 10-60 hours (highly parallelizable)
- Preprocessing: 5-15 minutes (parallel)
- Training: 2-5 hours (GPU)
- Ready for deployment!

---

## Next Steps

1. **Start small**: Test with 10 clean + 10 infected samples first
2. **Validate visually**: Check entropy maps look reasonable
3. **Monitor training**: Watch for overfitting (val_loss increasing)
4. **Iterate**: Tune hyperparameters based on your specific designs
5. **Deploy**: Integrate into your build system
6. **Maintain**: Retrain periodically as designs evolve

For questions or issues, see README.md or open a GitHub issue.
