# ML-Assisted Intel FPGA Bitstream Analysis

*Draft technical documentation*

---

## Abstract

This document describes a machine learning-based system for analyzing Intel FPGA bitstreams to detect anomalous modifications (Hardware Trojans). The system combines:

1. **Static bitstream forensics** using CNNs trained on 2D entropy representations
2. **Automated dataset generation** via Quartus ECO-based Trojan injection
3. **Side-channel analysis** for encrypted designs using power trace classification

Target platforms: Intel Arria 10 (BittWare A10PED) and Stratix 10 (commercial miners).

---

## 1. Introduction

### 1.1 Problem Statement

Modern FPGA designs face supply chain risks:

- **Third-party IP cores**: May contain hidden backdoors
- **Untrusted compilation**: Cloud-based Quartus may be compromised
- **Firmware updates**: Over-the-air updates could inject malicious logic
- **Used hardware**: Second-hand boards may have modified bitstreams

Traditional verification methods (formal verification, gate-level inspection) are:

- **Too slow**: Can't analyze full bitstreams in reasonable time
- **Too limited**: Require golden reference or source code
- **Ineffective on encrypted designs**: Modern FPGAs use bitstream encryption

### 1.2 Proposed Solution

Use machine learning to:

1. Learn "normal" bitstream patterns from clean designs
2. Detect statistical anomalies indicative of Trojans
3. Work on raw bitstreams without source code
4. Extend to side-channel analysis for encrypted designs

---

## 2. System Architecture

### 2.1 Overview

```
┌─────────────────────┐
│  Quartus Pro        │
│  + Tcl Scripts      │
│                     │
│  Generate:          │
│  - Clean designs    │
│  - Trojan variants  │
└──────────┬──────────┘
           │
           │ .sof/.rbf files
           ▼
┌─────────────────────┐
│  Preprocessing      │
│                     │
│  1. Extract config  │
│  2. Bytes → bits    │
│  3. Guess width     │
│  4. Fold to 2D      │
│  5. Compute entropy │
└──────────┬──────────┘
           │
           │ PNG images
           ▼
┌─────────────────────┐
│  CNN Training       │
│                     │
│  BitstreamCNN:      │
│  - 3 conv layers    │
│  - Global pooling   │
│  - Binary classify  │
└──────────┬──────────┘
           │
           │ trained model
           ▼
┌─────────────────────┐
│  Inference          │
│                     │
│  Analyze unknown    │
│  bitstreams         │
└─────────────────────┘
```

### 2.2 Static Analysis Pipeline

#### Step 1: Bitstream Extraction

Intel `.sof` (SRAM Object File) format:

```
Header (0x18 bytes)
├── Magic: 0x00 0x09 0x0f 0xf0
├── Pointers to sections
└── Metadata

Section 1: SDM Firmware
Section 2: Design Configuration ← WE WANT THIS
Section 3: HPS software (if applicable)
```

Extract only the configuration section that programs the FPGA fabric.

#### Step 2: Bit Expansion

Convert bytes to bit vector:

```
0xA5 → [1, 0, 1, 0, 0, 1, 0, 1]  (MSB first)
```

Result: 1D array of ~50-500 million bits (depending on device size).

#### Step 3: Width Detection

**Problem**: Configuration bitstreams have 2D structure (frames × frame_width), but width is not documented.

**Solution**: Autocorrelation-based heuristic:

```python
for candidate_width in [256, 512, 768, 1024, 1536, 2048, ...]:
    reshape bits to (H, candidate_width)
    compute column-wise entropy
    compute row autocorrelation
    score = variance(col_entropy) + max(autocorr)

best_width = argmax(score)
```

Intuition: Correct width aligns LAB/ALM columns vertically, creating periodic structure.

#### Step 4: Entropy Mapping

Compute local Shannon entropy in sliding windows:

```python
for window in sliding_windows(bit_image, size=16, stride=4):
    hist = histogram(window, bins=2)  # {0, 1}
    H = -Σ p(x) log₂ p(x)
    entropy_map[i, j] = H
```

Result: 2D grayscale image where bright = high entropy (random/dense logic).

#### Step 5: CNN Classification

**Input**: Entropy map (1 channel, variable size)
**Architecture**:

```
Conv2d(1, 16, k=3) + BN + ReLU + MaxPool(2)
Conv2d(16, 32, k=3) + BN + ReLU + MaxPool(2)
Conv2d(32, 64, k=3) + BN + ReLU + MaxPool(2)
AdaptiveAvgPool2d((4, 4))
Flatten
Linear(64*4*4, 128) + ReLU + Dropout(0.5)
Linear(128, 2)  # [clean, trojan]
```

**Training**:
- Dataset: 1000s of clean + Trojan variants from A10PED
- Loss: Cross-entropy
- Optimizer: Adam, lr=1e-3
- Augmentation: Random flips, crops

**Inference**:
- Input: Unknown bitstream from K10 miner
- Output: P(trojan) ∈ [0, 1]

### 2.3 Trojan Injection (Dataset Generation)

Use Quartus ECO (Engineering Change Order) flow:

```tcl
# Load compiled design
project_open my_design
initialize_eco

# Find a high-fanout net (e.g., clock, reset)
set target_net [get_nets -high_fanout 1]

# Create new LUT instance
create_node -type lut -name trojan_lut_0

# Connect to target net (leaks information)
make_connection -from $target_net -to trojan_lut_0 -pin IN[0]

# Route output to unused IO or internal node
make_connection -from trojan_lut_0 -to unused_io_pin

finalize_eco
project_close

# Recompile incrementally
execute_flow -compile my_design
```

Result: Bitstream with small, localized modification (1-100 LUTs).

### 2.4 Side-Channel Branch (Encrypted Designs)

For Stratix 10 with SDM encryption:

```
Power Supply
     │
     ├─── Shunt Resistor (0.1Ω)
     │         │
     │    Differential Amp
     │         │
     │    ADC (12-bit, 1 GSPS)
     │         │
     ▼         ▼
   FPGA   Oscilloscope
            │
            ▼
    Capture Traces (1D time series)
            │
            ▼
    Preprocessing:
    - DC offset removal
    - Z-score normalization
    - Bandpass filter (1-100 MHz)
            │
            ▼
    1D CNN Classifier:
    Conv1d(1, 16, k=9) + ReLU + MaxPool(2)
    Conv1d(16, 32, k=9) + ReLU + MaxPool(2)
    Conv1d(32, 64, k=9) + ReLU + MaxPool(2)
    AdaptiveAvgPool1d(16)
    Flatten
    Linear(64*16, 128) + ReLU
    Linear(128, 2)
```

**Key Insight**: Even if bitstream is encrypted, power consumption during configuration or runtime may leak information about Trojan presence.

---

## 3. Experimental Results

*(To be filled in after running experiments)*

### 3.1 Dataset Statistics

- Clean designs: XXX samples
- Trojan designs: XXX samples
- Trojan types: LUT insertion, routing modification, IO leakage

### 3.2 Static Analysis Results

- Training accuracy: XX%
- Validation accuracy: XX%
- Test accuracy on A10PED: XX%
- Test accuracy on K10 (if extractable): XX%

### 3.3 Side-Channel Results

- Trace length: XXX samples
- Sampling rate: XXX GSPS
- Classification accuracy: XX%

---

## 4. Limitations and Future Work

### 4.1 Current Limitations

1. **Width detection is heuristic**: May fail on highly optimized or sparse designs
2. **Trojan insertion is synthetic**: Real Trojans may have different signatures
3. **No golden reference**: Requires training on known-clean designs
4. **Scalability**: Large Stratix 10 designs (>1M LUTs) produce huge bitstreams

### 4.2 Future Directions

1. **Transfer learning**: Pre-train on Arria 10, fine-tune on Stratix 10
2. **Unsupervised anomaly detection**: Use autoencoders to detect outliers without labels
3. **Multi-modal fusion**: Combine static + side-channel features
4. **Real Trojan benchmarks**: Test on Trust-Hub or other academic Trojan collections
5. **Timing analysis**: Correlate bitstream features with critical path delays

---

## 5. Conclusion

This system demonstrates feasibility of ML-based FPGA bitstream analysis for Hardware Trojan detection. By combining static forensics with side-channel analysis, we can handle both unencrypted (A10) and encrypted (S10) designs.

**Key Contributions**:

1. Automated dataset generation via Quartus ECO
2. Width-agnostic 2D bitstream representation
3. End-to-end pipeline from `.sof` to classification
4. Dual-mode architecture (static + side-channel)

---

## References

1. Intel Arria 10 Device Handbook
2. Intel Stratix 10 Configuration User Guide
3. Trust-Hub Hardware Trojan Benchmarks
4. Relevant academic papers on ML for hardware security
