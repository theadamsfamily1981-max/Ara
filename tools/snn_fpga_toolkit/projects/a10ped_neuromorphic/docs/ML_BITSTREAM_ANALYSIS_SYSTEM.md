# ML-Assisted Bitstream Analysis System for Intel FPGAs

**Static Analysis Pipeline for Self-Diagnostics and Integrity Checking**

This document describes the ML-assisted static analysis pipeline for Intel FPGA bitstreams used in HNTF. The system operates strictly on self-compiled designs for validation, anomaly detection, and research purposes.

---

## Executive Summary

**Goal**: Build an ML-assisted static analysis pipeline that can:
1. Validate configuration structure of compiled bitstreams
2. Detect anomalous modifications (faults, synthetic trojans)
3. Provide research framework for understanding vendor formats
4. Enable pre-deployment integrity checking

**Scope**:
- ✅ Self-compiled designs on owned hardware
- ✅ Unencrypted, uncompressed bitstreams
- ✅ Defensive security and diagnostics
- ❌ Not for attacking third-party IP
- ❌ Not for circumventing encryption

**Devices**:
- **Arria 10** (A10PED, HawkEye): Primary target, simpler format
- **Stratix 10** (future): More complex with SDM firmware layer
- **Stratix V** (Azure card): Legacy format, research only

---

## 1. Paper-Ready Section

**For inclusion in HNTF paper methods section:**

### 7.x ML-Assisted Bitstream Analysis for Intel Arria/Stratix Devices

Our goal is to build an ML-assisted static analysis pipeline for Intel FPGA bitstreams that can support *self-diagnostics* and *integrity checking* on heterogeneous boards (Arria 10, Stratix 10) used in the HNTF. This system operates strictly on designs we compile ourselves, and is intended to (a) validate configuration structure, (b) detect anomalous modifications (e.g. injected faults or synthetic Trojans), and (c) provide a research framework for understanding vendor formats.

#### 7.x.1 Input Formats and Structural Constraints

We focus on two closely related configuration artifacts:

* **Arria 10**: `.sof` (SRAM Object File) and `.rbf` (raw binary file). These contain configuration packets for the core fabric, including LAB/ALM configuration and I/O, with no Secure Device Manager (SDM) layer.
* **Stratix 10**: `.sof` and `.rbf` with a more complex, layered structure. In particular:

  * A **firmware section** that configures the on-chip Secure Device Manager (SDM), followed by
  * One or more **design sections** that configure the user logic fabric.

In Stratix 10 files we must therefore **strip SDM firmware and header blocks** (e.g. fixed-size pointer block headers) before we can interpret the bitstream as a representation of user logic. This motivates a dedicated preprocessing stage.

We adopt a conservative security stance: **encrypted and/or compressed bitstreams are treated as opaque**. Where compression or encryption is enabled, we restrict ourselves to header-level analysis and meta-data; ML feature extraction is only performed on **uncompressed, unencrypted** configuration data (e.g. `.sof` before compression, or `.rbf` generated with compression disabled).

#### 7.x.2 Preprocessing Pipeline

The preprocessing stage normalizes heterogeneous Intel formats into a canonical byte sequence suitable for downstream ML.

1. **Format detection and parsing**

   * Inspect magic values and header signatures to distinguish:

     * `.sof` vs `.rbf`,
     * Arria 10 vs Stratix 10,
     * presence of SDM firmware sections.
   * For Stratix 10, use documented pointer block header sizes and offsets to locate the first design section.

2. **Header and firmware stripping**

   * For Arria 10, discard top-level container headers while preserving the configuration packet stream.
   * For Stratix 10, remove SDM firmware and non-fabric metadata so that the remaining bytes correspond as closely as possible to configuration RAM (CRAM) contents.

3. **Compression handling**

   * Detect whether bitstream compression is enabled (via configuration options in Quartus and/or header flags).
   * When possible, **generate uncompressed variants** via Quartus tooling (e.g. `quartus_cpf` with compression disabled) and treat the uncompressed data as the primary ML input.
   * If compression cannot be disabled for a given device or flow, we treat these artifacts as out-of-scope for structural ML (high-entropy compressed streams are not informative about logic structure).

4. **Normalization**

   * Normalize endianness and packet ordering where necessary.
   * Optionally pad or truncate to a fixed length window for batching; longer designs are split into segments.

The output of this stage is a **clean, uncompressed, header-stripped byte sequence** that approximates the configuration of the user logic fabric.

#### 7.x.3 Recovering Configuration Geometry

A central technical challenge is that Intel does not publicly specify the **configuration frame geometry** (frame length, sector size, logical-to-physical mapping) for Arria 10 and Stratix 10. This geometry is required if we want to reshape 1-D byte streams into 2-D "images" that align with Logic Array Block (LAB) columns for CNN-based analysis.

Instead of hard-coding vendor-specific values, we propose an **algorithmic geometry recovery module**:

* Given a 1-D byte sequence, we search over candidate "image widths" and use:

  * **Autocorrelation**: to detect periodicity corresponding to repeating LAB/column structures.
  * **Entropy minimization**: to find widths that produce images with visible structure (lower local entropy) rather than uniform noise.
* The system selects one or a small set of candidate widths that maximize structural regularity according to these metrics, and uses those widths to generate 2-D representations.

This approach avoids relying on proprietary documentation and allows the same ML infrastructure to adapt across device families.

#### 7.x.4 Feature Extraction

Once we have a normalized byte stream (and optionally a 2-D arrangement), we extract several feature families:

1. **Image-like features**

   * Interpret the bitstream (or segments) as 2-D grayscale images:

     * Raw byte values,
     * Entropy maps (local entropy over sliding windows),
     * Gradient / edge maps.
   * Feed these into CNNs or vision transformers trained to distinguish "normal" vs "mutated" or "anomalous" configurations.

2. **Sequence-based features**

   * Treat the bitstream as a 1-D sequence and extract:

     * N-gram frequency statistics,
     * 1-D convolutional features (similar to malware binary classifiers),
     * Learned embeddings from 1-D CNNs.

3. **Global statistical features**

   * Overall entropy, byte histograms, run-length distributions, etc., used as low-dimensional features for simpler models (e.g., random forests, SVMs) or as auxiliary inputs to deep models.

These features are designed to be **agnostic to specific LAB/ALM encodings** while still being sensitive to consistent structural changes.

#### 7.x.5 Dataset Generation via Automated Design Variants

To train and validate the ML models, we require labeled pairs of "baseline" and "modified" bitstreams. Rather than relying on third-party IP or real malicious designs, we generate **synthetic but controlled variants** of our own designs using Quartus' scripting and ECO (Engineering Change Order) capabilities:

* Use Tcl scripting to compile a base design (e.g., neuromorphic tile shell, small test cores) across many seeds and minor parameter variations.
* Use the ECO flow to introduce **small, localized netlist edits**, such as:

  * Inserting or removing LUT-level logic,
  * Modifying specific connections or constants,
  * Injecting small "Trojan-like" subcircuits from standard benchmark suites (e.g., Trust-Hub) where licensing permits.
* For each variant, regenerate the `.sof` / `.rbf` bitstreams with compression disabled.
* Label bitstreams as:

  * **Clean** (baseline or benign variations),
  * **Modified** (synthetic perturbations, Trojans, or other deliberate changes).

This yields a large-scale, fully controlled dataset that can be used to train both **supervised classifiers** (clean vs modified) and **unsupervised/anomaly detection** models.

#### 7.x.6 Model Training and Evaluation

We train multiple model families:

* Supervised CNNs / 1-D CNNs on image/sequence features for binary classification.
* Autoencoders and density models for unsupervised anomaly detection on new designs.
* Simple baselines (entropy thresholds, statistical tests) to quantify the value added by deep models.

Evaluation is performed under several conditions:

* **Intra-family**: train and test on the same device family and similar design types (e.g., Arria 10 neuromorphic tiles).
* **Cross-family**: train on Arria 10, test on Stratix 10 (or vice versa) to probe generalization.
* **Perturbation sensitivity**: measure detection performance as a function of perturbation size and location (e.g., small ECO edits vs larger Trojans).

The end result is not a security oracle but a **diagnostic tool**: given two bitstreams or a bitstream and a reference distribution, it can flag unusual structural deviations and provide a quantitative "distance" from expected designs.

#### 7.x.7 Scope and Limitations

We explicitly limit the system to:

* Designs we compile ourselves, on our own hardware.
* Unencrypted, uncompressed (or pre-decompression) configuration artifacts.
* Use as a **self-diagnostic, integrity-checking, and research tool**, not as a method for attacking third-party designs.

Encrypted bitstreams and side-channel analysis (e.g., power-based attacks on encryption keys) are treated as a separate research topic and are out-of-scope for the static ML pipeline described here.

---

## 2. Practical Implementation for A10PED (Arria 10)

### 2.1 Why Start with Arria 10?

**Advantages**:
- ✅ No Secure Device Manager (SDM) firmware section
- ✅ Simpler `.sof`/`.rbf` format
- ✅ Direct access to configuration data
- ✅ Compression is more controllable
- ✅ We have A10PED hardware for validation

**Strategy**:
1. Build preprocessing pipeline for Arria 10
2. Generate training dataset from A10PED tile variants
3. Train initial models
4. Extend to Stratix 10 when Gemini finishes format excavation

### 2.2 Arria 10 File Format Overview

**`.sof` (SRAM Object File)**:
```
┌─────────────────────────────────────┐
│ SOF Header                          │ ~4KB (metadata, timestamps, etc.)
├─────────────────────────────────────┤
│ Configuration Packet Stream         │ Main payload
│  - ALM/LAB configuration            │
│  - I/O configuration                │
│  - Routing configuration            │
│  - Hard IP settings (PCIe, DDR)     │
├─────────────────────────────────────┤
│ Footer/Checksum                     │ Small
└─────────────────────────────────────┘
```

**`.rbf` (Raw Binary File)**:
- More compact than `.sof`
- Less metadata overhead
- Closer to actual CRAM bits
- Preferred for ML analysis

**Key Characteristics**:
- Magic bytes: `0x00 0x09 0x0f 0xf0` (Altera SOF)
- Uncompressed size: ~50-200 MB for GX1150
- Compressed size: ~5-20 MB (if compression enabled)

### 2.3 Phase 1: Preprocessing Pipeline

**Goal**: Transform `.sof`/`.rbf` → clean byte sequence + metadata

**Implementation**: `tools/bitstream_analysis/arria10_preprocessor.py`

```python
#!/usr/bin/env python3
"""
Arria 10 Bitstream Preprocessor

Extracts clean configuration data from .sof/.rbf files for ML analysis.

Usage:
    python arria10_preprocessor.py input.sof --output output_dir/
"""

import struct
import numpy as np
from pathlib import Path
import json

class Arria10BitstreamPreprocessor:
    """Preprocess Arria 10 .sof/.rbf files"""

    # Magic bytes for SOF format
    SOF_MAGIC = b'\x00\x09\x0f\xf0'

    def __init__(self, input_path):
        self.input_path = Path(input_path)
        self.data = self.input_path.read_bytes()
        self.metadata = {}

    def detect_format(self):
        """Detect SOF vs RBF and compression"""
        if self.data[:4] == self.SOF_MAGIC:
            self.metadata['format'] = 'sof'

            # Check for compression indicators
            # (Look for high entropy or specific compression headers)
            entropy = self.compute_entropy(self.data[1024:5120])  # Sample
            self.metadata['likely_compressed'] = entropy > 7.5

        else:
            # Assume RBF (no standard magic)
            self.metadata['format'] = 'rbf'
            entropy = self.compute_entropy(self.data[:4096])
            self.metadata['likely_compressed'] = entropy > 7.5

        return self.metadata['format']

    def strip_headers(self):
        """Remove SOF container headers"""
        if self.metadata['format'] == 'sof':
            # SOF header is typically first ~4KB
            # Look for start of configuration packets

            # Heuristic: find first repeating pattern or low-entropy region
            # after initial header
            HEADER_SIZE = 4096  # Conservative estimate

            self.clean_data = self.data[HEADER_SIZE:]

            # Remove footer (last ~256 bytes often metadata/checksums)
            FOOTER_SIZE = 256
            self.clean_data = self.clean_data[:-FOOTER_SIZE]

        else:  # RBF
            # RBF has minimal overhead
            self.clean_data = self.data

        self.metadata['clean_size'] = len(self.clean_data)
        return self.clean_data

    def compute_entropy(self, data):
        """Compute Shannon entropy of byte sequence"""
        if len(data) == 0:
            return 0.0

        # Count byte frequencies
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts / len(data)

        # Remove zero probabilities
        probabilities = probabilities[probabilities > 0]

        # Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def entropy_profile(self, window_size=1024):
        """Compute entropy over sliding windows"""
        data = self.clean_data
        num_windows = len(data) // window_size

        entropies = []
        for i in range(num_windows):
            window = data[i*window_size:(i+1)*window_size]
            entropies.append(self.compute_entropy(window))

        return np.array(entropies)

    def find_geometry_candidates(self, max_width=2048):
        """Find candidate image widths via autocorrelation"""
        data = np.frombuffer(self.clean_data, dtype=np.uint8)

        # Limit to first 10MB for speed
        if len(data) > 10_000_000:
            data = data[:10_000_000]

        candidates = []

        # Try candidate widths
        for width in [256, 512, 1024, 1536, 2048]:
            if width > len(data):
                continue

            # Reshape into 2D
            height = len(data) // width
            if height < 10:
                continue

            image = data[:height*width].reshape(height, width)

            # Compute autocorrelation along columns
            # (Looking for repeating vertical structure)
            col_autocorr = np.mean([
                np.correlate(image[:, i], image[:, i], mode='same')[height//2]
                for i in range(min(width, 100))  # Sample columns
            ])

            # Compute local entropy variance
            # (Lower variance = more structure)
            local_entropy_var = np.var([
                self.compute_entropy(image[i:i+16, :].tobytes())
                for i in range(0, height-16, 16)
            ])

            score = col_autocorr / (local_entropy_var + 1e-6)

            candidates.append({
                'width': width,
                'score': score,
                'autocorr': col_autocorr,
                'entropy_var': local_entropy_var
            })

        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates

    def generate_2d_views(self, output_dir, top_n=3):
        """Generate 2D image views for top candidate widths"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        candidates = self.find_geometry_candidates()

        images = []
        for i, cand in enumerate(candidates[:top_n]):
            width = cand['width']
            data = np.frombuffer(self.clean_data, dtype=np.uint8)

            height = len(data) // width
            image = data[:height*width].reshape(height, width)

            # Save as PNG
            from PIL import Image
            img = Image.fromarray(image, mode='L')
            img_path = output_dir / f"geometry_w{width}_rank{i}.png"
            img.save(img_path)

            images.append({
                'path': str(img_path),
                'width': width,
                'height': height,
                'score': cand['score']
            })

        return images

    def generate_statistics(self):
        """Generate global statistics"""
        data = np.frombuffer(self.clean_data, dtype=np.uint8)

        stats = {
            'size_bytes': len(data),
            'entropy': self.compute_entropy(self.clean_data),
            'byte_histogram': np.bincount(data, minlength=256).tolist(),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': int(np.min(data)),
            'max': int(np.max(data)),
        }

        # Run-length encoding statistics
        runs = []
        current_byte = data[0]
        current_length = 1

        for byte in data[1:1000000]:  # Sample first 1MB
            if byte == current_byte:
                current_length += 1
            else:
                runs.append(current_length)
                current_byte = byte
                current_length = 1

        stats['run_length_mean'] = float(np.mean(runs)) if runs else 0
        stats['run_length_max'] = int(np.max(runs)) if runs else 0

        return stats

    def process(self, output_dir):
        """Complete preprocessing pipeline"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[1/6] Detecting format...")
        format_type = self.detect_format()
        print(f"      Format: {format_type}")
        print(f"      Likely compressed: {self.metadata['likely_compressed']}")

        if self.metadata['likely_compressed']:
            print("      ⚠️  Warning: Bitstream appears compressed")
            print("      Regenerate with compression disabled for better analysis")

        print(f"\n[2/6] Stripping headers...")
        self.strip_headers()
        print(f"      Clean size: {self.metadata['clean_size']:,} bytes")

        print(f"\n[3/6] Computing entropy profile...")
        entropy_profile = self.entropy_profile()
        np.save(output_dir / 'entropy_profile.npy', entropy_profile)

        # Plot entropy
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(entropy_profile)
        plt.xlabel('Window Index (1KB windows)')
        plt.ylabel('Shannon Entropy')
        plt.title('Entropy Profile')
        plt.grid(True)
        plt.savefig(output_dir / 'entropy_profile.png', dpi=150)
        plt.close()
        print(f"      Saved: entropy_profile.png")

        print(f"\n[4/6] Finding geometry candidates...")
        candidates = self.find_geometry_candidates()
        print(f"      Top candidates:")
        for i, c in enumerate(candidates[:3]):
            print(f"        {i+1}. Width {c['width']:4d} - Score: {c['score']:.2f}")

        print(f"\n[5/6] Generating 2D views...")
        images = self.generate_2d_views(output_dir, top_n=3)
        for img in images:
            print(f"      {img['path']} ({img['width']}x{img['height']})")

        print(f"\n[6/6] Computing statistics...")
        stats = self.generate_statistics()
        stats_path = output_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"      Saved: statistics.json")

        # Save metadata
        self.metadata['statistics'] = stats
        self.metadata['geometry_candidates'] = candidates
        self.metadata['images'] = images

        meta_path = output_dir / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"\n✅ Preprocessing complete!")
        print(f"   Output directory: {output_dir}")

        return self.metadata


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess Arria 10 bitstreams for ML analysis'
    )
    parser.add_argument('input', help='Input .sof or .rbf file')
    parser.add_argument('--output', default='./bitstream_analysis',
                        help='Output directory')

    args = parser.parse_args()

    preprocessor = Arria10BitstreamPreprocessor(args.input)
    preprocessor.process(args.output)


if __name__ == '__main__':
    main()
```

**Usage**:

```bash
# Process a bitstream
python tools/bitstream_analysis/arria10_preprocessor.py \
  out/a10ped/build/output_files/a10ped_tile0.sof \
  --output analysis/tile0_baseline/

# Output:
# analysis/tile0_baseline/
#   ├── entropy_profile.png        # Visualization
#   ├── entropy_profile.npy        # Raw data
#   ├── geometry_w256_rank0.png    # Best width candidate
#   ├── geometry_w512_rank1.png
#   ├── geometry_w1024_rank2.png
#   ├── statistics.json            # Global stats
#   └── metadata.json              # Complete metadata
```

---

## 3. Dataset Generation Workflow

### 3.1 Phase 1: Simple Variations (Start Here)

**Goal**: Build initial dataset without ECO/Trojans

**Method**: Vary compilation parameters

```python
# tools/bitstream_analysis/generate_training_corpus.py

import subprocess
from pathlib import Path

def build_variant(base_qpf, output_dir, **params):
    """Build a design variant with given parameters"""

    # Modify Quartus project
    # (Use Tcl scripting to change settings)

    tcl_script = f"""
    project_open {base_qpf}

    # Change fitter seed
    set_global_assignment -name SEED {params.get('seed', 1)}

    # Set optimization mode
    set_global_assignment -name OPTIMIZATION_MODE "{params.get('opt_mode', 'BALANCED')}"

    # Export and recompile
    export_assignments
    execute_flow -compile

    project_close
    """

    # Run Quartus
    subprocess.run(['quartus_sh', '-t', '-'], input=tcl_script.encode())

    # Copy outputs
    sof_path = Path(f"output_files/{base_qpf.stem}.sof")
    output_path = Path(output_dir) / f"variant_{params['name']}.sof"
    shutil.copy(sof_path, output_path)

    return output_path

# Generate corpus
variants = []

# Baseline
variants.append(build_variant('a10ped_tile.qpf', 'corpus/', name='baseline', seed=1))

# Seed variations
for seed in range(2, 11):
    variants.append(build_variant('a10ped_tile.qpf', 'corpus/',
                                   name=f'seed{seed}', seed=seed))

# Optimization mode variations
for opt in ['BALANCED', 'HIGH PERFORMANCE EFFORT', 'AGGRESSIVE AREA']:
    variants.append(build_variant('a10ped_tile.qpf', 'corpus/',
                                   name=f'opt_{opt.replace(" ", "_")}',
                                   seed=1, opt_mode=opt))

# Parameter variations (via HDL parameters)
# Modify Verilog parameters and recompile
# ...

print(f"Generated {len(variants)} variants")
```

### 3.2 Phase 2: Controlled Perturbations

**Goal**: Add small, known modifications

**Method**: Use Quartus ECO (Engineering Change Order)

```tcl
# tools/bitstream_analysis/eco_inject_trojan.tcl

# Open project
project_open a10ped_tile

# Read post-fit netlist
read_netlist output_files/a10ped_tile.vqm

# Add a small circuit (e.g., rare-trigger counter)
# This is simplified - actual ECO commands are device-specific

# Create new nodes
create_node trojan_counter[0] -type dff
create_node trojan_counter[1] -type dff
create_node trojan_trigger -type lut

# Connect to existing signals (very carefully)
# ...

# Write modified netlist
write_netlist output_files/a10ped_tile_modified.vqm

# Recompile from ECO
execute_flow -compile

project_close
```

**Python Wrapper**:

```python
# tools/bitstream_analysis/inject_perturbation.py

def inject_small_trojan(base_design, trojan_type, output_path):
    """
    Inject a small trojan circuit using ECO flow

    trojan_type: 'counter', 'toggle', 'payload'
    """

    # Run ECO script
    subprocess.run([
        'quartus_sh', '-t',
        'tools/bitstream_analysis/eco_inject_trojan.tcl',
        base_design, trojan_type
    ])

    # Copy modified bitstream
    shutil.copy('output_files/modified.sof', output_path)

    # Return metadata
    return {
        'base': base_design,
        'trojan_type': trojan_type,
        'location': 'auto',  # Could parse from ECO report
        'size_luts': estimate_size(trojan_type)
    }
```

### 3.3 Dataset Organization

```
bitstream_corpus/
├── metadata.json                    # Corpus index
├── clean/
│   ├── baseline_seed1.sof
│   ├── baseline_seed2.sof
│   ├── param_neurons512.sof
│   ├── param_neurons1024.sof
│   └── ...
├── modified/
│   ├── trojan_counter_v1.sof
│   ├── trojan_toggle_v1.sof
│   ├── eco_lut_insert_v1.sof
│   └── ...
└── processed/
    ├── clean/
    │   ├── baseline_seed1/
    │   │   ├── entropy_profile.npy
    │   │   ├── geometry_w512_rank0.png
    │   │   └── statistics.json
    │   └── ...
    └── modified/
        └── ...
```

**Metadata Format**:

```json
{
  "corpus_version": "1.0",
  "device": "10AX115N2F40E2LG",
  "date_generated": "2025-11-24",

  "samples": [
    {
      "id": "baseline_seed1",
      "label": "clean",
      "sof_path": "clean/baseline_seed1.sof",
      "processed_dir": "processed/clean/baseline_seed1/",
      "metadata": {
        "seed": 1,
        "opt_mode": "BALANCED",
        "parameters": {"neurons": 512}
      }
    },
    {
      "id": "trojan_counter_v1",
      "label": "modified",
      "sof_path": "modified/trojan_counter_v1.sof",
      "processed_dir": "processed/modified/trojan_counter_v1/",
      "metadata": {
        "base_design": "baseline_seed1",
        "modification": "trojan_counter",
        "size_luts": 15
      }
    }
  ]
}
```

---

## 4. Feature Extraction and ML Models

### 4.1 Feature Extractor

```python
# tools/bitstream_analysis/feature_extractor.py

import numpy as np
import torch
from PIL import Image

class BitstreamFeatureExtractor:
    """Extract features from preprocessed bitstreams"""

    def __init__(self, processed_dir):
        self.processed_dir = Path(processed_dir)

        # Load preprocessed data
        self.entropy_profile = np.load(self.processed_dir / 'entropy_profile.npy')

        with open(self.processed_dir / 'statistics.json', 'r') as f:
            self.statistics = json.load(f)

        # Load best geometry image
        self.image = Image.open(self.processed_dir / 'geometry_w512_rank0.png')
        self.image_array = np.array(self.image)

    def extract_global_features(self):
        """Extract global statistical features"""
        return np.array([
            self.statistics['entropy'],
            self.statistics['mean'],
            self.statistics['std'],
            self.statistics['run_length_mean'],
            self.statistics['run_length_max'],
            # Add more as needed
        ])

    def extract_image_features(self, cnn_model=None):
        """Extract CNN features from 2D image"""
        if cnn_model is None:
            # Use simple features
            return np.concatenate([
                self.image_array.mean(axis=0),  # Column means
                self.image_array.std(axis=0),   # Column stds
                self.image_array.mean(axis=1)[:100]  # Row means (first 100)
            ])
        else:
            # Use pre-trained CNN
            img_tensor = torch.from_numpy(self.image_array).float()
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel
            with torch.no_grad():
                features = cnn_model(img_tensor)
            return features.numpy().flatten()

    def extract_sequence_features(self, seq_length=1024):
        """Extract 1-D sequence features"""
        # Load raw bytes (first seq_length)
        raw_bytes = np.load(self.processed_dir / 'clean_bytes.npy')[:seq_length]

        # N-gram frequencies (2-grams)
        ngrams = {}
        for i in range(len(raw_bytes) - 1):
            bigram = (raw_bytes[i], raw_bytes[i+1])
            ngrams[bigram] = ngrams.get(bigram, 0) + 1

        # Convert to feature vector (top 100 most common)
        sorted_ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)
        ngram_features = np.array([count for _, count in sorted_ngrams[:100]])

        return np.concatenate([
            raw_bytes[:100],  # First 100 bytes
            ngram_features     # N-gram frequencies
        ])

    def extract_all_features(self):
        """Extract all feature types"""
        return {
            'global': self.extract_global_features(),
            'image': self.extract_image_features(),
            'sequence': self.extract_sequence_features()
        }
```

### 4.2 Simple Baseline Classifier

```python
# tools/bitstream_analysis/train_baseline_classifier.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_baseline_classifier(corpus_dir):
    """Train a simple Random Forest classifier"""

    # Load corpus
    with open(corpus_dir / 'metadata.json', 'r') as f:
        corpus = json.load(f)

    X = []
    y = []

    for sample in corpus['samples']:
        extractor = BitstreamFeatureExtractor(sample['processed_dir'])

        # Use global features only for baseline
        features = extractor.extract_global_features()
        X.append(features)

        # Label: 0 = clean, 1 = modified
        y.append(0 if sample['label'] == 'clean' else 1)

    X = np.array(X)
    y = np.array(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"Train Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy:  {test_acc:.2%}")

    # Feature importance
    importances = clf.feature_importances_
    print("\nTop 5 Important Features:")
    for i in np.argsort(importances)[-5:][::-1]:
        print(f"  Feature {i}: {importances[i]:.4f}")

    # Save model
    joblib.dump(clf, 'models/baseline_rf_classifier.pkl')

    return clf
```

### 4.3 CNN-Based Classifier

```python
# tools/bitstream_analysis/train_cnn_classifier.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BitstreamImageDataset(Dataset):
    """Dataset for 2D bitstream images"""

    def __init__(self, corpus_dir):
        with open(corpus_dir / 'metadata.json', 'r') as f:
            self.corpus = json.load(f)

        self.samples = self.corpus['samples']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img_path = Path(sample['processed_dir']) / 'geometry_w512_rank0.png'
        img = Image.open(img_path).convert('L')  # Grayscale
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add channel dim

        # Label
        label = 0 if sample['label'] == 'clean' else 1

        return img_tensor, label

class BitstreamCNN(nn.Module):
    """CNN for bitstream image classification"""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        # Adaptive pooling to handle variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)  # Binary classification

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = torch.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def train_cnn_classifier(corpus_dir, epochs=50):
    """Train CNN classifier"""

    # Load dataset
    dataset = BitstreamImageDataset(corpus_dir)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = BitstreamCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), 'models/bitstream_cnn_classifier.pth')

    return model
```

---

## 5. Integration with HNTF Build Flow

### 5.1 Pre-Deployment Check

```bash
# flows/quartus/a10ped/build_tile.sh (updated)

# ... existing build steps ...

# After successful Quartus build:
echo "[Post-Build] Analyzing bitstream integrity..."

python3 tools/bitstream_analysis/arria10_preprocessor.py \
  out/a10ped/build/output_files/a10ped_tile0.sof \
  --output out/a10ped/analysis/current_build/

python3 tools/bitstream_analysis/verify_bitstream.py \
  out/a10ped/analysis/current_build/ \
  --model models/bitstream_cnn_classifier.pth \
  --threshold 0.90

if [ $? -ne 0 ]; then
  echo "⚠️  Bitstream verification FAILED!"
  echo "    Build may be corrupted or unexpected"
  exit 1
fi

echo "✅ Bitstream verification passed"
```

### 5.2 Continuous Monitoring

```python
# tools/bitstream_analysis/verify_bitstream.py

def verify_bitstream(processed_dir, model_path, threshold=0.90):
    """Verify a bitstream against trained model"""

    # Load model
    model = BitstreamCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Extract features
    extractor = BitstreamFeatureExtractor(processed_dir)
    img_path = Path(processed_dir) / 'geometry_w512_rank0.png'
    img = Image.open(img_path).convert('L')
    img_tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0) / 255.0

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        is_clean = torch.argmax(probabilities, dim=1).item() == 0
        confidence = probabilities[0, 0 if is_clean else 1].item()

    print(f"Bitstream Classification:")
    print(f"  Prediction: {'CLEAN' if is_clean else 'MODIFIED'}")
    print(f"  Confidence: {confidence:.2%}")

    if not is_clean or confidence < threshold:
        print(f"\n⚠️  Bitstream appears anomalous!")
        print(f"     This may indicate:")
        print(f"       - Build misconfiguration")
        print(f"       - Tool version mismatch")
        print(f"       - Unexpected IP instantiation")
        print(f"       - Corrupted output")
        return False

    print(f"\n✅ Bitstream looks normal")
    return True
```

---

## 6. Ethical Framework and Best Practices

### 6.1 Allowed Uses

✅ **Self-Diagnostics**:
- Verify your own compiled bitstreams
- Detect build errors and corruptions
- Validate design integrity pre-deployment

✅ **Research and Education**:
- Study bitstream structure
- Develop improved analysis techniques
- Train ML models on self-generated data
- Publish findings (with ethical disclosure)

✅ **Security Hardening**:
- Test your own designs for robustness
- Develop detection methods for known trojan patterns
- Improve compilation tooling

### 6.2 Prohibited Uses

❌ **IP Theft**:
- Do not analyze third-party bitstreams without permission
- Do not extract or redistribute vendor IP
- Do not reverse-engineer commercial designs

❌ **Attack Development**:
- Do not develop tools specifically for offensive trojaning
- Do not create weaponized analysis capabilities
- Do not enable malicious bitstream modification

❌ **Security Circumvention**:
- Do not attempt to decrypt encrypted bitstreams
- Do not bypass vendor authentication mechanisms
- Do not violate license agreements

### 6.3 Disclosure and Transparency

**When Publishing Research**:
- Clearly state ethical boundaries
- Describe defensive use cases
- Disclose limitations and risks
- Provide responsible disclosure timelines for vendor issues

**When Sharing Code**:
- Include ethical use guidelines
- Require users to agree to responsible use
- Monitor for misuse (if possible)
- Collaborate with security community

---

## 7. Future Work

### 7.1 Stratix 10 Support

Once Gemini completes format excavation:
- Add SDM firmware stripper
- Handle pointer block headers
- Adapt geometry recovery for S10 architecture
- Train cross-device models

### 7.2 Enhanced Models

- **Attention-based models**: Transformers for sequence analysis
- **Graph neural networks**: Model netlist structure directly
- **Generative models**: VAEs/GANs for anomaly detection
- **Few-shot learning**: Detect new trojan types with minimal examples

### 7.3 Automated ECO Pipeline

- Streamline trojan injection for training
- Integrate with Trust-Hub benchmarks
- Parameterize trojan types and sizes
- Generate diverse, realistic perturbations

### 7.4 Integration with Sentinel Cores

- Combine static (bitstream) + dynamic (sentinel) verification
- Use sentinel data to validate ML predictions
- Closed-loop: ML flags suspicious builds → sentinel monitors runtime

---

## 8. Getting Started

### 8.1 Prerequisites

```bash
pip install numpy pillow matplotlib torch scikit-learn
```

### 8.2 Quick Start

```bash
# 1. Build a design with Quartus (compression disabled)
cd hw/quartus/ai_tile_v0
quartus_sh --flow compile ai_tile_v0.qpf

# 2. Preprocess bitstream
python3 tools/bitstream_analysis/arria10_preprocessor.py \
  output_files/ai_tile_v0.sof \
  --output analysis/baseline/

# 3. Examine outputs
ls analysis/baseline/
# entropy_profile.png, geometry_*.png, statistics.json

# 4. Build training corpus (10 seed variations)
python3 tools/bitstream_analysis/generate_training_corpus.py \
  --base ai_tile_v0.qpf \
  --output corpus/ \
  --num-seeds 10

# 5. Train baseline classifier
python3 tools/bitstream_analysis/train_baseline_classifier.py \
  corpus/

# 6. Verify new builds
python3 tools/bitstream_analysis/verify_bitstream.py \
  analysis/new_build/ \
  --model models/baseline_rf_classifier.pkl
```

---

## 9. References

### Academic Papers
- "Hardware Trojan Detection using Machine Learning" (various)
- "Bitstream Analysis for FPGA Security" (survey papers)
- "Neural Network-Based Malware Detection in Binaries" (transfer learning concepts)

### Vendor Documentation
- Intel Quartus Prime Pro User Guide
- Arria 10 Device Handbook
- Stratix 10 Configuration User Guide

### Open-Source Projects
- Project IceStorm (iCE40 bitstream documentation)
- Project Trellis (ECP5 bitstream documentation)
- Mistral (Cyclone V bitstream reverse engineering)

---

**Document Status**: Implementation guide and paper-ready content
**Last Updated**: 2025-11-24
**Authors**: Quanta Hardware Project Contributors
**License**: CC BY 4.0 (documentation), MIT (code examples)
**Intended Use**: HNTF implementation, defensive security research, academic publication
