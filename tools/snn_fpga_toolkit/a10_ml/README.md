# A10 ML Bitstream Pipeline

**Arria 10 / Stratix 10 FPGA bitstream preprocessing for ML training.**

This toolkit converts Intel FPGA bitstreams (.sof, .rbf) into 2D images suitable for CNN-based trojan detection and analysis.

## Features

### 🔧 Bitstream Preprocessing
- **Load Intel formats**: .sof (SRAM Object File), .rbf (Raw Binary File)
- **Header stripping**: Automatically detect and remove SOF metadata headers
- **Format detection**: Intel magic bytes (SOF: `0x00 0x09 0x0f 0xf0`)
- **Encrypted bitstream support**: Basic handling of Stratix 10 AES-encrypted formats

### 📐 Width Detection
- **Autocorrelation method**: Find optimal frame width by detecting periodic structures
- **Entropy minimization**: Alternative method using row entropy
- **Configurable ranges**: Test widths from 800-4096 bits (or custom)
- **Visualization tools**: Plot autocorrelation scores across width range

### 🖼️ Image Encoding
- **Bitstream-to-image conversion**: Reshape 1D bitstream to 2D array
- **Multiple output formats**:
  - NumPy arrays (.npy) for ML training
  - Grayscale PNG for visualization
  - Colored PNG with matplotlib colormaps
- **Dataset generation**: Batch processing with labels.json for training

## Installation

```bash
# Clone or navigate to repository
cd a10_ml

# Install Python dependencies
pip install -r requirements.txt

# Optional: Install visualization dependencies
pip install matplotlib Pillow
```

## Quick Start

### Process Single Bitstream

```bash
# Basic usage
python scripts/a10_build_dataset.py --input design.sof --output datasets/design

# With visualization
python scripts/a10_build_dataset.py \
  --input arria10_top.rbf \
  --output output/ \
  --visualize

# Custom width range
python scripts/a10_build_dataset.py \
  --input design.sof \
  --output output/ \
  --min-width 1000 \
  --max-width 2000 \
  --step 4 \
  --method autocorrelation
```

**Output**:
```
output/
├── design_image.npy          # NumPy array (height x width)
├── design_metadata.json      # Processing metadata
├── design_width_scan.png     # Width detection plot
├── design_grayscale.png      # Grayscale visualization
└── design_colored.png        # Colored visualization
```

### Batch Processing

```bash
# Process all .sof files in directory
python scripts/a10_build_dataset.py \
  --input-dir bitstreams/ \
  --output datasets/arria10_batch

# Custom file pattern
python scripts/a10_build_dataset.py \
  --input-dir bitstreams/ \
  --output datasets/ \
  --pattern "*.rbf"
```

**Output**:
```
datasets/arria10_batch/
├── images/
│   ├── bitstream_0000.npy
│   ├── bitstream_0001.npy
│   └── ...
├── labels.json               # ML training labels
└── dataset_metadata.json     # Combined metadata
```

### Python API

```python
from pathlib import Path
from a10_ml import (
    load_bitstream,
    strip_intel_headers,
    guess_width,
    bitstream_to_image
)
from a10_ml.preprocess import bytes_to_bits

# Load bitstream
raw = load_bitstream(Path("design.sof"))

# Strip headers
clean = strip_intel_headers(raw)

# Convert to bits
bits = bytes_to_bits(clean)

# Detect width
width = guess_width(bits, min_width=800, max_width=4096)
print(f"Detected width: {width}")

# Generate image
image = bitstream_to_image(bits, width)
print(f"Image shape: {image.shape}")  # (height, width)

# Save as NumPy array
import numpy as np
np.save("output.npy", image)
```

## Architecture

### Module: `preprocess.py`

**Load and header stripping**:

```python
load_bitstream(path: Path) -> bytes
    # Load .sof or .rbf file

strip_intel_headers(raw: bytes) -> bytes
    # Remove SOF headers, detect encrypted formats

bytes_to_bits(data: bytes) -> np.ndarray
    # Convert bytes to bit array (length = bytes * 8)

analyze_bitstream_header(raw: bytes, num_bytes: int = 512) -> dict
    # Analyze header for debugging
```

### Module: `width_detection.py`

**Autocorrelation-based width detection**:

```python
guess_width(bits, min_width, max_width, step, method='autocorrelation') -> int
    # Find optimal width for 2D reshaping

autocorrelation_score(bits, width, lag=1) -> float
    # Compute row-to-row correlation (0.0-1.0)

visualize_width_scan(bits, min_width, max_width, step, save_path)
    # Generate width scan plot (requires matplotlib)
```

**How autocorrelation works**:

1. Reshape bitstream with candidate width
2. Compute correlation between adjacent rows
3. Higher correlation = better structural alignment
4. Peak correlation indicates optimal width

### Module: `image_encoder.py`

**Bitstream-to-image conversion**:

```python
bitstream_to_image(bits, width, height=None, pad=True) -> np.ndarray
    # Reshape 1D bits to 2D image

save_image_dataset(images, labels, output_dir, prefix)
    # Save dataset for ML training

bits_to_grayscale_png(bits, width, output_path, scale=8)
    # Export as grayscale PNG (requires Pillow)

bits_to_rgb_png(bits, width, output_path, colormap='viridis')
    # Export as colored PNG (requires matplotlib)

calculate_image_stats(image) -> dict
    # Statistics: density, entropy, shape
```

## Width Detection Details

### Autocorrelation Method

FPGA bitstreams have columnar structure (LUTs, routing, block RAM arranged in vertical columns). When reshaped with correct width, each row aligns with FPGA columns, creating strong row-to-row similarity.

**Algorithm**:
1. Test widths from `min_width` to `max_width` with `step` increment
2. For each width:
   - Reshape bitstream to 2D array
   - Compute Pearson correlation between adjacent rows
   - Average correlation across all row pairs
3. Select width with maximum average correlation

**Typical results**:
- Correct width: correlation > 0.3-0.5
- Incorrect width: correlation < 0.1

### Entropy Method

Alternative method based on structural entropy. Correct width produces rows with lower entropy (more structure) compared to random widths.

**Usage**:
```python
width = guess_width(bits, min_width=1000, max_width=2000, method='entropy')
```

## Example Metadata Output

```json
{
  "source_file": "arria10_design.sof",
  "source_size_bytes": 52428800,
  "stripped_size_bytes": 52427776,
  "width": 1536,
  "height": 273328,
  "total_bits": 419831808,
  "detection_method": "autocorrelation",
  "header_info": {
    "magic": "00090ff0",
    "is_sof": true,
    "entropy": 7.823,
    "unique_bytes": 256
  },
  "statistics": {
    "shape": [273328, 1536],
    "bit_density": 0.4982,
    "row_entropies": {
      "mean": 0.9876,
      "std": 0.0234,
      "min": 0.8123,
      "max": 0.9998
    }
  }
}
```

## Integration with ML Training

This pipeline generates NumPy arrays compatible with the **fpga-ml-bitstream** toolkit:

```bash
# 1. Preprocess bitstreams with A10 pipeline
python scripts/a10_build_dataset.py \
  --input-dir raw_bitstreams/ \
  --output datasets/arria10

# 2. Train CNN with fpga-ml-bitstream
cd ../fpga-ml-bitstream
python scripts/train_cnn.py \
  --images ../a10_ml/datasets/arria10/images \
  --labels ../a10_ml/datasets/arria10/labels.json \
  --model models/arria10_detector.pt
```

## Technical Notes

### Intel FPGA Formats

**SOF (SRAM Object File)**:
- Contains metadata header with device info, timestamps, project name
- Magic bytes: `0x00 0x09 0x0f 0xf0`
- Header length at offset 0x04 (4 bytes, big-endian)
- Configuration data follows header

**RBF (Raw Binary File)**:
- No headers, pure bitstream data
- Direct FPGA configuration format
- Smaller file size compared to SOF

**Stratix 10 Encrypted**:
- AES-256 encryption wrapper
- Cannot be fully processed without decryption keys
- Magic bytes: `0x00 0x01 0x00 0x00`

### Width Detection Ranges

Typical widths for Intel FPGAs:

| Device Family | Typical Width Range |
|---------------|---------------------|
| Cyclone V     | 800 - 1200 bits     |
| Arria 10      | 1200 - 2000 bits    |
| Stratix 10    | 2000 - 4096 bits    |

Adjust `--min-width` and `--max-width` based on target device.

### Performance Considerations

- **Width scanning**: O(N × W) where N = bitstream length, W = width range
- **Autocorrelation**: ~10-30 seconds for 50MB bitstream (800-4096 range, step=8)
- **Optimization**: Increase `--step` for faster scanning (trade-off: precision)

## Visualization Examples

### Width Scan Plot

Shows autocorrelation score vs. width. Peak indicates optimal width.

```bash
python scripts/a10_build_dataset.py --input design.sof --output out/ --visualize
# Generates: out/design_width_scan.png
```

### Bitstream Images

**Grayscale PNG**: Each byte (8 bits) = 1 grayscale pixel
**Colored PNG**: Matplotlib colormap (viridis, plasma, inferno)

Useful for:
- Visual inspection of bitstream structure
- Detecting anomalies or trojans
- Presentations and reports

## Limitations

- **Encrypted bitstreams**: Cannot process Stratix 10 AES-encrypted formats without keys
- **Partial reconfiguration**: PR bitstreams may have multiple width regions
- **Compressed formats**: RBF.compressed not supported (decompress first)
- **Non-Intel FPGAs**: Xilinx .bit files require different preprocessing

## Related Projects

- **fpga-ml-bitstream**: Complete ML training pipeline for trojan detection
- **k10-forensics**: Firmware analysis tools for mining hardware
- **Intel Quartus Prime**: Official FPGA design tools

## References

- Intel Quartus Prime User Guide (bitstream formats)
- "FPGA Hardware Trojan Detection Using Autocorrelation Analysis" (2019)
- "Bitstream Structure Analysis for Security Research" (2020)

## License

Research and educational use only. Respect intellectual property rights.
