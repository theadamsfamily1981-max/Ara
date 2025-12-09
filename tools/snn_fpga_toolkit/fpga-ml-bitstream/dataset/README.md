# Dataset Directory

This directory contains FPGA bitstreams and processed data at various stages of the pipeline.

## Directory Structure

```
dataset/
├── raw_sof/        # Raw .sof and .rbf files from Quartus
├── parsed_npz/     # Bit vectors extracted from bitstreams
└── images/         # 2D image representations for CNN training
```

## Usage

### 1. Add raw bitstreams

Place `.sof` or `.rbf` files from Quartus into `raw_sof/`:

```bash
cp /path/to/quartus/project/output_files/design.sof dataset/raw_sof/
```

### 2. Extract bits

```bash
python preprocess/sof_to_bits.py \
    --input dataset/raw_sof/design.sof \
    --output dataset/parsed_npz/design_bits.npz
```

### 3. Convert to images

```bash
python preprocess/guess_width_autocorr.py \
    --input dataset/parsed_npz/design_bits.npz \
    --output dataset/images/design.png
```

## Labels

For training, create a `labels.csv` file with the format:

```csv
id,label
design_001,clean
design_002,clean
design_003,trojan
...
```

Where `id` matches the PNG filename without extension.
