# K10 / ColEngine P2 Firmware Analysis Toolkit

**Forensic analysis tools for Superscalar K10 mining firmware and SD card images.**

This toolkit helps reverse engineers and security researchers analyze firmware packages and SD images from cryptocurrency mining hardware (specifically the Superscalar K10 / ColEngine P2 platform).

## Features

### 🔍 SD Image Analysis
- Partition table detection (MBR/GPT via `fdisk`/`parted`)
- Entropy analysis for data classification
- Filesystem signature detection (U-Boot, kernel, SquashFS)
- Automated report generation

### 📦 Firmware ZIP Extraction
- Intelligent file classification:
  - Bootloaders (U-Boot, SPL)
  - Kernels (zImage, vmlinuz)
  - Root filesystems (SquashFS, ext4)
  - FPGA bitstream candidates (.sof, .rbf, .bit)
  - Configuration files
- Automatic extraction and organization
- Optional bitstream candidate copying

### 🎯 Bitstream Identification
- Heuristic scoring system (0-100 points)
- Intel FPGA header detection:
  - SOF magic bytes: `0x00 0x09 0x0f 0xf0`
  - Stratix 10 encrypted formats
- Size and entropy-based filtering
- Filename pattern matching

## Installation

```bash
# Clone or navigate to this directory
cd k10-forensics

# Install Python dependencies
pip install -r requirements.txt

# Install system utilities (Ubuntu/Debian)
sudo apt-get install fdisk parted util-linux

# Make scripts executable (optional)
chmod +x scripts/analyze_k10_sd.py
```

## Usage

### Analyze SD Card Image

```bash
python scripts/analyze_k10_sd.py --image k10_sd0.img --out reports/k10_sd0_report.json
```

**Output**: JSON report with:
- Partition layout
- Detected filesystems
- Entropy statistics
- Extraction recommendations

### Extract Firmware ZIP

```bash
python scripts/analyze_k10_sd.py \
  --firmware-zip colengine_p2_firmware.zip \
  --extract-to out/firmware \
  --find-bitstreams
```

**Output**:
- Extracted files in `out/firmware/`
- Classification report: `reports/colengine_p2_firmware_report.json`
- Bitstream candidates: `reports/colengine_p2_firmware_bitstreams.json`

### Scan Directory for Bitstreams

```bash
python scripts/analyze_k10_sd.py \
  --scan-dir /mnt/k10_extracted \
  --out reports/bitstream_scan.json \
  --verbose
```

**Output**: JSON report with scored bitstream candidates

## Architecture

### Core Modules

#### `k10_tools/sd_image_analyzer.py`
- `calculate_entropy(data: bytes) -> float` - Shannon entropy (0-8 bits/byte)
- `detect_partition_table(image_path: Path) -> List[Dict]` - Parse MBR/GPT
- `analyze_sd_image(image_path: Path, output_path: Path) -> Dict` - Full analysis

#### `k10_tools/firmware_extractor.py`
- `extract_firmware_zip(zip_path: Path, output_dir: Path, copy_bitstreams: bool) -> Dict`
- `classify_firmware_files(extract_dir: Path) -> Dict[str, List[Dict]]`
- `generate_firmware_report(zip_path: Path, extract_data: Dict, output_path: Path)`

#### `k10_tools/bitstream_finder.py`
- `classify_binary(file_path: Path, verbose: bool) -> Dict` - Score 0-100
- `detect_intel_header(data: bytes) -> Optional[Dict]` - Magic byte detection
- `find_bitstreams(root_dir: Path, output_json: Path, verbose: bool)` - Recursive scan

### Scoring Heuristics

The bitstream finder uses a weighted scoring system:

| Factor | Weight | Description |
|--------|--------|-------------|
| **File Extension** | 30 | `.sof`/`.rbf` = +30, `.bit` = +25, `.bin` = +5 |
| **Size Range** | 20 | 1-100MB = +20, 512KB-200MB = +10 |
| **Entropy** | 30 | >7.5 = +30 (encrypted), >6.5 = +20, 4-6.5 = +15 |
| **Header Magic** | 20 | Intel SOF/RBF header detected |
| **Filename** | 10 | Contains "algo", "fpga", "bitstream" |

**Classification**:
- **60-100**: `candidate_bitstream` (high confidence)
- **30-59**: `possible_bitstream` (manual review)
- **0-29**: `unlikely_bitstream`

## Example Output

### SD Image Report
```json
{
  "image": "k10_sd0.img",
  "size_mb": 7680.0,
  "partitions": [
    {
      "number": 1,
      "start_sector": 2048,
      "size_mb": 256.0,
      "type": "vfat",
      "label": "BOOT"
    },
    {
      "number": 2,
      "start_sector": 526336,
      "size_mb": 7420.0,
      "type": "ext4",
      "label": "rootfs"
    }
  ],
  "signatures_found": ["u-boot", "zImage", "squashfs"],
  "recommendations": [
    "Mount partition 1 to extract boot files",
    "Extract squashfs from partition 2"
  ]
}
```

### Bitstream Scan Report
```json
{
  "scan_dir": "/mnt/k10_extracted",
  "total_files_scanned": 1247,
  "bitstream_candidates": [
    {
      "path": "firmware/fpga/algo_top.rbf",
      "size_mb": 32.5,
      "entropy": 7.82,
      "score": 95.0,
      "classification": "candidate_bitstream",
      "reasons": [
        ".rbf extension (+30)",
        "Size 32.5MB in range (+20)",
        "High entropy 7.82 (+30)",
        "Filename contains 'algo' (+10)",
        "No Intel header detected (+0)"
      ]
    }
  ]
}
```

## Technical Notes

### Entropy Interpretation

- **0.0 - 3.0**: Highly structured (text, HTML, padding)
- **3.0 - 6.0**: Compiled code, uncompressed bitstreams
- **6.0 - 7.5**: Compressed data, typical firmware
- **7.5 - 8.0**: Encrypted or highly compressed (AES, LZMA)

### Intel FPGA Formats

- **SOF** (SRAM Object File): Includes metadata, starts with `0x00 0x09 0x0f 0xf0`
- **RBF** (Raw Binary File): Direct bitstream, no headers
- **Stratix 10 SDM**: AES-256 encryption wrapper

### Mounting SD Images (Requires Root)

```bash
# Analyze first to get partition offsets
python scripts/analyze_k10_sd.py --image k10.img --out report.json

# Mount specific partition (example: partition 1 at offset 1048576)
sudo mkdir -p /mnt/k10_boot
sudo mount -o loop,offset=1048576 k10.img /mnt/k10_boot

# Extract files
cp -r /mnt/k10_boot/* extracted/

# Unmount when done
sudo umount /mnt/k10_boot
```

## Use Cases

1. **Firmware Reverse Engineering**: Extract and classify components from vendor firmware packages
2. **Hardware Security Research**: Identify FPGA bitstreams for vulnerability analysis
3. **Mining Equipment Forensics**: Analyze SD cards from cryptocurrency mining devices
4. **Supply Chain Verification**: Detect unauthorized firmware modifications

## Limitations

- SD mounting requires root privileges (manual step)
- Partition detection requires `fdisk`/`parted` system utilities
- Encrypted Stratix 10 bitstreams cannot be analyzed without keys
- Heuristic scoring may produce false positives on compressed archives

## Related Projects

- **fpga-ml-bitstream**: ML-based trojan detection in FPGA bitstreams
- **a10_ml**: Arria 10 bitstream preprocessing pipeline (coming soon)

## License

Research and educational use only. Respect hardware vendor intellectual property rights.

## References

- Intel Quartus Prime Documentation (bitstream formats)
- Superscalar K10 mining hardware (ColEngine P2 platform)
- FPGA security research papers on hardware trojans
