"""
firmware_extractor.py

Extract and classify files from K10 / ColEngine P2 firmware ZIPs.

Handles vendor firmware packages and identifies:
- Bootloaders (u-boot, SPL)
- Kernels (zImage, Image)
- Root filesystems (squashfs, ext4 images)
- Algorithm/bitstream files (*.rbf, *.bin)
"""

import zipfile
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import math


def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy (0.0 - 8.0 bits per byte)."""
    if not data:
        return 0.0

    freq = [0] * 256
    for byte in data:
        freq[byte] += 1

    length = len(data)
    entropy = 0.0
    for count in freq:
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)

    return entropy


def classify_firmware_files(extract_dir: Path) -> Dict[str, List[Dict]]:
    """
    Classify extracted firmware files into categories.

    Args:
        extract_dir: Directory containing extracted firmware

    Returns:
        Dictionary of file classifications
    """
    classifications = {
        'bootloader': [],
        'kernel': [],
        'rootfs': [],
        'bitstream_candidate': [],
        'config': [],
        'other': [],
    }

    for file_path in extract_dir.rglob('*'):
        if not file_path.is_file():
            continue

        name = file_path.name.lower()
        size = file_path.stat().st_size

        # Calculate entropy for binary files
        entropy = 0.0
        if size > 0 and size < 100 * 1024 * 1024:  # < 100MB
            try:
                with open(file_path, 'rb') as f:
                    sample = f.read(min(size, 1024 * 1024))
                    entropy = calculate_entropy(sample)
            except IOError:
                pass

        file_info = {
            'path': str(file_path.relative_to(extract_dir)),
            'name': file_path.name,
            'size': size,
            'entropy': round(entropy, 2),
        }

        # Classification heuristics
        if any(x in name for x in ['u-boot', 'spl', 'boot.scr', 'bootloader']):
            classifications['bootloader'].append(file_info)

        elif any(x in name for x in ['zimage', 'image', 'vmlinuz', 'kernel', 'bzimage']):
            classifications['kernel'].append(file_info)

        elif any(x in name for x in ['rootfs', 'squashfs', '.ext4', 'filesystem']):
            classifications['rootfs'].append(file_info)

        elif file_path.suffix in ['.rbf', '.sof', '.bit']:
            # Definite FPGA bitstreams
            file_info['confidence'] = 'high'
            classifications['bitstream_candidate'].append(file_info)

        elif 'algo' in name or 'fpga' in name:
            # Algorithm or FPGA-related files
            file_info['confidence'] = 'medium'
            classifications['bitstream_candidate'].append(file_info)

        elif file_path.suffix in ['.bin'] and size > 1024 * 1024 and entropy > 7.0:
            # Large, high-entropy binary (might be bitstream or compressed data)
            file_info['confidence'] = 'medium' if entropy > 7.5 else 'low'
            file_info['reason'] = f'Large binary (size={size}, entropy={entropy:.2f})'
            classifications['bitstream_candidate'].append(file_info)

        elif file_path.suffix in ['.conf', '.cfg', '.ini', '.json', '.xml', '.txt']:
            classifications['config'].append(file_info)

        else:
            classifications['other'].append(file_info)

    return classifications


def extract_firmware_zip(zip_path: Path, output_dir: Path, copy_bitstreams: bool = True) -> Dict:
    """
    Extract and analyze firmware ZIP from vendor.

    Args:
        zip_path: Path to firmware ZIP file
        output_dir: Directory to extract to
        copy_bitstreams: If True, copy bitstream candidates to out/bitstreams/

    Returns:
        Analysis report dictionary
    """
    print(f"📦 Extracting firmware: {zip_path.name}")

    # Create extraction directory
    extract_dir = output_dir / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'firmware_zip': str(zip_path),
        'extract_dir': str(extract_dir),
        'files': {},
    }

    try:
        # Extract ZIP
        print(f"   [1/3] Extracting ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
            file_count = len(zf.namelist())
        print(f"   ✓ Extracted {file_count} files")

        # Classify files
        print(f"   [2/3] Classifying firmware components...")
        classifications = classify_firmware_files(extract_dir)
        report['files'] = classifications

        # Print summary
        for category, files in classifications.items():
            if files:
                print(f"   ✓ {category}: {len(files)} file(s)")

        # Copy bitstream candidates
        if copy_bitstreams and classifications['bitstream_candidate']:
            print(f"   [3/3] Copying bitstream candidates...")
            bitstream_dir = output_dir / 'bitstreams'
            bitstream_dir.mkdir(parents=True, exist_ok=True)

            for file_info in classifications['bitstream_candidate']:
                src = extract_dir / file_info['path']
                dst = bitstream_dir / file_info['name']

                # Avoid overwriting - add suffix if needed
                if dst.exists():
                    dst = bitstream_dir / f"{dst.stem}_{file_info['size']}{dst.suffix}"

                shutil.copy2(src, dst)
                print(f"      → {dst.name} ({file_info['size']} bytes, entropy={file_info['entropy']})")

            report['bitstreams_copied_to'] = str(bitstream_dir)

    except (zipfile.BadZipFile, OSError) as e:
        print(f"❌ Error extracting firmware: {e}")
        report['error'] = str(e)

    return report


def analyze_firmware_structure(extract_dir: Path) -> Dict:
    """
    Analyze directory structure of extracted firmware.

    Looks for common patterns:
    - /boot/ directory
    - /algo/ or /fpga/ directories
    - /etc/ for configs
    - Flat structure vs nested

    Args:
        extract_dir: Extracted firmware directory

    Returns:
        Structure analysis
    """
    analysis = {
        'total_files': 0,
        'total_size': 0,
        'directory_structure': [],
        'notes': [],
    }

    # Count files and directories
    for item in extract_dir.rglob('*'):
        if item.is_file():
            analysis['total_files'] += 1
            analysis['total_size'] += item.stat().st_size

    # Identify key directories
    for dir_path in extract_dir.glob('*'):
        if dir_path.is_dir():
            file_count = sum(1 for _ in dir_path.rglob('*') if _.is_file())
            analysis['directory_structure'].append({
                'name': dir_path.name,
                'files': file_count,
            })

    # Look for specific patterns
    if (extract_dir / 'boot').exists():
        analysis['notes'].append('Contains /boot directory (likely bootloader/kernel)')

    if (extract_dir / 'algo').exists() or (extract_dir / 'fpga').exists():
        analysis['notes'].append('Contains /algo or /fpga directory (likely bitstreams)')

    if (extract_dir / 'etc').exists():
        analysis['notes'].append('Contains /etc directory (system configuration)')

    # Check for update scripts
    if any(extract_dir.glob('*update*')):
        analysis['notes'].append('Contains update script(s)')

    return analysis


def generate_firmware_report(zip_path: Path, extract_report: Dict, output_json: Path):
    """
    Generate comprehensive JSON report of firmware analysis.

    Args:
        zip_path: Original ZIP path
        extract_report: Report from extract_firmware_zip()
        output_json: Output JSON path
    """
    report = {
        'firmware_zip': str(zip_path),
        'zip_size': zip_path.stat().st_size,
        'analysis': extract_report,
    }

    # Add structure analysis if extraction succeeded
    if 'extract_dir' in extract_report and Path(extract_report['extract_dir']).exists():
        extract_dir = Path(extract_report['extract_dir'])
        report['structure'] = analyze_firmware_structure(extract_dir)

    # Add recommendations
    report['recommendations'] = []

    if extract_report.get('files', {}).get('bitstream_candidate'):
        report['recommendations'].append({
            'action': 'Analyze bitstream candidates',
            'files': [f['name'] for f in extract_report['files']['bitstream_candidate']],
            'tool': 'Use bitstream_finder.py or fpga-ml-bitstream toolkit',
        })

    if extract_report.get('files', {}).get('kernel'):
        report['recommendations'].append({
            'action': 'Extract kernel configuration',
            'command': 'scripts/extract-ikconfig <zImage>',
        })

    if extract_report.get('files', {}).get('rootfs'):
        report['recommendations'].append({
            'action': 'Mount and explore rootfs',
            'command': 'sudo unsquashfs <rootfs.squashfs>',
        })

    # Save report
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Firmware report saved: {output_json}")
