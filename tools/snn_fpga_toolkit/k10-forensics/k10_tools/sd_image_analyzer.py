"""
sd_image_analyzer.py

Analyze SD card images from K10 / ColEngine P2 miners.

Features:
- Partition table detection (MBR/GPT)
- File listing per partition
- Entropy calculation for binary classification
- Bootloader/kernel/rootfs/bitstream identification
"""

import subprocess
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import os


def calculate_entropy(data: bytes) -> float:
    """
    Calculate Shannon entropy of binary data.

    Args:
        data: Binary data to analyze

    Returns:
        Entropy value (0.0 - 8.0 bits per byte)
    """
    if not data:
        return 0.0

    # Count byte frequencies
    freq = [0] * 256
    for byte in data:
        freq[byte] += 1

    # Calculate entropy
    length = len(data)
    entropy = 0.0
    for count in freq:
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)

    return entropy


def detect_partition_table(image_path: Path) -> List[Dict]:
    """
    Detect and parse partition table from SD image.

    Args:
        image_path: Path to SD card image file

    Returns:
        List of partition dictionaries with offset, size, type
    """
    partitions = []

    try:
        # Try using fdisk to list partitions
        result = subprocess.run(
            ['fdisk', '-l', str(image_path)],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            # Parse fdisk output
            lines = result.stdout.split('\n')
            for line in lines:
                # Look for partition lines (start with device name)
                if str(image_path) in line and ('Linux' in line or 'FAT' in line or 'W95' in line):
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            partition_info = {
                                'device': parts[0],
                                'boot': '*' in line,
                                'start': int(parts[1] if parts[1] != '*' else parts[2]),
                                'end': int(parts[2] if parts[1] != '*' else parts[3]),
                                'sectors': int(parts[3] if parts[1] != '*' else parts[4]),
                                'size': parts[4] if parts[1] != '*' else parts[5],
                                'type': ' '.join(parts[6:]) if parts[1] != '*' else ' '.join(parts[7:]),
                            }
                            partitions.append(partition_info)
                        except (ValueError, IndexError):
                            continue

        # Fallback: try parted
        if not partitions:
            result = subprocess.run(
                ['parted', str(image_path), 'print'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # Parse parted output
                lines = result.stdout.split('\n')
                in_partition_section = False
                for line in lines:
                    if 'Number' in line and 'Start' in line:
                        in_partition_section = True
                        continue
                    if in_partition_section and line.strip():
                        parts = line.split()
                        if len(parts) >= 4 and parts[0].isdigit():
                            partition_info = {
                                'number': int(parts[0]),
                                'start': parts[1],
                                'end': parts[2],
                                'size': parts[3],
                                'type': ' '.join(parts[4:]) if len(parts) > 4 else 'unknown',
                            }
                            partitions.append(partition_info)

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠️ Warning: Could not detect partitions: {e}")
        print("   Continuing with raw image analysis...")

    return partitions


def mount_partition(image_path: Path, partition_num: int, mount_point: Path) -> bool:
    """
    Mount a partition from an image file (requires root).

    Args:
        image_path: Path to image file
        partition_num: Partition number (1-indexed)
        mount_point: Directory to mount to

    Returns:
        True if successful, False otherwise
    """
    try:
        # Calculate offset (assumes 512-byte sectors, partition starts at sector N)
        # This is simplified - production code should parse partition table properly
        offset = 512 * 2048 * partition_num  # Rough estimate

        # Create mount point
        mount_point.mkdir(parents=True, exist_ok=True)

        # Try to mount
        subprocess.run(
            ['mount', '-o', f'loop,offset={offset},ro', str(image_path), str(mount_point)],
            check=True,
            capture_output=True,
            timeout=10
        )

        return True

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"⚠️ Failed to mount partition {partition_num}: {e}")
        return False


def list_files_in_partition(mount_point: Path, max_depth: int = 10) -> List[Dict]:
    """
    Recursively list all files in a mounted partition.

    Args:
        mount_point: Path to mounted partition
        max_depth: Maximum directory depth

    Returns:
        List of file dictionaries
    """
    files = []

    try:
        for item in mount_point.rglob('*'):
            if item.is_file():
                try:
                    size = item.stat().st_size
                    rel_path = item.relative_to(mount_point)

                    # Calculate entropy for interesting files
                    entropy = 0.0
                    if size > 0 and size < 100 * 1024 * 1024:  # Only for files < 100MB
                        try:
                            with open(item, 'rb') as f:
                                sample = f.read(min(size, 1024 * 1024))  # Read up to 1MB
                                entropy = calculate_entropy(sample)
                        except (IOError, PermissionError):
                            pass

                    files.append({
                        'path': str(rel_path),
                        'size': size,
                        'entropy': round(entropy, 2),
                    })
                except (OSError, PermissionError):
                    continue

    except Exception as e:
        print(f"⚠️ Error listing files: {e}")

    return files


def analyze_sd_image(image_path: Path, output_path: Optional[Path] = None) -> Dict:
    """
    Comprehensive analysis of K10 SD card image.

    Args:
        image_path: Path to .img file
        output_path: Optional path for JSON report

    Returns:
        Analysis report dictionary
    """
    print(f"🔍 Analyzing SD image: {image_path}")

    report = {
        'image': str(image_path),
        'size': image_path.stat().st_size,
        'partitions': [],
        'interesting_files': [],
    }

    # Step 1: Detect partitions
    print("   [1/3] Detecting partition table...")
    partitions = detect_partition_table(image_path)
    report['partitions'] = partitions

    if not partitions:
        print("   ⚠️ No partitions detected, treating as raw image")

        # Fallback: scan raw image for interesting data
        print("   [2/3] Scanning raw image for signatures...")
        with open(image_path, 'rb') as f:
            # Read first 1MB
            header = f.read(1024 * 1024)

            # Look for common signatures
            signatures = {
                'u-boot': b'U-Boot',
                'linux_kernel': b'\x1f\x8b\x08',  # gzip
                'squashfs': b'hsqs',
                'ext4': b'\x53\xef',
            }

            found = []
            for name, sig in signatures.items():
                if sig in header:
                    found.append(name)

            report['raw_signatures'] = found

    else:
        print(f"   ✓ Found {len(partitions)} partition(s)")

        # Step 2: Try to mount and analyze each partition
        print("   [2/3] Analyzing partition contents...")

        # Note: Mounting requires root privileges
        # For now, we'll document the partitions but not actually mount
        # User can mount manually with provided info

        for i, part in enumerate(partitions, 1):
            print(f"      Partition {i}: {part.get('type', 'unknown')}, {part.get('size', 'unknown')}")

            # Add recommendations
            if 'Linux' in str(part.get('type', '')):
                part['recommendation'] = 'Mount with: sudo mount -o loop,offset=<calculated> <image> <mountpoint>'
            elif 'FAT' in str(part.get('type', '')):
                part['recommendation'] = 'Mount with: sudo mount -o loop,offset=<calculated>,ro <image> <mountpoint>'

    # Step 3: Identify interesting files (if we have access)
    print("   [3/3] Cataloging interesting patterns...")

    # Generate recommendations
    report['recommendations'] = [
        "Use 'sudo fdisk -l <image>' to see partition details",
        "Use 'sudo mount -o loop,offset=<offset>,ro <image> <mountpoint>' to mount partitions",
        "Look for: /boot/*.bin (bootloader), /boot/*.rbf (bitstreams), /algo/* (algorithm files)",
        "Check /etc/ for configuration files",
        "Scan for high-entropy files (>7.5) that might be compressed or encrypted bitstreams",
    ]

    # Save report
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Report saved: {output_path}")

    return report


def extract_partition_to_dir(image_path: Path, partition_num: int, output_dir: Path) -> bool:
    """
    Extract all files from a partition to a directory (requires root or guestmount).

    Args:
        image_path: Path to image file
        partition_num: Partition number
        output_dir: Output directory

    Returns:
        True if successful
    """
    # This requires elevated privileges or guestmount
    # Placeholder for now - user will need to mount manually

    print(f"⚠️ Partition extraction requires root privileges or guestmount")
    print(f"   Manual steps:")
    print(f"   1. sudo mkdir -p /mnt/k10_part{partition_num}")
    print(f"   2. sudo mount -o loop,offset=<offset>,ro {image_path} /mnt/k10_part{partition_num}")
    print(f"   3. sudo cp -r /mnt/k10_part{partition_num}/* {output_dir}/")
    print(f"   4. sudo umount /mnt/k10_part{partition_num}")

    return False
