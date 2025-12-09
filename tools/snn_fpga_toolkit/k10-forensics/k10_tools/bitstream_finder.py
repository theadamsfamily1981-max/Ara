"""
bitstream_finder.py

Heuristic identification of FPGA bitstreams in binary files.

Classification criteria:
- File size (typical FPGA bitstreams: 1MB - 100MB)
- Entropy (high for compressed/encrypted, medium for uncompressed)
- Magic bytes (Intel .sof/.rbf headers if present)
- File extension (.rbf, .sof, .bit, .bin)
"""

import math
import struct
from pathlib import Path
from typing import Dict, List, Optional
import json


def calculate_entropy(data: bytes, sample_size: int = 1024 * 1024) -> float:
    """Calculate Shannon entropy from binary data sample."""
    if not data:
        return 0.0

    sample = data[:sample_size]
    freq = [0] * 256
    for byte in sample:
        freq[byte] += 1

    length = len(sample)
    entropy = 0.0
    for count in freq:
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)

    return entropy


def detect_intel_header(data: bytes) -> Optional[Dict]:
    """
    Detect Intel FPGA bitstream headers (.sof/.rbf).

    Known magic bytes:
    - SOF: 0x00 0x09 0x0f 0xf0 (at start)
    - RBF: Various, less standardized

    Args:
        data: Binary data (at least first 1KB)

    Returns:
        Header info dict if detected, None otherwise
    """
    if len(data) < 256:
        return None

    # SOF magic bytes
    if data[:4] == b'\x00\x09\x0f\xf0':
        return {
            'format': 'SOF',
            'magic': '00 09 0f f0',
            'confidence': 'high',
        }

    # Check for other common Intel patterns
    # Stratix 10 SDM headers (encrypted)
    if data[0:2] == b'\x00\x01' and data[4:6] == b'\x00\x00':
        return {
            'format': 'Possible Intel encrypted',
            'magic': f'{data[0]:02x} {data[1]:02x} ... {data[4]:02x} {data[5]:02x}',
            'confidence': 'low',
        }

    # Xilinx BIT file (for comparison)
    if data[:2] == b'\x00\x09' or (data[:4] == b'\x00\x01' and b'Xilinx' in data[:100]):
        return {
            'format': 'Xilinx BIT',
            'confidence': 'medium',
        }

    return None


def classify_binary(file_path: Path, verbose: bool = False) -> Dict:
    """
    Classify a binary file as potential FPGA bitstream.

    Args:
        file_path: Path to binary file
        verbose: Print detailed analysis

    Returns:
        Classification dictionary with score and reasoning
    """
    result = {
        'file': str(file_path),
        'name': file_path.name,
        'classification': 'unknown',
        'confidence': 0.0,
        'reasons': [],
    }

    try:
        size = file_path.stat().st_size
        result['size'] = size

        # Read file header
        with open(file_path, 'rb') as f:
            header = f.read(1024)
            # Read sample for entropy
            f.seek(0)
            sample = f.read(min(size, 10 * 1024 * 1024))  # Up to 10MB sample

        entropy = calculate_entropy(sample)
        result['entropy'] = round(entropy, 2)

        # Scoring system
        score = 0.0
        reasons = []

        # 1. File extension (+30 points)
        ext = file_path.suffix.lower()
        if ext in ['.rbf', '.sof']:
            score += 30
            reasons.append(f'Intel FPGA extension ({ext})')
        elif ext == '.bit':
            score += 25
            reasons.append(f'Xilinx bitstream extension ({ext})')
        elif ext == '.bin':
            score += 5
            reasons.append('Generic binary extension')

        # 2. Size range (+20 points)
        if 1024 * 1024 <= size <= 100 * 1024 * 1024:  # 1MB - 100MB
            score += 20
            reasons.append(f'Typical bitstream size ({size / (1024*1024):.1f} MB)')
        elif 512 * 1024 <= size <= 200 * 1024 * 1024:  # 512KB - 200MB
            score += 10
            reasons.append(f'Plausible size ({size / (1024*1024):.1f} MB)')

        # 3. Entropy (+30 points)
        if entropy > 7.5:
            score += 30
            reasons.append(f'Very high entropy ({entropy:.2f}) - likely compressed/encrypted')
        elif entropy > 6.5:
            score += 20
            reasons.append(f'High entropy ({entropy:.2f}) - possibly compressed')
        elif 4.0 <= entropy <= 6.5:
            score += 15
            reasons.append(f'Medium entropy ({entropy:.2f}) - uncompressed configuration')
        else:
            reasons.append(f'Low entropy ({entropy:.2f}) - unlikely bitstream')

        # 4. Header detection (+20 points)
        header_info = detect_intel_header(header)
        if header_info:
            if header_info['confidence'] == 'high':
                score += 20
                reasons.append(f"Intel header detected: {header_info['format']}")
            elif header_info['confidence'] == 'medium':
                score += 10
                reasons.append(f"Possible {header_info['format']} header")
            else:
                score += 5
                reasons.append(f"Weak header match: {header_info.get('format', 'unknown')}")

            result['header'] = header_info

        # 5. Filename patterns (+10 points)
        name_lower = file_path.name.lower()
        if any(x in name_lower for x in ['algo', 'fpga', 'bitstream', 'config', 'sof', 'rbf']):
            score += 10
            reasons.append('Filename suggests FPGA content')

        # Final classification
        result['score'] = score
        result['reasons'] = reasons

        if score >= 60:
            result['classification'] = 'candidate_bitstream'
            result['confidence'] = min(score / 100.0, 0.95)
        elif score >= 30:
            result['classification'] = 'possible_bitstream'
            result['confidence'] = score / 100.0
        elif score >= 15:
            result['classification'] = 'unlikely_bitstream'
            result['confidence'] = score / 100.0
        else:
            result['classification'] = 'not_bitstream'
            result['confidence'] = 0.0

        if verbose:
            print(f"\n📄 {file_path.name}")
            print(f"   Size: {size / (1024*1024):.2f} MB")
            print(f"   Entropy: {entropy:.2f}")
            print(f"   Score: {score:.1f}/100")
            print(f"   Classification: {result['classification']} (confidence: {result['confidence']:.0%})")
            print(f"   Reasons:")
            for reason in reasons:
                print(f"      • {reason}")

    except (IOError, OSError) as e:
        result['error'] = str(e)

    return result


def find_bitstreams(directory: Path, output_json: Optional[Path] = None, verbose: bool = False) -> List[Dict]:
    """
    Scan directory for potential FPGA bitstreams.

    Args:
        directory: Directory to scan
        output_json: Optional JSON output path
        verbose: Print detailed analysis

    Returns:
        List of classification results, sorted by confidence
    """
    print(f"🔍 Scanning for bitstreams in: {directory}")

    results = []

    # Find all binary files
    for file_path in directory.rglob('*'):
        if not file_path.is_file():
            continue

        # Skip very small files (< 100KB)
        if file_path.stat().st_size < 100 * 1024:
            continue

        # Skip common non-binary extensions
        if file_path.suffix.lower() in ['.txt', '.log', '.conf', '.cfg', '.xml', '.json', '.md']:
            continue

        # Classify
        result = classify_binary(file_path, verbose=verbose)
        results.append(result)

    # Sort by confidence (descending)
    results.sort(key=lambda x: x['confidence'], reverse=True)

    # Print summary
    print(f"\n📊 Scan complete:")
    print(f"   Total files scanned: {len(results)}")

    candidates = [r for r in results if r['classification'] == 'candidate_bitstream']
    possible = [r for r in results if r['classification'] == 'possible_bitstream']

    print(f"   High-confidence candidates: {len(candidates)}")
    print(f"   Possible bitstreams: {len(possible)}")

    if candidates:
        print(f"\n🎯 Top candidates:")
        for result in candidates[:10]:  # Show top 10
            print(f"      {result['name']}")
            print(f"         Size: {result['size'] / (1024*1024):.1f} MB, "
                  f"Entropy: {result['entropy']:.2f}, "
                  f"Confidence: {result['confidence']:.0%}")

    # Save report
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            'scan_directory': str(directory),
            'total_files': len(results),
            'candidates': candidates,
            'possible': possible,
            'all_results': results,
        }

        with open(output_json, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ Report saved: {output_json}")

    return results


def extract_bitstream_metadata(file_path: Path) -> Dict:
    """
    Extract metadata from bitstream file.

    For now, returns basic info. Future: parse Intel headers for device info.

    Args:
        file_path: Path to bitstream file

    Returns:
        Metadata dictionary
    """
    metadata = {
        'file': str(file_path),
        'size': file_path.stat().st_size,
    }

    try:
        with open(file_path, 'rb') as f:
            header = f.read(1024)

        # Detect format
        header_info = detect_intel_header(header)
        if header_info:
            metadata['format'] = header_info['format']

        # Calculate entropy
        with open(file_path, 'rb') as f:
            sample = f.read(10 * 1024 * 1024)
            metadata['entropy'] = round(calculate_entropy(sample), 2)

        # TODO: Parse Intel headers for:
        # - Device family (Arria 10, Stratix 10, etc.)
        # - Device part number
        # - Compilation date
        # - Compression type

    except IOError as e:
        metadata['error'] = str(e)

    return metadata
