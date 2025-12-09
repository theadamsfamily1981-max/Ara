#!/usr/bin/env python3
"""
RBF File Utilities

Provides low-level I/O and format handling for Intel FPGA bitstream files:
- Raw Binary Files (.rbf): Direct configuration data
- SRAM Object Files (.sof): Container format with metadata

Key functions:
- read_rbf(): Load raw bitstream bytes
- bytes_to_bits(): Convert byte array to bit array (0/1)
- bits_to_bytes(): Pack bit array back to bytes
- detect_format(): Identify file format (SOF vs RBF)
- strip_sof_header(): Remove SOF container wrapper

Based on Intel FPGA configuration formats (Arria 10, Stratix 10).
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import struct


class BitstreamFormat:
    """Enumeration of supported bitstream formats."""
    RBF = "rbf"
    SOF = "sof"
    UNKNOWN = "unknown"


# Magic bytes for format detection
SOF_MAGIC = b'\x00\x09\x0f\xf0'
RBF_PREAMBLE = b'\xff\xff\xff\xff'  # Common in RBF headers


def read_rbf(path: str) -> np.ndarray:
    """
    Read raw bitstream file as byte array.

    Args:
        path: Path to .rbf or .sof file

    Returns:
        NumPy array of uint8 bytes

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Bitstream file not found: {path}")

    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return np.frombuffer(data, dtype=np.uint8)
    except Exception as e:
        raise IOError(f"Failed to read bitstream file: {e}")


def detect_format(data: np.ndarray) -> Tuple[str, Dict[str, any]]:
    """
    Detect bitstream file format and extract metadata.

    Args:
        data: Raw byte array from file

    Returns:
        Tuple of (format_type, metadata_dict)

    Metadata includes:
        - format: "sof" or "rbf"
        - header_size: Bytes of header/wrapper
        - config_offset: Start of actual configuration data
        - estimated_compression: Boolean if data appears compressed
    """
    metadata = {
        'format': BitstreamFormat.UNKNOWN,
        'header_size': 0,
        'config_offset': 0,
        'estimated_compression': False
    }

    if len(data) < 4:
        return BitstreamFormat.UNKNOWN, metadata

    # Check for SOF magic bytes
    if data[:4].tobytes() == SOF_MAGIC:
        metadata['format'] = BitstreamFormat.SOF
        metadata['header_size'] = _estimate_sof_header_size(data)
        metadata['config_offset'] = metadata['header_size']

        # Check compression by entropy of payload
        payload = data[metadata['config_offset']:]
        if len(payload) > 4096:
            entropy = _compute_byte_entropy(payload[:4096])
            metadata['estimated_compression'] = entropy > 7.5  # High entropy = likely compressed

    # Check for RBF preamble
    elif RBF_PREAMBLE in data[:1024].tobytes():
        metadata['format'] = BitstreamFormat.RBF
        preamble_end = data[:1024].tobytes().index(RBF_PREAMBLE) + len(RBF_PREAMBLE)
        metadata['header_size'] = preamble_end
        metadata['config_offset'] = preamble_end

        # Check compression
        payload = data[preamble_end:]
        if len(payload) > 4096:
            entropy = _compute_byte_entropy(payload[:4096])
            metadata['estimated_compression'] = entropy > 7.5

    else:
        # Assume raw RBF with no header
        metadata['format'] = BitstreamFormat.RBF
        metadata['header_size'] = 0
        metadata['config_offset'] = 0

        # Check compression on start of file
        if len(data) > 4096:
            entropy = _compute_byte_entropy(data[:4096])
            metadata['estimated_compression'] = entropy > 7.5

    return metadata['format'], metadata


def _estimate_sof_header_size(data: np.ndarray) -> int:
    """
    Estimate SOF header size by finding the largest contiguous data section.

    SOF structure (simplified):
    - Header: Magic, version, metadata (variable length, ~256-4096 bytes)
    - Sections: Length-prefixed data blocks

    The largest section is typically the configuration payload.
    """
    try:
        offset = 0x100  # Skip fixed header (minimum 256 bytes)
        max_section_offset = offset
        max_section_size = 0

        # Scan for section length markers (big-endian 32-bit)
        while offset < min(len(data) - 4, 16384):  # Limit scan to first 16KB
            try:
                section_len = struct.unpack('>I', data[offset:offset+4].tobytes())[0]

                # Sanity check: reasonable section size
                if 1024 < section_len < len(data):
                    if section_len > max_section_size:
                        max_section_size = section_len
                        max_section_offset = offset + 4  # Start of data after length field
                    offset += 4 + section_len
                else:
                    offset += 256  # Skip ahead
            except:
                offset += 256

        if max_section_size > 0:
            return max_section_offset
        else:
            # Fallback: assume 4KB header
            return 0x1000
    except:
        return 0x1000  # Conservative default


def _compute_byte_entropy(data: np.ndarray) -> float:
    """
    Compute Shannon entropy of byte sequence.

    High entropy (>7.5) typically indicates compression or encryption.
    Low entropy (<4.0) indicates regular structures or padding.
    """
    if len(data) == 0:
        return 0.0

    # Count byte frequencies
    counts = np.bincount(data, minlength=256)
    probs = counts / len(data)

    # Remove zero probabilities
    probs = probs[probs > 0]

    # Shannon entropy
    return -np.sum(probs * np.log2(probs))


def strip_sof_header(data: np.ndarray) -> np.ndarray:
    """
    Remove SOF container wrapper, returning only configuration data.

    Args:
        data: Raw SOF file bytes

    Returns:
        Configuration payload (stripped of headers)
    """
    format_type, metadata = detect_format(data)

    if format_type == BitstreamFormat.SOF:
        config_offset = metadata['config_offset']
        return data[config_offset:]
    else:
        # Not SOF, return as-is
        return data


def bytes_to_bits(byte_array: np.ndarray, bit_order: str = 'msb') -> np.ndarray:
    """
    Convert byte array to bit array (0/1 values).

    Args:
        byte_array: NumPy array of uint8
        bit_order: 'msb' (most significant bit first) or 'lsb'

    Returns:
        NumPy array of uint8 with values 0 or 1

    Example:
        >>> bytes_to_bits(np.array([0xA5], dtype=np.uint8))
        array([1, 0, 1, 0, 0, 1, 0, 1], dtype=uint8)
    """
    # Unpack bits
    bits = np.unpackbits(byte_array)

    if bit_order == 'lsb':
        # Reverse each byte
        bits = bits.reshape(-1, 8)[:, ::-1].flatten()

    return bits


def bits_to_bytes(bit_array: np.ndarray, bit_order: str = 'msb') -> np.ndarray:
    """
    Pack bit array (0/1 values) back into bytes.

    Args:
        bit_array: NumPy array of 0/1 values
        bit_order: 'msb' or 'lsb'

    Returns:
        NumPy array of uint8 bytes
    """
    # Ensure bit_array is uint8
    bits = bit_array.astype(np.uint8)

    # Pad to multiple of 8
    remainder = len(bits) % 8
    if remainder != 0:
        bits = np.pad(bits, (0, 8 - remainder), 'constant')

    if bit_order == 'lsb':
        # Reverse each byte
        bits = bits.reshape(-1, 8)[:, ::-1].flatten()

    # Pack bits
    return np.packbits(bits)


def write_rbf(path: str, data: np.ndarray):
    """
    Write byte array to RBF file.

    Args:
        path: Output file path
        data: NumPy array of uint8 bytes
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as f:
        f.write(data.tobytes())


def get_bitstream_info(path: str) -> Dict[str, any]:
    """
    Get comprehensive information about a bitstream file.

    Returns dictionary with:
        - size_bytes: File size
        - size_bits: Configuration bits
        - format: Detected format
        - compression: Estimated compression state
        - entropy: Overall entropy
        - header_size: Header bytes
    """
    data = read_rbf(path)
    format_type, metadata = detect_format(data)

    config_data = data[metadata['config_offset']:]
    bits = bytes_to_bits(config_data)

    info = {
        'size_bytes': len(data),
        'size_bits': len(bits),
        'format': format_type,
        'compression': metadata['estimated_compression'],
        'entropy': _compute_byte_entropy(config_data[:min(len(config_data), 65536)]),
        'header_size': metadata['header_size'],
        'config_offset': metadata['config_offset']
    }

    return info


if __name__ == '__main__':
    # Simple CLI for testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rbf_utils.py <bitstream_file.rbf|.sof>")
        sys.exit(1)

    path = sys.argv[1]

    print(f"Analyzing: {path}")
    print("-" * 60)

    info = get_bitstream_info(path)

    print(f"Format:       {info['format']}")
    print(f"Size:         {info['size_bytes']:,} bytes ({info['size_bits']:,} bits)")
    print(f"Header:       {info['header_size']} bytes")
    print(f"Config Start: 0x{info['config_offset']:08x}")
    print(f"Entropy:      {info['entropy']:.4f}")
    print(f"Compressed:   {'Yes (likely)' if info['compression'] else 'No (likely)'}")

    if info['compression']:
        print("\nWARNING: High entropy detected - bitstream may be compressed.")
        print("For Arria 10 analysis, recompile with bitstream_compression=off")
