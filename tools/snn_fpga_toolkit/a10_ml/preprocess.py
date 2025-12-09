"""
preprocess.py

Load and preprocess Intel FPGA bitstreams (.sof, .rbf).

Handles:
- SOF (SRAM Object File) with metadata headers
- RBF (Raw Binary File) without headers
- Stratix 10 / Arria 10 encrypted formats
"""

import struct
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


# Intel FPGA magic bytes
SOF_MAGIC = b'\x00\x09\x0f\xf0'
STRATIX10_ENCRYPTED_MAGIC = b'\x00\x01\x00\x00'


def load_bitstream(path: Path) -> bytes:
    """
    Load bitstream file from disk.

    Args:
        path: Path to .sof or .rbf file

    Returns:
        Raw bytes of the file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is too small or invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Bitstream file not found: {path}")

    data = path.read_bytes()

    if len(data) < 1024:
        raise ValueError(f"File too small to be a valid bitstream: {len(data)} bytes")

    print(f"✅ Loaded {len(data):,} bytes from {path.name}")
    return data


def strip_intel_headers(raw: bytes) -> bytes:
    """
    Remove Intel FPGA headers and extract raw configuration bitstream.

    SOF format structure:
    - Magic: 0x00 0x09 0x0f 0xf0
    - Header with metadata (variable length)
    - Configuration data (actual bitstream)

    RBF format:
    - No headers, pure binary data

    Args:
        raw: Raw bytes from .sof or .rbf file

    Returns:
        Pure configuration bitstream (headers removed)
    """
    if len(raw) < 4:
        return raw

    # Check for SOF magic
    if raw[:4] == SOF_MAGIC:
        print("🔍 Detected Intel SOF format")
        return _strip_sof_header(raw)

    # Check for Stratix 10 encrypted format
    elif raw[:4] == STRATIX10_ENCRYPTED_MAGIC:
        print("🔒 Detected Stratix 10 encrypted format (limited support)")
        return _strip_stratix10_header(raw)

    # Assume RBF (raw binary file)
    else:
        print("📄 Assumed RBF format (no headers)")
        return raw


def _strip_sof_header(raw: bytes) -> bytes:
    """
    Strip SOF header to extract configuration section.

    SOF structure (simplified):
    - 0x00: Magic (4 bytes)
    - 0x04: Header length (4 bytes, big-endian)
    - 0x08: Metadata fields
    - Variable: Configuration data

    Returns:
        Configuration bitstream starting after header
    """
    if len(raw) < 8:
        return raw

    # Read header length (bytes 4-7, big-endian)
    try:
        header_len = struct.unpack('>I', raw[4:8])[0]

        # Sanity check
        if header_len < 8 or header_len > len(raw):
            print(f"⚠️  Invalid header length {header_len}, using offset 512")
            header_len = 512

        print(f"📏 SOF header length: {header_len} bytes")

        # Skip header and return configuration data
        return raw[header_len:]

    except Exception as e:
        print(f"⚠️  Error parsing SOF header: {e}")
        # Fallback: skip first 512 bytes (typical SOF header size)
        return raw[512:] if len(raw) > 512 else raw


def _strip_stratix10_header(raw: bytes) -> bytes:
    """
    Strip Stratix 10 encrypted bitstream header.

    Note: Encrypted bitstreams cannot be fully processed without keys.
    This function attempts to skip the encryption wrapper.

    Returns:
        Data after encryption header (still encrypted)
    """
    # Stratix 10 encrypted format has a variable-length header
    # Typical offset is 1024-2048 bytes
    # Without full format specification, we use a heuristic

    offset = 1024
    if len(raw) > offset:
        print(f"⚠️  Skipping {offset} byte encryption header")
        return raw[offset:]
    return raw


def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Convert bytes to bit array.

    Args:
        data: Raw bytes

    Returns:
        NumPy array of bits (0 or 1), length = len(data) * 8
    """
    # Unpack bytes to bits using NumPy
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    print(f"🔢 Converted {len(data):,} bytes → {len(bits):,} bits")
    return bits


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convert bit array back to bytes (for saving/verification).

    Args:
        bits: NumPy array of bits (0 or 1)

    Returns:
        Raw bytes
    """
    # Pad to multiple of 8
    if len(bits) % 8 != 0:
        padding = 8 - (len(bits) % 8)
        bits = np.concatenate([bits, np.zeros(padding, dtype=np.uint8)])

    # Pack bits to bytes
    byte_array = np.packbits(bits)
    return bytes(byte_array)


def analyze_bitstream_header(raw: bytes, num_bytes: int = 512) -> dict:
    """
    Analyze first N bytes of bitstream for debugging.

    Args:
        raw: Raw bitstream bytes
        num_bytes: Number of bytes to analyze

    Returns:
        Dictionary with analysis results
    """
    header = raw[:num_bytes]

    analysis = {
        'total_size': len(raw),
        'header_bytes': len(header),
        'magic': header[:4].hex(),
        'is_sof': header[:4] == SOF_MAGIC,
        'is_encrypted': header[:4] == STRATIX10_ENCRYPTED_MAGIC,
        'entropy': _calculate_entropy(header),
        'unique_bytes': len(set(header)),
        'first_16_bytes': header[:16].hex(' '),
    }

    return analysis


def _calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of byte sequence."""
    if not data:
        return 0.0

    # Count byte frequencies
    freq = np.zeros(256, dtype=int)
    for byte in data:
        freq[byte] += 1

    # Calculate probabilities and entropy
    total = len(data)
    entropy = 0.0
    for count in freq:
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)

    return entropy


# Example usage for testing
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <bitstream.sof>")
        sys.exit(1)

    # Load and analyze
    path = Path(sys.argv[1])
    raw = load_bitstream(path)

    print("\n" + "=" * 60)
    print("Bitstream Analysis")
    print("=" * 60)

    analysis = analyze_bitstream_header(raw)
    for key, value in analysis.items():
        print(f"{key:20s}: {value}")

    print("\n" + "=" * 60)
    print("Header Stripping")
    print("=" * 60)

    # Strip headers
    clean = strip_intel_headers(raw)
    print(f"📦 Original size: {len(raw):,} bytes")
    print(f"✂️  Stripped size: {len(clean):,} bytes")
    print(f"📉 Removed: {len(raw) - len(clean):,} bytes ({100 * (len(raw) - len(clean)) / len(raw):.1f}%)")

    # Convert to bits
    bits = bytes_to_bits(clean)
    print(f"🔢 Bit array length: {len(bits):,}")
