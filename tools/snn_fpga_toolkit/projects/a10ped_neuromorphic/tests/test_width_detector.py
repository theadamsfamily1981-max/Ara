#!/usr/bin/env python3
"""
Unit tests for algorithmic width detector.

Tests the autocorrelation + entropy minimization approach on synthetic
bitstreams with known geometry.
"""

import pytest
import numpy as np
import tempfile
import struct
from pathlib import Path
import sys

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools' / 'bitstream_analysis'))

from width_detector import WidthDetector


class TestSyntheticBitstreams:
    """Tests using synthetic bitstreams with known widths."""

    def create_synthetic_sof(self, width: int, num_rows: int) -> bytes:
        """
        Create synthetic .sof file with known frame width.

        Strategy: Generate correlated rows with consistent entropy
        to simulate FPGA configuration frames.
        """
        # SOF header
        header = WidthDetector.SOF_MAGIC
        header += struct.pack('>I', 0x00000001)  # Version
        header += struct.pack('>I', 0x00000100)  # Header length
        header += b'\x00' * (0x100 - len(header))  # Pad to 256 bytes

        # Generate configuration data with structure at target width
        config_data = bytearray()

        # Create a "template" row with some pattern
        np.random.seed(42)
        template = np.random.randint(0, 256, width, dtype=np.uint8)

        for i in range(num_rows):
            # Each row is similar to template but with some variation
            # This creates autocorrelation between rows
            noise = np.random.randint(-30, 30, width, dtype=np.int16)
            row = np.clip(template.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Add some column structure (simulates LAB columns)
            for col in range(0, width, 64):
                # Every 64 bytes, insert a "marker" pattern
                if col + 8 <= width:
                    row[col:col+8] = [0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00]

            config_data.extend(row.tobytes())

            # Periodically update template (simulates different regions)
            if i % 10 == 0:
                template = np.random.randint(0, 256, width, dtype=np.uint8)

        # Wrap config data in SOF section
        section_header = struct.pack('>I', len(config_data))
        full_data = header + section_header + config_data

        return bytes(full_data)

    def create_synthetic_rbf(self, width: int, num_rows: int) -> bytes:
        """Create synthetic uncompressed .rbf with known width."""
        # Simple RBF header
        header = b'\xff' * 512  # Preamble
        header += struct.pack('>I', 0xCAFEBABE)  # Marker

        # Generate config data (similar to SOF but without section wrapper)
        config_data = bytearray()

        np.random.seed(123)
        template = np.random.randint(0, 256, width, dtype=np.uint8)

        for i in range(num_rows):
            noise = np.random.randint(-20, 20, width, dtype=np.int16)
            row = np.clip(template.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Column markers
            for col in range(0, width, 128):
                if col + 4 <= width:
                    row[col:col+4] = [0xAA, 0x55, 0xAA, 0x55]

            config_data.extend(row.tobytes())

            if i % 15 == 0:
                template = np.random.randint(0, 256, width, dtype=np.uint8)

        return header + bytes(config_data)

    def test_sof_width_512(self):
        """Test detection of width=512 in synthetic SOF."""
        width = 512
        num_rows = 200

        # Create synthetic bitstream
        data = self.create_synthetic_sof(width, num_rows)

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix='.sof', delete=False) as f:
            f.write(data)
            temp_path = f.name

        try:
            # Run detector
            detector = WidthDetector(temp_path, verbose=False)
            assert detector.load_bitstream()

            results = detector.detect_width(candidates=[256, 512, 1024], top_k=3)

            # Check that 512 is top candidate
            assert len(results) >= 1
            detected_width, score = results[0]

            assert detected_width == width, \
                f"Expected width {width}, got {detected_width} (score {score:.4f})"
            assert score > 0.1, f"Score too low: {score:.4f}"

        finally:
            Path(temp_path).unlink()

    def test_sof_width_1024(self):
        """Test detection of width=1024 in synthetic SOF."""
        width = 1024
        num_rows = 150

        data = self.create_synthetic_sof(width, num_rows)

        with tempfile.NamedTemporaryFile(suffix='.sof', delete=False) as f:
            f.write(data)
            temp_path = f.name

        try:
            detector = WidthDetector(temp_path, verbose=False)
            assert detector.load_bitstream()

            results = detector.detect_width(candidates=[512, 1024, 2048], top_k=3)

            assert len(results) >= 1
            detected_width, score = results[0]

            assert detected_width == width, \
                f"Expected width {width}, got {detected_width}"
            assert score > 0.1, f"Score too low: {score:.4f}"

        finally:
            Path(temp_path).unlink()

    def test_rbf_width_768(self):
        """Test detection of width=768 in synthetic RBF."""
        width = 768
        num_rows = 180

        data = self.create_synthetic_rbf(width, num_rows)

        with tempfile.NamedTemporaryFile(suffix='.rbf', delete=False) as f:
            f.write(data)
            temp_path = f.name

        try:
            detector = WidthDetector(temp_path, verbose=False)
            assert detector.load_bitstream()

            results = detector.detect_width(candidates=[512, 768, 1024], top_k=3)

            assert len(results) >= 1
            detected_width, score = results[0]

            # For RBF, detection may be slightly less accurate
            assert detected_width == width, \
                f"Expected width {width}, got {detected_width}"

        finally:
            Path(temp_path).unlink()

    def test_top_k_ranking(self):
        """Test that top-k returns correctly ranked results."""
        width = 1024
        num_rows = 100

        data = self.create_synthetic_sof(width, num_rows)

        with tempfile.NamedTemporaryFile(suffix='.sof', delete=False) as f:
            f.write(data)
            temp_path = f.name

        try:
            detector = WidthDetector(temp_path, verbose=False)
            assert detector.load_bitstream()

            results = detector.detect_width(
                candidates=[256, 512, 768, 1024, 1536, 2048],
                top_k=6
            )

            # Check that results are sorted by score (descending)
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True), \
                "Results not sorted by score"

            # Check that correct width has highest score
            top_width, top_score = results[0]
            assert top_width == width, \
                f"Top candidate should be {width}, got {top_width}"

        finally:
            Path(temp_path).unlink()


class TestWidthDetectorMethods:
    """Test individual methods of WidthDetector."""

    def test_entropy_computation(self):
        """Test Shannon entropy calculation."""
        detector = WidthDetector("dummy.sof", verbose=False)

        # Uniform distribution -> high entropy
        uniform = bytes(range(256))
        entropy_uniform = detector._compute_entropy(uniform)
        assert entropy_uniform > 7.0, "Uniform distribution should have high entropy"

        # Single value -> zero entropy
        constant = bytes([42] * 256)
        entropy_constant = detector._compute_entropy(constant)
        assert entropy_constant == 0.0, "Constant should have zero entropy"

        # Binary distribution
        binary = bytes([0] * 128 + [255] * 128)
        entropy_binary = detector._compute_entropy(binary)
        assert 0.9 < entropy_binary < 1.1, "Binary should have entropy ~1.0"

    def test_autocorrelation_correlated_rows(self):
        """Test autocorrelation on strongly correlated rows."""
        detector = WidthDetector("dummy.sof", verbose=False)

        # Create matrix with highly correlated adjacent rows
        width = 256
        num_rows = 20

        matrix = np.zeros((num_rows, width), dtype=np.uint8)
        base_row = np.random.randint(0, 256, width, dtype=np.uint8)

        for i in range(num_rows):
            # Each row is base + small noise
            noise = np.random.randint(-10, 10, width, dtype=np.int16)
            matrix[i] = np.clip(base_row.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        autocorr = detector._compute_autocorrelation(matrix)
        assert autocorr > 0.5, f"Correlated rows should have high autocorrelation, got {autocorr:.4f}"

    def test_autocorrelation_uncorrelated_rows(self):
        """Test autocorrelation on uncorrelated rows."""
        detector = WidthDetector("dummy.sof", verbose=False)

        # Create matrix with random uncorrelated rows
        width = 256
        num_rows = 20

        np.random.seed(999)
        matrix = np.random.randint(0, 256, (num_rows, width), dtype=np.uint8)

        autocorr = detector._compute_autocorrelation(matrix)
        assert autocorr < 0.3, f"Uncorrelated rows should have low autocorrelation, got {autocorr:.4f}"

    def test_entropy_variance_consistent_rows(self):
        """Test entropy variance on rows with similar entropy."""
        detector = WidthDetector("dummy.sof", verbose=False)

        width = 256
        num_rows = 50

        matrix = np.zeros((num_rows, width), dtype=np.uint8)

        # All rows have similar distribution -> similar entropy
        for i in range(num_rows):
            # Random bytes with consistent distribution
            matrix[i] = np.random.randint(0, 256, width, dtype=np.uint8)

        entropy_var = detector._compute_entropy_variance(matrix)
        assert entropy_var < 0.5, f"Consistent entropy should have low variance, got {entropy_var:.4f}"

    def test_entropy_variance_inconsistent_rows(self):
        """Test entropy variance on rows with varying entropy."""
        detector = WidthDetector("dummy.sof", verbose=False)

        width = 256
        num_rows = 50

        matrix = np.zeros((num_rows, width), dtype=np.uint8)

        # Alternate between high and low entropy rows
        for i in range(num_rows):
            if i % 2 == 0:
                # High entropy (random)
                matrix[i] = np.random.randint(0, 256, width, dtype=np.uint8)
            else:
                # Low entropy (mostly constant)
                matrix[i] = np.full(width, 42, dtype=np.uint8)

        entropy_var = detector._compute_entropy_variance(matrix)
        assert entropy_var > 10.0, f"Inconsistent entropy should have high variance, got {entropy_var:.4f}"

    def test_format_detection_sof(self):
        """Test SOF format detection."""
        data = WidthDetector.SOF_MAGIC + b'\x00' * 1000

        with tempfile.NamedTemporaryFile(suffix='.sof', delete=False) as f:
            f.write(data)
            temp_path = f.name

        try:
            detector = WidthDetector(temp_path, verbose=False)
            detector.load_bitstream()

            assert detector.format_info['format'] == 'sof', \
                "Should detect SOF format"

        finally:
            Path(temp_path).unlink()

    def test_format_detection_rbf(self):
        """Test RBF format detection."""
        data = b'\x00' * 100 + b'\xff\xff\xff\xff' + b'\x00' * 1000

        with tempfile.NamedTemporaryFile(suffix='.rbf', delete=False) as f:
            f.write(data)
            temp_path = f.name

        try:
            detector = WidthDetector(temp_path, verbose=False)
            detector.load_bitstream()

            assert detector.format_info['format'] == 'rbf', \
                "Should detect RBF format"

        finally:
            Path(temp_path).unlink()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_file(self):
        """Test handling of empty bitstream file."""
        with tempfile.NamedTemporaryFile(suffix='.sof', delete=False) as f:
            temp_path = f.name

        try:
            detector = WidthDetector(temp_path, verbose=False)
            result = detector.load_bitstream()

            assert not result, "Should fail to load empty file"

        finally:
            Path(temp_path).unlink()

    def test_tiny_file(self):
        """Test handling of very small file."""
        data = b'\x00' * 100  # Too small to be valid bitstream

        with tempfile.NamedTemporaryFile(suffix='.sof', delete=False) as f:
            f.write(data)
            temp_path = f.name

        try:
            detector = WidthDetector(temp_path, verbose=False)
            detector.load_bitstream()

            # Should load but have minimal config data
            # Detection should handle gracefully
            results = detector.detect_width(candidates=[256], top_k=1)
            assert len(results) == 1  # Should return result even if low quality

        finally:
            Path(temp_path).unlink()

    def test_insufficient_rows(self):
        """Test width detection with insufficient rows."""
        width = 1024
        num_rows = 5  # Too few for good detection

        detector = WidthDetector("dummy.sof", verbose=False)

        # Create minimal matrix
        data = np.random.randint(0, 256, width * num_rows, dtype=np.uint8)
        detector.config_data = data.tobytes()

        results = detector.detect_width(candidates=[1024], top_k=1)

        # Should return result but with low score
        assert len(results) == 1
        _, score = results[0]
        # Score may be low but should not crash


class TestRealWorldScenarios:
    """Test scenarios similar to real FPGA bitstreams."""

    def test_mixed_region_bitstream(self):
        """
        Test bitstream with multiple distinct regions (simulating PCIe/EMIF/fabric).

        Real bitstreams have regions with different characteristics.
        """
        width = 1024
        num_rows = 300

        config_data = bytearray()

        np.random.seed(777)

        # Region 1: High entropy (fabric configuration, rows 0-99)
        for i in range(100):
            row = np.random.randint(0, 256, width, dtype=np.uint8)
            config_data.extend(row.tobytes())

        # Region 2: Medium entropy with structure (PCIe hard IP, rows 100-199)
        template = np.random.randint(0, 256, width, dtype=np.uint8)
        for i in range(100):
            noise = np.random.randint(-15, 15, width, dtype=np.int16)
            row = np.clip(template.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # Add column markers every 128 bytes
            for col in range(0, width, 128):
                if col + 8 <= width:
                    row[col:col+8] = [0xFF] * 8
            config_data.extend(row.tobytes())

        # Region 3: Low entropy (unused/default, rows 200-299)
        for i in range(100):
            row = np.full(width, 0x00, dtype=np.uint8)
            # Sparse markers
            row[0] = 0xFF
            row[width-1] = 0xFF
            config_data.extend(row.tobytes())

        # Create SOF wrapper
        header = WidthDetector.SOF_MAGIC + b'\x00' * 0xFC
        section_header = struct.pack('>I', len(config_data))
        full_data = header + section_header + bytes(config_data)

        with tempfile.NamedTemporaryFile(suffix='.sof', delete=False) as f:
            f.write(full_data)
            temp_path = f.name

        try:
            detector = WidthDetector(temp_path, verbose=False)
            assert detector.load_bitstream()

            results = detector.detect_width(candidates=[512, 768, 1024, 1536], top_k=3)

            # Should still detect correct width despite mixed regions
            detected_width, score = results[0]
            assert detected_width == width, \
                f"Expected {width}, got {detected_width}"

        finally:
            Path(temp_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
