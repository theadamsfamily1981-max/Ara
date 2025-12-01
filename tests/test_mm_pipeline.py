#!/usr/bin/env python3
"""
Multi-modal pipeline tests for TF-A-N.

Tests the text/audio/video fusion pipeline.
"""

import pytest
import sys


def test_imports():
    """Test that multi-modal imports work."""
    # These are skipped if dependencies not available
    try:
        from tfan.mm import TextAdapter, pack_and_mask
        print("✓ Multi-modal imports successful")
    except ImportError as e:
        pytest.skip(f"Multi-modal dependencies not available: {e}")


def test_text_adapter_basic():
    """Test basic text adapter."""
    try:
        from tfan.mm import TextAdapter
        adapter = TextAdapter(output_dim=128)
        result = adapter(['test sentence'])
        print(f"✓ Text adapter: {result.features.shape}")
    except ImportError as e:
        pytest.skip(f"Text adapter dependencies not available: {e}")


def test_pack_and_mask():
    """Test pack and mask fusion."""
    try:
        import torch
        from tfan.mm import pack_and_mask

        fused = pack_and_mask(
            {'text': torch.randn(1, 10, 128)},
            {'text': torch.linspace(0, 1, 10).unsqueeze(0)},
            d_model=128, n_heads=4
        )
        print(f"✓ Fusion: {fused.tokens.shape}")
    except ImportError as e:
        pytest.skip(f"Pack and mask dependencies not available: {e}")


if __name__ == '__main__':
    # Run basic tests
    test_imports()
    test_text_adapter_basic()
    test_pack_and_mask()
    print("\n✓ All multi-modal pipeline tests passed")
    sys.exit(0)
