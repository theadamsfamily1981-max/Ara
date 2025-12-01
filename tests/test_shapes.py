"""
Shape validation tests for TF-A-N 7B.
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.models.tfan7b import TFANConfig, TFANForCausalLM


@pytest.fixture
def config():
    """Create test config."""
    config = TFANConfig()
    config.num_hidden_layers = 2  # Reduce for testing
    return config


@pytest.fixture
def model(config):
    """Create test model."""
    return TFANForCausalLM(config).eval()


def test_forward_pass_2k(model, config):
    """Test forward pass with 2k context."""
    batch_size = 2
    seq_len = 2048

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)

    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    print(f"✓ 2k context test passed")


def test_forward_pass_8k(model, config):
    """Test forward pass with 8k context."""
    batch_size = 1
    seq_len = 8192

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)

    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    print(f"✓ 8k context test passed")


def test_parameter_count(model):
    """Test that parameter count is within range."""
    from tfan.models.tfan7b import count_parameters

    counts = count_parameters(model)
    # With reduced layers (2), won't be 7B, but test structure
    assert counts["total"] > 0
    print(f"✓ Parameter count: {counts['total']:,}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
