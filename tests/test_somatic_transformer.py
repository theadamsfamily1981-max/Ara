"""
Unit Tests for Somatic Transformer Components.

These tests verify that the somatic integration works correctly:
1. SomaticEmbedding produces correct shapes and responds to body state
2. SomaticAttention applies focus gating based on arousal
3. Full model integration passes body state through correctly

Run with:
    pytest tests/test_somatic_transformer.py -v
    python -m pytest tests/test_somatic_transformer.py -v --tb=short
"""

import pytest
import torch
import torch.nn as nn
import math

# Import somatic components
from tfan.models.tfan7b.somatic_embedding import (
    SomaticEmbedding,
    CortisolInjector,
    SomaticEncoder,
    create_somatic_tensor,
    somatic_from_hal,
)
from tfan.models.tfan7b.somatic_attention import (
    SomaticAttention,
    SomaticSSAAttention,
)
from tfan.models.tfan7b.modeling_tfan_somatic import (
    SomaticConfig,
    SomaticDecoderLayer,
    TFANSomaticModel,
    TFANSomaticForCausalLM,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    """Get test dtype."""
    return torch.float32


@pytest.fixture
def mini_config():
    """Minimal config for fast testing."""
    return SomaticConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        enable_somatic=True,
        somatic_dim=7,
        somatic_intermediate=32,
        use_cortisol=True,
        use_somatic_attention=True,
    )


@pytest.fixture
def calm_somatic():
    """Calm body state."""
    return create_somatic_tensor(
        pain=0.0, entropy=0.1, pad_p=0.5, pad_a=-0.5, pad_d=0.3
    )


@pytest.fixture
def stressed_somatic():
    """Stressed body state."""
    return create_somatic_tensor(
        pain=0.9, entropy=0.9, pad_p=-0.7, pad_a=0.8, pad_d=-0.5
    )


# ==============================================================================
# SomaticEmbedding Tests
# ==============================================================================

class TestSomaticEmbedding:
    """Tests for SomaticEmbedding module."""

    def test_output_shape(self, device, dtype):
        """Verify output shape is [batch, 1, hidden_size]."""
        hidden_size = 128
        batch_size = 2

        embed = SomaticEmbedding(hidden_size=hidden_size).to(device, dtype)
        somatic = create_somatic_tensor(pain=0.5, pad_a=0.3).to(device, dtype)
        somatic = somatic.expand(batch_size, -1)

        output = embed(somatic)

        assert output.shape == (batch_size, 1, hidden_size)
        assert output.device == device
        assert output.dtype == dtype

    def test_different_body_states_produce_different_embeddings(self, device, dtype):
        """Verify different body states produce different embeddings."""
        hidden_size = 64
        embed = SomaticEmbedding(hidden_size=hidden_size).to(device, dtype)

        # Calm state
        calm = create_somatic_tensor(pain=0.0, pad_a=-0.5).to(device, dtype)
        calm_emb = embed(calm)

        # Stressed state
        stressed = create_somatic_tensor(pain=0.9, pad_a=0.8).to(device, dtype)
        stressed_emb = embed(stressed)

        # Should be different
        assert not torch.allclose(calm_emb, stressed_emb, atol=1e-6)

        # Norms should be different (stressed typically higher due to cortisol)
        calm_norm = calm_emb.norm()
        stressed_norm = stressed_emb.norm()
        assert calm_norm != stressed_norm

    def test_return_components(self, device, dtype):
        """Verify return_components provides diagnostic info."""
        embed = SomaticEmbedding(hidden_size=64).to(device, dtype)
        somatic = create_somatic_tensor(pain=0.7, pad_a=0.5).to(device, dtype)

        output, components = embed(somatic, return_components=True)

        assert 'pain_level' in components
        assert 'arousal_level' in components
        assert 'bias_norm' in components
        assert components['pain_level'].item() == pytest.approx(0.7, abs=1e-5)
        assert components['arousal_level'].item() == pytest.approx(0.5, abs=1e-5)

    def test_gated_embedding(self, device, dtype):
        """Test SomaticEmbedding with gating enabled."""
        embed = SomaticEmbedding(hidden_size=64, use_gate=True).to(device, dtype)
        somatic = create_somatic_tensor(pain=0.5).to(device, dtype)

        output, components = embed(somatic, return_components=True)

        assert output.shape == (1, 1, 64)
        assert components['gate_value'] is not None


class TestCortisolInjector:
    """Tests for CortisolInjector module."""

    def test_no_injection_below_threshold(self, device, dtype):
        """Verify no cortisol injected when pain/arousal below threshold."""
        hidden_size = 64
        injector = CortisolInjector(hidden_size).to(device, dtype)

        hidden = torch.randn(1, 10, hidden_size, device=device, dtype=dtype)
        pain = torch.tensor([0.1], device=device, dtype=dtype)
        arousal = torch.tensor([0.2], device=device, dtype=dtype)

        output = injector(hidden, pain, arousal)

        # Should be nearly identical (small numerical differences possible)
        assert torch.allclose(output, hidden, atol=1e-4)

    def test_injection_above_threshold(self, device, dtype):
        """Verify cortisol injected when pain/arousal above threshold."""
        hidden_size = 64
        injector = CortisolInjector(hidden_size).to(device, dtype)

        hidden = torch.randn(1, 10, hidden_size, device=device, dtype=dtype)
        pain = torch.tensor([0.9], device=device, dtype=dtype)
        arousal = torch.tensor([0.9], device=device, dtype=dtype)

        output = injector(hidden, pain, arousal)

        # Should be different
        assert not torch.allclose(output, hidden, atol=1e-4)


# ==============================================================================
# SomaticAttention Tests
# ==============================================================================

class TestSomaticAttention:
    """Tests for SomaticAttention module."""

    def test_output_shape(self, device, dtype):
        """Verify attention output shape."""
        batch_size = 2
        seq_len = 16
        hidden_size = 64
        num_heads = 4

        attn = SomaticAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        ).to(device, dtype)

        hidden = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
        arousal = torch.tensor([0.5, -0.3], device=device, dtype=dtype)

        outputs = attn(hidden, arousal=arousal)

        assert outputs[0].shape == (batch_size, seq_len, hidden_size)

    def test_arousal_affects_attention(self, device, dtype):
        """Verify high arousal triggers tunnel vision (sparser attention)."""
        hidden_size = 64
        num_heads = 4
        seq_len = 32

        attn = SomaticAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            focus_threshold=0.3,
        ).to(device, dtype)

        hidden = torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype)

        # Calm arousal (dreamy, broad attention)
        calm_out = attn(hidden, arousal=torch.tensor([-0.8], device=device))[0]

        # High arousal (tunnel vision)
        stress_out = attn(hidden, arousal=torch.tensor([0.9], device=device))[0]

        # Outputs should differ due to different attention patterns
        assert not torch.allclose(calm_out, stress_out, atol=1e-3)

    def test_focus_scalar_computation(self, device, dtype):
        """Test focus scalar computation from arousal."""
        attn = SomaticAttention(hidden_size=64, num_heads=4).to(device, dtype)

        # High arousal -> low focus (tight)
        high_arousal = torch.tensor([0.9], device=device, dtype=dtype)
        focus_high = attn._compute_focus_scalar(high_arousal)
        assert focus_high.item() < 1.0

        # Low arousal -> high focus (broad)
        low_arousal = torch.tensor([-0.9], device=device, dtype=dtype)
        focus_low = attn._compute_focus_scalar(low_arousal)
        assert focus_low.item() > 1.0

    def test_no_arousal_default(self, device, dtype):
        """Test attention works without arousal (default behavior)."""
        attn = SomaticAttention(hidden_size=64, num_heads=4).to(device, dtype)
        hidden = torch.randn(1, 16, 64, device=device, dtype=dtype)

        # Should work without arousal
        outputs = attn(hidden, arousal=None)
        assert outputs[0].shape == (1, 16, 64)


# ==============================================================================
# Full Model Integration Tests
# ==============================================================================

class TestSomaticModel:
    """Integration tests for full somatic model."""

    def test_model_forward(self, mini_config, device, dtype):
        """Test full model forward pass."""
        model = TFANSomaticForCausalLM(mini_config).to(device, dtype)

        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, mini_config.vocab_size, (batch_size, seq_len), device=device)
        somatic = create_somatic_tensor(pain=0.3, pad_a=0.5).to(device, dtype)
        somatic = somatic.expand(batch_size, -1)

        outputs = model(input_ids, somatic_state=somatic, return_dict=True)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, mini_config.vocab_size)
        assert "somatic_info" in outputs

    def test_somatic_cache(self, mini_config, device, dtype):
        """Test somatic state caching for generation."""
        model = TFANSomaticForCausalLM(mini_config).to(device, dtype)

        # Set somatic state
        somatic = create_somatic_tensor(pain=0.5, pad_a=0.7).to(device, dtype)
        model.set_somatic_state(somatic)

        # Forward without explicit somatic_state (uses cache)
        input_ids = torch.randint(0, mini_config.vocab_size, (1, 4), device=device)
        outputs = model(input_ids, return_dict=True)

        assert outputs["logits"].shape == (1, 4, mini_config.vocab_size)

    def test_different_somatic_states_affect_output(self, mini_config, device, dtype):
        """Verify different body states produce different outputs."""
        model = TFANSomaticForCausalLM(mini_config).to(device, dtype)
        model.eval()

        input_ids = torch.randint(0, mini_config.vocab_size, (1, 8), device=device)

        # Calm state
        calm = create_somatic_tensor(pain=0.0, pad_a=-0.5).to(device, dtype)
        with torch.no_grad():
            calm_out = model(input_ids, somatic_state=calm, return_dict=True)["logits"]

        # Stressed state
        stressed = create_somatic_tensor(pain=0.9, pad_a=0.9).to(device, dtype)
        with torch.no_grad():
            stressed_out = model(input_ids, somatic_state=stressed, return_dict=True)["logits"]

        # Outputs should be different
        assert not torch.allclose(calm_out, stressed_out, atol=1e-3)

    def test_generate(self, mini_config, device, dtype):
        """Test generation with somatic modulation."""
        model = TFANSomaticForCausalLM(mini_config).to(device, dtype)
        model.eval()

        input_ids = torch.randint(0, mini_config.vocab_size, (1, 4), device=device)
        somatic = create_somatic_tensor(pain=0.3, pad_a=0.2).to(device, dtype)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=10,
                somatic_state=somatic,
                do_sample=False,
            )

        assert output_ids.shape[1] > input_ids.shape[1]
        assert output_ids.shape[1] <= 10


# ==============================================================================
# Helper Function Tests
# ==============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_somatic_tensor(self, device):
        """Test somatic tensor creation."""
        tensor = create_somatic_tensor(
            pain=0.5,
            entropy=0.3,
            flow_x=0.1,
            flow_y=-0.1,
            pad_p=0.2,
            pad_a=0.4,
            pad_d=-0.3,
            device=device,
        )

        assert tensor.shape == (1, 7)
        assert tensor[0, 0].item() == pytest.approx(0.5)  # pain
        assert tensor[0, 5].item() == pytest.approx(0.4)  # pad_a

    def test_somatic_from_hal(self, device):
        """Test creating somatic tensor from HAL state dict."""
        hal_state = {
            'pain': 0.7,
            'entropy': 0.4,
            'flow': (0.1, 0.2),
            'pad': {'p': 0.3, 'a': 0.5, 'd': -0.2},
        }

        tensor = somatic_from_hal(hal_state, device=device)

        assert tensor.shape == (1, 7)
        assert tensor[0, 0].item() == pytest.approx(0.7)  # pain
        assert tensor[0, 5].item() == pytest.approx(0.5)  # pad_a


# ==============================================================================
# Gradient Flow Tests
# ==============================================================================

class TestGradientFlow:
    """Tests for gradient flow through somatic components."""

    def test_embedding_gradients(self, device):
        """Verify gradients flow through SomaticEmbedding."""
        embed = SomaticEmbedding(hidden_size=64).to(device)
        somatic = torch.randn(1, 7, device=device, requires_grad=True)

        output = embed(somatic)
        loss = output.sum()
        loss.backward()

        assert somatic.grad is not None
        assert not torch.all(somatic.grad == 0)

    def test_attention_gradients(self, device):
        """Verify gradients flow through SomaticAttention."""
        attn = SomaticAttention(hidden_size=64, num_heads=4).to(device)
        hidden = torch.randn(1, 16, 64, device=device, requires_grad=True)
        arousal = torch.tensor([0.5], device=device, requires_grad=True)

        output = attn(hidden, arousal=arousal)[0]
        loss = output.sum()
        loss.backward()

        assert hidden.grad is not None
        # Note: arousal might not have grad if focus_scalar uses .item()
        # This is acceptable for inference but limits training


# ==============================================================================
# Edge Case Tests
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_sequence(self, mini_config, device, dtype):
        """Test handling of very short sequences."""
        model = TFANSomaticForCausalLM(mini_config).to(device, dtype)

        # Single token
        input_ids = torch.randint(0, mini_config.vocab_size, (1, 1), device=device)
        somatic = create_somatic_tensor().to(device, dtype)

        outputs = model(input_ids, somatic_state=somatic, return_dict=True)
        assert outputs["logits"].shape == (1, 1, mini_config.vocab_size)

    def test_extreme_arousal_values(self, device, dtype):
        """Test attention with extreme arousal values."""
        attn = SomaticAttention(hidden_size=64, num_heads=4).to(device, dtype)
        hidden = torch.randn(1, 16, 64, device=device, dtype=dtype)

        # Extreme positive
        out_high = attn(hidden, arousal=torch.tensor([1.0], device=device))[0]
        assert not torch.isnan(out_high).any()

        # Extreme negative
        out_low = attn(hidden, arousal=torch.tensor([-1.0], device=device))[0]
        assert not torch.isnan(out_low).any()

        # Beyond range (should be clamped)
        out_beyond = attn(hidden, arousal=torch.tensor([2.0], device=device))[0]
        assert not torch.isnan(out_beyond).any()

    def test_no_somatic_encoder(self, device, dtype):
        """Test model works without somatic encoder."""
        config = SomaticConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_kv_heads=1,
            intermediate_size=64,
            enable_somatic=False,  # Disabled
        )
        model = TFANSomaticForCausalLM(config).to(device, dtype)

        input_ids = torch.randint(0, 100, (1, 4), device=device)
        outputs = model(input_ids, return_dict=True)

        assert outputs["logits"].shape == (1, 4, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
