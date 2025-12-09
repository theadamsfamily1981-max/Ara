#!/usr/bin/env python3
"""
Sanity Test: Prosody Factorization
===================================

Verifies that the prosody tokenizer separates content from prosody.

Success criteria:
- Tokenizer produces valid ProsodyToken objects
- Different audio produces different tokens
- Subspaces are relatively independent

Run: python tests/sanity/test_prosody_factorization.py
"""

import sys
import numpy as np

# Add project root to path
sys.path.insert(0, '/home/user/Ara')

from ara.nervous.prosody import (
    ProsodyTokenizer,
    PROSODY_DIM,
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def generate_fake_audio(duration_ms: int = 80, seed: int = 42) -> np.ndarray:
    """Generate fake audio samples for testing."""
    np.random.seed(seed)
    sample_rate = 16000
    n_samples = int(sample_rate * duration_ms / 1000)
    return np.random.randn(n_samples).astype(np.float32)


def test_tokenizer_creates_valid_tokens():
    """Test that tokenizer produces correctly-shaped tokens."""
    print("Test: Tokenizer creates valid tokens...")

    tokenizer = ProsodyTokenizer()
    audio = generate_fake_audio(80)

    sequence = tokenizer.tokenize(audio)

    assert len(sequence.tokens) >= 1, f"Expected at least 1 token, got {len(sequence.tokens)}"

    token = sequence.tokens[0]

    assert token.phonetic_hv.shape == (512,), f"Expected (512,), got {token.phonetic_hv.shape}"
    assert token.pitch_hv.shape == (512,), f"Expected (512,), got {token.pitch_hv.shape}"
    assert token.timbre_hv.shape == (512,), f"Expected (512,), got {token.timbre_hv.shape}"
    assert token.prosodic_hv.shape == (512,), f"Expected (512,), got {token.prosodic_hv.shape}"
    assert token.token_hv.shape == (PROSODY_DIM,), f"Expected ({PROSODY_DIM},), got {token.token_hv.shape}"

    print(f"  Got {len(sequence.tokens)} tokens")
    print("  PASS: Token shapes correct")


def test_different_audio_different_tokens():
    """Test that different audio produces different tokens."""
    print("Test: Different audio produces different tokens...")

    tokenizer = ProsodyTokenizer()

    audio_a = generate_fake_audio(80, seed=1)
    audio_b = generate_fake_audio(80, seed=2)

    seq_a = tokenizer.tokenize(audio_a)
    seq_b = tokenizer.tokenize(audio_b)

    token_a = seq_a.tokens[0]
    token_b = seq_b.tokens[0]

    sim = cosine_similarity(token_a.token_hv, token_b.token_hv)

    # Different audio should produce different tokens
    assert sim < 0.95, f"Expected similarity < 0.95, got {sim:.3f}"

    print(f"  PASS: Token similarity = {sim:.3f} (expected < 0.95)")


def test_same_audio_same_tokens():
    """Test that same audio produces identical tokens."""
    print("Test: Same audio produces identical tokens...")

    tokenizer = ProsodyTokenizer()

    audio = generate_fake_audio(80, seed=42)

    seq_a = tokenizer.tokenize(audio)
    seq_b = tokenizer.tokenize(audio)

    token_a = seq_a.tokens[0]
    token_b = seq_b.tokens[0]

    sim = cosine_similarity(token_a.token_hv, token_b.token_hv)

    # Same audio should produce identical tokens
    assert sim > 0.99, f"Expected similarity > 0.99, got {sim:.3f}"

    print(f"  PASS: Token similarity = {sim:.3f} (expected > 0.99)")


def test_subspace_independence():
    """Test that different subspaces encode different information."""
    print("Test: Subspaces are relatively independent...")

    tokenizer = ProsodyTokenizer()

    # Generate several audio samples
    tokens = []
    for seed in range(10):
        audio = generate_fake_audio(80, seed=seed)
        seq = tokenizer.tokenize(audio)
        tokens.append(seq.tokens[0])

    # Check correlations between subspaces
    phonetic_prosodic_sims = []
    pitch_timbre_sims = []

    for t in tokens:
        # Phonetic vs prosodic should be relatively independent
        sim = cosine_similarity(t.phonetic_hv, t.prosodic_hv)
        phonetic_prosodic_sims.append(sim)

        # Pitch vs timbre should be relatively independent
        sim = cosine_similarity(t.pitch_hv, t.timbre_hv)
        pitch_timbre_sims.append(sim)

    avg_phonetic_prosodic = np.mean(np.abs(phonetic_prosodic_sims))
    avg_pitch_timbre = np.mean(np.abs(pitch_timbre_sims))

    # Subspaces should not be highly correlated
    print(f"  Avg |phonetic<->prosodic|: {avg_phonetic_prosodic:.3f}")
    print(f"  Avg |pitch<->timbre|: {avg_pitch_timbre:.3f}")

    # These are random projections, so correlation should be low
    assert avg_phonetic_prosodic < 0.5, f"Subspaces too correlated: {avg_phonetic_prosodic:.3f}"

    print("  PASS: Subspaces relatively independent")


def test_sequence_aggregation():
    """Test that sequence can be aggregated to single HV."""
    print("Test: Sequence aggregation...")

    tokenizer = ProsodyTokenizer()
    audio = generate_fake_audio(200, seed=42)  # Longer audio

    seq = tokenizer.tokenize(audio)

    # Should be able to get a single HV representing the sequence
    seq_hv = seq.to_sequence_hv()

    assert seq_hv.shape == (PROSODY_DIM,), f"Expected ({PROSODY_DIM},), got {seq_hv.shape}"
    assert np.linalg.norm(seq_hv) > 0, "Sequence HV should be non-zero"

    print(f"  Sequence has {len(seq.tokens)} tokens")
    print(f"  Aggregated to shape {seq_hv.shape}")
    print("  PASS: Sequence aggregation works")


def main():
    """Run all prosody factorization sanity tests."""
    print("=" * 60)
    print("SANITY TEST: Prosody Factorization")
    print("=" * 60)
    print()

    try:
        test_tokenizer_creates_valid_tokens()
        test_different_audio_different_tokens()
        test_same_audio_same_tokens()
        test_subspace_independence()
        test_sequence_aggregation()

        print()
        print("=" * 60)
        print("ALL PROSODY TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
