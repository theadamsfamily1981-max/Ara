#!/usr/bin/env python3
"""Example: Using Ara with TFAN Cognitive Architecture.

This example demonstrates how to upgrade Ara from a simple pipeline
to a cognitive entity that perceives through a unified sensory lattice.

Architecture Overview:
    Phase 1: Sensory Bridge (MultiModalIngestor)
        - Audio -> Log-Mel Spectrograms + Prosody
        - Video -> ViT Patches
        - Text  -> Token Embeddings

    Phase 2: Fusion Reactor (MultiModalFuser)
        - Interleaves streams with [AUDIO], [VIDEO], [FUSE] sentinels
        - TLS selects topologically significant landmarks

    Phase 3: Brain (SSAAttention)
        - O(N log N) selective sparse attention
        - Per-head landmark masks from TLS

    Phase 4: Sanity Guard (TopologyGate)
        - Validates output topology against input
        - CAT fallback on hallucination detection

Usage:
    python examples/cognitive_ara_example.py
"""

import asyncio
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_ai_workspace.src.integrations import (
    AraCognitiveBackend,
    AraCognitivePipeline,
    CognitiveFrame,
    create_cognitive_backend,
)


async def basic_cognitive_example():
    """Basic usage: Text-only cognitive processing."""
    print("=" * 60)
    print("Example 1: Basic Cognitive Backend (Text Only)")
    print("=" * 60)

    # Create cognitive backend with text modality
    backend = create_cognitive_backend(
        modalities=["text"],
        device="cpu",
    )

    # Send a message
    response = await backend.send_message(
        prompt="What is the topological significance of neural attention patterns?",
    )

    print(f"\nResponse: {response.content[:500]}...")
    print(f"\nMetadata: {response.metadata}")
    print(f"Latency: {response.latency_ms:.2f}ms")


async def multimodal_example():
    """Multi-modal: Audio + Text cognitive processing."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Modal Cognitive Processing (Audio + Text)")
    print("=" * 60)

    # Create cognitive backend with audio and text
    backend = create_cognitive_backend(
        modalities=["audio", "text"],
        device="cpu",
        d_model=768,
        n_heads=12,
        keep_ratio=0.33,
    )

    # Simulate audio buffer (2 seconds at 16kHz)
    sample_rate = 16000
    duration = 2.0
    audio_buffer = np.random.randn(int(sample_rate * duration)).astype(np.float32)

    # Send message with audio context
    response = await backend.send_message(
        prompt="Describe what you hear in the audio context.",
        audio_buffer=audio_buffer,
    )

    print(f"\nResponse: {response.content[:500]}...")

    # Check cognitive metrics
    metrics = response.metadata.get("cognitive_metrics", {})
    print(f"\nCognitive Metrics:")
    print(f"  - Ingest time: {metrics.get('ingest_time_ms', 0):.2f}ms")
    print(f"  - Fuse time: {metrics.get('fuse_time_ms', 0):.2f}ms")
    print(f"  - Total seq length: {metrics.get('total_seq_len', 0)}")
    print(f"  - Landmarks identified: {metrics.get('n_landmarks', 0)}")
    print(f"  - Gate passed: {response.metadata.get('gate_passed', True)}")


async def pipeline_only_example():
    """Direct pipeline usage: Cognitive processing without LLM."""
    print("\n" + "=" * 60)
    print("Example 3: Direct Pipeline Usage (No LLM)")
    print("=" * 60)

    try:
        # Initialize cognitive pipeline directly
        pipeline = AraCognitivePipeline(
            modalities=["audio", "text"],
            d_model=768,
            n_heads=12,
            keep_ratio=0.33,
            tls_alpha=0.7,
            device="cpu",
        )

        # Create a cognitive frame
        frame = CognitiveFrame(
            text="Hello, this is a test message for cognitive processing.",
            audio_buffer=np.random.randn(32000).astype(np.float32),  # 2 sec audio
            timestamp=0.0,
        )

        # Process through pipeline
        fused, metrics = pipeline.process_frame(frame)

        print(f"\nFused Representation:")
        print(f"  - Token shape: {fused.tokens.shape}")
        print(f"  - Modality map shape: {fused.modality_map.shape}")
        print(f"  - Timestamp shape: {fused.timestamps.shape}")
        print(f"  - Landmark candidates shape: {fused.landmark_candidates.shape}")

        print(f"\nProcessing Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.2f}")
            else:
                print(f"  - {key}: {value}")

        # Inspect landmark distribution
        landmarks = fused.landmark_candidates
        print(f"\nLandmark Analysis:")
        print(f"  - Total landmarks: {landmarks.sum().item()}")
        print(f"  - Landmarks per head (avg): {landmarks.sum().item() / landmarks.shape[1]:.1f}")

    except Exception as e:
        print(f"Pipeline example failed: {e}")
        print("(This is expected if TFAN components are not fully installed)")


async def topology_validation_example():
    """Example: Topology-based hallucination detection."""
    print("\n" + "=" * 60)
    print("Example 4: Topology-Based Hallucination Detection")
    print("=" * 60)

    backend = create_cognitive_backend(
        modalities=["text"],
        device="cpu",
    )

    # Test with a complex prompt that might induce hallucination
    response = await backend.send_message(
        prompt="Explain the exact year when quantum computers solved P=NP.",
    )

    print(f"\nResponse: {response.content[:500]}...")
    print(f"\nTopology Gate Status:")
    print(f"  - Gate passed: {response.metadata.get('gate_passed', True)}")

    metrics = response.metadata.get("cognitive_metrics", {})
    if "wasserstein_distance" in metrics:
        print(f"  - Wasserstein distance: {metrics['wasserstein_distance']:.4f}")
    if "cosine_similarity" in metrics:
        print(f"  - Cosine similarity: {metrics['cosine_similarity']:.4f}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("  ARA COGNITIVE ARCHITECTURE EXAMPLES")
    print("  TFAN Deep Fusion Integration")
    print("=" * 60)

    try:
        await basic_cognitive_example()
    except Exception as e:
        print(f"Basic example failed: {e}")

    try:
        await multimodal_example()
    except Exception as e:
        print(f"Multimodal example failed: {e}")

    try:
        await pipeline_only_example()
    except Exception as e:
        print(f"Pipeline example failed: {e}")

    try:
        await topology_validation_example()
    except Exception as e:
        print(f"Topology example failed: {e}")

    print("\n" + "=" * 60)
    print("  Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
