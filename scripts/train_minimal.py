#!/usr/bin/env python
"""
Minimal TF-A-N training example.

Demonstrates end-to-end training loop with all components:
- Multi-modal fusion
- Sparse attention
- Topology regularization
- FDT homeostat
- Emotion control
- PGU verification

Usage:
    python scripts/train_minimal.py --config configs/default.yaml
"""

import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn

from tfan import TFANConfig, TFANTrainer
from tfan.attention import SparseAttention
from tfan.topo import TopologyRegularizer
from tfan.emotion import EmotionHead
from tfan.mm import MultiModalIngestor, align_streams, pack_and_mask


class SimpleTransformer(nn.Module):
    """Simple transformer for demonstration."""

    def __init__(self, config: TFANConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(30522, config.d_model)  # BERT vocab size
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers with sparse attention
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": SparseAttention(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    keep_ratio=config.attention.keep_ratio,
                    alpha=config.attention.alpha_tls,
                    window_size=config.attention.window_size,
                ),
                "ff": nn.Sequential(
                    nn.LayerNorm(config.d_model),
                    nn.Linear(config.d_model, config.d_ff),
                    nn.GELU(),
                    nn.Linear(config.d_ff, config.d_model),
                    nn.Dropout(config.dropout),
                ),
                "norm1": nn.LayerNorm(config.d_model),
                "norm2": nn.LayerNorm(config.d_model),
            })
            for _ in range(config.n_layers)
        ])

        self.output_head = nn.Linear(config.d_model, 30522)

        self.temperature = config.base_temperature

    def forward(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape

        # Embeddings
        x = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # Transformer layers
        for layer in self.layers:
            # Attention
            attn_out, _ = layer["attn"](layer["norm1"](x), attn_mask=attention_mask)
            x = x + attn_out

            # Feed-forward
            x = x + layer["ff"](layer["norm2"](x))

        # Output
        logits = self.output_head(x)

        # Loss (simplified)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

        return {
            "loss": loss,
            "logits": logits,
            "latents": x,  # For emotion and topology
        }

    def set_temperature(self, temp):
        """Set temperature for softmax."""
        self.temperature = temp


def create_dummy_batch(batch_size=4, seq_len=512):
    """Create dummy batch for demonstration."""
    return {
        "input_ids": torch.randint(0, 30522, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal TF-A-N training example")
    parser.add_argument("--config", type=str, default="config_examples/default.yaml",
                        help="Config file path")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/minimal",
                        help="Checkpoint directory")
    args = parser.parse_args()

    print("=" * 80)
    print("TF-A-N Minimal Training Example")
    print("=" * 80)

    # Load config
    print(f"\nLoading config from: {args.config}")
    config = TFANConfig.from_yaml(args.config)

    # Override device
    config.device = args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        config.device = "cpu"

    # Validate config
    violations = config.validate_gates()
    if violations:
        print("⚠️  Config violations:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("✓ Config validated")

    # Create model
    print("\nCreating model...")
    model = SimpleTransformer(config).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Create trainer
    print("\nCreating trainer...")
    trainer = TFANTrainer(model, config)

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    # Training loop
    for step in range(args.steps):
        # Create batch
        batch = create_dummy_batch(
            batch_size=config.batch_size,
            seq_len=min(512, config.max_seq_len),  # Start small
        )

        # Move to device
        batch = {k: v.to(config.device) for k, v in batch.items()}

        # Training step
        metrics = trainer.training_step(batch)

        # Log every 100 steps
        if (step + 1) % 100 == 0:
            print(f"\nStep {step + 1}/{args.steps}:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  EPR-CV: {metrics['epr_cv']:.4f} {'✓' if metrics['epr_cv'] <= 0.15 else '✗'}")
            print(f"  LR: {metrics['lr']:.2e}")
            print(f"  Temperature: {metrics['temperature']:.3f}")
            print(f"  PGU: {'✓ proven' if metrics['pgu_proven'] else '✗ rejected'} "
                  f"({metrics['pgu_latency_ms']:.1f}ms)")

        # Validate gates every 500 steps
        if (step + 1) % 500 == 0:
            print(f"\n{'=' * 40}")
            print("Gate Validation")
            print("=" * 40)

            passes, results = trainer.validate_gates()

            print(f"\nEPR-CV: {results['epr_cv']['value']:.4f} / {results['epr_cv']['threshold']:.2f} "
                  f"{'✓' if results['epr_cv']['passes'] else '✗'}")

            print(f"PGU p95: {results['pgu']['p95_latency_ms']:.1f}ms / "
                  f"{config.pgu.p95_latency_max_ms:.0f}ms "
                  f"{'✓' if results['pgu']['passes'] else '✗'}")
            print(f"PGU cache hit: {results['pgu']['cache_hit_rate']:.1%} "
                  f"{'✓' if results['pgu']['cache_hit_rate'] >= 0.5 else '✗'}")

            print(f"\nOverall: {'✓ ALL GATES PASS' if passes else '✗ SOME GATES FAILED'}")

            # Save checkpoint
            ckpt_path = f"{args.checkpoint_dir}/step_{step+1}.pt"
            trainer.save_checkpoint(ckpt_path)
            print(f"\nCheckpoint saved: {ckpt_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)

    final_passes, final_results = trainer.validate_gates()

    print("\nFinal Gate Status:")
    print(f"  EPR-CV: {final_results['epr_cv']['value']:.4f} {'✓' if final_results['epr_cv']['passes'] else '✗'}")
    print(f"  PGU: {'✓' if final_results['pgu']['passes'] else '✗'}")

    print(f"\nFinal Metrics:")
    recent_metrics = trainer.metrics_history[-100:]
    print(f"  Avg Loss (last 100): {sum(m['loss'] for m in recent_metrics) / len(recent_metrics):.4f}")
    print(f"  Avg EPR-CV (last 100): {sum(m['epr_cv'] for m in recent_metrics) / len(recent_metrics):.4f}")

    print(f"\n{'=' * 80}")
    print(f"Status: {'✓ SUCCESS' if final_passes else '✗ FAILED (gates violated)'}")
    print("=" * 80)

    return 0 if final_passes else 1


if __name__ == "__main__":
    exit(main())
