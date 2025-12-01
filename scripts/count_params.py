"""
Parameter count validation for TF-A-N 7B.

Validates that parameter count is within 6.8-7.2B range.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.models.tfan7b import TFANConfig, TFANForCausalLM, count_parameters
import json


def main():
    print("="*60)
    print("TF-A-N 7B Parameter Count Validation")
    print("="*60)

    # Load config
    config = TFANConfig.from_json_file("tfan/models/tfan7b/config.json")

    print(f"\nModel configuration (PROFILE-A):")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  KV heads (GQA): {config.num_kv_heads}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  FFN multiplier: {config.ffn_mult}")
    print(f"  Vocabulary: {config.vocab_size}")
    print(f"  Max position embeddings: {config.max_position_embeddings}")

    # Create model
    print(f"\nCreating model...")
    model = TFANForCausalLM(config)

    # Count parameters
    counts = count_parameters(model)

    print(f"\nParameter counts:")
    print(f"  Total: {counts['total']:,}")
    print(f"  Trainable: {counts['trainable']:,}")
    print(f"  Non-trainable: {counts['non_trainable']:,}")
    print(f"\n  Total (millions): {counts['total_millions']:.2f}M")
    print(f"  Total (billions): {counts['total_billions']:.3f}B")

    # Validate against target range
    target_min = 6.8e9
    target_max = 7.2e9

    print(f"\n{'='*60}")
    print("Gate Validation: Parameter count in [6.8B, 7.2B]")
    print(f"{'='*60}")

    within_range = target_min <= counts['total'] <= target_max

    if within_range:
        print(f"✓ PASS: {counts['total_billions']:.3f}B is within target range")
    else:
        print(f"✗ FAIL: {counts['total_billions']:.3f}B is outside target range")
        if counts['total'] < target_min:
            print(f"  Model is {(target_min - counts['total'])/1e9:.3f}B too small")
        else:
            print(f"  Model is {(counts['total'] - target_max)/1e9:.3f}B too large")

    # Save results
    import os
    os.makedirs("artifacts/reports", exist_ok=True)

    results = {
        "config": {
            "num_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "num_heads": config.num_attention_heads,
            "num_kv_heads": config.num_kv_heads,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
        },
        "parameter_counts": counts,
        "target_range": {
            "min_billions": target_min / 1e9,
            "max_billions": target_max / 1e9,
        },
        "gate_pass": within_range,
    }

    with open("artifacts/reports/param_count.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to artifacts/reports/param_count.json")

    return 0 if within_range else 1


if __name__ == "__main__":
    exit(main())
