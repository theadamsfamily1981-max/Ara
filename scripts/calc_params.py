"""
Calculate TF-A-N 7B parameter count mathematically (no PyTorch required).
"""

import json


def calculate_params(config):
    """Calculate parameter count from config."""

    L = config["num_hidden_layers"]
    d = config["hidden_size"]
    H = config["num_attention_heads"]
    H_kv = config["num_kv_heads"]
    ffn_dim = config["intermediate_size"]
    V = config["vocab_size"]

    head_dim = d // H

    # Embeddings (tied with output)
    emb_params = V * d

    # Per-layer parameters
    # Attention: Q, K, V projections + output projection
    attn_qkv = d * d  # Q projection
    attn_qkv += d * (H_kv * head_dim)  # K projection (GQA)
    attn_qkv += d * (H_kv * head_dim)  # V projection (GQA)
    attn_out = d * d  # Output projection
    attn_params_per_layer = attn_qkv + attn_out

    # SwiGLU MLP: gate + value + output
    mlp_gate = d * ffn_dim
    mlp_value = d * ffn_dim
    mlp_out = ffn_dim * d
    mlp_params_per_layer = mlp_gate + mlp_value + mlp_out

    # RMSNorm: 2 per layer (pre-attn, pre-mlp)
    norm_params_per_layer = d * 2

    # Total per layer
    params_per_layer = attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer

    # Total model
    layer_params = L * params_per_layer
    final_norm = d
    total_params = emb_params + layer_params + final_norm

    return {
        "embeddings": emb_params,
        "per_layer": params_per_layer,
        "total_layers": layer_params,
        "final_norm": final_norm,
        "total": total_params,
        "total_millions": total_params / 1e6,
        "total_billions": total_params / 1e9,
    }


def main():
    print("="*60)
    print("TF-A-N 7B Parameter Count Calculation (PROFILE-A)")
    print("="*60)

    # Load config
    with open("tfan/models/tfan7b/config.json") as f:
        config = json.load(f)

    print(f"\nConfiguration:")
    print(f"  Layers (L): {config['num_hidden_layers']}")
    print(f"  Hidden size (d): {config['hidden_size']}")
    print(f"  Attention heads (H): {config['num_attention_heads']}")
    print(f"  KV heads (H_kv): {config['num_kv_heads']}")
    print(f"  Intermediate size: {config['intermediate_size']}")
    print(f"  Vocabulary (V): {config['vocab_size']}")

    # Calculate
    counts = calculate_params(config)

    print(f"\nParameter breakdown:")
    print(f"  Embeddings (tied): {counts['embeddings']:,}")
    print(f"  Per layer: {counts['per_layer']:,}")
    print(f"  Total layers: {counts['total_layers']:,}")
    print(f"  Final norm: {counts['final_norm']:,}")
    print(f"\n  TOTAL: {counts['total']:,}")
    print(f"  {counts['total_billions']:.3f}B")

    # Gate validation
    target_min = 6.8
    target_max = 7.2

    print(f"\n{'='*60}")
    print(f"Gate Validation: {target_min}B ≤ params ≤ {target_max}B")
    print(f"{'='*60}")

    within_range = target_min <= counts['total_billions'] <= target_max

    if within_range:
        print(f"✓ PASS: {counts['total_billions']:.3f}B is within target range")
    else:
        print(f"✗ FAIL: {counts['total_billions']:.3f}B is outside target range")

    # Save
    import os
    os.makedirs("artifacts/reports", exist_ok=True)

    results = {
        "config": config,
        "parameter_counts": counts,
        "gate_pass": within_range,
    }

    with open("artifacts/reports/param_count.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to artifacts/reports/param_count.json")

    return 0 if within_range else 1


if __name__ == "__main__":
    exit(main())
