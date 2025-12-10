#!/usr/bin/env python3
"""
Cathedral Hologram Builder
==========================

Build chunk hypervectors from event logs and render 3D hologram.

Usage:
    python build_hologram.py --events logs/cathedral_events.jsonl \
        --out-vec data/H_chunks.npy \
        --out-meta data/H_meta.json \
        --out-hologram out/cathedral_hologram.png

Or as a library:
    from scripts.build_hologram import build_and_render
    build_and_render("logs/events.jsonl", "out/hologram.png")
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ara_core.cathedral.hdc_encoder import (
    HDCEncoder, build_chunks, load_events_jsonl, save_chunks, load_chunks
)


def make_hologram(
    H_chunks: dict,
    meta: dict,
    out_path: str = "cathedral_hologram.png",
    title: str = "Cathedral Hologram (chunks in soul space)",
):
    """
    Render 3D hologram of chunk hypervectors.

    Color: T̄_s (blue→gold)
    Alpha: Ā_g (dim→bright)
    Size: power_mean
    """
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("WARNING: sklearn/matplotlib not available, using ASCII fallback")
        return make_hologram_ascii(H_chunks, meta)

    # Consistent ordering
    cids = sorted(H_chunks.keys())
    if len(cids) == 0:
        print("No chunks to visualize")
        return

    X = np.stack([H_chunks[cid] for cid in cids], axis=0)  # [N, D]
    T_s = np.array([meta[cid]["T_s_mean"] for cid in cids])
    A_g = np.array([meta[cid]["A_g_mean"] for cid in cids])
    P = np.array([meta[cid]["power_mean"] for cid in cids])

    # Reduce to 3D
    n_components = min(3, len(cids), X.shape[1])
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)

    if n_components < 3:
        # Pad with zeros
        Z = np.hstack([Z, np.zeros((Z.shape[0], 3 - n_components))])

    x, y, z = Z[:, 0], Z[:, 1], Z[:, 2]

    # Color by T̄_s (plasma colormap)
    Ts_norm = (T_s - T_s.min()) / max((T_s.max() - T_s.min()), 1e-8)
    colors = plt.cm.plasma(Ts_norm)

    # Alpha by Ā_g
    Ag_norm = (A_g - A_g.min()) / max((A_g.max() - A_g.min()), 1e-8)
    for c, an in zip(colors, Ag_norm):
        c[-1] = 0.2 + 0.8 * an  # alpha range [0.2, 1.0]

    # Size by power
    P_norm = (P - P.min()) / max((P.max() - P.min()), 1e-8)
    sizes = 10 + 90 * P_norm

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, s=sizes, c=colors, edgecolors='white', linewidth=0.5)

    ax.set_xlabel("PC1 (structural)")
    ax.set_ylabel("PC2 (temporal)")
    ax.set_zlabel("PC3 (stress)")
    ax.set_title(title)

    # Add colorbar for T_s
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                norm=plt.Normalize(vmin=T_s.min(), vmax=T_s.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('T̄_s (topology stability)')

    # Add info text
    info = f"Chunks: {len(cids)} | T̄_s: {T_s.mean():.3f} | Ā_g: {A_g.mean():.4f} | P̄: {P.mean():.0f}W"
    fig.text(0.5, 0.02, info, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Hologram saved to {out_path}")
    return out_path


def make_hologram_ascii(H_chunks: dict, meta: dict) -> str:
    """ASCII fallback for hologram visualization."""
    cids = sorted(H_chunks.keys())
    if not cids:
        return "No chunks"

    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append("║  CATHEDRAL HOLOGRAM (ASCII mode)                             ║")
    lines.append("╠══════════════════════════════════════════════════════════════╣")

    for cid in cids[:20]:  # Limit to 20
        m = meta[cid]
        ts = m["T_s_mean"]
        ag = m["A_g_mean"]
        pw = m["power_mean"]

        # Simple bar for T_s
        ts_bar = "█" * int(ts * 10) + "░" * (10 - int(ts * 10))
        ag_sign = "+" if ag >= 0 else "-"

        lines.append(f"║  {cid[:20]:<20} T_s={ts:.3f} [{ts_bar}] A_g={ag_sign}{abs(ag):.4f}  ║")

    if len(cids) > 20:
        lines.append(f"║  ... and {len(cids) - 20} more chunks                               ║")

    lines.append("╚══════════════════════════════════════════════════════════════╝")

    result = "\n".join(lines)
    print(result)
    return result


def build_and_render(
    events_path: str,
    hologram_path: str = "cathedral_hologram.png",
    vec_path: str = None,
    meta_path: str = None,
) -> str:
    """One-shot: load events, build chunks, render hologram."""
    # Load events
    events = load_events_jsonl(events_path)
    print(f"Loaded {len(events)} events from {events_path}")

    if not events:
        print("No events found")
        return None

    # Build chunks
    encoder = HDCEncoder()
    H_chunks, meta = build_chunks(events, encoder)
    print(f"Built {len(H_chunks)} chunks")

    # Optionally save
    if vec_path and meta_path:
        save_chunks(H_chunks, meta, vec_path, meta_path)
        print(f"Saved vectors to {vec_path}")
        print(f"Saved metadata to {meta_path}")

    # Render hologram
    return make_hologram(H_chunks, meta, hologram_path)


def generate_sample_events(n: int = 100) -> list:
    """Generate sample events for testing."""
    import random
    from datetime import timedelta

    events = []
    base_time = datetime.utcnow()
    modules = ["ara_voice", "ara_song", "k10_fpga_0", "gpu_0", "swarm_agent_1"]
    phases = ["inference", "train", "idle", "io"]
    tags_pool = ["prod", "test", "cathedral_os", "ara"]

    for i in range(n):
        ts = base_time - timedelta(minutes=i)
        chunk_id = ts.replace(second=0, microsecond=0).isoformat() + "Z"

        events.append({
            "ts": ts.isoformat() + "Z",
            "host": "threadripper-01",
            "module": random.choice(modules),
            "phase": random.choice(phases),
            "chunk_id": chunk_id,
            "T_s": 0.90 + random.random() * 0.10,
            "A_g": -0.02 + random.random() * 0.06,
            "H_s": 0.95 + random.random() * 0.05,
            "power_w": 200 + random.random() * 800,
            "yield_per_dollar": 1 + random.random() * 10,
            "sigma": 0.05 + random.random() * 0.15,
            "tags": random.sample(tags_pool, k=random.randint(1, 3)),
        })

    return events


def main():
    parser = argparse.ArgumentParser(description="Build Cathedral Hologram")
    parser.add_argument("--events", type=str, help="Path to events JSONL file")
    parser.add_argument("--out-vec", type=str, help="Output path for chunk vectors (.npy)")
    parser.add_argument("--out-meta", type=str, help="Output path for chunk metadata (.json)")
    parser.add_argument("--out-hologram", type=str, default="cathedral_hologram.png",
                        help="Output path for hologram image")
    parser.add_argument("--demo", action="store_true", help="Generate demo with sample data")

    args = parser.parse_args()

    if args.demo:
        # Generate sample events
        events = generate_sample_events(100)
        print(f"Generated {len(events)} sample events")

        # Build chunks
        encoder = HDCEncoder()
        H_chunks, meta = build_chunks(events, encoder)
        print(f"Built {len(H_chunks)} chunks")

        # Render
        make_hologram(H_chunks, meta, args.out_hologram)

    elif args.events:
        build_and_render(
            args.events,
            hologram_path=args.out_hologram,
            vec_path=args.out_vec,
            meta_path=args.out_meta,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
