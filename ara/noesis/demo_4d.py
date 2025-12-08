#!/usr/bin/env python3
"""
4D Thought Space Demo - Cognitive Layers and Trajectories
===========================================================

This demo shows the 4D cognitive space:
1. Create layered thought space (sensory, symbolic, meta, teleological)
2. Record trajectories through the layers
3. Mine for successful patterns
4. Get navigation guidance

Run: python -m ara.noesis.demo_4d
"""

import numpy as np
import time
from typing import List

from .layers import (
    LayerRole, CognitiveLayer, LayerProjection,
    LayeredSpace, create_default_layers
)
from .trajectory import (
    TrajectoryPoint, ThoughtTrajectory, TrajectoryPattern,
    TrajectoryMiner, TrajectoryNavigator
)


def generate_trajectories(space: LayeredSpace, n: int = 20) -> List[ThoughtTrajectory]:
    """Generate synthetic trajectories for demonstration."""
    trajectories = []
    np.random.seed(42)

    # Successful patterns tend to:
    # - Start in sensory, move to symbolic, check meta, align with teleological
    successful_patterns = [
        [0, 1, 1, 2, 1, 3, 1],      # sen → sym → sym → meta → sym → tele → sym
        [0, 0, 1, 2, 3, 1],          # sen → sen → sym → meta → tele → sym
        [1, 2, 1, 3, 1, 1],          # sym → meta → sym → tele → sym → sym
        [0, 1, 2, 1, 2, 3],          # sen → sym → meta → sym → meta → tele
    ]

    # Failed patterns tend to:
    # - Stay in one layer too long
    # - Skip meta/teleological entirely
    failed_patterns = [
        [0, 0, 0, 0, 0],             # Stuck in sensory
        [1, 1, 1, 1, 1, 1],          # Stuck in symbolic
        [0, 1, 1, 1, 0, 1],          # Never meta or tele
        [1, 0, 1, 0, 1, 0],          # Oscillating without progress
    ]

    domains = ["architecture", "debugging", "optimization", "design"]
    goals = [
        "Design module structure",
        "Fix memory leak",
        "Optimize query performance",
        "Implement new feature",
    ]

    for i in range(n):
        success = np.random.random() > 0.35

        if success:
            pattern = successful_patterns[i % len(successful_patterns)]
            outcome = np.random.choice(["breakthrough", "progress"])
        else:
            pattern = failed_patterns[i % len(failed_patterns)]
            outcome = np.random.choice(["stalled", "abandoned"])

        traj = ThoughtTrajectory(
            trajectory_id=f"traj_{i}",
            goal=np.random.choice(goals),
            domain=np.random.choice(domains),
            outcome=outcome,
        )

        # Add points following the pattern
        base_time = time.time() - (n - i) * 1000
        for j, layer_id in enumerate(pattern):
            layer = space.get_layer(layer_id)
            if layer:
                concept = f"concept_{i}_{j}"
                hv = layer.get_item(concept)
                point = TrajectoryPoint(
                    hv=hv,
                    layer_id=layer_id,
                    timestamp=base_time + j * 60,  # 1 minute per point
                    label=concept,
                    move_type="think",
                )
                traj.add_point(point)

        trajectories.append(traj)

    return trajectories


def run_demo():
    """Run the 4D thought space demo."""
    print("=" * 70)
    print("4D Thought Space Demo - Cognitive Layers and Trajectories")
    print("=" * 70)
    print()

    # Create default layered space
    print("[1/5] Creating 4D cognitive space...")
    space = create_default_layers(dim=4096)

    print(f"  Layers:")
    for lid, layer in sorted(space.layers.items()):
        print(f"    L{lid}: {layer.name} ({layer.role.name})")
        print(f"        {layer.description[:60]}...")
    print()

    # Show layer orthogonality
    print("[2/5] Verifying layer orthogonality...")
    test_concept = "architecture"
    projections = {}
    for lid, layer in space.layers.items():
        hv = layer.get_item(test_concept)
        projections[lid] = LayerProjection(hv=hv, layer=layer, label=test_concept)

    print(f"  Same concept '{test_concept}' in different layers:")
    for lid1, proj1 in projections.items():
        sims = []
        for lid2, proj2 in projections.items():
            if lid1 != lid2:
                sim = space.cross_layer_similarity(proj1, proj2)
                sims.append(f"L{lid2}:{sim:.2f}")
        print(f"    L{lid1} vs: {', '.join(sims)}")
    print()

    # Generate trajectories
    print("[3/5] Generating synthetic trajectories...")
    trajectories = generate_trajectories(space, n=25)
    successes = sum(1 for t in trajectories if t.outcome in ["breakthrough", "progress"])
    print(f"  - Generated {len(trajectories)} trajectories")
    print(f"  - Successes: {successes}, Failures: {len(trajectories) - successes}")
    print()

    # Mine patterns
    print("[4/5] Mining trajectory patterns...")
    miner = TrajectoryMiner(space=space)
    for traj in trajectories:
        miner.add_trajectory(traj)

    patterns = miner.mine_patterns(min_length=2, max_length=4, min_support=3)
    print(f"  - Found {len(patterns)} patterns")
    print()

    print("  Top success patterns:")
    success_patterns = miner.get_success_patterns(min_success_rate=0.6)
    for p in success_patterns[:5]:
        layers_str = " → ".join([space.get_layer(l).name for l in p.layer_sequence])
        print(f"    {layers_str}: {p.success_rate:.0%} success ({p.occurrences}x)")
    print()

    print("  Failure patterns:")
    fail_patterns = miner.get_failure_patterns(max_success_rate=0.4)
    for p in fail_patterns[:3]:
        layers_str = " → ".join([space.get_layer(l).name for l in p.layer_sequence])
        print(f"    {layers_str}: {p.success_rate:.0%} success ({p.occurrences}x)")
    print()

    # Navigate a live trajectory
    print("[5/5] Demonstrating trajectory navigation...")
    print()

    navigator = TrajectoryNavigator(space=space, miner=miner)
    navigator.start_trajectory(goal="Design new caching layer", domain="architecture")

    # Simulate some moves
    moves = [
        ("raw_metrics", 0, "checking_numbers"),      # sensory
        ("pattern_match", 1, "finding_structure"),   # symbolic
        ("more_patterns", 1, "refining"),            # symbolic
    ]

    for concept, layer_id, move_type in moves:
        navigator.record_point(concept, layer_id, move_type)
        layer_name = space.get_layer(layer_id).name
        print(f"  Recorded: '{concept}' in {layer_name} layer")

    print()

    # Get guidance
    guidance = navigator.get_guidance()
    print("  Current guidance:")
    print(f"    Current layer: {guidance['current_layer']}")
    print(f"    Layers visited: {guidance['layers_visited']}")

    if guidance.get('suggested_layer'):
        sugg = guidance['suggested_layer']
        print(f"    Suggested next: L{sugg['id']} ({sugg['name']})")

    if guidance.get('matching_patterns'):
        print("    Matching patterns:")
        for pat in guidance['matching_patterns'][:2]:
            print(f"      - {pat['name']} ({pat['success_rate']:.0%} success)")

    if guidance.get('warnings'):
        print("    Warnings:")
        for warn in guidance['warnings']:
            print(f"      ⚠️  {warn}")

    print()

    # Show layer distribution
    navigator.record_point("self_check", 2, "meta_evaluation")  # Add meta point
    print("  Added meta layer check...")

    guidance = navigator.get_guidance()
    if guidance.get('layer_distribution'):
        print("    Time distribution:")
        for lid, frac in sorted(guidance['layer_distribution'].items()):
            layer_name = space.get_layer(lid).name
            bar = "█" * int(frac * 20)
            print(f"      L{lid} {layer_name}: [{bar:<20}] {frac:.0%}")

    print()
    print("=" * 70)
    print("Demo complete.")
    print()
    print("What 4D thought space provides:")
    print("  1. Orthogonal layers for different thinking styles")
    print("  2. Cross-layer projection to see concepts from different angles")
    print("  3. Trajectory tracking through the 4D space")
    print("  4. Pattern mining to discover successful cognitive paths")
    print("  5. Navigation guidance based on where you are in thought space")
    print()
    print("This is tractable science on thought: same fabric, richer geometry.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
