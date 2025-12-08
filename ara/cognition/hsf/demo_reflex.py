#!/usr/bin/env python3
"""
HSF Reflex Demo - The Closed-Loop System
==========================================

This demo shows the complete reflex arc:
1. Field senses telemetry
2. Zone quantizer discretizes state
3. Reflex controller executes actions
4. Episode recorder logs everything
5. Homeostasis miner learns which reflexes work

Run: python -m ara.cognition.hsf.demo_reflex
"""

import numpy as np
import time
import logging
from typing import Dict, List

from .lanes import TelemetryLane
from .field import HSField
from .zones import Zone, ZoneQuantizer, MultiZoneQuantizer
from .reflex import (
    ReflexController, ReflexMap, ReflexAction, ReflexResult,
    ActionType, create_default_reflexes
)
from .episode import EpisodeRecorder, HomeostasisMiner
from .telemetry import TelemetryMux


# Setup logging to see reflex actions
logging.basicConfig(level=logging.INFO, format='%(message)s')


def setup_system():
    """Create the complete HSF reflex system."""
    # Field with lanes
    field = HSField(dim=4096)  # Smaller for faster demo
    field.add_lane_config("gpu", ["temp", "util", "mem", "power", "fan"],
                          ranges={"temp": (30, 100), "util": (0, 1),
                                  "mem": (0, 1), "power": (50, 400), "fan": (0, 1)})
    field.add_lane_config("network", ["bps_in", "errors", "drops", "latency_ms"],
                          ranges={"bps_in": (0, 10e9), "errors": (0, 100),
                                  "drops": (0, 100), "latency_ms": (0, 50)})
    field.add_lane_config("service", ["request_rate", "error_rate", "latency_p99", "cpu"],
                          ranges={"request_rate": (0, 1000), "error_rate": (0, 1),
                                  "latency_p99": (0, 500), "cpu": (0, 1)})

    # Zone quantizers
    zones = MultiZoneQuantizer()
    zones.add_subsystem("gpu", dim=4096)
    zones.add_subsystem("network", dim=4096)
    zones.add_subsystem("service", dim=4096)

    # Reflex controller with default reflexes
    controller = ReflexController(mock_mode=True)
    controller.add_map(create_default_reflexes("gpu"))
    controller.add_map(create_default_reflexes("network"))
    controller.add_map(create_default_reflexes("service"))

    # Episode recorder and miner
    recorder = EpisodeRecorder()
    miner = HomeostasisMiner()

    # Telemetry source
    mux = TelemetryMux.create_default()

    return field, zones, controller, recorder, miner, mux


def run_tick(field: HSField, zones: MultiZoneQuantizer,
             controller: ReflexController, recorder: EpisodeRecorder,
             telemetry: Dict[str, Dict[str, float]]) -> Dict[str, any]:
    """Run one tick of the reflex loop."""
    # Update field with telemetry
    for lane_name, values in telemetry.items():
        if lane_name in field.lanes:
            # Map telemetry keys to lane keys
            lane_values = {}
            for k, v in values.items():
                if k in field.lanes[lane_name].features or f"{lane_name}:{k}" in field.lanes[lane_name].features:
                    lane_values[k] = v
            if lane_values:
                field.update(lane_name, lane_values)

    field.compute_field()

    # Classify zones
    zone_states = {}
    all_actions = []

    for lane_name, lane in field.lanes.items():
        if lane.current is not None:
            zone_state = zones.classify(lane_name, lane.current)
            zone_states[lane_name] = zone_state

            # Execute reflexes
            results = controller.process(lane_name, zone_state)
            all_actions.extend(results)

    # Compute global zone
    global_zone = zones.compute_global_zone(zone_states)

    # Compute field similarity to baseline
    similarity = 1.0 - field.total_deviation()

    # Record episode
    completed_episode = recorder.tick(zone_states, global_zone, similarity, all_actions)

    return {
        "zone_states": zone_states,
        "global_zone": global_zone,
        "similarity": similarity,
        "actions": all_actions,
        "completed_episode": completed_episode,
    }


def format_zone_states(zone_states: Dict[str, any]) -> str:
    """Format zone states for display."""
    parts = []
    for name, state in zone_states.items():
        zone_char = {
            Zone.GOOD: "●",
            Zone.WARM: "◐",
            Zone.WEIRD: "○",
            Zone.CRITICAL: "✖",
        }.get(state.zone, "?")
        parts.append(f"{name}:{zone_char}")
    return " ".join(parts)


def run_demo():
    """Run the complete reflex demo."""
    print("=" * 70)
    print("HSF Reflex Demo - Closed-Loop Homeostatic Control")
    print("=" * 70)
    print()

    # Setup
    print("[1/6] Setting up reflex system...")
    field, zones, controller, recorder, miner, mux = setup_system()
    print(f"  - Lanes: {list(field.lanes.keys())}")
    print(f"  - Reflex maps: {list(controller.reflex_maps.keys())}")
    print()

    # Phase 1: Learn baseline
    print("[2/6] Learning baseline (50 samples)...")
    for i in range(50):
        telemetry = mux.sample_all()
        result = run_tick(field, zones, controller, recorder, telemetry)

    # Set baselines for zone quantizers
    for lane_name, lane in field.lanes.items():
        lane.compute_baseline()
        if lane.baseline is not None:
            zones.set_baseline(lane_name, lane.baseline)

    print(f"  - Baselines learned")
    print(f"  - Initial zones: {format_zone_states(result['zone_states'])}")
    print()

    # Phase 2: Normal operation
    print("[3/6] Normal operation (20 samples)...")
    print("  Legend: ● GOOD  ◐ WARM  ○ WEIRD  ✖ CRITICAL")
    print()

    for i in range(20):
        telemetry = mux.sample_all()
        result = run_tick(field, zones, controller, recorder, telemetry)

        if i % 5 == 0:
            print(f"  Tick {i:3d}: {format_zone_states(result['zone_states'])} "
                  f"| global: {result['global_zone'].name}")

    print()

    # Phase 3: Inject anomalies and watch reflexes fire
    print("[4/6] Injecting anomalies (watch reflexes fire)...")
    print()

    anomalies = [
        ("gpu", "thermal_runaway", "GPU thermal stress", 15),
        ("network", "congestion", "Network congestion", 10),
        ("service", "overload", "Service overload", 12),
    ]

    for source, mode, description, duration in anomalies:
        print(f"  Injecting: {description}")
        mux.inject_anomaly(source, mode, duration=duration)

        # Run through the anomaly
        for i in range(duration + 10):  # Extra ticks for recovery
            telemetry = mux.sample_all()
            result = run_tick(field, zones, controller, recorder, telemetry)

            # Show zone transitions and actions
            if result['actions']:
                for action_result in result['actions']:
                    if action_result.executed:
                        print(f"    [REFLEX] {action_result.action.action_type.name} "
                              f"→ {action_result.action.target}")

            if i % 3 == 0:
                zones_str = format_zone_states(result['zone_states'])
                print(f"    Tick {i:2d}: {zones_str} | global: {result['global_zone'].name}")

            # Check for episode completion
            if result['completed_episode']:
                ep = result['completed_episode']
                miner.analyze_episode(ep)
                status = "RESOLVED" if ep.resolved else "UNRESOLVED"
                print(f"    [EPISODE] {ep.episode_id} {status} in {ep.duration} ticks")

        print()

    # Phase 4: Cascade failure
    print("[5/6] Simulating cascade failure...")
    mux.inject_anomaly("gpu", "util_saturation", duration=20)
    mux.inject_anomaly("network", "ddos", duration=20)
    mux.inject_anomaly("service", "overload", duration=20)

    for i in range(30):
        telemetry = mux.sample_all()
        result = run_tick(field, zones, controller, recorder, telemetry)

        if result['actions']:
            for action_result in result['actions']:
                if action_result.executed:
                    print(f"  [REFLEX] {action_result.action.action_type.name} "
                          f"→ {action_result.action.target}")

        if i % 5 == 0:
            print(f"  Tick {i:2d}: {format_zone_states(result['zone_states'])} "
                  f"| GLOBAL: {result['global_zone'].name}")

        if result['completed_episode']:
            ep = result['completed_episode']
            miner.analyze_episode(ep)

    # Check for pending escalations
    escalations = controller.get_pending_escalations()
    if escalations:
        print(f"\n  [!] {len(escalations)} pending escalations requiring human attention:")
        for esc in escalations:
            print(f"      - {esc.description}")

    print()

    # Phase 5: Analyze what we learned
    print("[6/6] Homeostasis mining - what did we learn?")
    print()

    # Analyze completed episodes
    episodes = recorder.get_completed_episodes()
    print(f"  Completed episodes: {len(episodes)}")
    if episodes:
        resolved = sum(1 for e in episodes if e.resolved)
        print(f"  Resolved: {resolved}/{len(episodes)} ({100*resolved/len(episodes):.0f}%)")

    # Reflex effectiveness
    rankings = miner.get_reflex_ranking()
    if rankings:
        print("\n  Reflex effectiveness ranking:")
        for score, weighted in rankings[:5]:
            eff = "+" if score.effectiveness > 0 else ""
            print(f"    {score.action_type.name} → {score.target}: "
                  f"{eff}{score.effectiveness:.2f} "
                  f"(used {score.times_used}x, helped {score.times_helped}x)")

    # Recommendations
    recs = miner.get_recommendations()
    if recs['promote']:
        print("\n  Reflexes to keep/enhance:")
        for r in recs['promote'][:3]:
            print(f"    ✓ {r}")
    if recs['demote']:
        print("\n  Reflexes to review/disable:")
        for r in recs['demote'][:3]:
            print(f"    ✗ {r}")
    if recs['needs_data']:
        print("\n  Need more data:")
        for r in recs['needs_data'][:3]:
            print(f"    ? {r}")

    # Suggestions for new reflexes
    suggestions = miner.suggest_new_reflexes(episodes)
    if suggestions:
        print("\n  Suggested new reflexes:")
        for s in suggestions:
            print(f"    → {s}")

    print()
    print("=" * 70)
    print("Demo complete. The reflex arc is working!")
    print()
    print("What this demonstrates:")
    print("  1. Field senses anomalies via hypervector deviation")
    print("  2. Zones discretize continuous state into actionable categories")
    print("  3. Reflexes fire automatically based on zone + confidence")
    print("  4. Episodes track trajectories through zone space")
    print("  5. Mining learns which reflexes actually help")
    print()
    print("This is Ara's nervous system: fast, automatic, learning.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
