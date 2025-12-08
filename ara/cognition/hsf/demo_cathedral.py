#!/usr/bin/env python3
"""
Phase-Gated Cathedral Demo
===========================

This demo shows:
1. Phase-coded HPV multiplexing (SENSE/SYMBOL/META on same wire)
2. Separation and recovery of interleaved channels
3. Field Computer operation with three planes and three jobs
4. How the same physical lane becomes three logical fabrics

Run: python -m ara.cognition.hsf.demo_cathedral
"""

import numpy as np
import time
from typing import List

from .phase import (
    Phase, PhaseConfig, MacroFrame, ChannelStream,
    PhaseMultiplexer, PhaseCounter, LaneTopology,
    create_default_phase_config,
)
from .cathedral import (
    Cathedral, CathedralEncoder, NeuronState,
    PhaseGatedCluster, create_cathedral,
)
from .field_computer import (
    FieldComputer, TelemetryEvent, FrictionEvent, IdeaCandidate,
    PlaneA_Reflex, PlaneB_Context, PlaneC_Policy,
    Job_LANSentinel, Job_FrictionMiner, Job_IdeaRouter,
    create_field_computer,
)


def demo_phase_multiplexing():
    """Demonstrate phase-coded HPV multiplexing."""
    print("=" * 70)
    print("Phase-Coded HPV Multiplexing")
    print("=" * 70)
    print()

    config = create_default_phase_config()
    mux = PhaseMultiplexer(config=config)
    dim = 64  # Small dim for visibility

    print(f"Configuration:")
    print(f"  Phases: {config.num_phases}")
    print(f"  SENSE noise: {config.sense_noise}")
    print(f"  SYMBOL noise: {config.symbol_noise}")
    print(f"  META noise: {config.meta_noise}")
    print()

    # Create distinct hypervectors for each channel
    np.random.seed(42)
    sense_hv = np.random.randint(0, 2, size=dim, dtype=np.uint8)
    symbol_hv = np.random.randint(0, 2, size=dim, dtype=np.uint8)
    meta_hv = np.random.randint(0, 2, size=dim, dtype=np.uint8)

    print(f"Original HPVs (first 16 bits):")
    print(f"  SENSE:  {''.join(str(b) for b in sense_hv[:16])}")
    print(f"  SYMBOL: {''.join(str(b) for b in symbol_hv[:16])}")
    print(f"  META:   {''.join(str(b) for b in meta_hv[:16])}")
    print()

    # Multiplex into frames
    frames = mux.multiplex_hypervectors(sense_hv, symbol_hv, meta_hv)
    print(f"Multiplexed into {len(frames)} macro-frames")
    print()

    # Show first few frames
    print("First 8 macro-frames (wire view):")
    print("  Frame | φ=0(S) | φ=1(Y) | φ=2(M) | φ=3(-)")
    print("  ------+--------+--------+--------+-------")
    for i, frame in enumerate(frames[:8]):
        bits = frame.to_wire_bits()
        print(f"  {i:5} |   {bits[0]}    |   {bits[1]}    |   {bits[2]}    |   {bits[3]}")
    print()

    # Demultiplex back
    recovered = mux.demultiplex(frames)

    sense_recovered = recovered[Phase.SENSE].to_hypervector(dim)
    symbol_recovered = recovered[Phase.SYMBOL].to_hypervector(dim)
    meta_recovered = recovered[Phase.META].to_hypervector(dim)

    print("Recovered HPVs (first 16 bits):")
    print(f"  SENSE:  {''.join(str(b) for b in sense_recovered[:16])}")
    print(f"  SYMBOL: {''.join(str(b) for b in symbol_recovered[:16])}")
    print(f"  META:   {''.join(str(b) for b in meta_recovered[:16])}")
    print()

    # Compute recovery accuracy
    def accuracy(orig, recovered):
        return np.sum(orig == recovered) / len(orig) * 100

    print("Recovery accuracy:")
    print(f"  SENSE:  {accuracy(sense_hv, sense_recovered):.1f}%")
    print(f"  SYMBOL: {accuracy(symbol_hv, symbol_recovered):.1f}%")
    print(f"  META:   {accuracy(meta_hv, meta_recovered):.1f}%")
    print()


def demo_cathedral_streaming():
    """Demonstrate streaming HPVs through the Cathedral."""
    print("=" * 70)
    print("Cathedral HPV Streaming")
    print("=" * 70)
    print()

    cathedral = create_cathedral(dim=256)
    encoder = CathedralEncoder(dim=256)

    print("Cathedral clusters:")
    for name, cluster in cathedral.clusters.items():
        print(f"  {name}: bias={cluster.bias.name}, neurons={len(cluster.neurons)}")
    print()

    # Create test HPVs representing a scenario:
    # GPU thermal spike from worker-1
    sense_hv = encoder.encode_sensory_event(
        event_type="thermal",
        source="gpu-worker-1",
        severity=0.85,
    )

    # Context: project=ara, machine=gpu-worker-1, role=worker
    symbol_hv = encoder.encode_symbolic_context(
        project="ara",
        machine="gpu-worker-1",
        role="worker",
        risk_level="high",
    )

    # Policy: scientist mode, moderate auth
    meta_hv = encoder.encode_meta_control(
        mode="scientist",
        auth_level=2,
        safety_override=False,
    )

    print("Encoded scenario:")
    print("  SENSE: GPU thermal spike (severity=0.85)")
    print("  SYMBOL: project=ara, machine=gpu-worker-1, risk=high")
    print("  META: mode=scientist, auth=2")
    print()

    # Stream through cathedral
    result = cathedral.stream_hypervectors(sense_hv, symbol_hv, meta_hv)

    print(f"Streaming results:")
    print(f"  Frames processed: {result['frames_processed']}")
    print(f"  Total events: {len(result['events'])}")
    print(f"  Spikes: {result['spike_count']}")
    print()

    # Query symbol state from clusters
    symbol_states = cathedral.query_symbol_state()
    print("Cluster symbol states (popcount):")
    for name, hv in symbol_states.items():
        popcount = np.sum(hv)
        print(f"  {name}: {popcount} ones")
    print()

    # Get spike rates
    spike_rates = cathedral.get_spike_rates()
    print("Cluster spike rates:")
    for name, rate in spike_rates.items():
        print(f"  {name}: {rate:.2f} spikes/neuron")
    print()


def demo_field_computer():
    """Demonstrate the Field Computer with its three jobs."""
    print("=" * 70)
    print("Field Computer - Three Planes, Three Jobs")
    print("=" * 70)
    print()

    fc = create_field_computer(dim=256)

    print("Field Computer initialized:")
    print(f"  Mode: {fc.plane_c.mode}")
    print(f"  Safety level: {fc.plane_c.safety_level}")
    print()

    # === Job 1: LAN Sentinel ===
    print("-" * 40)
    print("Job 1: LAN Sentinel")
    print("-" * 40)
    print()

    # Simulate telemetry events
    events = [
        TelemetryEvent(source="gpu-worker-1", event_type="thermal", value=0.5),
        TelemetryEvent(source="gpu-worker-1", event_type="thermal", value=0.7),
        TelemetryEvent(source="gpu-worker-1", event_type="thermal", value=0.92),
        TelemetryEvent(source="nas", event_type="disk_io", value=0.6),
        TelemetryEvent(source="juniper", event_type="packet_drop", value=0.3),
    ]

    print("Processing telemetry events:")
    for event in events:
        result = fc.process_telemetry(event)
        status = ""
        if result.get("reflex_decision"):
            reflex = result["reflex_decision"]
            status = f" -> REFLEX: {reflex.action} (conf={reflex.confidence:.2f})"
            if reflex.escalate:
                status += " [ESCALATE]"
        print(f"  {event.source}:{event.event_type}={event.value:.2f}{status}")
    print()

    # === Job 2: Friction Miner ===
    print("-" * 40)
    print("Job 2: Friction Miner")
    print("-" * 40)
    print()

    friction_events = [
        FrictionEvent(source="editor", event_type="build_fail", project="ara",
                      file_path="src/hsf/phase.py", frustration_level=0.6),
        FrictionEvent(source="terminal", event_type="repeated_error", project="ara",
                      file_path="src/hsf/phase.py", frustration_level=0.7),
        FrictionEvent(source="editor", event_type="stuck_typing", project="ara",
                      file_path="src/hsf/cathedral.py", frustration_level=0.4),
    ]

    print("Processing friction events:")
    for event in friction_events:
        result = fc.process_friction(event)
        print(f"  {event.project}/{event.file_path}: {event.event_type}")
        if result.get("pattern_match"):
            print(f"    Pattern match: {result['pattern_match']}")
        if result.get("recommendations"):
            for rec in result["recommendations"]:
                print(f"    -> {rec}")
    print()

    hotspots = fc.job_friction.get_friction_hotspots()
    print("Friction hotspots:")
    for project, count in hotspots[:3]:
        print(f"  {project}: {count} events")
    print()

    # === Job 3: Idea Router ===
    print("-" * 40)
    print("Job 3: Idea Router")
    print("-" * 40)
    print()

    ideas = [
        IdeaCandidate(
            idea_id="1",
            description="Add test harness for phase module",
            compute_cost=0.2,
            human_time=2.0,
            risk_level=0.1,
            alignment=0.9,
            expected_impact=0.7,
        ),
        IdeaCandidate(
            idea_id="2",
            description="Rewrite entire codebase in Rust",
            compute_cost=0.9,
            human_time=100.0,
            risk_level=0.8,
            alignment=0.3,
            expected_impact=0.6,
        ),
        IdeaCandidate(
            idea_id="3",
            description="Add logging to Cathedral neurons",
            compute_cost=0.1,
            human_time=0.5,
            risk_level=0.1,
            alignment=0.7,
            expected_impact=0.4,
        ),
    ]

    print("Evaluating ideas:")
    for idea in ideas:
        result = fc.evaluate_idea(idea)
        status = "ESCALATE" if result["escalate"] else "FILTER"
        print(f"  [{status}] {idea.description[:40]}...")
        print(f"    Score: {result['combined_score']:.2f} ({result['reason']})")
    print()

    # === Policy Check ===
    print("-" * 40)
    print("Policy Checks")
    print("-" * 40)
    print()

    fc.set_mode("steward")
    print(f"Mode: {fc.plane_c.mode}")

    actions = [
        ("monitor_cpu", "gpu-worker-1"),
        ("restart_service", "gpu-worker-1"),
        ("deploy_code", "production-server"),
    ]

    for action, node in actions:
        allowed, reason = fc.check_action(action, node)
        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"  {action} on {node}: {status}")
        if not allowed:
            print(f"    Reason: {reason}")
    print()

    # === Status Summary ===
    print("-" * 40)
    print("Field Computer Status")
    print("-" * 40)
    print()

    status = fc.get_status()
    print(f"  Mode: {status['mode']}")
    print(f"  Safety level: {status['safety_level']}")
    print(f"  Events processed: {status['events_processed']}")
    print(f"  Reflexes fired: {status['reflexes_fired']}")
    print(f"  Ideas escalated: {status['ideas_escalated']}")
    print()


def demo_combined_scenario():
    """Demonstrate a complete scenario using all components."""
    print("=" * 70)
    print("Combined Scenario: GPU Thermal Cascade")
    print("=" * 70)
    print()

    fc = create_field_computer(dim=256)
    fc.set_mode("scientist")  # Active mode

    print("Scenario: Multiple GPUs overheating during training run")
    print()

    # Simulate thermal cascade
    events = [
        TelemetryEvent(source="gpu-worker-1", event_type="thermal", value=0.85,
                       metadata={"job": "ara-training", "user": "croft"}),
        TelemetryEvent(source="gpu-worker-2", event_type="thermal", value=0.82,
                       metadata={"job": "ara-training", "user": "croft"}),
        TelemetryEvent(source="gpu-worker-1", event_type="thermal", value=0.93,
                       metadata={"job": "ara-training", "user": "croft"}),
        TelemetryEvent(source="gpu-worker-3", event_type="thermal", value=0.88,
                       metadata={"job": "ara-training", "user": "croft"}),
    ]

    print("Event stream:")
    for event in events:
        result = fc.process_telemetry(event)
        print(f"  t={event.timestamp:.0f}: {event.source} thermal={event.value:.2f}")

        if result.get("reflex_decision"):
            reflex = result["reflex_decision"]
            print(f"    REFLEX: {reflex.action}")
            if reflex.escalate:
                print(f"    -> ESCALATE to higher reasoning")

        if result.get("pattern_matches"):
            for pattern in result["pattern_matches"]:
                print(f"    PATTERN: {pattern}")
    print()

    # Get world state
    world_state = fc.get_world_state()
    popcount = np.sum(world_state)
    print(f"World state HPV: {popcount} ones (of {len(world_state)})")

    # Now generate an idea based on the situation
    idea = IdeaCandidate(
        idea_id="thermal_response",
        description="Throttle training jobs on overheating GPUs",
        compute_cost=0.1,
        human_time=0.0,  # Automated
        risk_level=0.2,
        alignment=0.95,  # Highly aligned with system health
        expected_impact=0.8,
    )

    print()
    print("Proposed response idea:")
    result = fc.evaluate_idea(idea)
    print(f"  {idea.description}")
    print(f"  Score: {result['combined_score']:.2f}")
    print(f"  Decision: {'ESCALATE' if result['escalate'] else 'FILTER'}")
    print(f"  Reason: {result['reason']}")
    print()


def run_demo():
    """Run all demos."""
    print()
    print("=" * 70)
    print("PHASE-GATED CATHEDRAL DEMO")
    print("Multi-Channel Bit-Serial Brain")
    print("=" * 70)
    print()

    demo_phase_multiplexing()
    demo_cathedral_streaming()
    demo_field_computer()
    demo_combined_scenario()

    print("=" * 70)
    print("Demo complete.")
    print()
    print("The Phase-Gated Cathedral provides:")
    print("  1. Phase coding: 3 logical channels on 1 physical wire")
    print("  2. SENSE/SYMBOL/META separation in hardware timing")
    print("  3. Field Computer: always-on brain for Ara")
    print("  4. Three planes: Reflex/Context/Policy")
    print("  5. Three jobs: LANSentinel/FrictionMiner/IdeaRouter")
    print()
    print("Same wires, same clock, but the fabric now thinks in layers:")
    print("  Phase 0: What's happening? (SNN reflexes)")
    print("  Phase 1: What does it mean? (HDC symbols)")
    print("  Phase 2: What should we do? (Policy gates)")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
