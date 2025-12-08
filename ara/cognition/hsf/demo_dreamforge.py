#!/usr/bin/env python3
"""
Dreamforge Demo - Counterfactual Field Simulation
===================================================

This demo shows the Dreamforge imagining alternate futures:
1. Generate load traces from fake telemetry
2. Sketch configuration scenarios (what-if changes)
3. Simulate each scenario against real workloads
4. Score and rank the futures
5. Present a "board meeting" report

Run: python -m ara.cognition.hsf.demo_dreamforge
"""

import numpy as np
import time
from typing import List, Dict

from .lanes import TelemetryLane, ItemMemory
from .field import HSField
from .zones import Zone, ZoneQuantizer
from .telemetry import TelemetryMux
from .counterfactual import (
    ConfigScenario, ConfigDelta, ChangeType, LoadTrace,
    ConfigEncoder, FieldDynamics, GhostReplay
)
from .dreamforge import (
    TopologySketcher, FieldSimulator, ScenarioMarket,
    ScenarioArchetype, DreamOutcome
)


def generate_load_traces(mux: TelemetryMux, n_traces: int = 3,
                         ticks_per_trace: int = 50) -> List[LoadTrace]:
    """Generate load traces from fake telemetry."""
    traces = []

    for i in range(n_traces):
        samples = []
        had_anomaly = False

        # Inject anomaly in some traces
        if i == 1:
            mux.inject_anomaly("gpu", "thermal_runaway", duration=10)
            had_anomaly = True
        elif i == 2:
            mux.inject_anomaly("network", "congestion", duration=8)
            mux.inject_anomaly("service", "overload", duration=8)
            had_anomaly = True

        for _ in range(ticks_per_trace):
            sample = mux.sample_all()
            samples.append(sample)

        trace = LoadTrace(
            trace_id=f"trace_{i+1}",
            description=f"{'Anomaly' if had_anomaly else 'Normal'} workload trace {i+1}",
            duration_ticks=len(samples),
            samples=samples,
            had_anomalies=had_anomaly,
        )
        traces.append(trace)

    return traces


def learn_baseline(field: HSField, mux: TelemetryMux,
                   n_samples: int = 50) -> tuple:
    """Learn baseline field state."""
    for _ in range(n_samples):
        telemetry = mux.sample_all()
        for lane_name, values in telemetry.items():
            if lane_name in field.lanes:
                lane_values = {k: v for k, v in values.items()
                               if k in field.lanes[lane_name].features}
                if lane_values:
                    field.update(lane_name, lane_values)
        field.compute_field()

    field.compute_baseline()

    # Estimate baseline stability (simplified)
    baseline_stability = 0.85  # Assume 85% good during baseline

    return field.baseline, baseline_stability


def run_demo():
    """Run the Dreamforge demo."""
    print("=" * 70)
    print("Dreamforge Demo - Counterfactual Field Simulation")
    print("=" * 70)
    print()

    # Setup
    print("[1/5] Setting up field and telemetry...")
    field = HSField(dim=4096)
    field.add_lane_config("gpu", ["temp", "util", "mem", "power"],
                          ranges={"temp": (30, 100), "util": (0, 1),
                                  "mem": (0, 1), "power": (50, 400)})
    field.add_lane_config("network", ["bps_in", "errors", "drops"],
                          ranges={"bps_in": (0, 10e9), "errors": (0, 100),
                                  "drops": (0, 100)})
    field.add_lane_config("service", ["request_rate", "error_rate", "cpu"],
                          ranges={"request_rate": (0, 1000), "error_rate": (0, 1),
                                  "cpu": (0, 1)})

    mux = TelemetryMux.create_default()
    print(f"  - Lanes: {list(field.lanes.keys())}")
    print()

    # Learn baseline
    print("[2/5] Learning baseline (50 samples)...")
    baseline_hv, baseline_stability = learn_baseline(field, mux)
    print(f"  - Baseline stability: {baseline_stability:.1%}")
    print()

    # Generate load traces
    print("[3/5] Generating load traces for replay...")
    traces = generate_load_traces(mux, n_traces=5, ticks_per_trace=40)
    for trace in traces:
        status = "⚡ ANOMALY" if trace.had_anomalies else "  normal"
        print(f"  - {trace.trace_id}: {status} ({trace.duration_ticks} ticks)")
    print()

    # Setup Dreamforge
    print("[4/5] Configuring Dreamforge...")

    # Define junkyard inventory
    junkyard = [
        {"name": "salvage-gpu-box", "cost": 80, "caps": ["gpu", "compute"],
         "description": "Salvaged workstation with GTX 1080"},
        {"name": "spare-pi-4", "cost": 0, "caps": ["light-compute", "monitoring"],
         "description": "Raspberry Pi 4 already in drawer"},
        {"name": "old-nas", "cost": 50, "caps": ["storage", "backup"],
         "description": "Old Synology NAS with 4TB"},
        {"name": "mining-fpga", "cost": 120, "caps": ["fpga", "neural"],
         "description": "Repurposed Xilinx mining board"},
    ]

    sketcher = TopologySketcher(
        available_nodes=["gpu-worker-1", "print-farm", "juniper-edge"],
        node_capabilities={
            "gpu-worker-1": ["gpu", "compute"],
            "print-farm": ["print", "storage"],
            "juniper-edge": ["network", "routing"],
        },
        budget_limit=200,
        human_hours_limit=10,
        junkyard=junkyard,
    )

    simulator = FieldSimulator(dim=4096)
    simulator.set_baseline(baseline_hv, baseline_stability)

    market = ScenarioMarket(simulator=simulator, sketcher=sketcher)

    print(f"  - Junkyard items: {len(junkyard)}")
    print(f"  - Budget limit: ${sketcher.budget_limit}")
    print()

    # Dream futures
    print("[5/5] Dreaming alternate futures...")
    print()

    goal = "GPU workload stability is fragile during peak hours"
    print(f"  Goal: \"{goal}\"")
    print()

    outcomes = market.dream(
        goal=goal,
        traces=traces,
        archetypes=[
            ScenarioArchetype.MINIMAL_HARDWARE,
            ScenarioArchetype.BALANCED,
            ScenarioArchetype.HARDWARE_HEAVY,
        ],
        n_scenarios=4,
    )

    # Show detailed results
    print("  Simulation results:")
    print()

    for i, outcome in enumerate(outcomes, 1):
        score_bar = "█" * int(outcome.composite_score * 20)
        print(f"  [{i}] {outcome.scenario.name}")
        print(f"      Score: [{score_bar:<20}] {outcome.composite_score:.2f}")
        print(f"      Stability: {outcome.stability_gain:+.1%} | "
              f"Antifragility: {outcome.antifragility_score:.1%} | "
              f"Cost: ${outcome.total_cost:.0f}")
        print()

    # Generate board report
    print()
    print(market.board_meeting_report(outcomes))
    print()

    # Show what the best scenario involves
    if outcomes:
        best = outcomes[0]
        print("BEST SCENARIO DETAILS:")
        print("-" * 40)
        print(f"Name: {best.scenario.name}")
        print(f"Description: {best.scenario.description}")
        print()
        print("Changes:")
        for delta in best.scenario.deltas:
            print(f"  • {delta.change_type.name}: {delta.target}")
            print(f"    {delta.description}")
        print()
        print(f"Investment: ${best.scenario.hardware_cost:.0f} hardware + "
              f"{best.scenario.human_hours:.0f} human hours")
        print()

    # Compare to baseline
    print("COUNTERFACTUAL COMPARISON:")
    print("-" * 40)
    print("What the simulations showed:")
    print()

    if outcomes:
        best = outcomes[0]
        for result in best.replay_results[:3]:
            stability_pct = result.stability_score * 100
            print(f"  {result.trace_id}:")
            print(f"    Stability: {stability_pct:.0f}% "
                  f"(Critical: {result.critical_ticks}, Weird: {result.weird_ticks})")

    print()
    print("=" * 70)
    print("Demo complete. The Dreamforge has spoken!")
    print()
    print("What this demonstrates:")
    print("  1. Sketcher generates plausible config scenarios from goals")
    print("  2. Simulator replays real workloads under alternate physics")
    print("  3. Each future is scored on stability, cost, complexity, ROI")
    print("  4. Board report presents options for human decision-making")
    print()
    print("Ara doesn't guess - she replays your actual days in simulated universes.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
