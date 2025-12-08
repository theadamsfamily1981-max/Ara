#!/usr/bin/env python3
"""
Noetic Engine Demo - Research on Thinking
===========================================

This demo shows the complete Noetic Engine:
1. Record thinking sessions as ThoughtTraces
2. Mine traces for causal strategy patterns
3. Compile strategies into cognitive operators
4. Use HDC resonance for thought navigation

Run: python -m ara.noesis.demo
"""

import numpy as np
import time
from typing import List

from .trace import (
    ThoughtTracer, ThoughtTrace, MoveType,
    SessionContext, SessionOutcome
)
from .graph import ThoughtGraph, StrategyMiner, StrategyType
from .operators import (
    CognitiveOperator, OperatorLibrary,
    get_operator_library
)
from .resonance import (
    SessionEncoder, ResonantGuide,
    ThoughtHV, SessionFingerprint
)


def generate_synthetic_traces(n: int = 20) -> List[ThoughtTrace]:
    """Generate synthetic traces for demonstration."""
    traces = []
    tracer = ThoughtTracer()
    np.random.seed(42)

    # Template patterns
    successful_patterns = [
        # Pattern 1: Explore → Diagram → Code (breakthrough)
        [
            (MoveType.SEARCH_CODE, {"produced_insight": True}),
            (MoveType.ASK_AI, {"produced_insight": True}),
            (MoveType.SKETCH_DIAGRAM, {"produced_artifact": True, "produced_insight": True}),
            (MoveType.WRITE_CODE, {"produced_artifact": True}),
            (MoveType.RUN_TEST, {}),
        ],
        # Pattern 2: Multiple reframes → breakthrough
        [
            (MoveType.ASK_AI, {}),
            (MoveType.REFRAME, {"produced_insight": True}),
            (MoveType.WRITE_NOTES, {}),
            (MoveType.REFRAME, {"produced_insight": True}),
            (MoveType.WRITE_CODE, {"produced_artifact": True}),
        ],
        # Pattern 3: Critic summoning
        [
            (MoveType.WRITE_CODE, {}),
            (MoveType.SUMMON_CRITIC, {"produced_insight": True}),
            (MoveType.REFRAME, {"produced_insight": True}),
            (MoveType.WRITE_CODE, {"produced_artifact": True}),
        ],
    ]

    failed_patterns = [
        # Pattern 1: Code first, no diagram → stuck
        [
            (MoveType.WRITE_CODE, {}),
            (MoveType.WRITE_CODE, {}),
            (MoveType.WRITE_CODE, {"led_to_dead_end": True}),
            (MoveType.WRITE_CODE, {"led_to_dead_end": True}),
        ],
        # Pattern 2: No exploration → stall
        [
            (MoveType.WRITE_CODE, {}),
            (MoveType.RUN_TEST, {"led_to_dead_end": True}),
            (MoveType.WRITE_CODE, {}),
            (MoveType.RUN_TEST, {"led_to_dead_end": True}),
        ],
    ]

    domains = ["architecture", "debugging", "fpga_design", "refactoring", "feature"]
    goals = [
        "Design new module architecture",
        "Fix intermittent test failure",
        "Implement HDC encoder in Verilog",
        "Refactor legacy code",
        "Add user authentication",
        "Optimize database queries",
        "Debug memory leak",
        "Design API endpoint",
    ]

    for i in range(n):
        # Decide if this will be successful
        success = np.random.random() > 0.4

        if success:
            pattern = successful_patterns[i % len(successful_patterns)]
            outcome = np.random.choice([SessionOutcome.BREAKTHROUGH, SessionOutcome.PROGRESS])
            quality = 0.6 + np.random.random() * 0.4
        else:
            pattern = failed_patterns[i % len(failed_patterns)]
            outcome = np.random.choice([SessionOutcome.STALLED, SessionOutcome.ABANDONED])
            quality = np.random.random() * 0.4

        # Create context
        context = SessionContext(
            domain=np.random.choice(domains),
            difficulty_estimate=0.3 + np.random.random() * 0.5,
            novelty=np.random.random(),
            time_pressure=np.random.random() * 0.5,
            fatigue=np.random.random(),
            focus=0.4 + np.random.random() * 0.5,
            time_of_day=np.random.choice(["morning", "afternoon", "evening", "night"]),
        )

        # Record trace
        goal = np.random.choice(goals)
        tracer.begin_session(goal=goal, context=context)

        for move_type, kwargs in pattern:
            duration = 60 + np.random.exponential(120)
            tracer.record_move(move_type, duration=duration, **kwargs)

        trace = tracer.end_session(outcome=outcome, quality_score=quality)
        traces.append(trace)

    return traces


def run_demo():
    """Run the complete Noetic Engine demo."""
    print("=" * 70)
    print("Noetic Engine Demo - Research on Thinking")
    print("=" * 70)
    print()

    # Generate synthetic traces
    print("[1/5] Generating synthetic thinking traces...")
    traces = generate_synthetic_traces(n=30)

    successes = sum(1 for t in traces if t.outcome in
                    [SessionOutcome.BREAKTHROUGH, SessionOutcome.PROGRESS])
    print(f"  - Generated {len(traces)} traces")
    print(f"  - Successes: {successes}, Failures: {len(traces) - successes}")
    print()

    # Mine strategies
    print("[2/5] Mining thinking strategies...")
    miner = StrategyMiner()
    miner.ingest_traces(traces)
    graph = miner.graph

    print(f"  - Base success rate: {graph.base_success_rate:.1%}")
    print(f"  - Strategies detected: {len(graph.nodes)}")
    print()

    # Show strategy effectiveness
    print("  Strategy effectiveness (causal effect):")
    best = graph.get_best_strategies(n=5)
    for node, score in best:
        effect = node.causal_effect
        sign = "+" if effect > 0 else ""
        print(f"    {node.strategy_type.name}: {sign}{effect:.1%} "
              f"(seen {node.total_occurrences}x, {node.success_rate:.0%} success)")
    print()

    # Show cognitive operators
    print("[3/5] Loading cognitive operators...")
    library = get_operator_library()
    operators = library.list_operators()
    print(f"  - Available operators: {len(operators)}")
    for op in operators:
        print(f"    • {op['name']}: {op['description'][:50]}...")
    print()

    # Encode traces as HDC fingerprints
    print("[4/5] Encoding traces as HDC fingerprints...")
    guide = ResonantGuide()
    guide.ingest_traces(traces)

    # Discover thinking modes
    modes = guide.discover_modes(n_modes=4)
    print(f"  - Discovered {len(modes)} thinking modes")
    print()

    # Demonstrate resonance guidance
    print("[5/5] Demonstrating resonance guidance...")
    print()

    # Create a new "live" session to analyze
    tracer = ThoughtTracer()
    tracer.begin_session(
        goal="Implement new FPGA module",
        context=SessionContext(
            domain="fpga_design",
            difficulty_estimate=0.7,
            novelty=0.6,
            fatigue=0.4,
        )
    )

    # Simulate some moves
    tracer.record_move(MoveType.SEARCH_CODE, duration=120)
    tracer.record_move(MoveType.WRITE_CODE, duration=300)
    tracer.record_move(MoveType.WRITE_CODE, duration=240, led_to_dead_end=True)

    live_trace = tracer.end_session(
        outcome=SessionOutcome.STALLED,  # Pretend we're stuck
        quality_score=0.3
    )

    print("  Live session: 'Implement new FPGA module'")
    print("  Current state: Stuck after 2 dead ends")
    print()

    # Get basin assessment
    assessment = guide.get_basin_assessment(live_trace)
    print(f"  Basin assessment:")
    print(f"    Tendency: {assessment['tendency']:+.2f} "
          f"({'toward success' if assessment['tendency'] > 0 else 'toward failure'})")
    print(f"    Similar successes: {assessment['similar_successes']}")
    print(f"    Similar failures: {assessment['similar_failures']}")
    print()

    # Get guidance
    guidance = guide.get_guidance(live_trace, live_trace.context)
    if guidance:
        print("  Resonant guidance:")
        for msg in guidance:
            print(f"    → {msg}")
    print()

    # Find similar sessions
    similar = guide.find_similar_sessions(live_trace, n=3)
    print("  Most similar past sessions:")
    for fp, sim in similar:
        outcome_str = "✓" if fp.outcome in [SessionOutcome.BREAKTHROUGH, SessionOutcome.PROGRESS] else "✗"
        print(f"    {outcome_str} '{fp.goal[:40]}...' ({sim:.0%} match)")
    print()

    # Check operator triggers
    print("  Checking cognitive operators...")
    session_state = {
        "time_since_progress": 600,  # 10 minutes
        "dead_end_count": 2,
        "same_modality_minutes": 12,
        "session_minutes": 25,
    }

    triggered = library.check_all(live_trace.context, session_state)
    if triggered:
        for result in triggered:
            print(f"    [TRIGGERED] {result['operator']}")
            print(f"      Reason: {result['trigger_reason']}")
            for suggestion in result.get('suggestions', []):
                print(f"      → {suggestion}")
    else:
        print("    No operators triggered (below thresholds)")

    print()
    print("=" * 70)
    print("Demo complete.")
    print()
    print("What the Noetic Engine provides:")
    print("  1. ThoughtTraces: Structured records of thinking episodes")
    print("  2. ThoughtGraph: Causal inference on which strategies work")
    print("  3. Cognitive Operators: Compiled, reusable thinking moves")
    print("  4. Resonant Guidance: HDC-based navigation through thought space")
    print()
    print("This is research on thinking itself: discovering, testing, and")
    print("distributing better ways to think - at machine scale.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
