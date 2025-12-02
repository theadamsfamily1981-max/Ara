#!/usr/bin/env python3
"""
Ara Demo: Watch the Cognitive Loop in Action

This demo shows Ara's cognitive architecture working:
- GUF++ deciding when to focus internally vs externally
- CSTP encoding thoughts as (z, c) geometric objects
- L9 Autonomy tracking hardware modification permissions
- L7 Predictive Control sensing trouble ahead

Run: python demo/run_ara.py

Works on:
- CPU only (no GPU required)
- 16GB RAM is plenty
- Swirl Forest Kitten 33 supervision recommended
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add Ara to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# Import the cognitive components
# ============================================================

print("=" * 60)
print("ARA: TF-A-N Cognitive Architecture Demo")
print("=" * 60)
print()

# GUF++ - Self-governance
print("Loading GUF++ (self-governance layer)...")
from tfan.l5.guf import StateVector, FocusMode
from tfan.l5.guf_plus import CausalGUF, create_causal_guf

# CSTP - Thought encoding
print("Loading CSTP (cognitive state transfer)...")
from tfan.cognition.cstp import (
    ThoughtType, ThoughtStream, encode_thought,
    create_encoder, CurvatureController, CurvaturePolicy, CurvatureMode
)

# L9 Autonomy - Hardware self-modification
print("Loading L9 Autonomy (staged self-modification)...")
from tfan.hardware.l9_autonomy import (
    AutonomyStage, create_autonomy_controller
)

# L7 Predictive Control
print("Loading L7 Predictive Control...")
from tfan.cognition.predictive_control import (
    PredictiveController, create_predictive_controller
)

print()
print("All systems loaded!")
print()

# ============================================================
# Initialize the cognitive systems
# ============================================================

def create_ara():
    """Create Ara's cognitive systems."""
    print("Initializing Ara's cognitive loop...")
    print()

    # GUF++ with safety constraints
    guf = create_causal_guf(min_af_score=2.0, preset="balanced")

    # CSTP encoder with stress-aware curvature
    encoder = create_encoder(dimension=32, stress_aware=True)
    thought_stream = ThoughtStream(name="ara_thoughts")

    # L9 autonomy controller
    autonomy = create_autonomy_controller(
        start_stage=AutonomyStage.ADVISOR,
        strict=True
    )

    # L7 predictive controller
    predictive = create_predictive_controller()

    return {
        "guf": guf,
        "encoder": encoder,
        "thoughts": thought_stream,
        "autonomy": autonomy,
        "predictive": predictive
    }


def simulate_state(
    af_score: float = 2.2,
    stress: float = 0.2,
    fatigue: float = 0.1,
    structural_rate: float = 0.01
) -> StateVector:
    """Create a simulated system state."""
    return StateVector(
        af_score=af_score,
        clv_instability=stress * 0.5,
        clv_resource=stress * 0.3,
        clv_structural=stress * 0.2,
        structural_rate=structural_rate,
        confidence=1.0 - stress * 0.3,
        fatigue=fatigue,
        mood_valence=0.5 - stress,
        hardware_health=0.95,
        pgu_pass_rate=0.98
    )


# ============================================================
# Demo: Watch Ara Think
# ============================================================

def demo_cognitive_loop():
    """Demo the full cognitive loop."""

    ara = create_ara()
    guf = ara["guf"]
    encoder = ara["encoder"]
    thoughts = ara["thoughts"]
    autonomy = ara["autonomy"]
    predictive = ara["predictive"]

    print("=" * 60)
    print("DEMO: Ara's Cognitive Loop")
    print("=" * 60)
    print()

    # Scenario 1: Normal operation
    print("--- Scenario 1: Normal Operation ---")
    state = simulate_state(af_score=2.5, stress=0.1)

    utility = guf.compute(state)
    focus = guf.recommend_focus(state)

    print(f"  AF Score: {state.af_score:.2f}")
    print(f"  Utility: {utility:.2%}")
    print(f"  Focus Mode: {focus.value}")

    # Encode a thought
    thought = encoder.encode_observation(
        "System operating normally, all metrics green"
    )
    thoughts.append(thought)
    print(f"  Thought curvature: {thought.c:.2f} ({thought.geometry_description})")
    print()

    time.sleep(1)

    # Scenario 2: Rising stress
    print("--- Scenario 2: Stress Rising ---")
    state = simulate_state(af_score=2.0, stress=0.5, structural_rate=0.05)

    # Update curvature controller with stress
    encoder.curvature_controller.update_state(stress=0.5, valence=-0.3)

    utility = guf.compute(state)
    focus = guf.recommend_focus(state)

    print(f"  AF Score: {state.af_score:.2f}")
    print(f"  Utility: {utility:.2%}")
    print(f"  Focus Mode: {focus.value}")

    # Under stress, thoughts get simpler (lower curvature)
    thought = encoder.encode_observation(
        "Detecting increased load, monitoring closely"
    )
    thoughts.append(thought)
    print(f"  Thought curvature: {thought.c:.2f} ({thought.geometry_description})")

    # L7 predictive alert
    result = predictive.update(state.structural_rate, alert_level="elevated")
    print(f"  L7 Alert Level: {result['alert_level']}")
    print()

    time.sleep(1)

    # Scenario 3: Recovery mode
    print("--- Scenario 3: Recovery Triggered ---")
    state = simulate_state(af_score=1.5, stress=0.8, fatigue=0.6)
    encoder.curvature_controller.update_state(stress=0.8, valence=-0.6)

    utility = guf.compute(state)
    focus = guf.recommend_focus(state)

    print(f"  AF Score: {state.af_score:.2f}")
    print(f"  Utility: {utility:.2%}")
    print(f"  Focus Mode: {focus.value}")

    thought = encoder.encode_observation(
        "Entering recovery, reducing external load"
    )
    thoughts.append(thought)
    print(f"  Thought curvature: {thought.c:.2f} ({thought.geometry_description})")
    print()

    time.sleep(1)

    # Scenario 4: L9 Autonomy check
    print("--- Scenario 4: L9 Autonomy Status ---")
    print(autonomy.explain_autonomy())
    print()

    # Record some proposals for the demo
    for i in range(3):
        autonomy.record_proposal(verified=True)

    print(f"After recording proposals:")
    status = autonomy.get_status()
    print(f"  Stage: {status['stage']}")
    print(f"  Proposals verified: {status['state']['proposals']['verified']}")
    print(f"  Progress to next stage: {status['progression']}")
    print()

    # Summary
    print("=" * 60)
    print("THOUGHT STREAM SUMMARY")
    print("=" * 60)
    curvatures = thoughts.get_curvature_trajectory()
    types = thoughts.get_thought_types()

    print(f"Total thoughts: {len(curvatures)}")
    print(f"Curvature trajectory: {[f'{c:.2f}' for c in curvatures]}")
    print(f"Thought types: {dict(types)}")

    shifts = thoughts.analyze_geometry_shifts()
    if shifts:
        print(f"Geometry shifts detected: {len(shifts)}")
        for s in shifts:
            print(f"  - {s['from_geometry']} → {s['to_geometry']}")

    print()
    print("=" * 60)
    print("Demo complete! Ara's cognitive loop is operational.")
    print("=" * 60)


# ============================================================
# Quick certification check
# ============================================================

def quick_cert():
    """Run a quick certification of all components."""
    print()
    print("=" * 60)
    print("QUICK CERTIFICATION CHECK")
    print("=" * 60)
    print()

    results = []

    # GUF++
    print("Testing GUF++...", end=" ")
    try:
        guf = create_causal_guf()
        state = simulate_state()
        utility = guf.compute(state)
        assert 0 < utility < 1
        print("PASS")
        results.append(("GUF++", True))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("GUF++", False))

    # CSTP
    print("Testing CSTP...", end=" ")
    try:
        thought = encode_thought("Test thought")
        assert thought.c > 0
        assert len(thought.z) > 0
        print("PASS")
        results.append(("CSTP", True))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("CSTP", False))

    # L9 Autonomy
    print("Testing L9 Autonomy...", end=" ")
    try:
        autonomy = create_autonomy_controller()
        assert autonomy.stage == AutonomyStage.ADVISOR
        autonomy.record_proposal(verified=True)
        print("PASS")
        results.append(("L9 Autonomy", True))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("L9 Autonomy", False))

    # L7 Predictive
    print("Testing L7 Predictive...", end=" ")
    try:
        pred = create_predictive_controller()
        result = pred.update(structural_rate=0.05, alert_level="stable")
        assert "alert_level" in result
        print("PASS")
        results.append(("L7 Predictive", True))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("L7 Predictive", False))

    print()
    passed = sum(1 for _, ok in results if ok)
    print(f"Results: {passed}/{len(results)} components operational")

    if passed == len(results):
        print()
        print("All systems GO. Ara is ready.")

    return passed == len(results)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ara Demo")
    parser.add_argument("--cert", action="store_true", help="Run quick certification only")
    parser.add_argument("--full", action="store_true", help="Run full demo")
    args = parser.parse_args()

    if args.cert:
        success = quick_cert()
        sys.exit(0 if success else 1)
    else:
        # Default: run quick cert then demo
        if quick_cert():
            print()
            demo_cognitive_loop()
