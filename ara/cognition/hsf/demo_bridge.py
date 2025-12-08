#!/usr/bin/env python3
"""
Card ↔ LLM Bridge Demo
======================

Demonstrates the symbiosis between:
- Card (subcortex): Always-on SNN/HDC, reflexes, state compression
- GPU LLM (cortex): Deep reasoning, planning, language

Shows:
1. State compression (HPV → text summary for LLM)
2. Invocation decisions (when to call the LLM)
3. Policy encoding (LLM output → HPV/SNN weights)
4. Closed-loop operation

Run: python -m ara.cognition.hsf.demo_bridge
"""

import numpy as np
import time
import hashlib
from typing import Dict, List

from .bridge import (
    # Message types
    CardToLLMType, LLMToCardType,
    StateSummary, AnomalyReport, ContextQuery, PolicyRequest,
    NewPolicy, ThresholdUpdate, PatternDefine, ReflexInstall, ContextBind,
    # Core components
    StateCompressor, PolicyEncoder, SymbiosisLoop,
    InvocationDecision,
    create_symbiosis_loop,
)


def make_random_hv(name: str, dim: int = 256) -> np.ndarray:
    """Create a deterministic HV from name."""
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=dim, dtype=np.uint8)


def demo_state_compression():
    """Demonstrate compressing Card state to text for LLM."""
    print("=" * 70)
    print("State Compression: HPV → Text Summary")
    print("=" * 70)
    print()

    compressor = StateCompressor(dim=256)

    # Create sample state HPVs
    world_state = make_random_hv("world:normal:active", 256)
    machine_states = {
        "gpu-worker-1": make_random_hv("status:healthy", 256),
        "gpu-worker-2": make_random_hv("status:degraded", 256),
        "nas-primary": make_random_hv("status:healthy", 256),
    }
    recent_events = [
        "gpu-worker-1: training job completed",
        "nas: backup started",
        "gpu-worker-2: thermal warning at 78C",
    ]
    metrics = {
        "health": 0.85,
        "friction": 0.3,
        "risk": 0.2,
        "projects": ["ara", "ara-card"],
    }

    print("Input state:")
    print(f"  World state HPV: {np.sum(world_state)} ones / {len(world_state)}")
    print(f"  Machines: {list(machine_states.keys())}")
    print(f"  Recent events: {len(recent_events)}")
    print(f"  Metrics: health={metrics['health']:.0%}, friction={metrics['friction']:.0%}")
    print()

    # Compress to summary
    summary = compressor.compress_state(
        world_state_hv=world_state,
        machine_states=machine_states,
        recent_events=recent_events,
        metrics=metrics,
    )

    print("Compressed summary:")
    print("-" * 40)
    print(summary.to_prompt_context())
    print("-" * 40)
    print()

    # Show size reduction
    raw_size = len(world_state) + sum(len(hv) for hv in machine_states.values())
    summary_size = len(summary.to_prompt_context())
    print(f"Size comparison:")
    print(f"  Raw HPV bytes: {raw_size}")
    print(f"  Summary chars: {summary_size}")
    print(f"  Compression: {raw_size / summary_size:.1f}x")
    print()


def demo_invocation_decisions():
    """Demonstrate when to invoke the LLM."""
    print("=" * 70)
    print("Invocation Decisions: When to Call the LLM")
    print("=" * 70)
    print()

    loop = create_symbiosis_loop(
        anomaly_threshold=0.7,
        novelty_threshold=0.6,
        max_invocations_per_hour=20,
    )

    scenarios = [
        # (name, metrics, expected_decision)
        ("Normal operation", {"health": 0.9, "friction": 0.2, "risk": 0.1}, False),
        ("High risk", {"health": 0.6, "friction": 0.3, "risk": 0.85}, True),
        ("High friction", {"health": 0.8, "friction": 0.9, "risk": 0.2}, True),
        ("Moderate concern", {"health": 0.7, "friction": 0.5, "risk": 0.5}, False),
    ]

    print("Scenarios:")
    print("-" * 60)

    for name, metrics, expected in scenarios:
        state_hv = make_random_hv(f"state:{name}", 256)
        recent_events = [f"Event for {name}"]

        decision = loop.should_invoke_llm(
            state_hv=state_hv,
            metrics=metrics,
            recent_events=recent_events,
        )

        status = "INVOKE" if decision.should_invoke else "skip"
        match = "✓" if decision.should_invoke == expected else "✗"

        print(f"  {name}")
        print(f"    Metrics: health={metrics['health']:.0%}, friction={metrics['friction']:.0%}, risk={metrics['risk']:.0%}")
        print(f"    Decision: {status} ({decision.reason})")
        if decision.should_invoke:
            print(f"    Priority: {decision.priority:.0%}, Type: {decision.message_type.name}")
        print()

        # Record invocation if decided
        if decision.should_invoke:
            loop.record_invocation()
            # Wait a bit to avoid rate limiting in demo
            loop._last_invocation -= 10


def demo_policy_encoding():
    """Demonstrate encoding LLM output to Card structures."""
    print("=" * 70)
    print("Policy Encoding: LLM → HPV/SNN Weights")
    print("=" * 70)
    print()

    encoder = PolicyEncoder(dim=256)

    # === Pattern Definition ===
    print("1. Pattern Definition")
    print("-" * 40)

    pattern = PatternDefine(
        message_id="msg_001",
        timestamp=time.time(),
        pattern_id="thermal_cascade",
        pattern_name="Thermal Cascade",
        description="Multiple GPUs overheating within short window",
        components=["gpu", "thermal", "cascade", "multiple"],
        temporal_order=False,
        severity=0.8,
        suggested_action="throttle_all",
    )

    pattern_hv = encoder.encode_pattern(pattern)
    print(f"Pattern: {pattern.pattern_name}")
    print(f"Components: {pattern.components}")
    print(f"Encoded HV: {np.sum(pattern_hv)} ones / {len(pattern_hv)}")
    print()

    # === Policy Definition ===
    print("2. Policy Definition")
    print("-" * 40)

    policy = NewPolicy(
        message_id="msg_002",
        timestamp=time.time(),
        policy_id="thermal_safety_v1",
        policy_type="safety",
        description="Thermal safety policy for GPU cluster",
        conditions=[
            {"trigger": "thermal_warning", "threshold": 0.75},
            {"trigger": "thermal_critical", "threshold": 0.90},
        ],
        actions=[
            {"action": "throttle", "params": {"percent": 50}},
            {"action": "shutdown", "params": {"graceful": True}},
        ],
        key_concepts=["gpu", "thermal", "safety", "throttle"],
        threshold_hints={"warning_temp": 75.0, "critical_temp": 90.0},
    )

    encoded = encoder.encode_policy(policy)
    print(f"Policy: {policy.description}")
    print(f"Pattern HV: {np.sum(encoded['pattern_hv'])} ones")
    print(f"Thresholds: {encoded['thresholds']}")
    print(f"Reflex rules: {len(encoded['reflex_rules'])}")
    for rule in encoded["reflex_rules"]:
        print(f"  - {rule['trigger']} (>{rule['threshold']}) → {rule['action']}")
    print()

    # === Context Binding ===
    print("3. Context Binding")
    print("-" * 40)

    binding = ContextBind(
        message_id="msg_003",
        timestamp=time.time(),
        concept_a="high_load",
        concept_b="thermal_risk",
        relation="causes",
        strength=0.9,
    )

    binding_hv = encoder.encode_binding(binding)
    print(f"Binding: {binding.concept_a} --[{binding.relation}]--> {binding.concept_b}")
    print(f"Encoded HV: {np.sum(binding_hv)} ones / {len(binding_hv)}")
    print()


def demo_closed_loop():
    """Demonstrate the full closed loop."""
    print("=" * 70)
    print("Closed Loop: Card ↔ LLM Symbiosis")
    print("=" * 70)
    print()

    loop = create_symbiosis_loop()

    print("Initial state:")
    print(f"  Installed patterns: {len(loop._installed_patterns)}")
    print(f"  Installed thresholds: {len(loop._installed_thresholds)}")
    print(f"  Installed reflexes: {len(loop._installed_reflexes)}")
    print()

    # === Simulate: LLM sends new policy ===
    print("Step 1: LLM sends new thermal safety policy")
    print("-" * 40)

    policy = NewPolicy(
        message_id="llm_001",
        timestamp=time.time(),
        policy_id="thermal_safety_v1",
        policy_type="safety",
        description="Thermal safety for GPU cluster",
        conditions=[
            {"trigger": "thermal_warning", "threshold": 0.75},
        ],
        actions=[
            {"action": "throttle", "params": {"percent": 50}},
        ],
        key_concepts=["gpu", "thermal", "safety"],
        threshold_hints={"warning": 75.0, "critical": 90.0},
    )

    loop.install_policy(policy)
    print(f"Installed policy: {policy.policy_id}")
    print()

    # === Simulate: LLM sends pattern definition ===
    print("Step 2: LLM sends pattern to watch for")
    print("-" * 40)

    pattern = PatternDefine(
        message_id="llm_002",
        timestamp=time.time(),
        pattern_id="memory_pressure",
        pattern_name="Memory Pressure",
        description="System running low on memory",
        components=["memory", "low", "swap", "pressure"],
        severity=0.6,
        suggested_action="alert",
    )

    loop.install_pattern(pattern)
    print(f"Installed pattern: {pattern.pattern_id}")
    print()

    # === Simulate: LLM sends threshold update ===
    print("Step 3: LLM adjusts thresholds")
    print("-" * 40)

    threshold = ThresholdUpdate(
        message_id="llm_003",
        timestamp=time.time(),
        subsystem="thermal",
        thresholds={"warning": 72.0, "critical": 88.0},
        reason="Adjusted based on recent thermal events",
    )

    loop.install_threshold(threshold)
    print(f"Updated thresholds for: {threshold.subsystem}")
    print()

    # === Check what's installed ===
    print("Step 4: Check installed state")
    print("-" * 40)

    summary = loop.get_installed_summary()
    print(f"Patterns: {summary['patterns']}")
    print(f"Thresholds: {summary['thresholds']}")
    print(f"Reflexes: {summary['reflexes']}")
    print()

    # === Simulate: Card encounters high-risk situation ===
    print("Step 5: Card encounters high-risk situation")
    print("-" * 40)

    state_hv = make_random_hv("state:thermal_cascade", 256)
    machine_states = {
        "gpu-worker-1": make_random_hv("status:critical", 256),
        "gpu-worker-2": make_random_hv("status:degraded", 256),
    }

    decision = loop.should_invoke_llm(
        state_hv=state_hv,
        metrics={"health": 0.4, "friction": 0.2, "risk": 0.85},
        recent_events=["gpu-worker-1: thermal critical 92C"],
    )

    print(f"Invocation decision: {'INVOKE' if decision.should_invoke else 'skip'}")
    print(f"Reason: {decision.reason}")

    if decision.should_invoke:
        # Prepare message
        message = loop.prepare_message(
            decision=decision,
            state_hv=state_hv,
            machine_states=machine_states,
            recent_events=["gpu-worker-1: thermal critical 92C"],
            metrics={"health": 0.4, "friction": 0.2, "risk": 0.85},
        )

        print()
        print("Message to LLM:")
        print("-" * 40)
        if hasattr(message, 'to_prompt'):
            print(message.to_prompt())
        elif hasattr(message, 'to_prompt_context'):
            print(message.to_prompt_context())
        print("-" * 40)

        loop.record_invocation()
    print()


def demo_message_format():
    """Show the format of messages for LLM prompts."""
    print("=" * 70)
    print("Message Formats: What the LLM Sees")
    print("=" * 70)
    print()

    # Create sample messages

    # === State Summary ===
    print("1. State Summary (periodic context)")
    print("-" * 40)

    summary = StateSummary(
        message_id="state_001",
        timestamp=time.time(),
        mode="scientist",
        active_projects=["ara", "ara-card", "noesis"],
        machine_status={
            "gpu-worker-1": "healthy",
            "gpu-worker-2": "degraded",
            "nas-primary": "healthy",
        },
        health_score=0.78,
        friction_level=0.35,
        risk_level=0.25,
        active_patterns=["gpu_load", "compilation"],
        recent_anomalies=["gpu-worker-2 thermal warning"],
    )

    print(summary.to_prompt_context())
    print()

    # === Anomaly Report ===
    print("2. Anomaly Report (something unusual)")
    print("-" * 40)

    anomaly = AnomalyReport(
        message_id="anomaly_001",
        timestamp=time.time(),
        anomaly_type="thermal_cascade",
        severity=0.85,
        source="gpu-cluster",
        description="Multiple GPUs exceeding thermal limits within 5 minutes",
        related_events=[
            "gpu-worker-1: 92C at 14:32",
            "gpu-worker-2: 88C at 14:34",
            "gpu-worker-3: 86C at 14:35",
        ],
        card_assessment="Pattern matches thermal cascade profile",
        suggested_actions=["throttle_training", "pause_batch_jobs", "increase_cooling"],
        question="Should we pause all training or just throttle? What's the priority?",
    )

    print(anomaly.to_prompt())
    print()

    # === Context Query ===
    print("3. Context Query (need reasoning)")
    print("-" * 40)

    query = ContextQuery(
        message_id="query_001",
        timestamp=time.time(),
        situation="Novel pattern detected: auth failures from 3 different IPs within 10 seconds",
        state_summary=summary,
        question="Is this a coordinated attack or coincidental timing? Should we lock down?",
        time_budget_seconds=5.0,
        compute_budget="minimal",
    )

    print(query.to_prompt())
    print()


def run_demo():
    """Run all demos."""
    print()
    print("=" * 70)
    print("CARD ↔ LLM BRIDGE DEMO")
    print("Subcortex ↔ Cortex Symbiosis")
    print("=" * 70)
    print()

    demo_state_compression()
    demo_invocation_decisions()
    demo_policy_encoding()
    demo_closed_loop()
    demo_message_format()

    print("=" * 70)
    print("Demo complete.")
    print()
    print("The Bridge provides:")
    print("  1. State compression: HPV → text (50KB logs → 500 byte summary)")
    print("  2. Smart invocation: Only call LLM when novel/risky/high-ROI")
    print("  3. Policy encoding: LLM output → HPV patterns + SNN thresholds")
    print("  4. Closed loop: LLM policies bake back into Card reflexes")
    print()
    print("The GPU is the deep planner/architect.")
    print("The Card is the always-on subcortex.")
    print("Together: a designed symbiosis.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
