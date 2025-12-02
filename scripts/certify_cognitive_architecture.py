#!/usr/bin/env python3
"""
Phase 6: Cognitive Architecture Certification

This script certifies the cognitive components that move the system
from "smart control" toward genuine cognition:

1. L4 Cognitive Memory (Knowledge Graph)
   - World model with beliefs and uncertainty
   - Hardware health tracking
   - Experiment recording

2. Episodic Memory
   - Episode recording and retrieval
   - Temporal queries
   - Autobiographical narration

3. SelfState (Meta-cognition)
   - Mood and risk awareness
   - Confidence tracking
   - Self-description

4. Goal Vector
   - Explicit value representation
   - Reward computation
   - Priority explanation

5. Deliberation Loop
   - Multi-step thinking
   - Risk-based depth
   - Thought traces

Usage:
    python scripts/certify_cognitive_architecture.py
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(title: str) -> None:
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_result(name: str, passed: bool, details: str = "") -> None:
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} {name}")
    if details:
        print(f"         {details}")


def certify_l4_knowledge_graph() -> tuple:
    """Certify L4 Cognitive Memory."""
    print_header("L4 Cognitive Memory (Knowledge Graph)")

    from tfan.l4 import (
        CognitiveKnowledgeGraph, NodeType, EdgeType, HealthStatus,
        create_lab_kg, query_for_task
    )

    passed = 0
    total = 0

    # Test 1: Graph creation
    total += 1
    try:
        kg = CognitiveKnowledgeGraph()
        print_result("Graph initialization", True, f"empty graph created")
        passed += 1
    except Exception as e:
        print_result("Graph initialization", False, str(e))
        return passed, total

    # Test 2: Node operations
    total += 1
    try:
        node = kg.add_node(NodeType.FPGA, "Test FPGA", {"clock": "250MHz"})
        retrieved = kg.get_node(node.id)
        success = retrieved is not None and retrieved.name == "Test FPGA"
        print_result("Node add/get", success, f"id={node.id[:16]}...")
        if success:
            passed += 1
    except Exception as e:
        print_result("Node add/get", False, str(e))

    # Test 3: Belief operations
    total += 1
    try:
        belief = kg.add_belief(node.id, "FPGA is operational", 0.95, "health_check")
        retrieved = kg.get_node(node.id)
        success = len(retrieved.beliefs) == 1 and retrieved.beliefs[0].confidence == 0.95
        print_result("Belief add", success, f"confidence={belief.confidence}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Belief add", False, str(e))

    # Test 4: Edge operations
    total += 1
    try:
        node2 = kg.add_node(NodeType.EXPERIMENT, "Test Experiment")
        edge = kg.add_edge(node2.id, node.id, EdgeType.RAN_ON)
        edges = kg.get_edges(node.id)
        success = len(edges) == 1 and edges[0].edge_type == EdgeType.RAN_ON
        print_result("Edge add/query", success, f"edge_type={edge.edge_type.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Edge add/query", False, str(e))

    # Test 5: Health tracking
    total += 1
    try:
        node.update_health(HealthStatus.DEGRADED, "test_failure")
        success = node.health == HealthStatus.DEGRADED
        print_result("Health tracking", success, f"health={node.health.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Health tracking", False, str(e))

    # Test 6: Verified truth cache (PGU integration)
    total += 1
    try:
        kg.cache_verified_truth("2+2=4", True, 1.0)
        result = kg.check_verified_truth("2+2=4")
        success = result == (True, 1.0)
        print_result("Verified truth cache", success, f"cached truth: 2+2=4")
        if success:
            passed += 1
    except Exception as e:
        print_result("Verified truth cache", False, str(e))

    # Test 7: Query operations
    total += 1
    try:
        # Path query
        node3 = kg.add_node(NodeType.WORKLOAD, "Test Workload")
        kg.add_edge(node3.id, node2.id, EdgeType.REQUIRES)
        paths = kg.query("path", source=node3.id, target=node.id)
        success = len(paths) > 0
        print_result("Path query", success, f"found {len(paths)} path(s)")
        if success:
            passed += 1
    except Exception as e:
        print_result("Path query", False, str(e))

    # Test 8: Lab KG convenience function
    total += 1
    try:
        lab_kg = create_lab_kg()
        stats = lab_kg.stats
        success = stats["node_count"] > 0
        print_result("Lab KG creation", success, f"nodes={stats['node_count']}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Lab KG creation", False, str(e))

    # Test 9: Risk-aware retrieval
    total += 1
    try:
        high_risk = query_for_task(lab_kg, "test", "high")
        low_risk = query_for_task(lab_kg, "test", "low")
        success = high_risk["mode"] == "conservative" and low_risk["mode"] == "exploratory"
        print_result("Risk-aware retrieval", success,
                    f"high_risk={high_risk['mode']}, low_risk={low_risk['mode']}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Risk-aware retrieval", False, str(e))

    return passed, total


def certify_episodic_memory() -> tuple:
    """Certify Episodic Memory."""
    print_header("Episodic Memory")

    from tfan.memory.episodic import (
        EpisodicMemory, EpisodeType, OutcomeType,
        create_episodic_memory, record_certification_episode
    )

    passed = 0
    total = 0

    # Test 1: Memory creation
    total += 1
    try:
        memory = EpisodicMemory()
        print_result("Memory initialization", True)
        passed += 1
    except Exception as e:
        print_result("Memory initialization", False, str(e))
        return passed, total

    # Test 2: Episode lifecycle
    total += 1
    try:
        episode = memory.start_episode(
            EpisodeType.EXPERIMENT,
            "Test Experiment",
            task_description="Testing episodic memory",
            config={"param": 1.0},
            tags=["test"]
        )
        current = memory.get_current_episode()
        success = current is not None and current.id == episode.id
        print_result("Episode start", success, f"id={episode.id}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Episode start", False, str(e))

    # Test 3: Episode decisions
    total += 1
    try:
        decision = episode.add_decision(
            "backend_selection",
            "pgu_verified",
            alternatives=["dense", "sparse"],
            rationale="High risk situation"
        )
        success = len(episode.decisions) == 1
        print_result("Decision recording", success, f"choice={decision.choice}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Decision recording", False, str(e))

    # Test 4: Episode close
    total += 1
    try:
        closed = memory.close_current_episode(
            OutcomeType.SUCCESS,
            metrics={"af_score": 2.21},
            lessons=["PGU verification helps"]
        )
        success = closed.outcome == OutcomeType.SUCCESS
        print_result("Episode close", success, f"outcome={closed.outcome.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Episode close", False, str(e))

    # Test 5: Episode retrieval
    total += 1
    try:
        # Add more episodes
        for i in range(3):
            ep = memory.start_episode(
                EpisodeType.WORKLOAD,
                f"Workload {i}",
                tags=["batch"]
            )
            memory.close_current_episode(
                OutcomeType.SUCCESS if i % 2 == 0 else OutcomeType.FAILURE,
                metrics={"latency": 10 + i}
            )

        recent = memory.get_recent(3)
        success = len(recent) == 3
        print_result("Episode retrieval", success, f"found {len(recent)} episodes")
        if success:
            passed += 1
    except Exception as e:
        print_result("Episode retrieval", False, str(e))

    # Test 6: Query by type
    total += 1
    try:
        workloads = memory.get_by_type(EpisodeType.WORKLOAD)
        success = len(workloads) == 3
        print_result("Query by type", success, f"workloads={len(workloads)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Query by type", False, str(e))

    # Test 7: Query by tag
    total += 1
    try:
        tagged = memory.get_by_tag("batch")
        success = len(tagged) == 3
        print_result("Query by tag", success, f"batch_tagged={len(tagged)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Query by tag", False, str(e))

    # Test 8: Failure query
    total += 1
    try:
        failures = memory.get_failures()
        success = len(failures) >= 1
        print_result("Failure query", success, f"failures={len(failures)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Failure query", False, str(e))

    # Test 9: Period summary
    total += 1
    try:
        summary = memory.summarize_period(datetime.now() - timedelta(hours=1))
        success = summary["episode_count"] == 4
        print_result("Period summary", success,
                    f"episodes={summary['episode_count']}, success_rate={summary['success_rate']:.0%}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Period summary", False, str(e))

    # Test 10: Narrative generation
    total += 1
    try:
        episode = memory.get_recent(1)[0]
        narrative = memory.generate_narrative(episode)
        success = len(narrative) > 50
        print_result("Narrative generation", success, f"length={len(narrative)} chars")
        if success:
            passed += 1
    except Exception as e:
        print_result("Narrative generation", False, str(e))

    return passed, total


def certify_self_state() -> tuple:
    """Certify SelfState and Meta-cognition."""
    print_header("SelfState (Meta-cognition)")

    from tfan.cognition import (
        SelfState, PADState, CLVState, MoodState, ConfidenceLevel,
        HardwareTrust, TrustLevel, MetaPolicyEngine, create_self_state
    )

    passed = 0
    total = 0

    # Test 1: SelfState creation
    total += 1
    try:
        state = create_self_state()
        print_result("SelfState creation", True, f"mood={state.mood.value}")
        passed += 1
    except Exception as e:
        print_result("SelfState creation", False, str(e))
        return passed, total

    # Test 2: PAD → Mood mapping
    total += 1
    try:
        state.update_pad(valence=-0.5, arousal=0.8, dominance=0.3)
        success = state.mood == MoodState.STRESSED
        print_result("PAD → Mood", success, f"stressed at v=-0.5, a=0.8")
        if success:
            passed += 1
    except Exception as e:
        print_result("PAD → Mood", False, str(e))

    # Test 3: CLV → Risk mapping
    total += 1
    try:
        state.update_clv(instability=0.7, resource=0.5, structural=0.4)
        success = state.risk_level == "high"
        print_result("CLV → Risk", success, f"risk={state.risk_level}")
        if success:
            passed += 1
    except Exception as e:
        print_result("CLV → Risk", False, str(e))

    # Test 4: Confidence tracking
    total += 1
    try:
        state.confidence = 0.8
        state.update_confidence(-0.5, "world model uncertainty")
        success = state.confidence_level == ConfidenceLevel.LOW
        print_result("Confidence tracking", success,
                    f"conf={state.confidence:.1f}, level={state.confidence_level.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Confidence tracking", False, str(e))

    # Test 5: Needs caution check
    total += 1
    try:
        # State is stressed, high risk, low confidence
        success = state.needs_caution == True
        print_result("Needs caution detection", success, f"needs_caution={state.needs_caution}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Needs caution detection", False, str(e))

    # Test 6: Hardware trust
    total += 1
    try:
        hw = HardwareTrust(device_id="fpga_1", device_type="FPGA")
        hw.record_failure("overheat")
        hw.record_failure("timing_error")
        success = hw.trust == TrustLevel.DISTRUSTED
        print_result("Hardware trust degradation", success,
                    f"trust={hw.trust.value} after 2 failures")
        if success:
            passed += 1
    except Exception as e:
        print_result("Hardware trust degradation", False, str(e))

    # Test 7: Self-description
    total += 1
    try:
        desc = state.describe()
        success = len(desc) > 20 and "stress" in desc.lower()
        print_result("Self-description", success, f"'{desc[:60]}...'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Self-description", False, str(e))

    # Test 8: Meta-policy engine
    total += 1
    try:
        engine = MetaPolicyEngine()
        active = engine.get_active_policies(state)
        actions = engine.get_recommended_actions(state)
        success = len(active) >= 2  # Should have stressed + high_risk policies
        print_result("Meta-policy activation", success, f"active_policies={len(active)}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Meta-policy activation", False, str(e))

    # Test 9: Can explore check (should be False when stressed)
    total += 1
    try:
        success = state.can_explore == False
        print_result("Exploration gating", success, f"can_explore={state.can_explore}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Exploration gating", False, str(e))

    # Test 10: Recovery to calm state
    total += 1
    try:
        state.update_pad(valence=0.3, arousal=0.2, dominance=0.6)
        state.update_clv(instability=0.1, resource=0.1, structural=0.1)
        state.confidence = 0.9
        success = state.mood == MoodState.CALM and state.can_explore
        print_result("Recovery to calm", success,
                    f"mood={state.mood.value}, can_explore={state.can_explore}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Recovery to calm", False, str(e))

    return passed, total


def certify_goal_vector() -> tuple:
    """Certify Goal Vector system."""
    print_header("Goal Vector (Explicit Values)")

    from tfan.cognition import GoalVector, create_goal_vector

    passed = 0
    total = 0

    # Test 1: Goal vector creation
    total += 1
    try:
        goals = create_goal_vector("balanced")
        success = abs(goals.stability_weight + goals.latency_weight +
                     goals.energy_weight + goals.exploration_weight - 1.0) < 0.01
        print_result("Goal vector creation", success,
                    f"sum={goals.stability_weight + goals.latency_weight + goals.energy_weight + goals.exploration_weight:.2f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Goal vector creation", False, str(e))
        return passed, total

    # Test 2: Reward computation
    total += 1
    try:
        reward = goals.compute_reward(
            af_score=2.21,
            delta_p99=50.0,
            energy_efficiency=0.8,
            exploration_gain=0.5
        )
        success = 0.0 < reward < 1.0
        print_result("Reward computation", success, f"reward={reward:.3f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Reward computation", False, str(e))

    # Test 3: Priority explanation
    total += 1
    try:
        explanation = goals.explain_priority()
        success = "stability" in explanation.lower()
        print_result("Priority explanation", success, f"'{explanation[:50]}...'")
        if success:
            passed += 1
    except Exception as e:
        print_result("Priority explanation", False, str(e))

    # Test 4: Task adjustment
    total += 1
    try:
        safety_goals = goals.adjust_for_task("safety_critical")
        success = safety_goals.safety_modifier > 0 and safety_goals.exploration_weight == 0
        print_result("Task adjustment (safety)", success,
                    f"safety_mod={safety_goals.safety_modifier}, explore={safety_goals.exploration_weight}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Task adjustment (safety)", False, str(e))

    # Test 5: Preset comparison
    total += 1
    try:
        safety_first = create_goal_vector("safety_first")
        performance = create_goal_vector("performance")
        success = safety_first.stability_weight > performance.stability_weight
        print_result("Preset comparison", success,
                    f"safety.stab={safety_first.stability_weight:.2f} > perf.stab={performance.stability_weight:.2f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Preset comparison", False, str(e))

    # Test 6: Priority check
    total += 1
    try:
        success = goals.should_prioritize("stability") == True
        print_result("Priority check", success, f"prioritize_stability={goals.should_prioritize('stability')}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Priority check", False, str(e))

    return passed, total


def certify_deliberation() -> tuple:
    """Certify Deliberation Loop."""
    print_header("Deliberation Loop (Multi-Step Thinking)")

    from tfan.cognition import (
        Deliberator, SelfState, GoalVector, ThoughtType,
        create_self_state, create_goal_vector, create_deliberator
    )

    passed = 0
    total = 0

    # Test 1: Deliberator creation
    total += 1
    try:
        deliberator = create_deliberator(max_iterations=3)
        success = deliberator.max_iterations == 3
        print_result("Deliberator creation", success, f"max_iter={deliberator.max_iterations}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Deliberator creation", False, str(e))
        return passed, total

    # Test 2: Low-risk deliberation (1 step)
    total += 1
    try:
        state = create_self_state()
        state.update_clv(0.1, 0.1, 0.1)  # Low risk
        goals = create_goal_vector("balanced")

        def mock_proposal(task, goals, previous=None):
            return "use_dense_backend"

        result = deliberator.deliberate(
            "Select backend",
            state,
            goals,
            mock_proposal
        )
        success = result.iterations == 1 and not result.used_pgu
        print_result("Low-risk deliberation", success,
                    f"iterations={result.iterations}, used_pgu={result.used_pgu}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Low-risk deliberation", False, str(e))

    # Test 3: High-risk deliberation (multiple steps)
    total += 1
    try:
        state.update_clv(0.8, 0.6, 0.5)  # High risk
        state.update_pad(-0.3, 0.8, 0.3)  # Stressed

        def mock_checker(proposal):
            return (True, "Constraints satisfied")

        result = deliberator.deliberate(
            "Select backend",
            state,
            goals,
            mock_proposal,
            check_fn=mock_checker
        )
        success = result.used_pgu == True
        print_result("High-risk deliberation", success,
                    f"iterations={result.iterations}, used_pgu={result.used_pgu}")
        if success:
            passed += 1
    except Exception as e:
        print_result("High-risk deliberation", False, str(e))

    # Test 4: Thought trace
    total += 1
    try:
        trace = result.get_trace()
        success = "Deliberation trace" in trace and result.final_decision in trace
        print_result("Thought trace generation", success, f"trace_length={len(trace)} chars")
        if success:
            passed += 1
    except Exception as e:
        print_result("Thought trace generation", False, str(e))

    # Test 5: Quick decide
    total += 1
    try:
        options = ["conservative", "balanced", "aggressive"]
        goals_stable = create_goal_vector("safety_first")
        choice = deliberator.quick_decide("Select mode", options, goals_stable)
        success = choice == "conservative"  # Should pick first when stability prioritized
        print_result("Quick decide", success, f"choice={choice}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Quick decide", False, str(e))

    # Test 6: Refinement iteration
    total += 1
    try:
        iteration_count = 0

        def refining_proposal(task, goals, previous=None):
            nonlocal iteration_count
            iteration_count += 1
            if previous is None:
                return "initial_proposal"
            return f"refined_{iteration_count}"

        def failing_then_passing_checker(proposal):
            if "initial" in proposal:
                return (False, "Initial proposal failed")
            return (True, "Refined proposal passed")

        state.update_clv(0.9, 0.7, 0.6)  # Critical risk
        result = deliberator.deliberate(
            "Complex task",
            state,
            goals,
            refining_proposal,
            check_fn=failing_then_passing_checker
        )
        success = result.iterations >= 2
        print_result("Refinement iteration", success,
                    f"iterations={result.iterations}, final={result.final_decision}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Refinement iteration", False, str(e))

    return passed, total


def main():
    """Run all Phase 6 certifications."""
    print("=" * 70)
    print("  PHASE 6: COGNITIVE ARCHITECTURE CERTIFICATION")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}
    total_passed = 0
    total_tests = 0

    # Run all certifications
    for name, cert_fn in [
        ("L4 Cognitive Memory", certify_l4_knowledge_graph),
        ("Episodic Memory", certify_episodic_memory),
        ("SelfState", certify_self_state),
        ("Goal Vector", certify_goal_vector),
        ("Deliberation Loop", certify_deliberation),
    ]:
        try:
            passed, total = cert_fn()
            results[name] = {"passed": passed, "total": total}
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f"\n  ❌ ERROR in {name}: {e}")
            results[name] = {"passed": 0, "total": 1, "error": str(e)}
            total_tests += 1

    # Print summary
    print_header("CERTIFICATION SUMMARY")

    for name, result in results.items():
        p, t = result["passed"], result["total"]
        status = "✅ CERTIFIED" if p == t else "❌ FAILED"
        print(f"  {status}  {name} ({p}/{t})")

    print(f"\n  Total: {total_passed}/{total_tests} tests passed")

    all_passed = total_passed == total_tests

    if all_passed:
        print("""
  ╔════════════════════════════════════════════════════════════════╗
  ║                                                                ║
  ║   ✓ PHASE 6 COGNITIVE ARCHITECTURE CERTIFIED                   ║
  ║                                                                ║
  ║   The system demonstrates cognitive capabilities:              ║
  ║   • L4: Persistent world model with beliefs                    ║
  ║   • Episodic: Autobiographical memory & narrative              ║
  ║   • SelfState: Meta-cognitive self-awareness                   ║
  ║   • Goals: Explicit values & explainable priorities            ║
  ║   • Deliberation: Multi-step thinking with traces              ║
  ║                                                                ║
  ╚════════════════════════════════════════════════════════════════╝
""")
    else:
        print(f"\n  ⚠️  {total_tests - total_passed} test(s) failed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
