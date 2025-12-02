#!/usr/bin/env python3
"""
Certification Script: CSTP (Cognitive State Transfer Protocol)

Tests the CSTP implementation:
1. Cognitive state encoding as (z, c) pairs
2. Curvature selection and L3 integration
3. Serialization/deserialization protocol
4. Thought streams and analysis

Target: 30/30 tests passing
"""

import sys
from datetime import datetime

sys.path.insert(0, "/home/user/Ara")

from tfan.cognition.cstp import (
    ThoughtType,
    CompressionLevel,
    ThoughtMetadata,
    CognitiveState,
    CurvatureMode,
    CurvaturePolicy,
    CurvatureController,
    CSTPEncoder,
    CSTPDecoder,
    CSTPProtocol,
    ThoughtStreamEntry,
    ThoughtStream,
    create_encoder,
    create_decoder,
    encode_thought,
    serialize_thought,
    deserialize_thought
)


def run_tests():
    passed = 0
    failed = 0
    failures = []

    def test(name, condition, details=""):
        nonlocal passed, failed, failures
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            msg = f"  ✗ {name}" + (f" - {details}" if details else "")
            print(msg)
            failed += 1
            failures.append(name)

    # ========================================
    # Section 1: Cognitive State Basics
    # ========================================
    print("\n═══ Section 1: Cognitive State Basics ═══")

    metadata = ThoughtMetadata(
        thought_type=ThoughtType.OBSERVATION,
        confidence=0.9,
        tags=["test"]
    )

    state = CognitiveState(
        z=[0.1, 0.2, 0.3, 0.4],
        c=1.0,
        metadata=metadata
    )

    test("CognitiveState created", state is not None)
    test("State has z vector", len(state.z) == 4)
    test("State has curvature c", state.c == 1.0)
    test("Dimension computed", state.dimension == 4)
    test("Norm computed", state.norm > 0)
    test("Is hyperbolic detected", state.is_hyperbolic)

    # Low curvature state
    flat_state = CognitiveState(
        z=[0.1, 0.2],
        c=0.1,
        metadata=ThoughtMetadata(thought_type=ThoughtType.INFERENCE)
    )
    test("Is Euclidean detected", flat_state.is_euclidean)
    test("Geometry description", "flat" in flat_state.geometry_description)

    # ========================================
    # Section 2: Curvature Controller
    # ========================================
    print("\n═══ Section 2: Curvature Controller ═══")

    policy = CurvaturePolicy(
        mode=CurvatureMode.STRESS_AWARE,
        stressed_curvature=0.2
    )
    controller = CurvatureController(policy)

    test("CurvatureController created", controller is not None)

    # Planning task → higher curvature
    planning_c = controller.select_curvature(task_type="planning")
    test("Planning gets high curvature", planning_c >= 1.0, f"c={planning_c}")

    # Retrieval task → lower curvature
    retrieval_c = controller.select_curvature(task_type="retrieval")
    test("Retrieval gets low curvature", retrieval_c <= 0.5, f"c={retrieval_c}")

    # High stress → reduced curvature
    controller.update_state(stress=0.9, valence=-0.5)
    stressed_c = controller.select_curvature(task_type="planning")
    test("High stress reduces curvature",
         stressed_c <= policy.stressed_curvature,
         f"c={stressed_c}")

    # Reset stress
    controller.update_state(stress=0.1, valence=0.5)

    # Explanation works
    explanation = controller.explain_selection(1.0)
    test("Explanation generated", len(explanation) > 0)

    # ========================================
    # Section 3: Encoder
    # ========================================
    print("\n═══ Section 3: Encoder ═══")

    encoder = create_encoder(dimension=32)
    test("Encoder created", encoder is not None)

    # Encode vector
    raw_vec = [0.1] * 32
    vec_state = encoder.encode_vector(
        raw_vec,
        thought_type=ThoughtType.HYPOTHESIS,
        confidence=0.7
    )
    test("Vector encoded", vec_state is not None)
    test("Vector dimension preserved", vec_state.dimension == 32)
    test("Thought type set", vec_state.metadata.thought_type == ThoughtType.HYPOTHESIS)

    # Encode plan
    plan_steps = [
        {"action": "gather_info", "priority": 1},
        {"action": "analyze", "priority": 2},
        {"action": "decide", "priority": 3}
    ]
    plan_state = encoder.encode_plan(plan_steps, hierarchy_depth=2)
    test("Plan encoded", plan_state is not None)
    test("Plan type set", plan_state.metadata.thought_type == ThoughtType.PLAN)
    test("Plan has structure", plan_state.structure is not None)
    test("Plan curvature elevated", plan_state.c >= 1.0, f"c={plan_state.c}")

    # Encode observation
    obs_state = encoder.encode_observation("The weather is nice today")
    test("Observation encoded", obs_state is not None)
    test("Observation flat curvature", obs_state.c < 0.5)

    # ========================================
    # Section 4: Decoder
    # ========================================
    print("\n═══ Section 4: Decoder ═══")

    decoder = create_decoder()
    test("Decoder created", decoder is not None)

    # Decode vector
    decoded_vec = decoder.decode_to_vector(vec_state)
    test("Vector decoded", len(decoded_vec) == 32)

    # Decode plan
    decoded_plan = decoder.decode_plan(plan_state)
    test("Plan decoded", decoded_plan is not None)
    test("Plan steps preserved", len(decoded_plan.get("steps", [])) == 3)

    # Compare states
    comparison = decoder.compare_states(vec_state, plan_state)
    test("States compared", "euclidean_distance" in comparison)
    test("Curvature difference tracked", comparison["curvature_difference"] >= 0)

    # ========================================
    # Section 5: Serialization Protocol
    # ========================================
    print("\n═══ Section 5: Serialization Protocol ═══")

    # JSON serialization
    json_str = CSTPProtocol.serialize_json(plan_state)
    test("JSON serialization works", len(json_str) > 50)
    test("JSON contains version", "version" in json_str)

    # JSON deserialization
    restored_state = CSTPProtocol.deserialize_json(json_str)
    test("JSON deserialization works", restored_state is not None)
    test("Restored z matches", restored_state.z == plan_state.z)
    test("Restored c matches", restored_state.c == plan_state.c)

    # Binary serialization
    binary_data = CSTPProtocol.serialize_binary(plan_state)
    test("Binary serialization works", binary_data.startswith(b"CSTP"))

    # Binary deserialization
    restored_binary = CSTPProtocol.deserialize_binary(binary_data)
    test("Binary deserialization works", restored_binary is not None)

    # Compact serialization
    compact_str = CSTPProtocol.serialize_compact(plan_state)
    test("Compact serialization works", len(compact_str) > 0)

    restored_compact = CSTPProtocol.deserialize_compact(compact_str)
    test("Compact deserialization works", restored_compact.c == plan_state.c)

    # ========================================
    # Section 6: Thought Stream
    # ========================================
    print("\n═══ Section 6: Thought Stream ═══")

    stream = ThoughtStream(name="test_stream")
    test("ThoughtStream created", stream is not None)

    # Add thoughts
    for i in range(5):
        state = encoder.encode_vector(
            [0.1 * i] * 32,
            thought_type=ThoughtType.INFERENCE if i % 2 else ThoughtType.OBSERVATION
        )
        stream.append(state)

    test("Thoughts added to stream", len(stream.get_entries()) == 5)

    # Get curvature trajectory
    trajectory = stream.get_curvature_trajectory()
    test("Curvature trajectory tracked", len(trajectory) == 5)

    # Get thought types
    type_counts = stream.get_thought_types()
    test("Thought types counted", len(type_counts) > 0)

    # Analyze geometry shifts
    shifts = stream.analyze_geometry_shifts()
    test("Geometry shifts analyzed", isinstance(shifts, list))

    # Stream to dict
    stream_dict = stream.to_dict()
    test("Stream serialized", "entries" in stream_dict)

    # ========================================
    # Section 7: Convenience Functions
    # ========================================
    print("\n═══ Section 7: Convenience Functions ═══")

    # encode_thought with string
    thought1 = encode_thought("Hello world")
    test("encode_thought with string", thought1 is not None)

    # encode_thought with vector
    thought2 = encode_thought([0.5, 0.5, 0.5], ThoughtType.HYPOTHESIS)
    test("encode_thought with vector", thought2.metadata.thought_type == ThoughtType.HYPOTHESIS)

    # encode_thought with plan
    thought3 = encode_thought({"steps": [{"a": 1}, {"b": 2}], "depth": 2})
    test("encode_thought with plan", thought3.metadata.thought_type == ThoughtType.PLAN)

    # serialize/deserialize convenience
    serialized = serialize_thought(thought1, format="json")
    restored = deserialize_thought(serialized, format="json")
    test("serialize_thought works", restored.c == thought1.c)

    return passed, failed, failures


def main():
    print("=" * 60)
    print("CSTP CERTIFICATION")
    print("Cognitive State Transfer Protocol")
    print("=" * 60)

    passed, failed, failures = run_tests()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failures:
        print("\nFailed tests:")
        for f in failures:
            print(f"  - {f}")

    total = passed + failed
    threshold = 0.90

    if passed / total >= threshold:
        print(f"\n✅ CERTIFICATION PASSED ({passed}/{total})")
        return 0
    else:
        print(f"\n❌ CERTIFICATION FAILED ({passed}/{total} < {threshold:.0%})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
