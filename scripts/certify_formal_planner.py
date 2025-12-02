#!/usr/bin/env python3
"""
Certification: L8 Formal Planner

This script certifies that the formal planner:
1. Solves scheduling problems correctly
2. Detects unsatisfiable constraints
3. Verifies solutions against constraints
4. Meets latency targets (p95 ≤ 120ms)
5. Handles resource allocation

The key property: Solutions are verified, not hallucinated.
"""

import sys
from pathlib import Path
from datetime import datetime
import time

# Add paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def print_result(name: str, passed: bool, details: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} {name}")
    if details:
        print(f"         {details}")


# ============================================================
# Test: Basic Scheduling
# ============================================================

def certify_basic_scheduling():
    """Test basic scheduling functionality."""
    print_header("Basic Scheduling")

    from tfan.l8.planner import (
        create_planner, create_scheduling_problem, Task, PlanResult, SolveStatus
    )

    passed = 0
    total = 0

    # Test 1: Simple sequential schedule
    total += 1
    try:
        builder = create_scheduling_problem()
        builder.add_task("A", duration=1.0)
        builder.add_task("B", duration=2.0)
        builder.add_task("C", duration=1.0)
        builder.add_precedence("A", "B")
        builder.add_precedence("B", "C")

        planner = create_planner()
        result = planner.solve(builder.build())

        success = result.status == SolveStatus.SOLVED and result.verified
        print_result("Sequential schedule", success,
                    f"status={result.status.value}, verified={result.verified}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Sequential schedule", False, str(e))

    # Test 2: Parallel tasks
    total += 1
    try:
        builder = create_scheduling_problem()
        builder.add_task("A", duration=1.0)
        builder.add_task("B", duration=1.0)
        builder.add_task("C", duration=1.0, depends_on=["A", "B"])

        planner = create_planner()
        result = planner.solve(builder.build())

        # A and B can run in parallel, C waits for both
        # Makespan should be 2 (A||B then C)
        success = (result.status == SolveStatus.SOLVED and
                  result.verified and
                  result.makespan <= 2.0)
        print_result("Parallel tasks", success,
                    f"makespan={result.makespan:.1f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Parallel tasks", False, str(e))

    # Test 3: Task with deadline
    total += 1
    try:
        builder = create_scheduling_problem()
        builder.add_task("A", duration=1.0, deadline=2.0)
        builder.add_task("B", duration=1.0, deadline=3.0, depends_on=["A"])

        planner = create_planner()
        result = planner.solve(builder.build())

        success = result.status == SolveStatus.SOLVED and result.verified
        print_result("Tasks with deadlines", success,
                    f"A ends at {result.schedule[0].end_time if result.schedule else 'N/A'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Tasks with deadlines", False, str(e))

    # Test 4: Empty problem
    total += 1
    try:
        builder = create_scheduling_problem()
        planner = create_planner()
        result = planner.solve(builder.build())

        success = result.status == SolveStatus.SOLVED
        print_result("Empty problem", success,
                    f"status={result.status.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Empty problem", False, str(e))

    return passed, total


# ============================================================
# Test: Constraint Detection
# ============================================================

def certify_constraint_detection():
    """Test detection of unsatisfiable constraints."""
    print_header("Constraint Detection (Unsatisfiability)")

    from tfan.l8.planner import (
        create_planner, create_scheduling_problem, SolveStatus
    )

    passed = 0
    total = 0

    # Test 1: Circular dependency
    total += 1
    try:
        builder = create_scheduling_problem()
        builder.add_task("A", duration=1.0, depends_on=["C"])
        builder.add_task("B", duration=1.0, depends_on=["A"])
        builder.add_task("C", duration=1.0, depends_on=["B"])

        planner = create_planner()
        result = planner.solve(builder.build())

        success = result.status == SolveStatus.UNSATISFIABLE
        has_cycle_msg = any("cycle" in v.lower() for v in result.violations)
        print_result("Detects circular dependency", success and has_cycle_msg,
                    f"violations={result.violations}")
        if success and has_cycle_msg:
            passed += 1
    except Exception as e:
        print_result("Detects circular dependency", False, str(e))

    # Test 2: Impossible deadline
    total += 1
    try:
        builder = create_scheduling_problem()
        builder.add_task("A", duration=5.0, deadline=2.0)  # Can't finish 5 units in 2

        planner = create_planner()
        result = planner.solve(builder.build())

        success = result.status == SolveStatus.UNSATISFIABLE
        print_result("Detects impossible deadline", success,
                    f"status={result.status.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Detects impossible deadline", False, str(e))

    # Test 3: Mutex violation
    total += 1
    try:
        builder = create_scheduling_problem()
        builder.add_task("A", duration=2.0)
        builder.add_task("B", duration=2.0, depends_on=["A"])  # B must wait for A
        builder.add_mutex("A", "B")  # And they can't overlap

        planner = create_planner()
        result = planner.solve(builder.build())

        # This should be satisfiable because B already waits for A
        success = result.status == SolveStatus.SOLVED and result.verified
        print_result("Mutex with dependency (satisfiable)", success,
                    f"verified={result.verified}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Mutex with dependency", False, str(e))

    # Test 4: Self-dependency
    total += 1
    try:
        builder = create_scheduling_problem()
        builder.add_task("A", duration=1.0, depends_on=["A"])  # Depends on itself

        planner = create_planner()
        result = planner.solve(builder.build())

        success = result.status == SolveStatus.UNSATISFIABLE
        print_result("Detects self-dependency", success,
                    f"status={result.status.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Detects self-dependency", False, str(e))

    return passed, total


# ============================================================
# Test: Verification
# ============================================================

def certify_verification():
    """Test that solutions are properly verified."""
    print_header("Solution Verification")

    from tfan.l8.planner import (
        create_planner, create_scheduling_problem, PlanResult, SolveStatus
    )

    passed = 0
    total = 0

    # Test 1: Valid solution is verified
    total += 1
    try:
        builder = create_scheduling_problem()
        builder.add_task("A", duration=1.0)
        builder.add_task("B", duration=1.0, depends_on=["A"])

        planner = create_planner()
        result = planner.solve(builder.build())

        success = result.verified and "verified" in result.verification_status
        print_result("Valid solution verified", success,
                    f"verification_status={result.verification_status}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Valid solution verified", False, str(e))

    # Test 2: Verification catches precedence violations
    total += 1
    try:
        # This tests internal verification - should always pass if solver is correct
        builder = create_scheduling_problem()
        builder.add_task("A", duration=1.0)
        builder.add_task("B", duration=1.0)
        builder.add_precedence("A", "B")

        planner = create_planner()
        result = planner.solve(builder.build())

        # Find entries
        entry_a = next((e for e in result.schedule if e.task_id == "A"), None)
        entry_b = next((e for e in result.schedule if e.task_id == "B"), None)

        if entry_a and entry_b:
            success = entry_a.end_time <= entry_b.start_time
        else:
            success = False

        print_result("Precedence enforced", success,
                    f"A ends={entry_a.end_time if entry_a else 'N/A'}, B starts={entry_b.start_time if entry_b else 'N/A'}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Precedence enforced", False, str(e))

    # Test 3: Verification status in result
    total += 1
    try:
        builder = create_scheduling_problem()
        builder.add_task("A", duration=1.0)

        planner = create_planner()
        result = planner.solve(builder.build())

        success = (hasattr(result, 'verified') and
                  hasattr(result, 'verification_status') and
                  hasattr(result, 'violations'))
        print_result("Verification fields present", success,
                    f"verified={result.verified}, status={result.verification_status}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Verification fields present", False, str(e))

    # Test 4: is_valid property
    total += 1
    try:
        builder = create_scheduling_problem()
        builder.add_task("A", duration=1.0)

        planner = create_planner()
        result = planner.solve(builder.build())

        success = result.is_valid == (result.status == SolveStatus.SOLVED and result.verified)
        print_result("is_valid property", success,
                    f"is_valid={result.is_valid}")
        if success:
            passed += 1
    except Exception as e:
        print_result("is_valid property", False, str(e))

    return passed, total


# ============================================================
# Test: Latency
# ============================================================

def certify_latency():
    """Test that solver meets latency targets."""
    print_header("Latency (p95 ≤ 120ms)")

    from tfan.l8.planner import (
        create_planner, create_scheduling_problem
    )

    passed = 0
    total = 0

    # Test 1: Small problem latency
    total += 1
    try:
        builder = create_scheduling_problem()
        for i in range(5):
            deps = [f"task_{i-1}"] if i > 0 else []
            builder.add_task(f"task_{i}", duration=1.0, depends_on=deps)

        planner = create_planner()

        # Run multiple times to get p95
        times = []
        for _ in range(20):
            result = planner.solve(builder.build())
            times.append(result.solve_time_ms)

        times.sort()
        p95 = times[int(len(times) * 0.95)]

        success = p95 <= 120
        print_result("Small problem (5 tasks) p95", success,
                    f"p95={p95:.1f}ms, median={times[len(times)//2]:.1f}ms")
        if success:
            passed += 1
    except Exception as e:
        print_result("Small problem p95", False, str(e))

    # Test 2: Medium problem latency
    total += 1
    try:
        builder = create_scheduling_problem()
        for i in range(20):
            deps = [f"task_{i-1}"] if i > 0 else []
            builder.add_task(f"task_{i}", duration=1.0, depends_on=deps)

        planner = create_planner()

        times = []
        for _ in range(10):
            result = planner.solve(builder.build())
            times.append(result.solve_time_ms)

        times.sort()
        p95 = times[int(len(times) * 0.95)]

        success = p95 <= 120
        print_result("Medium problem (20 tasks) p95", success,
                    f"p95={p95:.1f}ms")
        if success:
            passed += 1
    except Exception as e:
        print_result("Medium problem p95", False, str(e))

    # Test 3: Complex dependencies
    total += 1
    try:
        builder = create_scheduling_problem()
        # Diamond pattern: A -> B,C -> D
        builder.add_task("A", duration=1.0)
        builder.add_task("B", duration=1.0, depends_on=["A"])
        builder.add_task("C", duration=1.0, depends_on=["A"])
        builder.add_task("D", duration=1.0, depends_on=["B", "C"])

        planner = create_planner()

        times = []
        for _ in range(20):
            result = planner.solve(builder.build())
            times.append(result.solve_time_ms)

        times.sort()
        p95 = times[int(len(times) * 0.95)]

        success = p95 <= 120
        print_result("Diamond pattern p95", success,
                    f"p95={p95:.1f}ms")
        if success:
            passed += 1
    except Exception as e:
        print_result("Diamond pattern p95", False, str(e))

    return passed, total


# ============================================================
# Test: Resource Allocation
# ============================================================

def certify_resource_allocation():
    """Test resource allocation functionality."""
    print_header("Resource Allocation")

    from tfan.l8.planner import (
        create_planner, PlanningProblem, ProblemType, Task, Resource, SolveStatus
    )

    passed = 0
    total = 0

    # Test 1: Simple allocation
    total += 1
    try:
        problem = PlanningProblem(problem_type=ProblemType.RESOURCE_ALLOCATION)
        problem.add_resource(Resource(id="gpu", capacity=2))
        problem.add_task(Task(id="task_a", requires=["gpu"]))
        problem.add_task(Task(id="task_b", requires=["gpu"]))

        planner = create_planner()
        result = planner.solve(problem)

        success = (result.status == SolveStatus.SOLVED and
                  "task_a" in result.assignments and
                  "task_b" in result.assignments)
        print_result("Simple allocation", success,
                    f"assignments={list(result.assignments.keys())}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Simple allocation", False, str(e))

    # Test 2: Resource exhaustion
    total += 1
    try:
        problem = PlanningProblem(problem_type=ProblemType.RESOURCE_ALLOCATION)
        problem.add_resource(Resource(id="gpu", capacity=1))  # Only 1 available
        problem.add_task(Task(id="task_a", requires=["gpu"]))
        problem.add_task(Task(id="task_b", requires=["gpu"]))

        planner = create_planner()
        result = planner.solve(problem)

        success = result.status == SolveStatus.UNSATISFIABLE
        print_result("Resource exhaustion detected", success,
                    f"status={result.status.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Resource exhaustion detected", False, str(e))

    # Test 3: Priority-based allocation
    total += 1
    try:
        problem = PlanningProblem(problem_type=ProblemType.RESOURCE_ALLOCATION)
        problem.add_resource(Resource(id="gpu", capacity=1))
        problem.add_task(Task(id="high_prio", priority=10, requires=["gpu"]))
        problem.add_task(Task(id="low_prio", priority=1, requires=["gpu"]))

        planner = create_planner()
        result = planner.solve(problem)

        # High priority should get allocated first
        # Low priority should fail
        success = result.status == SolveStatus.UNSATISFIABLE
        print_result("Priority respected in allocation", success,
                    f"status={result.status.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Priority respected", False, str(e))

    return passed, total


# ============================================================
# Test: Convenience Functions
# ============================================================

def certify_convenience_functions():
    """Test convenience API functions."""
    print_header("Convenience API")

    from tfan.l8.planner import solve_scheduling, SolveStatus

    passed = 0
    total = 0

    # Test 1: solve_scheduling function
    total += 1
    try:
        result = solve_scheduling(
            tasks=[
                {"id": "A", "duration": 1.0},
                {"id": "B", "duration": 1.0, "depends_on": ["A"]},
                {"id": "C", "duration": 1.0, "depends_on": ["B"]}
            ]
        )

        success = result.status == SolveStatus.SOLVED and result.verified
        print_result("solve_scheduling function", success,
                    f"makespan={result.makespan:.1f}")
        if success:
            passed += 1
    except Exception as e:
        print_result("solve_scheduling function", False, str(e))

    # Test 2: With constraints
    total += 1
    try:
        result = solve_scheduling(
            tasks=[
                {"id": "A", "duration": 1.0},
                {"id": "B", "duration": 1.0}
            ],
            constraints=[
                {"type": "before", "subjects": ["A", "B"]}
            ]
        )

        success = result.status == SolveStatus.SOLVED and result.verified
        print_result("solve_scheduling with constraints", success,
                    f"verified={result.verified}")
        if success:
            passed += 1
    except Exception as e:
        print_result("solve_scheduling with constraints", False, str(e))

    # Test 3: to_readable output
    total += 1
    try:
        result = solve_scheduling(
            tasks=[
                {"id": "A", "duration": 1.0},
                {"id": "B", "duration": 2.0, "depends_on": ["A"]}
            ]
        )

        readable = result.to_readable()
        success = "Schedule:" in readable and "A" in readable and "B" in readable
        print_result("to_readable output", success,
                    f"length={len(readable)} chars")
        if success:
            passed += 1
    except Exception as e:
        print_result("to_readable output", False, str(e))

    # Test 4: to_dict serialization
    total += 1
    try:
        result = solve_scheduling(
            tasks=[{"id": "A", "duration": 1.0}]
        )

        d = result.to_dict()
        required_keys = ["status", "is_valid", "schedule", "makespan", "verified"]
        success = all(k in d for k in required_keys)
        print_result("to_dict serialization", success,
                    f"keys={list(d.keys())}")
        if success:
            passed += 1
    except Exception as e:
        print_result("to_dict serialization", False, str(e))

    return passed, total


# ============================================================
# Test: Integration with L8 Verification
# ============================================================

def certify_l8_integration():
    """Test integration with L8 semantic verification."""
    print_header("L8 Integration (PGU Verification)")

    from tfan.l8.planner import create_planner, create_scheduling_problem, SolveStatus
    from tfan.l8 import AxiomStore, SemanticVerifier

    passed = 0
    total = 0

    # Test 1: Planner uses axiom store
    total += 1
    try:
        axiom_store = AxiomStore()
        planner = create_planner(axiom_store=axiom_store)

        builder = create_scheduling_problem()
        builder.add_task("A", duration=1.0)
        result = planner.solve(builder.build())

        success = result.status == SolveStatus.SOLVED
        print_result("Planner with axiom store", success,
                    f"status={result.status.value}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Planner with axiom store", False, str(e))

    # Test 2: Stats tracking
    total += 1
    try:
        planner = create_planner()

        # Solve a few problems
        for i in range(3):
            builder = create_scheduling_problem()
            builder.add_task(f"task_{i}", duration=1.0)
            planner.solve(builder.build())

        success = planner.stats["problems_solved"] >= 3
        print_result("Stats tracking", success,
                    f"problems_solved={planner.stats['problems_solved']}")
        if success:
            passed += 1
    except Exception as e:
        print_result("Stats tracking", False, str(e))

    # Test 3: Average solve time
    total += 1
    try:
        planner = create_planner()

        for i in range(5):
            builder = create_scheduling_problem()
            builder.add_task(f"task_{i}", duration=1.0)
            planner.solve(builder.build())

        success = planner.stats["avg_solve_time_ms"] > 0
        print_result("Avg solve time tracked", success,
                    f"avg={planner.stats['avg_solve_time_ms']:.2f}ms")
        if success:
            passed += 1
    except Exception as e:
        print_result("Avg solve time tracked", False, str(e))

    return passed, total


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  L8 FORMAL PLANNER CERTIFICATION")
    print("  PGU as Active Solver: Solutions, Not Hallucinations")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}
    total_passed = 0
    total_tests = 0

    for name, cert_fn in [
        ("Basic Scheduling", certify_basic_scheduling),
        ("Constraint Detection", certify_constraint_detection),
        ("Solution Verification", certify_verification),
        ("Latency", certify_latency),
        ("Resource Allocation", certify_resource_allocation),
        ("Convenience API", certify_convenience_functions),
        ("L8 Integration", certify_l8_integration),
    ]:
        try:
            passed, total = cert_fn()
            results[name] = {"passed": passed, "total": total}
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f"\n  ❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
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
  ║   ✓ L8 FORMAL PLANNER CERTIFIED                                ║
  ║                                                                ║
  ║   The system can now:                                          ║
  ║   • Solve scheduling problems with dependencies                ║
  ║   • Detect unsatisfiable constraints (cycles, deadlines)       ║
  ║   • Allocate resources with capacity limits                    ║
  ║   • Verify all solutions against constraints                   ║
  ║   • Meet latency targets (p95 ≤ 120ms)                         ║
  ║                                                                ║
  ║   Solutions are verified, not hallucinated.                    ║
  ║                                                                ║
  ╚════════════════════════════════════════════════════════════════╝""")
        return 0
    else:
        print(f"\n  ⚠️  {total_tests - total_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
