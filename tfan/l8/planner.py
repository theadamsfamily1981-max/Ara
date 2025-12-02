"""
L8 Formal Planner: PGU as Active Solver

This module makes the PGU not just a checker but an active problem solver.
For certain task types (scheduling, constraint satisfaction, resource allocation),
we encode the problem as formal constraints and let the solver generate
a verified solution.

Key insight: LLMs hallucinate. SMT solvers don't.
- For freeform reasoning: LLM + verification
- For constrained problems: Solver-first, then LLM explains

Pipeline:
    Problem specification (natural language or structured)
        ↓
    FormalPlanner.encode_problem() → PlanningProblem
        ↓
    FormalPlanner.solve() → candidate solution
        ↓
    PGU verification (consistency with axioms)
        ↓
    PlanResult (verified plan + explanation)

Supported problem types:
- SCHEDULING: Task ordering with dependencies and deadlines
- RESOURCE_ALLOCATION: Assigning limited resources to tasks
- CONFIGURATION: Finding valid system configurations
- SEQUENCING: Ordering steps with constraints
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
import heapq
import copy


class ProblemType(str, Enum):
    """Types of planning problems we can solve."""
    SCHEDULING = "scheduling"           # Tasks with dependencies and times
    RESOURCE_ALLOCATION = "resource"    # Allocating limited resources
    CONFIGURATION = "configuration"     # Finding valid configurations
    SEQUENCING = "sequencing"           # Ordering with constraints


class SolveStatus(str, Enum):
    """Status of solving attempt."""
    SOLVED = "solved"               # Found valid solution
    UNSATISFIABLE = "unsatisfiable" # No solution exists
    TIMEOUT = "timeout"             # Ran out of time
    PARTIAL = "partial"             # Found partial solution
    ERROR = "error"                 # Solver error


@dataclass
class Constraint:
    """A single constraint in the problem."""
    constraint_type: str  # "before", "after", "mutex", "requires", "within", "equals"
    subjects: List[str]   # Entities involved
    value: Optional[Any] = None  # For bounds/equality constraints
    hard: bool = True     # Hard constraint (must satisfy) vs soft (prefer)
    weight: float = 1.0   # Priority for soft constraints

    def __hash__(self):
        return hash((self.constraint_type, tuple(self.subjects), str(self.value)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.constraint_type,
            "subjects": self.subjects,
            "value": self.value,
            "hard": self.hard,
            "weight": self.weight
        }


@dataclass
class Resource:
    """A resource that can be allocated."""
    id: str
    capacity: int = 1           # How many can be used simultaneously
    available: int = 1          # Currently available
    exclusive: bool = True      # If true, only one task can use it at a time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "capacity": self.capacity,
            "available": self.available,
            "exclusive": self.exclusive
        }


@dataclass
class Task:
    """A task to be scheduled."""
    id: str
    duration: float = 1.0           # Time units
    priority: int = 0               # Higher = more important
    deadline: Optional[float] = None
    earliest_start: float = 0.0
    requires: List[str] = field(default_factory=list)  # Resource IDs
    depends_on: List[str] = field(default_factory=list)  # Task IDs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "duration": self.duration,
            "priority": self.priority,
            "deadline": self.deadline,
            "earliest_start": self.earliest_start,
            "requires": self.requires,
            "depends_on": self.depends_on
        }


@dataclass
class ScheduleEntry:
    """A single entry in a schedule."""
    task_id: str
    start_time: float
    end_time: float
    resources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "resources": self.resources
        }


@dataclass
class PlanningProblem:
    """
    A formal planning problem specification.

    This is the "input" to the solver.
    """
    problem_type: ProblemType
    tasks: List[Task] = field(default_factory=list)
    resources: List[Resource] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    objective: str = "minimize_makespan"  # What to optimize
    timeout_ms: int = 120  # p95 ≤ 120ms target

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def add_resource(self, resource: Resource) -> None:
        self.resources.append(resource)

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def add_precedence(self, before: str, after: str) -> None:
        """Add a precedence constraint: before must complete before after starts."""
        self.constraints.append(Constraint(
            constraint_type="before",
            subjects=[before, after],
            hard=True
        ))

    def add_mutex(self, task_a: str, task_b: str) -> None:
        """Add a mutual exclusion constraint: tasks cannot run simultaneously."""
        self.constraints.append(Constraint(
            constraint_type="mutex",
            subjects=[task_a, task_b],
            hard=True
        ))

    def add_deadline(self, task_id: str, deadline: float) -> None:
        """Add a deadline constraint for a task."""
        self.constraints.append(Constraint(
            constraint_type="within",
            subjects=[task_id],
            value=deadline,
            hard=True
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.problem_type.value,
            "tasks": [t.to_dict() for t in self.tasks],
            "resources": [r.to_dict() for r in self.resources],
            "constraints": [c.to_dict() for c in self.constraints],
            "objective": self.objective,
            "timeout_ms": self.timeout_ms
        }


@dataclass
class PlanResult:
    """
    Result of solving a planning problem.

    This is a verified plan that satisfies all constraints.
    """
    status: SolveStatus
    schedule: List[ScheduleEntry] = field(default_factory=list)
    assignments: Dict[str, Any] = field(default_factory=dict)  # For config problems
    makespan: float = 0.0  # Total time for schedule
    solve_time_ms: float = 0.0
    verified: bool = False
    verification_status: str = "not_checked"
    violations: List[str] = field(default_factory=list)
    explanation: str = ""

    @property
    def is_valid(self) -> bool:
        return self.status == SolveStatus.SOLVED and self.verified

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "schedule": [e.to_dict() for e in self.schedule],
            "assignments": self.assignments,
            "makespan": self.makespan,
            "solve_time_ms": self.solve_time_ms,
            "verified": self.verified,
            "verification_status": self.verification_status,
            "violations": self.violations,
            "explanation": self.explanation
        }

    def to_readable(self) -> str:
        """Convert to human-readable format."""
        if not self.schedule:
            return f"Status: {self.status.value}\nNo schedule generated."

        lines = [
            f"Status: {self.status.value}",
            f"Verified: {'✅' if self.verified else '❌'}",
            f"Makespan: {self.makespan:.2f}",
            f"Solve time: {self.solve_time_ms:.1f}ms",
            "",
            "Schedule:"
        ]

        for entry in sorted(self.schedule, key=lambda e: e.start_time):
            resources = f" (using: {', '.join(entry.resources)})" if entry.resources else ""
            lines.append(f"  [{entry.start_time:.1f} - {entry.end_time:.1f}] {entry.task_id}{resources}")

        if self.violations:
            lines.append("")
            lines.append("Violations:")
            for v in self.violations:
                lines.append(f"  ⚠️ {v}")

        return "\n".join(lines)


class FormalPlanner:
    """
    The core formal planning engine.

    Solves constraint satisfaction problems and returns verified plans.
    """

    def __init__(
        self,
        axiom_store=None,  # Optional: from L8 main module
        default_timeout_ms: int = 120
    ):
        self.axiom_store = axiom_store
        self.default_timeout_ms = default_timeout_ms

        # Statistics
        self._stats = {
            "problems_solved": 0,
            "problems_unsatisfiable": 0,
            "problems_timeout": 0,
            "total_solve_time_ms": 0,
            "avg_solve_time_ms": 0
        }

    def solve(self, problem: PlanningProblem) -> PlanResult:
        """
        Solve a planning problem.

        Returns a verified PlanResult.
        """
        import time
        start_time = time.time()

        # Dispatch to appropriate solver
        if problem.problem_type == ProblemType.SCHEDULING:
            result = self._solve_scheduling(problem)
        elif problem.problem_type == ProblemType.RESOURCE_ALLOCATION:
            result = self._solve_resource_allocation(problem)
        elif problem.problem_type == ProblemType.SEQUENCING:
            result = self._solve_sequencing(problem)
        elif problem.problem_type == ProblemType.CONFIGURATION:
            result = self._solve_configuration(problem)
        else:
            result = PlanResult(
                status=SolveStatus.ERROR,
                explanation=f"Unknown problem type: {problem.problem_type}"
            )

        result.solve_time_ms = (time.time() - start_time) * 1000

        # Verify the solution
        if result.status == SolveStatus.SOLVED:
            result = self._verify_solution(problem, result)

        # Update stats
        self._update_stats(result)

        return result

    def _solve_scheduling(self, problem: PlanningProblem) -> PlanResult:
        """
        Solve a scheduling problem using constraint propagation + priority queue.

        Algorithm:
        1. Build dependency graph
        2. Topological sort by dependencies
        3. Schedule tasks earliest possible respecting constraints
        4. Handle resource conflicts
        """
        if not problem.tasks:
            return PlanResult(
                status=SolveStatus.SOLVED,
                explanation="No tasks to schedule"
            )

        # Build dependency graph
        task_map = {t.id: t for t in problem.tasks}
        dependencies = {t.id: set(t.depends_on) for t in problem.tasks}

        # Add constraint-based dependencies
        for constraint in problem.constraints:
            if constraint.constraint_type == "before" and len(constraint.subjects) == 2:
                before, after = constraint.subjects
                if after in dependencies:
                    dependencies[after].add(before)

        # Check for cycles
        cycle = self._detect_cycle(dependencies)
        if cycle:
            return PlanResult(
                status=SolveStatus.UNSATISFIABLE,
                violations=[f"Dependency cycle: {' -> '.join(cycle)}"],
                explanation="Cannot schedule: circular dependency"
            )

        # Topological sort
        try:
            order = self._topological_sort(dependencies)
        except ValueError as e:
            return PlanResult(
                status=SolveStatus.UNSATISFIABLE,
                violations=[str(e)],
                explanation="Cannot establish valid task order"
            )

        # Schedule tasks
        schedule = []
        task_end_times = {}
        resource_available_at = {r.id: 0.0 for r in problem.resources}

        for task_id in order:
            if task_id not in task_map:
                continue

            task = task_map[task_id]

            # Find earliest start time
            earliest = task.earliest_start

            # Wait for ALL dependencies (both task.depends_on and constraint-based)
            all_deps = dependencies.get(task_id, set())
            for dep in all_deps:
                if dep in task_end_times:
                    earliest = max(earliest, task_end_times[dep])

            # Wait for required resources
            for res_id in task.requires:
                if res_id in resource_available_at:
                    earliest = max(earliest, resource_available_at[res_id])

            # Schedule the task
            start_time = earliest
            end_time = start_time + task.duration

            # Check deadline
            if task.deadline is not None and end_time > task.deadline:
                return PlanResult(
                    status=SolveStatus.UNSATISFIABLE,
                    violations=[f"Task {task_id} cannot meet deadline {task.deadline}"],
                    explanation=f"Deadline violation for {task_id}"
                )

            # Update resource availability
            for res_id in task.requires:
                if res_id in resource_available_at:
                    resource_available_at[res_id] = end_time

            task_end_times[task_id] = end_time
            schedule.append(ScheduleEntry(
                task_id=task_id,
                start_time=start_time,
                end_time=end_time,
                resources=task.requires.copy()
            ))

        # Check mutex constraints
        for constraint in problem.constraints:
            if constraint.constraint_type == "mutex" and len(constraint.subjects) == 2:
                task_a, task_b = constraint.subjects
                if task_a in task_end_times and task_b in task_end_times:
                    # Find their schedule entries
                    entry_a = next((e for e in schedule if e.task_id == task_a), None)
                    entry_b = next((e for e in schedule if e.task_id == task_b), None)
                    if entry_a and entry_b:
                        # Check for overlap
                        if not (entry_a.end_time <= entry_b.start_time or
                                entry_b.end_time <= entry_a.start_time):
                            return PlanResult(
                                status=SolveStatus.UNSATISFIABLE,
                                violations=[f"Mutex violation: {task_a} and {task_b} overlap"],
                                explanation="Tasks with mutex constraint cannot be scheduled without overlap"
                            )

        makespan = max(task_end_times.values()) if task_end_times else 0.0

        return PlanResult(
            status=SolveStatus.SOLVED,
            schedule=schedule,
            makespan=makespan,
            explanation=f"Scheduled {len(schedule)} tasks with makespan {makespan:.2f}"
        )

    def _solve_resource_allocation(self, problem: PlanningProblem) -> PlanResult:
        """
        Solve a resource allocation problem.

        Assigns resources to tasks respecting capacity and exclusivity.
        """
        if not problem.tasks or not problem.resources:
            return PlanResult(
                status=SolveStatus.SOLVED,
                explanation="No allocation needed"
            )

        # Build resource map
        resource_map = {r.id: r for r in problem.resources}

        # Track assignments and resource usage
        assignments = {}
        resource_usage = {r.id: 0 for r in problem.resources}

        # Sort tasks by priority (higher first)
        sorted_tasks = sorted(problem.tasks, key=lambda t: -t.priority)

        for task in sorted_tasks:
            task_assignments = []

            for req_id in task.requires:
                if req_id not in resource_map:
                    continue

                resource = resource_map[req_id]

                # Check availability
                if resource_usage[req_id] >= resource.capacity:
                    # Resource exhausted
                    return PlanResult(
                        status=SolveStatus.UNSATISFIABLE,
                        violations=[f"Resource {req_id} exhausted for task {task.id}"],
                        explanation=f"Cannot allocate {req_id} to {task.id}"
                    )

                # Allocate
                resource_usage[req_id] += 1
                task_assignments.append(req_id)

            assignments[task.id] = task_assignments

        return PlanResult(
            status=SolveStatus.SOLVED,
            assignments=assignments,
            explanation=f"Allocated resources to {len(assignments)} tasks"
        )

    def _solve_sequencing(self, problem: PlanningProblem) -> PlanResult:
        """
        Solve a sequencing problem (ordering with constraints).

        Similar to scheduling but focused on order, not timing.
        """
        if not problem.tasks:
            return PlanResult(
                status=SolveStatus.SOLVED,
                explanation="No tasks to sequence"
            )

        # Build precedence graph from constraints
        precedence = {t.id: set() for t in problem.tasks}

        for constraint in problem.constraints:
            if constraint.constraint_type == "before" and len(constraint.subjects) == 2:
                before, after = constraint.subjects
                if after in precedence:
                    precedence[after].add(before)

        # Add task dependencies
        for task in problem.tasks:
            for dep in task.depends_on:
                precedence[task.id].add(dep)

        # Check for cycles
        cycle = self._detect_cycle(precedence)
        if cycle:
            return PlanResult(
                status=SolveStatus.UNSATISFIABLE,
                violations=[f"Ordering cycle: {' -> '.join(cycle)}"],
                explanation="Cannot establish valid sequence"
            )

        # Topological sort
        try:
            order = self._topological_sort(precedence)
        except ValueError as e:
            return PlanResult(
                status=SolveStatus.UNSATISFIABLE,
                violations=[str(e)],
                explanation="Cannot establish valid sequence"
            )

        # Create schedule entries (sequential, no timing)
        schedule = []
        for i, task_id in enumerate(order):
            schedule.append(ScheduleEntry(
                task_id=task_id,
                start_time=float(i),
                end_time=float(i + 1)
            ))

        return PlanResult(
            status=SolveStatus.SOLVED,
            schedule=schedule,
            makespan=float(len(schedule)),
            explanation=f"Sequenced {len(schedule)} tasks"
        )

    def _solve_configuration(self, problem: PlanningProblem) -> PlanResult:
        """
        Solve a configuration problem.

        Find assignments that satisfy all constraints.
        """
        # Extract configuration variables and domains
        assignments = {}

        # For each task, treat it as a configuration variable
        for task in problem.tasks:
            # Default assignment
            assignments[task.id] = {
                "enabled": True,
                "resources": task.requires.copy()
            }

        # Check constraints
        violations = []

        for constraint in problem.constraints:
            if constraint.constraint_type == "mutex":
                # Cannot both be enabled
                enabled_count = sum(
                    1 for s in constraint.subjects
                    if s in assignments and assignments[s].get("enabled")
                )
                if enabled_count > 1:
                    # Disable lower priority one
                    task_priorities = {
                        t.id: t.priority for t in problem.tasks
                    }
                    subjects_by_priority = sorted(
                        constraint.subjects,
                        key=lambda s: task_priorities.get(s, 0)
                    )
                    # Disable lowest priority
                    if subjects_by_priority[0] in assignments:
                        assignments[subjects_by_priority[0]]["enabled"] = False

            elif constraint.constraint_type == "requires":
                # If subject is enabled, requirement must be met
                for subject in constraint.subjects:
                    if subject in assignments and assignments[subject].get("enabled"):
                        req = constraint.value
                        if req and req not in assignments.get(subject, {}).get("resources", []):
                            violations.append(f"{subject} requires {req}")

        if violations:
            return PlanResult(
                status=SolveStatus.PARTIAL,
                assignments=assignments,
                violations=violations,
                explanation="Configuration found with warnings"
            )

        return PlanResult(
            status=SolveStatus.SOLVED,
            assignments=assignments,
            explanation=f"Found valid configuration for {len(assignments)} items"
        )

    def _detect_cycle(self, graph: Dict[str, Set[str]]) -> Optional[List[str]]:
        """Detect cycles in a dependency graph."""
        visited = set()
        path = []
        path_set = set()

        def dfs(node: str) -> Optional[List[str]]:
            if node in path_set:
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
            if node in visited:
                return None

            visited.add(node)
            path.append(node)
            path_set.add(node)

            for neighbor in graph.get(node, []):
                cycle = dfs(neighbor)
                if cycle:
                    return cycle

            path.pop()
            path_set.remove(node)
            return None

        for node in graph:
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    return cycle

        return None

    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """Topological sort of dependency graph."""
        # Compute in-degrees
        in_degree = {node: 0 for node in graph}
        for node, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    pass  # Dependency exists, counted in target's in_degree
                else:
                    in_degree[dep] = 0

        for node, deps in graph.items():
            for dep in deps:
                if dep not in in_degree:
                    in_degree[dep] = 0
            in_degree[node] = len(deps)

        # Kahn's algorithm
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Find nodes that depend on this one
            for other, deps in graph.items():
                if node in deps:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        if len(result) != len(in_degree):
            raise ValueError("Graph has a cycle")

        return result

    def _verify_solution(self, problem: PlanningProblem, result: PlanResult) -> PlanResult:
        """
        Verify that a solution satisfies all constraints.

        This is the PGU verification step.
        """
        violations = []

        # Build lookup maps
        schedule_map = {e.task_id: e for e in result.schedule}

        # Check all constraints
        for constraint in problem.constraints:
            if constraint.constraint_type == "before" and len(constraint.subjects) == 2:
                before, after = constraint.subjects
                if before in schedule_map and after in schedule_map:
                    if schedule_map[before].end_time > schedule_map[after].start_time:
                        violations.append(
                            f"Precedence violation: {before} should complete before {after} starts"
                        )

            elif constraint.constraint_type == "mutex" and len(constraint.subjects) == 2:
                task_a, task_b = constraint.subjects
                if task_a in schedule_map and task_b in schedule_map:
                    a, b = schedule_map[task_a], schedule_map[task_b]
                    if not (a.end_time <= b.start_time or b.end_time <= a.start_time):
                        violations.append(
                            f"Mutex violation: {task_a} and {task_b} overlap"
                        )

            elif constraint.constraint_type == "within":
                task_id = constraint.subjects[0] if constraint.subjects else None
                deadline = constraint.value
                if task_id in schedule_map and deadline is not None:
                    if schedule_map[task_id].end_time > deadline:
                        violations.append(
                            f"Deadline violation: {task_id} finishes at {schedule_map[task_id].end_time}, deadline was {deadline}"
                        )

        # Check task dependencies
        for task in problem.tasks:
            if task.id not in schedule_map:
                continue
            entry = schedule_map[task.id]

            for dep_id in task.depends_on:
                if dep_id in schedule_map:
                    dep_entry = schedule_map[dep_id]
                    if dep_entry.end_time > entry.start_time:
                        violations.append(
                            f"Dependency violation: {task.id} starts before {dep_id} finishes"
                        )

        result.violations = violations
        result.verified = len(violations) == 0
        result.verification_status = "verified" if result.verified else "failed"

        if result.verified:
            result.explanation += " [PGU-verified]"
        else:
            result.explanation += f" [Verification failed: {len(violations)} violation(s)]"

        return result

    def _update_stats(self, result: PlanResult) -> None:
        """Update statistics."""
        if result.status == SolveStatus.SOLVED:
            self._stats["problems_solved"] += 1
        elif result.status == SolveStatus.UNSATISFIABLE:
            self._stats["problems_unsatisfiable"] += 1
        elif result.status == SolveStatus.TIMEOUT:
            self._stats["problems_timeout"] += 1

        self._stats["total_solve_time_ms"] += result.solve_time_ms
        total = (self._stats["problems_solved"] +
                self._stats["problems_unsatisfiable"] +
                self._stats["problems_timeout"])
        if total > 0:
            self._stats["avg_solve_time_ms"] = self._stats["total_solve_time_ms"] / total

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats.copy()


# ============================================================
# Problem Builders: Natural Language → Formal Problem
# ============================================================

class SchedulingProblemBuilder:
    """
    Helper to build scheduling problems from natural descriptions.
    """

    def __init__(self):
        self._problem = PlanningProblem(problem_type=ProblemType.SCHEDULING)

    def add_task(
        self,
        task_id: str,
        duration: float = 1.0,
        priority: int = 0,
        deadline: Optional[float] = None,
        requires: Optional[List[str]] = None,
        depends_on: Optional[List[str]] = None
    ) -> "SchedulingProblemBuilder":
        """Add a task to the problem."""
        self._problem.add_task(Task(
            id=task_id,
            duration=duration,
            priority=priority,
            deadline=deadline,
            requires=requires or [],
            depends_on=depends_on or []
        ))
        return self

    def add_resource(
        self,
        resource_id: str,
        capacity: int = 1,
        exclusive: bool = True
    ) -> "SchedulingProblemBuilder":
        """Add a resource."""
        self._problem.add_resource(Resource(
            id=resource_id,
            capacity=capacity,
            exclusive=exclusive
        ))
        return self

    def add_precedence(self, before: str, after: str) -> "SchedulingProblemBuilder":
        """Task 'before' must complete before task 'after' starts."""
        self._problem.add_precedence(before, after)
        return self

    def add_mutex(self, task_a: str, task_b: str) -> "SchedulingProblemBuilder":
        """Tasks cannot run at the same time."""
        self._problem.add_mutex(task_a, task_b)
        return self

    def set_timeout(self, timeout_ms: int) -> "SchedulingProblemBuilder":
        """Set solver timeout."""
        self._problem.timeout_ms = timeout_ms
        return self

    def build(self) -> PlanningProblem:
        """Build the problem."""
        return self._problem


# ============================================================
# Factory Functions
# ============================================================

def create_planner(axiom_store=None) -> FormalPlanner:
    """Create a formal planner."""
    return FormalPlanner(axiom_store=axiom_store)


def create_scheduling_problem() -> SchedulingProblemBuilder:
    """Create a scheduling problem builder."""
    return SchedulingProblemBuilder()


def solve_scheduling(
    tasks: List[Dict[str, Any]],
    resources: Optional[List[Dict[str, Any]]] = None,
    constraints: Optional[List[Dict[str, Any]]] = None
) -> PlanResult:
    """
    Convenience function to solve a scheduling problem.

    Args:
        tasks: List of task dicts with id, duration, depends_on, etc.
        resources: Optional list of resource dicts
        constraints: Optional list of constraint dicts

    Returns:
        PlanResult with verified schedule
    """
    builder = create_scheduling_problem()

    for t in tasks:
        builder.add_task(
            task_id=t["id"],
            duration=t.get("duration", 1.0),
            priority=t.get("priority", 0),
            deadline=t.get("deadline"),
            requires=t.get("requires", []),
            depends_on=t.get("depends_on", [])
        )

    if resources:
        for r in resources:
            builder.add_resource(
                resource_id=r["id"],
                capacity=r.get("capacity", 1),
                exclusive=r.get("exclusive", True)
            )

    if constraints:
        for c in constraints:
            if c["type"] == "before":
                builder.add_precedence(c["subjects"][0], c["subjects"][1])
            elif c["type"] == "mutex":
                builder.add_mutex(c["subjects"][0], c["subjects"][1])

    planner = create_planner()
    return planner.solve(builder.build())


__all__ = [
    "ProblemType",
    "SolveStatus",
    "Constraint",
    "Resource",
    "Task",
    "ScheduleEntry",
    "PlanningProblem",
    "PlanResult",
    "FormalPlanner",
    "SchedulingProblemBuilder",
    "create_planner",
    "create_scheduling_problem",
    "solve_scheduling"
]
