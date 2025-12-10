# ara_hive/patterns/aflow.py
"""
AFlow - Automated Workflow Generation
=====================================

AFlow is inspired by MetaGPT's AFlow framework for automated
workflow generation through Monte Carlo Tree Search (MCTS).

Key Concepts:
    - WorkflowCandidate: A potential workflow configuration
    - Mutation: Changes to a workflow (add step, remove step, reorder)
    - Search: Explore workflow space to find optimal configurations
    - Evaluation: Score workflows based on performance

The search process:
    1. Start with base workflow
    2. Generate mutations (variations)
    3. Evaluate candidates
    4. Keep best performers
    5. Repeat until convergence

This enables:
    - Automatic SOP optimization
    - Task-specific workflow generation
    - Continuous improvement from execution feedback

Usage:
    from ara_hive.patterns.aflow import AFlowGenerator, AFlowConfig

    generator = AFlowGenerator(config=AFlowConfig(
        max_iterations=50,
        population_size=10,
    ))

    # Generate optimized workflow
    workflow = await generator.generate(
        task_type="code_review",
        seed_workflow=base_workflow,
        evaluation_fn=my_evaluator,
    )
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..src.queen import QueenOrchestrator
    from .sop import SOP, SOPStep, SOPPhase

log = logging.getLogger("Hive.Patterns.AFlow")


# =============================================================================
# Types
# =============================================================================

class MutationType(str, Enum):
    """Types of workflow mutations."""
    ADD_STEP = "add_step"              # Add a new step
    REMOVE_STEP = "remove_step"        # Remove an existing step
    REORDER_STEPS = "reorder_steps"    # Change step order
    MODIFY_STEP = "modify_step"        # Change step parameters
    ADD_PHASE = "add_phase"            # Add a new phase
    REMOVE_PHASE = "remove_phase"      # Remove a phase
    MERGE_PHASES = "merge_phases"      # Combine two phases
    SPLIT_PHASE = "split_phase"        # Split a phase
    ADD_BRANCH = "add_branch"          # Add conditional branch
    ADD_LOOP = "add_loop"              # Add iteration
    PARALLELIZE = "parallelize"        # Make steps parallel
    SERIALIZE = "serialize"            # Make steps sequential


class SearchStrategy(str, Enum):
    """Search strategies for workflow generation."""
    RANDOM = "random"                  # Random mutations
    GREEDY = "greedy"                  # Always take best
    MCTS = "mcts"                      # Monte Carlo Tree Search
    EVOLUTIONARY = "evolutionary"      # Genetic algorithm style
    BEAM = "beam"                      # Beam search


# =============================================================================
# Workflow Candidate
# =============================================================================

@dataclass
class WorkflowCandidate:
    """
    A candidate workflow under evaluation.

    Tracks the workflow configuration and its performance metrics.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    workflow: Optional[SOP] = None

    # Performance metrics
    score: float = 0.0
    success_rate: float = 0.0
    avg_duration_ms: float = 0.0
    cost: float = 0.0

    # Evaluation history
    evaluations: int = 0
    successes: int = 0
    failures: int = 0

    # Genealogy
    parent_id: Optional[str] = None
    mutation: Optional[MutationType] = None
    generation: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_evaluated: Optional[datetime] = None

    def update_metrics(
        self,
        success: bool,
        duration_ms: float,
        score: float = 0.0,
    ) -> None:
        """Update metrics after an evaluation."""
        self.evaluations += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1

        self.success_rate = self.successes / self.evaluations
        self.avg_duration_ms = (
            (self.avg_duration_ms * (self.evaluations - 1) + duration_ms)
            / self.evaluations
        )

        # Update aggregate score
        self.score = (
            0.5 * score +
            0.3 * self.success_rate +
            0.2 * (1.0 - min(duration_ms / 10000, 1.0))  # Normalize duration
        )

        self.last_evaluated = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "success_rate": self.success_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "evaluations": self.evaluations,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "mutation": self.mutation.value if self.mutation else None,
        }


# =============================================================================
# Mutations
# =============================================================================

@dataclass
class WorkflowMutation:
    """
    A mutation operation on a workflow.

    Mutations are the building blocks for workflow evolution.
    """
    mutation_type: MutationType
    target: Optional[str] = None  # Phase or step name
    params: Dict[str, Any] = field(default_factory=dict)

    def apply(self, workflow: SOP) -> SOP:
        """
        Apply mutation to a workflow.

        Returns a new workflow with the mutation applied.
        """
        from .sop import SOP, SOPPhase, SOPStep, SOPStepType, SOPPhaseType
        import copy

        # Deep copy to avoid modifying original
        new_workflow = copy.deepcopy(workflow)

        if self.mutation_type == MutationType.ADD_STEP:
            self._add_step(new_workflow)
        elif self.mutation_type == MutationType.REMOVE_STEP:
            self._remove_step(new_workflow)
        elif self.mutation_type == MutationType.REORDER_STEPS:
            self._reorder_steps(new_workflow)
        elif self.mutation_type == MutationType.MODIFY_STEP:
            self._modify_step(new_workflow)
        elif self.mutation_type == MutationType.ADD_PHASE:
            self._add_phase(new_workflow)
        elif self.mutation_type == MutationType.REMOVE_PHASE:
            self._remove_phase(new_workflow)
        elif self.mutation_type == MutationType.PARALLELIZE:
            self._parallelize(new_workflow)
        elif self.mutation_type == MutationType.SERIALIZE:
            self._serialize(new_workflow)

        return new_workflow

    def _add_step(self, workflow: SOP) -> None:
        """Add a new step to a phase."""
        from .sop import SOPStep, SOPStepType

        phase_name = self.target or (
            workflow.phases[-1].name if workflow.phases else None
        )
        if not phase_name:
            return

        phase = workflow.get_phase(phase_name)
        if not phase:
            return

        step_name = self.params.get("step_name", f"step_{len(phase.steps)}")
        action = self.params.get("action", SOPStepType.GENERATE)
        tool = self.params.get("tool")

        new_step = SOPStep(
            name=step_name,
            action=action,
            tool=tool,
            description=self.params.get("description", ""),
        )
        phase.steps.append(new_step)

    def _remove_step(self, workflow: SOP) -> None:
        """Remove a step from a phase."""
        if not self.target:
            return

        for phase in workflow.phases:
            phase.steps = [s for s in phase.steps if s.name != self.target]

    def _reorder_steps(self, workflow: SOP) -> None:
        """Reorder steps within a phase."""
        phase_name = self.target or (
            workflow.phases[0].name if workflow.phases else None
        )
        if not phase_name:
            return

        phase = workflow.get_phase(phase_name)
        if phase and len(phase.steps) > 1:
            # Random shuffle or specific order
            order = self.params.get("order")
            if order:
                step_map = {s.name: s for s in phase.steps}
                phase.steps = [step_map[name] for name in order if name in step_map]
            else:
                random.shuffle(phase.steps)

    def _modify_step(self, workflow: SOP) -> None:
        """Modify step parameters."""
        if not self.target:
            return

        for phase in workflow.phases:
            for step in phase.steps:
                if step.name == self.target:
                    # Update parameters
                    if "timeout_seconds" in self.params:
                        step.timeout_seconds = self.params["timeout_seconds"]
                    if "max_retries" in self.params:
                        step.max_retries = self.params["max_retries"]
                    if "tool" in self.params:
                        step.tool = self.params["tool"]
                    return

    def _add_phase(self, workflow: SOP) -> None:
        """Add a new phase."""
        from .sop import SOPPhase, SOPPhaseType

        phase_name = self.params.get("phase_name", f"phase_{len(workflow.phases)}")
        phase_type = self.params.get("phase_type", SOPPhaseType.EXECUTION)

        new_phase = SOPPhase(
            name=phase_name,
            phase_type=phase_type,
            steps=[],
        )

        # Insert at position or append
        position = self.params.get("position", len(workflow.phases))
        workflow.phases.insert(position, new_phase)

    def _remove_phase(self, workflow: SOP) -> None:
        """Remove a phase."""
        if not self.target:
            return

        workflow.phases = [p for p in workflow.phases if p.name != self.target]

    def _parallelize(self, workflow: SOP) -> None:
        """Make a phase's steps run in parallel."""
        phase_name = self.target or (
            workflow.phases[0].name if workflow.phases else None
        )
        if not phase_name:
            return

        phase = workflow.get_phase(phase_name)
        if phase:
            phase.parallel_steps = True

    def _serialize(self, workflow: SOP) -> None:
        """Make a phase's steps run sequentially."""
        phase_name = self.target or (
            workflow.phases[0].name if workflow.phases else None
        )
        if not phase_name:
            return

        phase = workflow.get_phase(phase_name)
        if phase:
            phase.parallel_steps = False


class MutationGenerator:
    """Generates random mutations for workflows."""

    def __init__(self, allowed_mutations: Optional[List[MutationType]] = None):
        self.allowed_mutations = allowed_mutations or list(MutationType)

    def generate(
        self,
        workflow: SOP,
        num_mutations: int = 1,
    ) -> List[WorkflowMutation]:
        """Generate random mutations for a workflow."""
        mutations = []

        for _ in range(num_mutations):
            mutation_type = random.choice(self.allowed_mutations)
            mutation = self._create_mutation(workflow, mutation_type)
            if mutation:
                mutations.append(mutation)

        return mutations

    def _create_mutation(
        self,
        workflow: SOP,
        mutation_type: MutationType,
    ) -> Optional[WorkflowMutation]:
        """Create a specific mutation."""
        from .sop import SOPStepType, SOPPhaseType

        if mutation_type == MutationType.ADD_STEP:
            if not workflow.phases:
                return None
            phase = random.choice(workflow.phases)
            return WorkflowMutation(
                mutation_type=mutation_type,
                target=phase.name,
                params={
                    "step_name": f"step_{random.randint(100, 999)}",
                    "action": random.choice(list(SOPStepType)),
                },
            )

        elif mutation_type == MutationType.REMOVE_STEP:
            all_steps = workflow.get_all_steps()
            if not all_steps:
                return None
            step = random.choice(all_steps)
            return WorkflowMutation(
                mutation_type=mutation_type,
                target=step.name,
            )

        elif mutation_type == MutationType.REORDER_STEPS:
            if not workflow.phases:
                return None
            phase = random.choice(workflow.phases)
            if len(phase.steps) < 2:
                return None
            return WorkflowMutation(
                mutation_type=mutation_type,
                target=phase.name,
            )

        elif mutation_type == MutationType.MODIFY_STEP:
            all_steps = workflow.get_all_steps()
            if not all_steps:
                return None
            step = random.choice(all_steps)
            return WorkflowMutation(
                mutation_type=mutation_type,
                target=step.name,
                params={
                    "timeout_seconds": random.choice([30, 60, 120, 300]),
                    "max_retries": random.choice([0, 1, 2, 3]),
                },
            )

        elif mutation_type == MutationType.ADD_PHASE:
            return WorkflowMutation(
                mutation_type=mutation_type,
                params={
                    "phase_name": f"phase_{random.randint(100, 999)}",
                    "phase_type": random.choice(list(SOPPhaseType)),
                    "position": random.randint(0, len(workflow.phases)),
                },
            )

        elif mutation_type == MutationType.REMOVE_PHASE:
            if len(workflow.phases) < 2:  # Keep at least one phase
                return None
            phase = random.choice(workflow.phases)
            return WorkflowMutation(
                mutation_type=mutation_type,
                target=phase.name,
            )

        elif mutation_type in [MutationType.PARALLELIZE, MutationType.SERIALIZE]:
            if not workflow.phases:
                return None
            phase = random.choice(workflow.phases)
            return WorkflowMutation(
                mutation_type=mutation_type,
                target=phase.name,
            )

        return None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AFlowConfig:
    """Configuration for AFlow generator."""
    # Search parameters
    max_iterations: int = 50
    population_size: int = 10
    elite_size: int = 2  # Top candidates to preserve
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2

    # Search strategy
    strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY

    # Evaluation
    evaluations_per_candidate: int = 3
    evaluation_timeout_seconds: int = 300

    # Convergence
    convergence_threshold: float = 0.01  # Stop if improvement < this
    convergence_window: int = 10  # Iterations to check for convergence

    # Mutation settings
    mutations_per_generation: int = 2
    allowed_mutations: Optional[List[MutationType]] = None


# =============================================================================
# AFlow Generator
# =============================================================================

class AFlowGenerator:
    """
    Generates and optimizes workflows using search.

    Uses evolutionary algorithms to explore the workflow space
    and find configurations that maximize performance.
    """

    def __init__(
        self,
        config: Optional[AFlowConfig] = None,
        queen: Optional[QueenOrchestrator] = None,
    ):
        self.config = config or AFlowConfig()
        self.queen = queen
        self.mutation_generator = MutationGenerator(
            allowed_mutations=self.config.allowed_mutations
        )

        # Search state
        self.population: List[WorkflowCandidate] = []
        self.best_candidate: Optional[WorkflowCandidate] = None
        self.history: List[Dict[str, Any]] = []

    async def generate(
        self,
        task_type: str,
        seed_workflow: Optional[SOP] = None,
        evaluation_fn: Optional[Callable[[SOP], Tuple[bool, float, float]]] = None,
    ) -> SOP:
        """
        Generate an optimized workflow.

        Args:
            task_type: Type of task the workflow is for
            seed_workflow: Starting workflow to evolve
            evaluation_fn: Function to evaluate workflows (success, score, duration)

        Returns:
            Optimized SOP workflow
        """
        log.info(f"AFlow starting generation for task_type='{task_type}'")

        # Initialize population
        self._initialize_population(seed_workflow)

        # Evaluation function
        evaluator = evaluation_fn or self._default_evaluator

        # Evolution loop
        for iteration in range(self.config.max_iterations):
            log.debug(f"AFlow iteration {iteration + 1}/{self.config.max_iterations}")

            # Evaluate population
            await self._evaluate_population(evaluator)

            # Select best
            self._update_best()

            # Record history
            self._record_iteration(iteration)

            # Check convergence
            if self._check_convergence():
                log.info(f"AFlow converged at iteration {iteration + 1}")
                break

            # Generate next generation
            self._evolve_population()

        log.info(
            f"AFlow completed: best_score={self.best_candidate.score:.3f}, "
            f"generations={len(self.history)}"
        )

        return self.best_candidate.workflow

    def _initialize_population(self, seed_workflow: Optional[SOP]) -> None:
        """Initialize the population with seed and variations."""
        from .sop import SOP, SOPPhase, SOPStep, SOPStepType, SOPPhaseType

        self.population = []

        # Create seed if not provided
        if not seed_workflow:
            seed_workflow = SOP(
                name="generated_workflow",
                phases=[
                    SOPPhase(
                        name="main",
                        phase_type=SOPPhaseType.EXECUTION,
                        steps=[
                            SOPStep(name="step_1", action=SOPStepType.GENERATE),
                        ],
                    ),
                ],
            )

        # Add seed to population
        seed_candidate = WorkflowCandidate(
            workflow=seed_workflow,
            generation=0,
        )
        self.population.append(seed_candidate)

        # Generate variations
        for i in range(self.config.population_size - 1):
            mutations = self.mutation_generator.generate(
                seed_workflow,
                num_mutations=self.config.mutations_per_generation,
            )

            mutated = seed_workflow
            for mutation in mutations:
                mutated = mutation.apply(mutated)

            candidate = WorkflowCandidate(
                workflow=mutated,
                parent_id=seed_candidate.id,
                mutation=mutations[0].mutation_type if mutations else None,
                generation=0,
            )
            self.population.append(candidate)

    async def _evaluate_population(
        self,
        evaluator: Callable[[SOP], Tuple[bool, float, float]],
    ) -> None:
        """Evaluate all candidates in the population."""
        for candidate in self.population:
            if candidate.evaluations >= self.config.evaluations_per_candidate:
                continue  # Already evaluated enough

            try:
                # Run evaluation
                success, score, duration_ms = await asyncio.wait_for(
                    self._run_evaluation(candidate.workflow, evaluator),
                    timeout=self.config.evaluation_timeout_seconds,
                )

                candidate.update_metrics(success, duration_ms, score)

            except asyncio.TimeoutError:
                log.warning(f"Candidate {candidate.id} evaluation timed out")
                candidate.update_metrics(False, self.config.evaluation_timeout_seconds * 1000, 0.0)

            except Exception as e:
                log.warning(f"Candidate {candidate.id} evaluation failed: {e}")
                candidate.update_metrics(False, 0.0, 0.0)

    async def _run_evaluation(
        self,
        workflow: SOP,
        evaluator: Callable[[SOP], Tuple[bool, float, float]],
    ) -> Tuple[bool, float, float]:
        """Run a single evaluation."""
        if asyncio.iscoroutinefunction(evaluator):
            return await evaluator(workflow)
        else:
            return evaluator(workflow)

    def _default_evaluator(self, workflow: SOP) -> Tuple[bool, float, float]:
        """Default workflow evaluator."""
        # Simple heuristic-based evaluation
        errors = workflow.validate()
        if errors:
            return False, 0.0, 0.0

        # Score based on structure
        num_steps = len(workflow.get_all_steps())
        num_phases = len(workflow.phases)

        # Prefer workflows with 3-10 steps
        step_score = 1.0 - abs(num_steps - 5) / 10.0
        step_score = max(0.0, min(1.0, step_score))

        # Prefer 2-4 phases
        phase_score = 1.0 - abs(num_phases - 3) / 5.0
        phase_score = max(0.0, min(1.0, phase_score))

        score = 0.6 * step_score + 0.4 * phase_score
        duration_ms = num_steps * 100  # Estimate

        return True, score, duration_ms

    def _update_best(self) -> None:
        """Update best candidate."""
        for candidate in self.population:
            if self.best_candidate is None or candidate.score > self.best_candidate.score:
                self.best_candidate = candidate

    def _record_iteration(self, iteration: int) -> None:
        """Record iteration history."""
        scores = [c.score for c in self.population]
        self.history.append({
            "iteration": iteration,
            "best_score": max(scores) if scores else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "population_size": len(self.population),
            "best_id": self.best_candidate.id if self.best_candidate else None,
        })

    def _check_convergence(self) -> bool:
        """Check if search has converged."""
        if len(self.history) < self.config.convergence_window:
            return False

        recent = self.history[-self.config.convergence_window:]
        improvements = [
            recent[i + 1]["best_score"] - recent[i]["best_score"]
            for i in range(len(recent) - 1)
        ]

        avg_improvement = sum(improvements) / len(improvements)
        return avg_improvement < self.config.convergence_threshold

    def _evolve_population(self) -> None:
        """Create next generation through selection and mutation."""
        # Sort by score
        self.population.sort(key=lambda c: c.score, reverse=True)

        # Keep elite
        new_population = self.population[:self.config.elite_size]
        generation = (self.population[0].generation if self.population else 0) + 1

        # Fill rest with mutations of best candidates
        while len(new_population) < self.config.population_size:
            # Select parent from top half
            parent_idx = random.randint(0, len(self.population) // 2)
            parent = self.population[parent_idx]

            if random.random() < self.config.mutation_rate:
                # Mutate
                mutations = self.mutation_generator.generate(
                    parent.workflow,
                    num_mutations=self.config.mutations_per_generation,
                )

                mutated = parent.workflow
                for mutation in mutations:
                    mutated = mutation.apply(mutated)

                child = WorkflowCandidate(
                    workflow=mutated,
                    parent_id=parent.id,
                    mutation=mutations[0].mutation_type if mutations else None,
                    generation=generation,
                )
            else:
                # Clone parent
                import copy
                child = WorkflowCandidate(
                    workflow=copy.deepcopy(parent.workflow),
                    parent_id=parent.id,
                    generation=generation,
                )

            new_population.append(child)

        self.population = new_population

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "population_size": len(self.population),
            "generations": len(self.history),
            "best_score": self.best_candidate.score if self.best_candidate else 0.0,
            "best_id": self.best_candidate.id if self.best_candidate else None,
            "history": self.history[-10:],  # Last 10 iterations
        }


# =============================================================================
# Workflow Templates
# =============================================================================

def generate_template_workflow(
    task_type: str,
    complexity: str = "medium",
) -> SOP:
    """
    Generate a template workflow for a task type.

    Args:
        task_type: Type of task (code, research, content, etc.)
        complexity: simple, medium, complex

    Returns:
        Template SOP
    """
    from .sop import SOP, SOPPhase, SOPStep, SOPStepType, SOPPhaseType

    # Define templates
    templates = {
        "code": {
            "simple": [
                ("implement", SOPPhaseType.EXECUTION, [
                    ("generate_code", SOPStepType.GENERATE),
                ]),
            ],
            "medium": [
                ("plan", SOPPhaseType.PLANNING, [
                    ("analyze_requirements", SOPStepType.ANALYZE),
                ]),
                ("implement", SOPPhaseType.EXECUTION, [
                    ("generate_code", SOPStepType.GENERATE),
                    ("write_tests", SOPStepType.GENERATE),
                ]),
                ("validate", SOPPhaseType.VALIDATION, [
                    ("run_tests", SOPStepType.VALIDATE),
                ]),
            ],
            "complex": [
                ("plan", SOPPhaseType.PLANNING, [
                    ("analyze_requirements", SOPStepType.ANALYZE),
                    ("design_architecture", SOPStepType.GENERATE),
                ]),
                ("implement", SOPPhaseType.EXECUTION, [
                    ("generate_code", SOPStepType.GENERATE),
                    ("write_tests", SOPStepType.GENERATE),
                    ("write_docs", SOPStepType.GENERATE),
                ]),
                ("validate", SOPPhaseType.VALIDATION, [
                    ("run_tests", SOPStepType.VALIDATE),
                    ("code_review", SOPStepType.VALIDATE),
                ]),
                ("finalize", SOPPhaseType.COMPLETION, [
                    ("integrate", SOPStepType.TRANSFORM),
                ]),
            ],
        },
        "research": {
            "simple": [
                ("research", SOPPhaseType.EXECUTION, [
                    ("search", SOPStepType.ANALYZE),
                    ("summarize", SOPStepType.GENERATE),
                ]),
            ],
            "medium": [
                ("gather", SOPPhaseType.PLANNING, [
                    ("search", SOPStepType.ANALYZE),
                    ("filter", SOPStepType.TRANSFORM),
                ]),
                ("analyze", SOPPhaseType.EXECUTION, [
                    ("synthesize", SOPStepType.GENERATE),
                ]),
                ("report", SOPPhaseType.COMPLETION, [
                    ("write_report", SOPStepType.GENERATE),
                ]),
            ],
        },
        "content": {
            "simple": [
                ("create", SOPPhaseType.EXECUTION, [
                    ("write", SOPStepType.GENERATE),
                ]),
            ],
            "medium": [
                ("outline", SOPPhaseType.PLANNING, [
                    ("create_outline", SOPStepType.GENERATE),
                ]),
                ("write", SOPPhaseType.EXECUTION, [
                    ("write_draft", SOPStepType.GENERATE),
                ]),
                ("review", SOPPhaseType.VALIDATION, [
                    ("review_content", SOPStepType.VALIDATE),
                ]),
            ],
        },
    }

    # Get template
    task_templates = templates.get(task_type, templates["code"])
    template = task_templates.get(complexity, task_templates["medium"])

    # Build SOP
    phases = []
    for phase_name, phase_type, steps in template:
        sop_steps = [
            SOPStep(name=name, action=action)
            for name, action in steps
        ]
        phases.append(SOPPhase(
            name=phase_name,
            phase_type=phase_type,
            steps=sop_steps,
        ))

    return SOP(
        name=f"{task_type}_{complexity}",
        description=f"Template workflow for {complexity} {task_type} tasks",
        phases=phases,
        triggers=[task_type],
    )


__all__ = [
    # Types
    "MutationType",
    "SearchStrategy",
    # Candidates
    "WorkflowCandidate",
    # Mutations
    "WorkflowMutation",
    "MutationGenerator",
    # Config
    "AFlowConfig",
    # Generator
    "AFlowGenerator",
    # Templates
    "generate_template_workflow",
]
