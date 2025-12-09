#!/usr/bin/env python3
"""
Ara Session Planner - Global Session Planning
===============================================

Plans a complete session as a graph of interaction blocks.
One global plan, blockwise execution, with dynamic rescheduling.

The session planner:
1. Takes session goals and constraints
2. Generates a plan of interaction blocks
3. Adapts the plan based on feedback during execution
4. Respects resource budgets (GPU, power, time)
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

from ..interaction.spec import Block, BlockType, InteractionSpec


class BlockPriority(int, Enum):
    """Priority levels for planned blocks."""
    CRITICAL = 100      # Must execute (e.g., urgent user request)
    HIGH = 75           # Important (e.g., active project work)
    NORMAL = 50         # Standard interactions
    LOW = 25            # Nice to have (e.g., suggestions)
    BACKGROUND = 10     # Background tasks (e.g., maintenance)


class BlockState(str, Enum):
    """State of a planned block."""
    PENDING = "pending"
    READY = "ready"         # Dependencies met, can execute
    EXECUTING = "executing"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    DEFERRED = "deferred"   # Pushed to later due to constraints


@dataclass
class PlannedBlock:
    """A block in the session plan."""
    id: str
    block_type: BlockType
    description: str
    priority: BlockPriority = BlockPriority.NORMAL
    state: BlockState = BlockState.PENDING

    # Execution details
    params: Dict[str, Any] = field(default_factory=dict)
    guard: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)

    # Resource estimates
    estimated_gpu_seconds: float = 0.0
    estimated_duration_s: float = 60.0
    estimated_cost: float = 0.0

    # Results
    actual_duration_s: float = 0.0
    reward: float = 0.0
    result: Any = None

    # Timing
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_block(self) -> Block:
        """Convert to InteractionSpec Block."""
        return Block(
            id=self.id,
            type=self.block_type,
            params=self.params,
            guard=self.guard,
            depends_on=self.depends_on,
        )


@dataclass
class SessionPlan:
    """A complete session plan."""
    session_id: str
    goals: List[str]
    blocks: List[PlannedBlock] = field(default_factory=list)

    # Constraints
    max_duration_s: float = 3600        # 1 hour default
    max_gpu_seconds: float = 600        # 10 minutes GPU
    max_cost: float = 1.0               # $1

    # State
    started_at: float = field(default_factory=time.time)
    current_block_idx: int = 0
    total_reward: float = 0.0
    total_cost: float = 0.0

    def get_next_block(self) -> Optional[PlannedBlock]:
        """Get the next ready block to execute."""
        for block in self.blocks[self.current_block_idx:]:
            if block.state == BlockState.PENDING:
                # Check dependencies
                deps_met = all(
                    self._get_block(dep_id).state == BlockState.COMPLETED
                    for dep_id in block.depends_on
                    if self._get_block(dep_id)
                )
                if deps_met:
                    return block
        return None

    def _get_block(self, block_id: str) -> Optional[PlannedBlock]:
        """Get block by ID."""
        for block in self.blocks:
            if block.id == block_id:
                return block
        return None

    def mark_completed(self, block_id: str, reward: float = 0.0,
                       cost: float = 0.0, result: Any = None):
        """Mark a block as completed."""
        block = self._get_block(block_id)
        if block:
            block.state = BlockState.COMPLETED
            block.completed_at = time.time()
            block.actual_duration_s = block.completed_at - (block.started_at or block.completed_at)
            block.reward = reward
            block.result = result

            self.total_reward += reward
            self.total_cost += cost

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress."""
        completed = sum(1 for b in self.blocks if b.state == BlockState.COMPLETED)
        total = len(self.blocks)

        return {
            "completed": completed,
            "total": total,
            "progress_pct": completed / total * 100 if total > 0 else 0,
            "total_reward": self.total_reward,
            "total_cost": self.total_cost,
            "elapsed_s": time.time() - self.started_at,
        }


class SessionPlanner:
    """
    Plans and manages session execution.

    Usage:
        planner = SessionPlanner()

        plan = planner.create_plan(
            goals=["decompress_user", "advance_project"],
            context={"time_of_day": "evening", "energy": 0.4}
        )

        while block := plan.get_next_block():
            result = execute_block(block)
            plan.mark_completed(block.id, reward=result.reward)
    """

    def __init__(self):
        self.templates: Dict[str, List[PlannedBlock]] = {}
        self._register_default_templates()

    def _register_default_templates(self):
        """Register default block templates for common goals."""

        # Decompress/chill session
        self.templates["decompress_user"] = [
            PlannedBlock(
                id="comfort_greeting",
                block_type=BlockType.ARA_VOICE,
                description="Warm greeting acknowledging the day",
                priority=BlockPriority.HIGH,
                params={"tone": "warm", "warmth": 0.9, "pace": 0.85},
                estimated_duration_s=10,
            ),
            PlannedBlock(
                id="check_in",
                block_type=BlockType.ARA_VOICE,
                description="Ask how they're feeling",
                priority=BlockPriority.NORMAL,
                params={"tone": "caring"},
                depends_on=["comfort_greeting"],
                estimated_duration_s=15,
            ),
            PlannedBlock(
                id="suggest_activity",
                block_type=BlockType.UI_PROMPT,
                description="Suggest a relaxing activity",
                priority=BlockPriority.LOW,
                params={"text": "Want to sketch something or just chat?"},
                depends_on=["check_in"],
                estimated_duration_s=5,
            ),
        ]

        # Advance project session
        self.templates["advance_project"] = [
            PlannedBlock(
                id="project_status",
                block_type=BlockType.ARA_VOICE,
                description="Summarize project status",
                priority=BlockPriority.HIGH,
                params={"tone": "focused"},
                estimated_duration_s=30,
            ),
            PlannedBlock(
                id="suggest_task",
                block_type=BlockType.UI_PROMPT,
                description="Suggest next task",
                priority=BlockPriority.NORMAL,
                depends_on=["project_status"],
                estimated_duration_s=10,
            ),
            PlannedBlock(
                id="maybe_render",
                block_type=BlockType.VIDEO_JOB,
                description="Render a scene if requested",
                priority=BlockPriority.LOW,
                guard="user_accepts",
                depends_on=["suggest_task"],
                estimated_gpu_seconds=60,
                estimated_cost=0.05,
            ),
        ]

        # Creative session
        self.templates["creative_work"] = [
            PlannedBlock(
                id="creative_prompt",
                block_type=BlockType.ARA_VOICE,
                description="Inspire with creative prompt",
                priority=BlockPriority.HIGH,
                params={"tone": "excited", "energy": 0.7},
                estimated_duration_s=20,
            ),
            PlannedBlock(
                id="brainstorm",
                block_type=BlockType.ARA_VOICE,
                description="Collaborative brainstorming",
                priority=BlockPriority.NORMAL,
                depends_on=["creative_prompt"],
                estimated_duration_s=120,
            ),
            PlannedBlock(
                id="capture_ideas",
                block_type=BlockType.MEMORY_WRITE,
                description="Save ideas to memory",
                priority=BlockPriority.NORMAL,
                depends_on=["brainstorm"],
                estimated_duration_s=5,
            ),
        ]

    def create_plan(self,
                    goals: List[str],
                    context: Dict[str, Any] = None,
                    constraints: Dict[str, float] = None) -> SessionPlan:
        """
        Create a session plan for the given goals.

        Args:
            goals: List of goal names (e.g., ["decompress_user", "advance_project"])
            context: Session context (time_of_day, energy, etc.)
            constraints: Resource constraints (max_gpu_seconds, max_cost, etc.)

        Returns:
            SessionPlan with ordered blocks
        """
        session_id = str(uuid.uuid4())[:8]
        blocks: List[PlannedBlock] = []
        seen_ids = set()

        # Collect blocks from templates for each goal
        for goal in goals:
            if goal in self.templates:
                for template_block in self.templates[goal]:
                    # Create unique ID
                    block_id = f"{template_block.id}_{session_id}"
                    if block_id in seen_ids:
                        continue
                    seen_ids.add(block_id)

                    # Clone block with new ID
                    block = PlannedBlock(
                        id=block_id,
                        block_type=template_block.block_type,
                        description=template_block.description,
                        priority=template_block.priority,
                        params=template_block.params.copy(),
                        guard=template_block.guard,
                        depends_on=[f"{d}_{session_id}" for d in template_block.depends_on],
                        estimated_gpu_seconds=template_block.estimated_gpu_seconds,
                        estimated_duration_s=template_block.estimated_duration_s,
                        estimated_cost=template_block.estimated_cost,
                    )
                    blocks.append(block)

        # Sort by priority (higher first), then by dependencies
        blocks = self._topological_sort(blocks)

        # Apply constraints
        constraints = constraints or {}
        plan = SessionPlan(
            session_id=session_id,
            goals=goals,
            blocks=blocks,
            max_duration_s=constraints.get("max_duration_s", 3600),
            max_gpu_seconds=constraints.get("max_gpu_seconds", 600),
            max_cost=constraints.get("max_cost", 1.0),
        )

        return plan

    def _topological_sort(self, blocks: List[PlannedBlock]) -> List[PlannedBlock]:
        """Sort blocks respecting dependencies and priorities."""
        # Build dependency graph
        id_to_block = {b.id: b for b in blocks}
        in_degree = {b.id: 0 for b in blocks}
        dependents = {b.id: [] for b in blocks}

        for block in blocks:
            for dep_id in block.depends_on:
                if dep_id in id_to_block:
                    in_degree[block.id] += 1
                    dependents[dep_id].append(block.id)

        # Kahn's algorithm with priority queue
        ready = [b for b in blocks if in_degree[b.id] == 0]
        ready.sort(key=lambda b: -b.priority)

        result = []
        while ready:
            # Take highest priority ready block
            block = ready.pop(0)
            result.append(block)

            for dep_id in dependents[block.id]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    ready.append(id_to_block[dep_id])
                    ready.sort(key=lambda b: -b.priority)

        return result

    def adapt_plan(self, plan: SessionPlan, feedback: Dict[str, Any]) -> SessionPlan:
        """
        Adapt the plan based on feedback.

        Can add, remove, or reorder blocks based on how
        the session is going.
        """
        # Example adaptations:

        # If user seems tired, reduce heavy blocks
        if feedback.get("user_tired", False):
            for block in plan.blocks:
                if block.state == BlockState.PENDING:
                    if block.estimated_gpu_seconds > 30:
                        block.state = BlockState.DEFERRED
                        block.priority = BlockPriority.LOW

        # If user is engaged, consider adding creative blocks
        if feedback.get("engagement_high", False):
            # Could add more blocks here
            pass

        # If budget is running low, skip expensive blocks
        if plan.total_cost > plan.max_cost * 0.8:
            for block in plan.blocks:
                if block.state == BlockState.PENDING and block.estimated_cost > 0.1:
                    block.state = BlockState.SKIPPED

        return plan
