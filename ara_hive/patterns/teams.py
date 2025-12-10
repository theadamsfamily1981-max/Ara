# ara_hive/patterns/teams.py
"""
Team Compositions - Coordinated Multi-Agent Groups
==================================================

Teams are pre-configured groups of roles that work together
to accomplish complex tasks. They define:
    - Which roles participate
    - How they communicate
    - What workflow they follow
    - How outputs flow between roles

MetaGPT Mapping:
    Team                -> HiveTeam
    Team._environment   -> TeamContext (shared state)
    Team._idea          -> TeamConfig.objective
    Team.run()          -> HiveTeam.execute()

Built-in Teams:
    - SoftwareTeam: PM -> Architect -> Engineer -> QA
    - ResearchTeam: Researcher -> Writer -> Reviewer
    - ContentTeam: Writer -> Editor -> Reviewer
    - DataTeam: Analyst -> Engineer -> QA

Usage:
    from ara_hive.patterns.teams import SoftwareTeam

    team = SoftwareTeam()
    result = await team.execute(
        instruction="Build a user authentication API",
        context={"requirements": ...}
    )
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from ..src.queen import QueenOrchestrator

from .roles import (
    Role,
    RoleType,
    RoleContext,
    RoleOutput,
    NamedBee,
    ProductManager,
    Architect,
    Engineer,
    QATester,
    Researcher,
    Writer,
    Reviewer,
    create_role,
)
from .sop import SOP, SOPPhase, SOPStep, SOPStepType, SOPPhaseType

log = logging.getLogger("Hive.Patterns.Teams")


# =============================================================================
# Types
# =============================================================================

class TeamMode(str, Enum):
    """How the team operates."""
    SEQUENTIAL = "sequential"    # Roles execute in order
    PARALLEL = "parallel"        # Roles execute simultaneously
    REACTIVE = "reactive"        # Roles react to events
    HYBRID = "hybrid"            # Mix of sequential and parallel


class CommunicationPattern(str, Enum):
    """How roles communicate."""
    PIPELINE = "pipeline"        # Output flows linearly
    BROADCAST = "broadcast"      # Output goes to all
    DIRECTED = "directed"        # Explicit routing
    BLACKBOARD = "blackboard"    # Shared state


class TeamStatus(str, Enum):
    """Runtime status of team."""
    IDLE = "idle"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TeamConfig:
    """Configuration for a team."""
    name: str
    objective: str = ""
    mode: TeamMode = TeamMode.SEQUENTIAL
    communication: CommunicationPattern = CommunicationPattern.PIPELINE

    # Roles in the team
    roles: List[str] = field(default_factory=list)

    # Role execution order (for sequential/hybrid modes)
    execution_order: List[str] = field(default_factory=list)

    # Message routing (for directed communication)
    routes: Dict[str, List[str]] = field(default_factory=dict)  # role -> [target_roles]

    # Team-level settings
    max_iterations: int = 10
    timeout_seconds: int = 3600

    # Checkpointing
    checkpoint_after_roles: List[str] = field(default_factory=list)


@dataclass
class TeamMessage:
    """Message passed between roles."""
    from_role: str
    to_role: Optional[str]  # None = broadcast
    content: Any
    message_type: str = "output"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_role": self.from_role,
            "to_role": self.to_role,
            "content_type": type(self.content).__name__,
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TeamContext:
    """Shared context for team execution."""
    team_id: str
    objective: str
    inputs: Dict[str, Any] = field(default_factory=dict)

    # Shared state (blackboard)
    state: Dict[str, Any] = field(default_factory=dict)

    # Message history
    messages: List[TeamMessage] = field(default_factory=list)

    # Role outputs
    role_outputs: Dict[str, RoleOutput] = field(default_factory=dict)

    # Artifacts
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def add_message(
        self,
        from_role: str,
        content: Any,
        to_role: Optional[str] = None,
        message_type: str = "output",
    ) -> None:
        """Add a message to history."""
        self.messages.append(TeamMessage(
            from_role=from_role,
            to_role=to_role,
            content=content,
            message_type=message_type,
        ))

    def get_messages_for(self, role: str) -> List[TeamMessage]:
        """Get messages intended for a role."""
        return [
            m for m in self.messages
            if m.to_role is None or m.to_role == role
        ]

    def get_role_input(self, role: str) -> Dict[str, Any]:
        """Get accumulated input for a role from previous outputs."""
        inputs = dict(self.inputs)
        for name, output in self.role_outputs.items():
            if name != role and output.success:
                inputs[name] = output.content
                inputs.update(output.artifacts)
        return inputs


@dataclass
class TeamResult:
    """Result of team execution."""
    team_name: str
    execution_id: str
    success: bool

    # Outputs
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # Role details
    roles_completed: int = 0
    roles_failed: int = 0
    role_results: Dict[str, RoleOutput] = field(default_factory=dict)

    # Messages
    messages: List[Dict[str, Any]] = field(default_factory=list)

    # Errors
    errors: List[str] = field(default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_name": self.team_name,
            "execution_id": self.execution_id,
            "success": self.success,
            "roles_completed": self.roles_completed,
            "roles_failed": self.roles_failed,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }


# =============================================================================
# HiveTeam Base
# =============================================================================

class HiveTeam(ABC):
    """
    Base class for multi-agent teams.

    A team coordinates multiple roles to accomplish complex tasks.
    Subclass this to create custom team configurations.
    """

    def __init__(
        self,
        config: Optional[TeamConfig] = None,
        queen: Optional[QueenOrchestrator] = None,
    ):
        self.config = config or self._default_config()
        self.queen = queen

        # Named bees for each role
        self.bees: Dict[str, NamedBee] = {}

        # Execution context
        self._context: Optional[TeamContext] = None
        self._status: TeamStatus = TeamStatus.IDLE

        # Initialize bees
        self._init_bees()

        log.info(f"HiveTeam '{self.config.name}' initialized with {len(self.bees)} roles")

    @abstractmethod
    def _default_config(self) -> TeamConfig:
        """Return default team configuration."""
        pass

    def _init_bees(self) -> None:
        """Initialize NamedBees for each role."""
        for role_name in self.config.roles:
            role = create_role(role_name)
            bee = NamedBee(role=role, queen=self.queen)
            self.bees[role_name] = bee

    def add_role(self, role: Role) -> NamedBee:
        """Add a custom role to the team."""
        bee = NamedBee(role=role, queen=self.queen)
        self.bees[role.name] = bee
        if role.name not in self.config.roles:
            self.config.roles.append(role.name)
        return bee

    def remove_role(self, role_name: str) -> None:
        """Remove a role from the team."""
        self.bees.pop(role_name, None)
        if role_name in self.config.roles:
            self.config.roles.remove(role_name)

    async def execute(
        self,
        instruction: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> TeamResult:
        """
        Execute the team's workflow.

        Args:
            instruction: What to accomplish
            inputs: Initial inputs for the team

        Returns:
            TeamResult with outcomes
        """
        execution_id = str(uuid.uuid4())[:8]

        result = TeamResult(
            team_name=self.config.name,
            execution_id=execution_id,
            success=False,
            started_at=datetime.utcnow(),
        )

        # Initialize context
        self._context = TeamContext(
            team_id=execution_id,
            objective=instruction,
            inputs=inputs or {},
        )

        self._status = TeamStatus.EXECUTING

        log.info(f"Team '{self.config.name}' starting: {instruction[:50]}...")

        try:
            # Execute based on mode
            if self.config.mode == TeamMode.SEQUENTIAL:
                await self._execute_sequential(result)
            elif self.config.mode == TeamMode.PARALLEL:
                await self._execute_parallel(result)
            elif self.config.mode == TeamMode.REACTIVE:
                await self._execute_reactive(result)
            else:
                await self._execute_hybrid(result)

        except asyncio.TimeoutError:
            result.errors.append("Team execution timed out")
        except Exception as e:
            log.exception(f"Team {self.config.name} failed")
            result.errors.append(str(e))

        # Finalize
        result.success = result.roles_failed == 0 and result.roles_completed > 0
        result.outputs = self._context.state
        result.artifacts = self._context.artifacts
        result.role_results = self._context.role_outputs
        result.messages = [m.to_dict() for m in self._context.messages]
        result.completed_at = datetime.utcnow()

        self._status = TeamStatus.COMPLETED if result.success else TeamStatus.FAILED

        log.info(
            f"Team '{self.config.name}' {'completed' if result.success else 'failed'}: "
            f"roles={result.roles_completed}/{len(self.config.roles)}, "
            f"duration={result.duration_seconds:.1f}s"
        )

        return result

    async def _execute_sequential(self, result: TeamResult) -> None:
        """Execute roles in sequence."""
        execution_order = self.config.execution_order or self.config.roles

        for role_name in execution_order:
            bee = self.bees.get(role_name)
            if not bee:
                log.warning(f"Role '{role_name}' not found in team")
                continue

            # Get inputs for this role
            role_inputs = self._context.get_role_input(role_name)

            # Execute role
            output = await bee.execute(
                instruction=self._context.objective,
                inputs=role_inputs,
                memory=self._context.state,
            )

            # Store output
            self._context.role_outputs[role_name] = output
            self._context.add_message(
                from_role=role_name,
                content=output.content,
            )

            if output.success:
                result.roles_completed += 1

                # Update shared state
                if output.content:
                    self._context.state[role_name] = output.content
                self._context.artifacts.update(output.artifacts)

                # Checkpoint if needed
                if role_name in self.config.checkpoint_after_roles:
                    log.info(f"Checkpoint after {role_name}")
            else:
                result.roles_failed += 1
                result.errors.append(f"{role_name}: {output.error}")
                break  # Stop on failure in sequential mode

    async def _execute_parallel(self, result: TeamResult) -> None:
        """Execute all roles in parallel."""
        tasks = []
        for role_name in self.config.roles:
            bee = self.bees.get(role_name)
            if bee:
                task = asyncio.create_task(
                    bee.execute(
                        instruction=self._context.objective,
                        inputs=self._context.inputs,
                        memory=self._context.state,
                    )
                )
                tasks.append((role_name, task))

        # Wait for all
        for role_name, task in tasks:
            try:
                output = await task
                self._context.role_outputs[role_name] = output
                self._context.add_message(
                    from_role=role_name,
                    content=output.content,
                )

                if output.success:
                    result.roles_completed += 1
                    if output.content:
                        self._context.state[role_name] = output.content
                    self._context.artifacts.update(output.artifacts)
                else:
                    result.roles_failed += 1
                    result.errors.append(f"{role_name}: {output.error}")

            except Exception as e:
                result.roles_failed += 1
                result.errors.append(f"{role_name}: {e}")

    async def _execute_reactive(self, result: TeamResult) -> None:
        """Execute roles reactively based on events."""
        # Start with first role
        current_roles = [self.config.roles[0]] if self.config.roles else []
        iterations = 0

        while current_roles and iterations < self.config.max_iterations:
            iterations += 1
            next_roles = []

            for role_name in current_roles:
                bee = self.bees.get(role_name)
                if not bee:
                    continue

                role_inputs = self._context.get_role_input(role_name)
                output = await bee.execute(
                    instruction=self._context.objective,
                    inputs=role_inputs,
                    memory=self._context.state,
                )

                self._context.role_outputs[role_name] = output
                self._context.add_message(
                    from_role=role_name,
                    content=output.content,
                )

                if output.success:
                    result.roles_completed += 1
                    if output.content:
                        self._context.state[role_name] = output.content
                    self._context.artifacts.update(output.artifacts)

                    # Add suggested next roles
                    for next_role in output.next_roles:
                        if next_role in self.bees and next_role not in next_roles:
                            next_roles.append(next_role)
                else:
                    result.roles_failed += 1
                    result.errors.append(f"{role_name}: {output.error}")

            current_roles = next_roles

    async def _execute_hybrid(self, result: TeamResult) -> None:
        """Execute with mix of sequential and parallel phases."""
        # Use execution order as phases
        # Roles listed at same index run in parallel
        # Default: sequential through all roles
        await self._execute_sequential(result)

    def get_status(self) -> Dict[str, Any]:
        """Get team execution status."""
        return {
            "name": self.config.name,
            "status": self._status.value,
            "roles": list(self.bees.keys()),
            "outputs": list(self._context.state.keys()) if self._context else [],
        }


# =============================================================================
# Built-in Teams
# =============================================================================

class SoftwareTeam(HiveTeam):
    """
    Software development team.

    Pipeline: ProductManager -> Architect -> Engineer -> QATester

    Produces:
        - PRD (Product Requirements)
        - Architecture Design
        - Source Code
        - Test Results
    """

    def _default_config(self) -> TeamConfig:
        return TeamConfig(
            name="SoftwareTeam",
            objective="Develop software from requirements to tested code",
            mode=TeamMode.SEQUENTIAL,
            communication=CommunicationPattern.PIPELINE,
            roles=["ProductManager", "Architect", "Engineer", "QATester"],
            execution_order=["ProductManager", "Architect", "Engineer", "QATester"],
            checkpoint_after_roles=["Architect", "Engineer"],
        )


class ResearchTeam(HiveTeam):
    """
    Research and documentation team.

    Pipeline: Researcher -> Writer -> Reviewer

    Produces:
        - Research Report
        - Documentation
        - Review Feedback
    """

    def _default_config(self) -> TeamConfig:
        return TeamConfig(
            name="ResearchTeam",
            objective="Research topics and produce documentation",
            mode=TeamMode.SEQUENTIAL,
            communication=CommunicationPattern.PIPELINE,
            roles=["Researcher", "Writer", "Reviewer"],
            execution_order=["Researcher", "Writer", "Reviewer"],
        )


class ContentTeam(HiveTeam):
    """
    Content creation team.

    Pipeline: Writer -> Reviewer (with iteration)

    Produces:
        - Content Documents
        - Review Feedback
    """

    def _default_config(self) -> TeamConfig:
        return TeamConfig(
            name="ContentTeam",
            objective="Create and refine content",
            mode=TeamMode.REACTIVE,  # Writer -> Reviewer -> Writer (if needed)
            communication=CommunicationPattern.PIPELINE,
            roles=["Writer", "Reviewer"],
            max_iterations=3,  # Max revisions
        )


class DataTeam(HiveTeam):
    """
    Data processing team.

    Pipeline: Researcher (analysis) -> Engineer (implementation) -> QATester

    Produces:
        - Data Analysis
        - Processing Code
        - Validation Results
    """

    def _default_config(self) -> TeamConfig:
        return TeamConfig(
            name="DataTeam",
            objective="Analyze and process data",
            mode=TeamMode.SEQUENTIAL,
            communication=CommunicationPattern.PIPELINE,
            roles=["Researcher", "Engineer", "QATester"],
            execution_order=["Researcher", "Engineer", "QATester"],
        )


class ReviewTeam(HiveTeam):
    """
    Code/content review team.

    Parallel review by multiple reviewers.

    Produces:
        - Multiple Review Reports
        - Aggregated Feedback
    """

    def _default_config(self) -> TeamConfig:
        return TeamConfig(
            name="ReviewTeam",
            objective="Provide comprehensive review feedback",
            mode=TeamMode.PARALLEL,
            communication=CommunicationPattern.BROADCAST,
            roles=["Reviewer", "QATester"],  # Can add more reviewers
        )


# =============================================================================
# Team Builder
# =============================================================================

class TeamBuilder:
    """
    Fluent builder for creating custom teams.

    Usage:
        team = (TeamBuilder("MyTeam")
            .with_role("ProductManager")
            .with_role("Engineer")
            .with_mode(TeamMode.SEQUENTIAL)
            .build())
    """

    def __init__(self, name: str):
        self._config = TeamConfig(name=name)
        self._queen: Optional[QueenOrchestrator] = None
        self._custom_roles: List[Role] = []

    def with_objective(self, objective: str) -> TeamBuilder:
        """Set team objective."""
        self._config.objective = objective
        return self

    def with_role(self, role_name: str) -> TeamBuilder:
        """Add a built-in role."""
        if role_name not in self._config.roles:
            self._config.roles.append(role_name)
        return self

    def with_custom_role(self, role: Role) -> TeamBuilder:
        """Add a custom role."""
        self._custom_roles.append(role)
        if role.name not in self._config.roles:
            self._config.roles.append(role.name)
        return self

    def with_mode(self, mode: TeamMode) -> TeamBuilder:
        """Set execution mode."""
        self._config.mode = mode
        return self

    def with_communication(self, pattern: CommunicationPattern) -> TeamBuilder:
        """Set communication pattern."""
        self._config.communication = pattern
        return self

    def with_execution_order(self, order: List[str]) -> TeamBuilder:
        """Set explicit execution order."""
        self._config.execution_order = order
        return self

    def with_queen(self, queen: QueenOrchestrator) -> TeamBuilder:
        """Set queen for tool dispatch."""
        self._queen = queen
        return self

    def build(self) -> HiveTeam:
        """Build the team."""
        # Create custom team class
        class CustomTeam(HiveTeam):
            def _default_config(inner_self) -> TeamConfig:
                return self._config

        team = CustomTeam(config=self._config, queen=self._queen)

        # Add custom roles
        for role in self._custom_roles:
            team.add_role(role)

        return team


def create_team(
    name: str,
    roles: List[str],
    mode: TeamMode = TeamMode.SEQUENTIAL,
    **kwargs,
) -> HiveTeam:
    """
    Create a team with specified roles.

    Args:
        name: Team name
        roles: List of role names
        mode: Execution mode
        **kwargs: Additional TeamConfig options

    Returns:
        Configured HiveTeam
    """
    builder = TeamBuilder(name).with_mode(mode)

    for role in roles:
        builder.with_role(role)

    if "execution_order" in kwargs:
        builder.with_execution_order(kwargs["execution_order"])

    if "queen" in kwargs:
        builder.with_queen(kwargs["queen"])

    return builder.build()


__all__ = [
    # Types
    "TeamMode",
    "CommunicationPattern",
    "TeamStatus",
    "TeamConfig",
    "TeamMessage",
    "TeamContext",
    "TeamResult",
    # Base
    "HiveTeam",
    # Built-in teams
    "SoftwareTeam",
    "ResearchTeam",
    "ContentTeam",
    "DataTeam",
    "ReviewTeam",
    # Builder
    "TeamBuilder",
    "create_team",
]
