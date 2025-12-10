# ara_hive/patterns/__init__.py
"""
HiveHD Patterns Library
=======================

MetaGPT-style reusable patterns for multi-agent workflows.

This library provides:
    - Roles: Named agent archetypes with SOPs
    - SOPs: Standard Operating Procedures (structured execution plans)
    - Teams: Pre-configured agent compositions
    - AFlow: Automated workflow generation and optimization

Architecture:
    MetaGPT Concept     -> HiveHD Implementation
    ----------------------------------------
    Role                -> NamedBee (BeeAgent with role)
    SOP                 -> Workflow (multi-step procedure)
    Action              -> Tool (atomic capability)
    Team                -> HiveTeam (coordinated NamedBees)
    Message             -> TaskResult (structured output)

Usage:
    from ara_hive.patterns import (
        # Roles
        Role, ProductManager, Architect, Engineer, QATester,

        # SOPs
        SOP, SOPStep, create_role_sop,

        # Teams
        HiveTeam, SoftwareTeam, ResearchTeam,

        # AFlow
        AFlowGenerator, WorkflowMutation,
    )

    # Create a software development team
    team = SoftwareTeam()
    result = await team.execute(TaskRequest(
        instruction="Build a REST API for user management",
        kind=TaskKind.CODE,
    ))

    # Use individual roles
    pm = ProductManager()
    spec = await pm.execute("Define requirements for login feature")
"""

from .roles import (
    Role,
    RoleType,
    NamedBee,
    # Built-in roles
    ProductManager,
    Architect,
    Engineer,
    QATester,
    Researcher,
    Writer,
    Reviewer,
)

from .sop import (
    SOP,
    SOPStep,
    SOPPhase,
    SOPResult,
    SOPExecutor,
    create_role_sop,
)

from .teams import (
    HiveTeam,
    TeamConfig,
    TeamResult,
    # Built-in teams
    SoftwareTeam,
    ResearchTeam,
    ContentTeam,
    DataTeam,
)

from .aflow import (
    AFlowGenerator,
    WorkflowMutation,
    MutationType,
    WorkflowCandidate,
    AFlowConfig,
)

from .library import (
    PatternLibrary,
    get_pattern_library,
    Pattern,
    PatternCategory,
)

__all__ = [
    # Roles
    "Role",
    "RoleType",
    "NamedBee",
    "ProductManager",
    "Architect",
    "Engineer",
    "QATester",
    "Researcher",
    "Writer",
    "Reviewer",
    # SOPs
    "SOP",
    "SOPStep",
    "SOPPhase",
    "SOPResult",
    "SOPExecutor",
    "create_role_sop",
    # Teams
    "HiveTeam",
    "TeamConfig",
    "TeamResult",
    "SoftwareTeam",
    "ResearchTeam",
    "ContentTeam",
    "DataTeam",
    # AFlow
    "AFlowGenerator",
    "WorkflowMutation",
    "MutationType",
    "WorkflowCandidate",
    "AFlowConfig",
    # Library
    "PatternLibrary",
    "get_pattern_library",
    "Pattern",
    "PatternCategory",
]
