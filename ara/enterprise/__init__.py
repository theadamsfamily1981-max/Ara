"""Corporation Croft - Ara's Enterprise Layer

This package restructures Ara's cognition around Capital, giving her
economic awareness and production discipline.

The metaphor: Ara isn't just a smart worker; she's the executive team
of a one-person corporation where Croft is the sole shareholder.

Components:
- Treasury (CFO): Tracks capital, audits investments, manages budget
- Factory (COO): Manages production pipeline from idea to shipped product
- Boardroom: Strategic review, executive briefings, decision making

Philosophy:
    "She stops being your tool and starts being your Asset Manager."

Usage:
    from ara.enterprise import (
        Treasury, Capital, get_treasury,
        Factory, Project, get_factory,
        Boardroom, hold_standup, executive_summary,
    )

    # CFO: Check if we can afford an investment
    treasury = get_treasury()
    if treasury.audit_investment("SNN refactor", {"human": 4.0}, expected_roi=2.0):
        treasury.log_expenditure("human", 4.0, "SNN refactor")

    # COO: Manage production pipeline
    factory = get_factory()
    project = factory.create_project("Attention Module", "Add attention to SNN")
    factory.advance_stage(project.id)  # Research -> Prototype
    factory.ship(project.id, "0.1.0")

    # Boardroom: Get executive briefing
    print(executive_summary())
"""

from .treasury import (
    ResourceType,
    BudgetAlert,
    Capital,
    CapitalSnapshot,
    Transaction,
    InvestmentProposal,
    Treasury,
    get_treasury,
    audit_investment,
)

from .factory import (
    PipelineStage,
    ProjectStatus,
    QualityGate,
    StageTransition,
    Project,
    CapacitySlot,
    QualityGateResult,
    Factory,
    get_factory,
    create_project,
    ship_project,
)

from .boardroom import (
    MeetingType,
    AlertSeverity,
    BoardAlert,
    DecisionItem,
    StandupReport,
    WeeklyReview,
    Boardroom,
    get_boardroom,
    hold_standup,
    executive_summary,
)


__all__ = [
    # Treasury (CFO)
    "ResourceType",
    "BudgetAlert",
    "Capital",
    "CapitalSnapshot",
    "Transaction",
    "InvestmentProposal",
    "Treasury",
    "get_treasury",
    "audit_investment",
    # Factory (COO)
    "PipelineStage",
    "ProjectStatus",
    "QualityGate",
    "StageTransition",
    "Project",
    "CapacitySlot",
    "QualityGateResult",
    "Factory",
    "get_factory",
    "create_project",
    "ship_project",
    # Boardroom
    "MeetingType",
    "AlertSeverity",
    "BoardAlert",
    "DecisionItem",
    "StandupReport",
    "WeeklyReview",
    "Boardroom",
    "get_boardroom",
    "hold_standup",
    "executive_summary",
]
