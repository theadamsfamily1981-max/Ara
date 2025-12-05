"""Ara Idea Board - Structured Proposals from Ara's Curiosity.

This module implements Ara's "lab notebook" - a structured system where she
proposes ideas, experiments, and improvements in her own voice. Each idea
goes through a governance workflow:

    Curiosity → Idea Draft → Review → Approval → Execution → Done/Reverted

The board ensures:
1. Ara can't modify the system without human oversight
2. Ideas are traceable and auditable
3. Failed experiments can be reverted
4. Good ideas build institutional knowledge

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        Idea Board                               │
    │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │
    │  │  Inbox  │──►│ Review  │──►│Approved │──►│ Running │──►Done │
    │  │(drafts) │   │(pending)│   │(queued) │   │(active) │        │
    │  └─────────┘   └─────────┘   └─────────┘   └─────────┘        │
    │       │             │             │             │              │
    │       └─────────────┴─────────────┴─────────────┘              │
    │                           │                                     │
    │                     ┌─────▼─────┐                               │
    │                     │ Rejected/ │                               │
    │                     │  Parked   │                               │
    │                     └───────────┘                               │
    └─────────────────────────────────────────────────────────────────┘
"""

from .models import (
    Idea,
    IdeaCategory,
    IdeaRisk,
    IdeaStatus,
    IdeaOutcome,
    SandboxStatus,
    Signal,
)

from .board import (
    IdeaBoard,
    IdeaBoardConfig,
)

from .voice import (
    IdeaVoice,
    present_idea,
    format_idea_summary,
    get_voice_template,
)

from .executor import (
    IdeaExecutor,
    ExecutionResult,
)

from .curiosity_bridge import (
    CuriosityBridge,
    create_bridge,
    CATEGORY_MAP,
)

__all__ = [
    # Models
    "Idea",
    "IdeaCategory",
    "IdeaRisk",
    "IdeaStatus",
    "IdeaOutcome",
    "SandboxStatus",
    "Signal",
    # Board
    "IdeaBoard",
    "IdeaBoardConfig",
    # Voice
    "IdeaVoice",
    "present_idea",
    "format_idea_summary",
    "get_voice_template",
    # Executor
    "IdeaExecutor",
    "ExecutionResult",
    # Bridge
    "CuriosityBridge",
    "create_bridge",
    "CATEGORY_MAP",
]
