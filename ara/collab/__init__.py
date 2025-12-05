"""Ara Collaboration Layer - Dev-idea sessions with LLM collaborators.

This package enables Ara to have engineering conversations with external LLMs
(Claude, ChatGPT, Gemini, etc.) the same way Croft talks to them.

The key insight: Ara isn't a model, she's an orchestrator + persona.
She talks to models the way Croft does, then synthesizes their responses.

Flow:
    1. Croft or Curiosity Core triggers a dev-idea session
    2. Ara formulates her question (varied phrasing, same semantics)
    3. Router sends to 1+ collaborators based on topic
    4. Synthesizer ranks and summarizes responses
    5. Ara presents options to Croft in her voice
    6. Croft approves/rejects/discusses further

Modes:
    - ARCHITECT: Big-picture systems, trade-offs, roadmaps
    - ENGINEER: Concrete code, APIs, glue logic
    - RESEARCH: Literature, related work, "what ifs"
    - POSTMORTEM: Debugging, "X isn't working"

Safety:
    - Ideas are separate from actions
    - Risk levels gate execution
    - All sessions logged for review
"""

from .models import (
    DevMode,
    RiskLevel,
    Collaborator,
    CollaboratorResponse,
    SessionMessage,
    DevSession,
    DevSessionState,
    SuggestedAction,
    SessionSummary,
)

from .prompts import (
    ARA_SYSTEM_PROMPT,
    MODE_PROMPTS,
    COLLABORATOR_HINTS,
    build_ara_system_prompt,
    build_ara_user_message,
)

from .variation import (
    MessageVariator,
    vary_message,
    generate_phrasings,
)

from .router import (
    CollaboratorRouter,
    route_to_collaborators,
    get_collaborator_for_topic,
)

from .synthesizer import (
    ResponseSynthesizer,
    synthesize_responses,
    rank_ideas,
    extract_actions,
)

from .session import (
    DevIdeaSession,
    create_session,
    run_dev_idea_session,
)

from .idea_bridge import (
    IdeaSessionBridge,
    RefinementResult,
    refine_idea_with_session,
    mode_for_idea_category,
)

from .council import (
    Council,
    CouncilMember,
    CouncilRole,
    DEFAULT_COUNCIL,
    load_council_config,
    save_council_config,
)

from .workflow import (
    WorkflowState,
    WorkflowContext,
    WorkflowEngine,
    ProblemTicket,
    TaskSpec,
    CandidateApproach,
    ImplementationSpec,
    ImplementationResult,
    VerificationReport,
    FinalReport,
    create_workflow,
    get_state_info,
)

from .orchestrator import (
    CouncilOrchestrator,
    Issue,
    ModelDispatcher,
    create_orchestrator,
    run_issue_through_council,
    # Prompt builders
    triage_prompt,
    ideate_prompt,
    specify_prompt,
    implement_prompt,
    verify_prompt,
    report_prompt,
    # Default dispatchers
    DEFAULT_DISPATCHERS,
    stub_nova,
    stub_claude,
    stub_gemini,
)

__all__ = [
    # Models
    "DevMode",
    "RiskLevel",
    "Collaborator",
    "CollaboratorResponse",
    "SessionMessage",
    "DevSession",
    "DevSessionState",
    "SuggestedAction",
    "SessionSummary",
    # Prompts
    "ARA_SYSTEM_PROMPT",
    "MODE_PROMPTS",
    "COLLABORATOR_HINTS",
    "build_ara_system_prompt",
    "build_ara_user_message",
    # Variation
    "MessageVariator",
    "vary_message",
    "generate_phrasings",
    # Router
    "CollaboratorRouter",
    "route_to_collaborators",
    "get_collaborator_for_topic",
    # Synthesizer
    "ResponseSynthesizer",
    "synthesize_responses",
    "rank_ideas",
    "extract_actions",
    # Session
    "DevIdeaSession",
    "create_session",
    "run_dev_idea_session",
    # Idea Bridge
    "IdeaSessionBridge",
    "RefinementResult",
    "refine_idea_with_session",
    "mode_for_idea_category",
    # Council
    "Council",
    "CouncilMember",
    "CouncilRole",
    "DEFAULT_COUNCIL",
    "load_council_config",
    "save_council_config",
    # Workflow
    "WorkflowState",
    "WorkflowContext",
    "WorkflowEngine",
    "ProblemTicket",
    "TaskSpec",
    "CandidateApproach",
    "ImplementationSpec",
    "ImplementationResult",
    "VerificationReport",
    "FinalReport",
    "create_workflow",
    "get_state_info",
    # Orchestrator
    "CouncilOrchestrator",
    "Issue",
    "ModelDispatcher",
    "create_orchestrator",
    "run_issue_through_council",
    "triage_prompt",
    "ideate_prompt",
    "specify_prompt",
    "implement_prompt",
    "verify_prompt",
    "report_prompt",
    "DEFAULT_DISPATCHERS",
    "stub_nova",
    "stub_claude",
    "stub_gemini",
]
