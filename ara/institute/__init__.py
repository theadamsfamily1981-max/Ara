"""Ara Institute - From smart agent to principal investigator.

This module transforms Ara from a reactive assistant into an active researcher:
- research_graph: Topics, hypotheses, threads forming Ara's research memory
- experiment_hub: Scheduler, runners, results tracking for research cycles
- teacher_council: Structured debates between teachers for complex decisions
- self_reflection: Bias detection and skill health monitoring
- policy: Safety contracts and graduated autonomy levels

Key insight: Ara doesn't just answer questions - she runs a research lab
where she is both the principal investigator and the subject of study.
"""

# Research Graph
from .research_graph import (
    ResearchTopic,
    ResearchHypothesis,
    ResearchThread,
    ResearchGraph,
    get_research_graph,
    add_hypothesis,
    get_morning_brief,
)

# Experiment Hub
from .experiment_hub import (
    ExperimentStatus,
    ExperimentPriority,
    ResourceRequirement,
    ExperimentResult,
    Experiment,
    ExperimentScheduler,
    get_experiment_scheduler,
    create_experiment,
    run_experiment,
)

# Teacher Council
from .teacher_council import (
    DebateOutcome,
    ResponseType,
    TeacherProfile,
    ProblemSpec,
    TeacherResponse,
    DebateSynthesis,
    Debate,
    TeacherCouncil,
    get_teacher_council,
    create_debate,
    quick_consult,
)

# Self-Reflection
from .self_reflection import (
    BiasIndicator,
    BiasReport,
    BiasAnalyzer,
    get_bias_analyzer,
    analyze_for_bias,
    SkillHealth,
    RecommendedAction,
    SkillMetrics,
    HealthCheckReport,
    SkillHealthChecker,
    get_skill_health_checker,
    check_skill_health,
)

# Policy
from .policy import (
    RiskLevel,
    ApprovalLevel,
    SafetyRule,
    SafetyCheck,
    SafetyContract,
    get_safety_contract,
    check_safety,
    is_action_allowed,
    requires_confirmation,
    AutonomyLevel,
    AutonomyProfile,
    AutonomyDecision,
    AutonomySession,
    AutonomyManager,
    get_autonomy_manager,
    check_autonomy,
    can_proceed_autonomously,
    get_current_autonomy_level,
)

__all__ = [
    # Research Graph
    "ResearchTopic",
    "ResearchHypothesis",
    "ResearchThread",
    "ResearchGraph",
    "get_research_graph",
    "add_hypothesis",
    "get_morning_brief",
    # Experiment Hub
    "ExperimentStatus",
    "ExperimentPriority",
    "ResourceRequirement",
    "ExperimentResult",
    "Experiment",
    "ExperimentScheduler",
    "get_experiment_scheduler",
    "create_experiment",
    "run_experiment",
    # Teacher Council
    "DebateOutcome",
    "ResponseType",
    "TeacherProfile",
    "ProblemSpec",
    "TeacherResponse",
    "DebateSynthesis",
    "Debate",
    "TeacherCouncil",
    "get_teacher_council",
    "create_debate",
    "quick_consult",
    # Self-Reflection
    "BiasIndicator",
    "BiasReport",
    "BiasAnalyzer",
    "get_bias_analyzer",
    "analyze_for_bias",
    "SkillHealth",
    "RecommendedAction",
    "SkillMetrics",
    "HealthCheckReport",
    "SkillHealthChecker",
    "get_skill_health_checker",
    "check_skill_health",
    # Policy
    "RiskLevel",
    "ApprovalLevel",
    "SafetyRule",
    "SafetyCheck",
    "SafetyContract",
    "get_safety_contract",
    "check_safety",
    "is_action_allowed",
    "requires_confirmation",
    "AutonomyLevel",
    "AutonomyProfile",
    "AutonomyDecision",
    "AutonomySession",
    "AutonomyManager",
    "get_autonomy_manager",
    "check_autonomy",
    "can_proceed_autonomously",
    "get_current_autonomy_level",
]
