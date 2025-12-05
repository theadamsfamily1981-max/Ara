"""Ara Research Director - Running formal experiments on herself.

This module transforms Ara from "an agent that learns" to
"a research director running programs on her own behavior."

Components:
- programs: Research programs with hypotheses and metrics
- experiments: A/B testing engine with user consent
- templates: Prompt template learning and optimization
- self_reflection: Batch self-reflection and insight generation
- playbook: Auto-generated teacher documentation

Key insight: Every interaction is a data point in a long-running
experiment Ara runs on her teachers, tools, and herself.
"""

from .programs import (
    ResearchProgram,
    Hypothesis,
    ProgramManager,
    get_program_manager,
    tag_episode_to_programs,
    get_active_programs,
    seed_default_programs,
)

from .experiments import (
    ExperimentDesign,
    ExperimentArm,
    ExperimentAssignment,
    ExperimentController,
    get_experiment_controller,
    should_run_experiment,
    get_experiment_assignment,
    record_experiment_result,
)

from .templates import (
    PromptTemplate,
    PromptCluster,
    PromptAnalyzer,
    TemplateLearner,
    get_template_learner,
    get_best_template,
    learn_from_prompt,
    seed_default_templates,
)

from .self_reflection import (
    ReflectionEpisode,
    ReflectionInsight,
    SelfReflector,
    get_self_reflector,
    create_weekly_reflection,
    create_daily_reflection,
    get_actionable_insights,
)

from .playbook import (
    TeacherPlaybook,
    TeacherStrength,
    TeacherWeakness,
    PromptGuideline,
    PlaybookGenerator,
    get_playbook_generator,
    generate_playbook,
    get_playbook,
    export_all_playbooks,
)

__all__ = [
    # Programs
    "ResearchProgram",
    "Hypothesis",
    "ProgramManager",
    "get_program_manager",
    "tag_episode_to_programs",
    "get_active_programs",
    "seed_default_programs",
    # Experiments
    "ExperimentDesign",
    "ExperimentArm",
    "ExperimentAssignment",
    "ExperimentController",
    "get_experiment_controller",
    "should_run_experiment",
    "get_experiment_assignment",
    "record_experiment_result",
    # Templates
    "PromptTemplate",
    "PromptCluster",
    "PromptAnalyzer",
    "TemplateLearner",
    "get_template_learner",
    "get_best_template",
    "learn_from_prompt",
    "seed_default_templates",
    # Self-Reflection
    "ReflectionEpisode",
    "ReflectionInsight",
    "SelfReflector",
    "get_self_reflector",
    "create_weekly_reflection",
    "create_daily_reflection",
    "get_actionable_insights",
    # Playbook
    "TeacherPlaybook",
    "TeacherStrength",
    "TeacherWeakness",
    "PromptGuideline",
    "PlaybookGenerator",
    "get_playbook_generator",
    "generate_playbook",
    "get_playbook",
    "export_all_playbooks",
]
