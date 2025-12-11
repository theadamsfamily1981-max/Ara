"""Ara Academy - Ara stops just studying her teachers and starts teaching.

This module transforms Ara from "student of teachers" to "teacher of skills":
- skills/: Learned skills registry and extraction
- students/: Tiny local helpers for pattern mining
- curriculum/: What to internalize vs outsource

Key insight: Ara compresses what works from teachers into local, reusable
skills - only escalating when things get weird or novel.
"""

from .skills.registry import (
    LearnedSkill,
    SkillRegistry,
    SkillImplementation,
    EscalationPolicy,
    get_skill_registry,
    find_skill,
    register_skill,
    seed_default_skills,
)

from .skills.extractor import (
    SkillExtractor,
    SessionPattern,
    SkillProposal,
    PatternDetector,
    get_skill_extractor,
    extract_skills_from_logs,
)

from .curriculum.internalization import (
    Curriculum,
    CurriculumManager,
    InternalizationCriteria,
    AvoidanceRule,
    get_curriculum_manager,
    should_internalize,
)

from .students.codegen_lite import (
    CodegenLite,
    CodePattern,
    SkillTemplate,
    get_codegen_lite,
    analyze_and_generate_templates,
)

from .dojo import (
    Dojo,
    Shadow,
    SkillSpec as DojoSkillSpec,
    StressCase,
    HardeningReport,
    HardeningResult,
    StressCaseType,
    get_dojo,
    harden_skill,
)

from .pipeline import (
    SkillPipeline,
    SkillCandidate,
    PipelineResult,
    PipelineStage,
    RejectionReason,
    get_pipeline,
    run_pipeline,
    discover_skills,
)

from .skills.architect import (
    Architect,
    Episode,
    SkillSpec,
    get_architect,
    generalize_episodes,
)

from .session_graph import (
    NodeType,
    EdgeType,
    Node,
    Edge,
    SessionGraph,
    SessionGraphBuilder,
    SessionStyleClassifier,
)

__all__ = [
    # Skills
    "LearnedSkill",
    "SkillRegistry",
    "SkillImplementation",
    "EscalationPolicy",
    "get_skill_registry",
    "find_skill",
    "register_skill",
    "seed_default_skills",
    # Extraction
    "SkillExtractor",
    "SessionPattern",
    "SkillProposal",
    "PatternDetector",
    "get_skill_extractor",
    "extract_skills_from_logs",
    # Curriculum
    "Curriculum",
    "CurriculumManager",
    "InternalizationCriteria",
    "AvoidanceRule",
    "get_curriculum_manager",
    "should_internalize",
    # Students
    "CodegenLite",
    "CodePattern",
    "SkillTemplate",
    "get_codegen_lite",
    "analyze_and_generate_templates",
    # Dojo
    "Dojo",
    "Shadow",
    "DojoSkillSpec",
    "StressCase",
    "HardeningReport",
    "HardeningResult",
    "StressCaseType",
    "get_dojo",
    "harden_skill",
    # Pipeline
    "SkillPipeline",
    "SkillCandidate",
    "PipelineResult",
    "PipelineStage",
    "RejectionReason",
    "get_pipeline",
    "run_pipeline",
    "discover_skills",
    # Architect
    "Architect",
    "Episode",
    "SkillSpec",
    "get_architect",
    "generalize_episodes",
    # Session Graph (Visual Cortex)
    "NodeType",
    "EdgeType",
    "Node",
    "Edge",
    "SessionGraph",
    "SessionGraphBuilder",
    "SessionStyleClassifier",
]
