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
]
