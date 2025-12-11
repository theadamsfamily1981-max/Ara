# ara_hive/patterns/library.py
"""
Pattern Library - Reusable Templates and Configurations
=======================================================

The Pattern Library stores and retrieves reusable patterns:
    - Role configurations
    - SOP templates
    - Team compositions
    - Workflow blueprints

This enables:
    - Knowledge accumulation over time
    - Best practice sharing
    - Quick task setup
    - Continuous improvement

Categories:
    - development: Software development patterns
    - research: Research and analysis patterns
    - content: Content creation patterns
    - data: Data processing patterns
    - operations: DevOps and operations patterns

Usage:
    from ara_hive.patterns.library import get_pattern_library

    library = get_pattern_library()

    # Get a pattern
    pattern = library.get("software_development_basic")

    # List available patterns
    patterns = library.list_patterns(category="development")

    # Register custom pattern
    library.register(my_pattern)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from .roles import (
    Role,
    RoleType,
    ProductManager,
    Architect,
    Engineer,
    QATester,
    Researcher,
    Writer,
    Reviewer,
)
from .sop import SOP, SOPPhase, SOPStep, SOPStepType, SOPPhaseType
from .teams import TeamConfig, TeamMode, CommunicationPattern, HiveTeam

log = logging.getLogger("Hive.Patterns.Library")


# =============================================================================
# Types
# =============================================================================

class PatternCategory(str, Enum):
    """Categories of patterns."""
    DEVELOPMENT = "development"
    RESEARCH = "research"
    CONTENT = "content"
    DATA = "data"
    OPERATIONS = "operations"
    AUTOMATION = "automation"
    CUSTOM = "custom"


class PatternType(str, Enum):
    """Types of patterns."""
    ROLE = "role"
    SOP = "sop"
    TEAM = "team"
    WORKFLOW = "workflow"


# =============================================================================
# Pattern
# =============================================================================

@dataclass
class Pattern:
    """
    A reusable pattern in the library.

    Patterns can be roles, SOPs, teams, or complete workflows.
    """
    id: str
    name: str
    pattern_type: PatternType
    category: PatternCategory

    # The actual pattern content
    content: Union[Role, SOP, TeamConfig, Dict[str, Any]]

    # Metadata
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)

    # Usage tracking
    usage_count: int = 0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None

    # Relations
    depends_on: List[str] = field(default_factory=list)  # Other pattern IDs
    conflicts_with: List[str] = field(default_factory=list)

    def use(self) -> Any:
        """Mark pattern as used and return content."""
        self.usage_count += 1
        self.last_used = datetime.utcnow()
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.pattern_type.value,
            "category": self.category.value,
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
        }


# =============================================================================
# Pattern Library
# =============================================================================

class PatternLibrary:
    """
    Central repository of reusable patterns.

    Stores and retrieves patterns for quick task setup.
    """

    def __init__(self):
        self._patterns: Dict[str, Pattern] = {}
        self._by_category: Dict[PatternCategory, List[str]] = {
            c: [] for c in PatternCategory
        }
        self._by_type: Dict[PatternType, List[str]] = {
            t: [] for t in PatternType
        }
        self._by_tag: Dict[str, List[str]] = {}

        # Load built-in patterns
        self._load_builtin_patterns()

        log.info(f"PatternLibrary initialized with {len(self._patterns)} patterns")

    # =========================================================================
    # Registration
    # =========================================================================

    def register(self, pattern: Pattern) -> None:
        """Register a pattern in the library."""
        if pattern.id in self._patterns:
            log.warning(f"Overwriting existing pattern: {pattern.id}")

        self._patterns[pattern.id] = pattern

        # Index by category
        if pattern.id not in self._by_category[pattern.category]:
            self._by_category[pattern.category].append(pattern.id)

        # Index by type
        if pattern.id not in self._by_type[pattern.pattern_type]:
            self._by_type[pattern.pattern_type].append(pattern.id)

        # Index by tags
        for tag in pattern.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            if pattern.id not in self._by_tag[tag]:
                self._by_tag[tag].append(pattern.id)

        log.info(f"Registered pattern: {pattern.id} ({pattern.pattern_type.value})")

    def unregister(self, pattern_id: str) -> bool:
        """Remove a pattern from the library."""
        if pattern_id not in self._patterns:
            return False

        pattern = self._patterns.pop(pattern_id)

        # Remove from indices
        if pattern_id in self._by_category[pattern.category]:
            self._by_category[pattern.category].remove(pattern_id)
        if pattern_id in self._by_type[pattern.pattern_type]:
            self._by_type[pattern.pattern_type].remove(pattern_id)
        for tag in pattern.tags:
            if tag in self._by_tag and pattern_id in self._by_tag[tag]:
                self._by_tag[tag].remove(pattern_id)

        log.info(f"Unregistered pattern: {pattern_id}")
        return True

    # =========================================================================
    # Retrieval
    # =========================================================================

    def get(self, pattern_id: str) -> Optional[Pattern]:
        """Get a pattern by ID."""
        return self._patterns.get(pattern_id)

    def get_content(self, pattern_id: str) -> Optional[Any]:
        """Get pattern content and mark as used."""
        pattern = self._patterns.get(pattern_id)
        if pattern:
            return pattern.use()
        return None

    def list_patterns(
        self,
        category: Optional[PatternCategory] = None,
        pattern_type: Optional[PatternType] = None,
        tag: Optional[str] = None,
    ) -> List[Pattern]:
        """List patterns with optional filtering."""
        pattern_ids = set(self._patterns.keys())

        if category:
            pattern_ids &= set(self._by_category.get(category, []))
        if pattern_type:
            pattern_ids &= set(self._by_type.get(pattern_type, []))
        if tag:
            pattern_ids &= set(self._by_tag.get(tag, []))

        return [self._patterns[pid] for pid in pattern_ids]

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Pattern]:
        """Search patterns by keyword."""
        query_lower = query.lower()
        results = []

        for pattern in self._patterns.values():
            score = 0

            # Check name
            if query_lower in pattern.name.lower():
                score += 3

            # Check description
            if query_lower in pattern.description.lower():
                score += 2

            # Check tags
            for tag in pattern.tags:
                if query_lower in tag.lower():
                    score += 1

            if score > 0:
                results.append((score, pattern))

        # Sort by score and return top results
        results.sort(key=lambda x: (-x[0], -x[1].usage_count))
        return [p for _, p in results[:limit]]

    def get_by_category(self, category: PatternCategory) -> List[Pattern]:
        """Get all patterns in a category."""
        return self.list_patterns(category=category)

    def get_by_type(self, pattern_type: PatternType) -> List[Pattern]:
        """Get all patterns of a type."""
        return self.list_patterns(pattern_type=pattern_type)

    def get_by_tag(self, tag: str) -> List[Pattern]:
        """Get all patterns with a tag."""
        return self.list_patterns(tag=tag)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        return {
            "total_patterns": len(self._patterns),
            "by_category": {c.value: len(ids) for c, ids in self._by_category.items() if ids},
            "by_type": {t.value: len(ids) for t, ids in self._by_type.items() if ids},
            "most_used": self._get_most_used(5),
            "tags": list(self._by_tag.keys()),
        }

    def _get_most_used(self, limit: int) -> List[Dict[str, Any]]:
        """Get most used patterns."""
        sorted_patterns = sorted(
            self._patterns.values(),
            key=lambda p: p.usage_count,
            reverse=True,
        )
        return [p.to_dict() for p in sorted_patterns[:limit]]

    # =========================================================================
    # Built-in Patterns
    # =========================================================================

    def _load_builtin_patterns(self) -> None:
        """Load built-in patterns into the library."""
        self._load_role_patterns()
        self._load_sop_patterns()
        self._load_team_patterns()
        self._load_workflow_patterns()

    def _load_role_patterns(self) -> None:
        """Load built-in role patterns."""
        roles = [
            ("pm", ProductManager(), "Product Manager role for requirements"),
            ("architect", Architect(), "Architect role for system design"),
            ("engineer", Engineer(), "Engineer role for implementation"),
            ("qa", QATester(), "QA Tester role for validation"),
            ("researcher", Researcher(), "Researcher role for information gathering"),
            ("writer", Writer(), "Writer role for content creation"),
            ("reviewer", Reviewer(), "Reviewer role for feedback"),
        ]

        for role_id, role, description in roles:
            self.register(Pattern(
                id=f"role_{role_id}",
                name=role.name,
                pattern_type=PatternType.ROLE,
                category=self._role_category(role.role_type),
                content=role,
                description=description,
                tags=["role", role.role_type.value],
            ))

    def _role_category(self, role_type: RoleType) -> PatternCategory:
        """Map role type to pattern category."""
        mapping = {
            RoleType.PLANNING: PatternCategory.DEVELOPMENT,
            RoleType.DESIGN: PatternCategory.DEVELOPMENT,
            RoleType.IMPLEMENTATION: PatternCategory.DEVELOPMENT,
            RoleType.VALIDATION: PatternCategory.DEVELOPMENT,
            RoleType.RESEARCH: PatternCategory.RESEARCH,
            RoleType.CONTENT: PatternCategory.CONTENT,
            RoleType.OPERATIONS: PatternCategory.OPERATIONS,
        }
        return mapping.get(role_type, PatternCategory.CUSTOM)

    def _load_sop_patterns(self) -> None:
        """Load built-in SOP patterns."""
        # Code Review SOP
        code_review_sop = SOP(
            name="code_review",
            description="Standard code review procedure",
            phases=[
                SOPPhase(
                    name="preparation",
                    phase_type=SOPPhaseType.PLANNING,
                    steps=[
                        SOPStep(
                            name="checkout_code",
                            action=SOPStepType.TRANSFORM,
                            description="Get the code to review",
                        ),
                        SOPStep(
                            name="understand_context",
                            action=SOPStepType.ANALYZE,
                            description="Understand the PR context and requirements",
                        ),
                    ],
                ),
                SOPPhase(
                    name="review",
                    phase_type=SOPPhaseType.EXECUTION,
                    steps=[
                        SOPStep(
                            name="check_logic",
                            action=SOPStepType.VALIDATE,
                            description="Verify business logic correctness",
                        ),
                        SOPStep(
                            name="check_style",
                            action=SOPStepType.VALIDATE,
                            description="Check code style and conventions",
                        ),
                        SOPStep(
                            name="check_security",
                            action=SOPStepType.VALIDATE,
                            description="Look for security issues",
                        ),
                        SOPStep(
                            name="check_tests",
                            action=SOPStepType.VALIDATE,
                            description="Verify test coverage",
                        ),
                    ],
                ),
                SOPPhase(
                    name="feedback",
                    phase_type=SOPPhaseType.COMPLETION,
                    steps=[
                        SOPStep(
                            name="write_comments",
                            action=SOPStepType.GENERATE,
                            description="Write review comments",
                        ),
                        SOPStep(
                            name="summarize_review",
                            action=SOPStepType.GENERATE,
                            description="Create review summary",
                        ),
                    ],
                ),
            ],
            triggers=["code_review", "pull_request", "pr_review"],
            tags=["development", "review", "code"],
        )

        self.register(Pattern(
            id="sop_code_review",
            name="Code Review",
            pattern_type=PatternType.SOP,
            category=PatternCategory.DEVELOPMENT,
            content=code_review_sop,
            description="Standard operating procedure for code review",
            tags=["sop", "code", "review", "development"],
        ))

        # Research SOP
        research_sop = SOP(
            name="research_task",
            description="Standard research procedure",
            phases=[
                SOPPhase(
                    name="gather",
                    phase_type=SOPPhaseType.PLANNING,
                    steps=[
                        SOPStep(
                            name="define_scope",
                            action=SOPStepType.ANALYZE,
                            description="Define research scope and questions",
                        ),
                        SOPStep(
                            name="search_sources",
                            action=SOPStepType.ANALYZE,
                            description="Search for relevant sources",
                            tool="web_search",
                        ),
                    ],
                ),
                SOPPhase(
                    name="analyze",
                    phase_type=SOPPhaseType.EXECUTION,
                    steps=[
                        SOPStep(
                            name="read_sources",
                            action=SOPStepType.ANALYZE,
                            description="Read and analyze sources",
                        ),
                        SOPStep(
                            name="extract_insights",
                            action=SOPStepType.TRANSFORM,
                            description="Extract key insights",
                        ),
                    ],
                ),
                SOPPhase(
                    name="synthesize",
                    phase_type=SOPPhaseType.COMPLETION,
                    steps=[
                        SOPStep(
                            name="compile_findings",
                            action=SOPStepType.GENERATE,
                            description="Compile research findings",
                        ),
                        SOPStep(
                            name="write_report",
                            action=SOPStepType.GENERATE,
                            description="Write research report",
                        ),
                    ],
                ),
            ],
            triggers=["research", "investigate", "analyze"],
            tags=["research", "analysis"],
        )

        self.register(Pattern(
            id="sop_research",
            name="Research Task",
            pattern_type=PatternType.SOP,
            category=PatternCategory.RESEARCH,
            content=research_sop,
            description="Standard operating procedure for research tasks",
            tags=["sop", "research", "analysis"],
        ))

        # Bug Fix SOP
        bug_fix_sop = SOP(
            name="bug_fix",
            description="Standard bug fix procedure",
            phases=[
                SOPPhase(
                    name="investigate",
                    phase_type=SOPPhaseType.PLANNING,
                    steps=[
                        SOPStep(
                            name="reproduce_bug",
                            action=SOPStepType.VALIDATE,
                            description="Reproduce the bug",
                        ),
                        SOPStep(
                            name="identify_cause",
                            action=SOPStepType.ANALYZE,
                            description="Identify root cause",
                        ),
                    ],
                ),
                SOPPhase(
                    name="fix",
                    phase_type=SOPPhaseType.EXECUTION,
                    steps=[
                        SOPStep(
                            name="implement_fix",
                            action=SOPStepType.GENERATE,
                            description="Implement the fix",
                        ),
                        SOPStep(
                            name="write_test",
                            action=SOPStepType.GENERATE,
                            description="Write regression test",
                        ),
                    ],
                ),
                SOPPhase(
                    name="verify",
                    phase_type=SOPPhaseType.VALIDATION,
                    steps=[
                        SOPStep(
                            name="run_tests",
                            action=SOPStepType.VALIDATE,
                            description="Run all tests",
                        ),
                        SOPStep(
                            name="verify_fix",
                            action=SOPStepType.VALIDATE,
                            description="Verify bug is fixed",
                        ),
                    ],
                ),
            ],
            triggers=["bug", "fix", "issue", "defect"],
            tags=["development", "bug", "fix"],
        )

        self.register(Pattern(
            id="sop_bug_fix",
            name="Bug Fix",
            pattern_type=PatternType.SOP,
            category=PatternCategory.DEVELOPMENT,
            content=bug_fix_sop,
            description="Standard operating procedure for fixing bugs",
            tags=["sop", "bug", "fix", "development"],
        ))

    def _load_team_patterns(self) -> None:
        """Load built-in team patterns."""
        # Software Development Team
        software_team = TeamConfig(
            name="SoftwareTeam",
            objective="Develop software from requirements to tested code",
            mode=TeamMode.SEQUENTIAL,
            communication=CommunicationPattern.PIPELINE,
            roles=["ProductManager", "Architect", "Engineer", "QATester"],
            execution_order=["ProductManager", "Architect", "Engineer", "QATester"],
            checkpoint_after_roles=["Architect", "Engineer"],
        )

        self.register(Pattern(
            id="team_software",
            name="Software Development Team",
            pattern_type=PatternType.TEAM,
            category=PatternCategory.DEVELOPMENT,
            content=software_team,
            description="Full-stack software development team",
            tags=["team", "software", "development"],
        ))

        # Research Team
        research_team = TeamConfig(
            name="ResearchTeam",
            objective="Research and document findings",
            mode=TeamMode.SEQUENTIAL,
            communication=CommunicationPattern.PIPELINE,
            roles=["Researcher", "Writer", "Reviewer"],
            execution_order=["Researcher", "Writer", "Reviewer"],
        )

        self.register(Pattern(
            id="team_research",
            name="Research Team",
            pattern_type=PatternType.TEAM,
            category=PatternCategory.RESEARCH,
            content=research_team,
            description="Research and documentation team",
            tags=["team", "research", "documentation"],
        ))

        # Quick Code Team (smaller)
        quick_code_team = TeamConfig(
            name="QuickCodeTeam",
            objective="Quickly implement small features",
            mode=TeamMode.SEQUENTIAL,
            communication=CommunicationPattern.PIPELINE,
            roles=["Engineer", "QATester"],
            execution_order=["Engineer", "QATester"],
        )

        self.register(Pattern(
            id="team_quick_code",
            name="Quick Code Team",
            pattern_type=PatternType.TEAM,
            category=PatternCategory.DEVELOPMENT,
            content=quick_code_team,
            description="Minimal team for quick implementations",
            tags=["team", "code", "quick"],
        ))

    def _load_workflow_patterns(self) -> None:
        """Load built-in workflow patterns."""
        # Feature Development Workflow
        feature_workflow = {
            "name": "feature_development",
            "description": "End-to-end feature development workflow",
            "stages": [
                {
                    "name": "requirements",
                    "role": "ProductManager",
                    "sop": "requirements_gathering",
                },
                {
                    "name": "design",
                    "role": "Architect",
                    "sop": "system_design",
                },
                {
                    "name": "implementation",
                    "role": "Engineer",
                    "sop": "code_generation",
                },
                {
                    "name": "testing",
                    "role": "QATester",
                    "sop": "test_execution",
                },
                {
                    "name": "review",
                    "role": "Reviewer",
                    "sop": "code_review",
                },
            ],
            "transitions": {
                "requirements": ["design"],
                "design": ["implementation"],
                "implementation": ["testing"],
                "testing": ["review", "implementation"],  # Can loop back
                "review": ["implementation", "complete"],
            },
        }

        self.register(Pattern(
            id="workflow_feature",
            name="Feature Development",
            pattern_type=PatternType.WORKFLOW,
            category=PatternCategory.DEVELOPMENT,
            content=feature_workflow,
            description="Complete workflow for feature development",
            tags=["workflow", "feature", "development"],
        ))

        # Data Pipeline Workflow
        data_workflow = {
            "name": "data_pipeline",
            "description": "Data processing pipeline workflow",
            "stages": [
                {
                    "name": "extract",
                    "tools": ["data_fetch", "web_scrape"],
                },
                {
                    "name": "transform",
                    "tools": ["data_transform", "data_clean"],
                },
                {
                    "name": "validate",
                    "tools": ["data_validate", "data_test"],
                },
                {
                    "name": "load",
                    "tools": ["data_store", "data_export"],
                },
            ],
            "parallel_stages": ["extract"],  # These can run in parallel
        }

        self.register(Pattern(
            id="workflow_data",
            name="Data Pipeline",
            pattern_type=PatternType.WORKFLOW,
            category=PatternCategory.DATA,
            content=data_workflow,
            description="ETL data processing workflow",
            tags=["workflow", "data", "etl", "pipeline"],
        ))


# =============================================================================
# Singleton
# =============================================================================

_default_library: Optional[PatternLibrary] = None


def get_pattern_library() -> PatternLibrary:
    """Get the global pattern library."""
    global _default_library
    if _default_library is None:
        _default_library = PatternLibrary()
    return _default_library


__all__ = [
    # Types
    "PatternCategory",
    "PatternType",
    # Pattern
    "Pattern",
    # Library
    "PatternLibrary",
    "get_pattern_library",
]
