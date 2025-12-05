"""Prompt Template Learner - Ara optimizes HOW she talks to teachers.

Not just WHAT workflow to pick, but HOW to phrase requests.
For each teacher × intent pair, she:
1. Clusters successful prompts
2. Extracts skeleton templates
3. Runs small mutations to improve

Example template:
  template_id: claude_debug_stepwise_v2
  skeleton: "[TASK] {problem}\n[CONTEXT] {context}\n[CONSTRAINTS] {constraints}"
  success_rate: 0.87
  samples: 42
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A learned prompt template."""

    id: str
    teacher: str
    intent: str
    name: str

    # The template structure
    skeleton: str  # With {placeholders}
    example_prompt: str  # A concrete example
    sections: List[str] = field(default_factory=list)  # e.g., ["TASK", "CONTEXT"]

    # Statistics
    sample_count: int = 0
    success_count: int = 0
    total_reward: float = 0.0

    # Versioning
    version: int = 1
    parent_id: Optional[str] = None  # Evolved from
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def success_rate(self) -> Optional[float]:
        if self.sample_count == 0:
            return None
        return self.success_count / self.sample_count

    @property
    def avg_reward(self) -> Optional[float]:
        if self.sample_count == 0:
            return None
        return self.total_reward / self.sample_count

    def record_usage(self, success: bool, reward: float = 0.0) -> None:
        """Record a usage of this template."""
        self.sample_count += 1
        if success:
            self.success_count += 1
        self.total_reward += reward

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "teacher": self.teacher,
            "intent": self.intent,
            "name": self.name,
            "skeleton": self.skeleton,
            "example_prompt": self.example_prompt,
            "sections": self.sections,
            "sample_count": self.sample_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "avg_reward": self.avg_reward,
            "version": self.version,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        return cls(
            id=data["id"],
            teacher=data["teacher"],
            intent=data["intent"],
            name=data.get("name", data["id"]),
            skeleton=data.get("skeleton", ""),
            example_prompt=data.get("example_prompt", ""),
            sections=data.get("sections", []),
            sample_count=data.get("sample_count", 0),
            success_count=data.get("success_count", 0),
            total_reward=data.get("total_reward", 0.0),
            version=data.get("version", 1),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )


@dataclass
class PromptCluster:
    """A cluster of similar prompts."""

    id: str
    teacher: str
    intent: str

    # Cluster characteristics
    pattern: str  # Regex or structural pattern
    common_sections: List[str] = field(default_factory=list)
    avg_length: float = 0.0

    # Members
    prompt_samples: List[str] = field(default_factory=list)
    sample_count: int = 0
    success_count: int = 0

    @property
    def success_rate(self) -> Optional[float]:
        if self.sample_count == 0:
            return None
        return self.success_count / self.sample_count


class PromptAnalyzer:
    """Analyzes prompt structure and extracts patterns."""

    # Common section headers in structured prompts
    SECTION_PATTERNS = [
        r"\[([A-Z_]+)\]",  # [TASK], [CONTEXT]
        r"##\s*([A-Za-z_]+)",  # ## Task, ## Context
        r"\*\*([A-Za-z_]+)\*\*:",  # **Task**:
        r"^([A-Z][a-z]+):",  # Task:, Context:
    ]

    @classmethod
    def extract_sections(cls, prompt: str) -> List[str]:
        """Extract section headers from a prompt."""
        sections = []
        for pattern in cls.SECTION_PATTERNS:
            matches = re.findall(pattern, prompt, re.MULTILINE)
            sections.extend(matches)
        return list(dict.fromkeys(sections))  # Dedupe preserving order

    @classmethod
    def extract_skeleton(cls, prompt: str) -> str:
        """Extract a skeleton template from a prompt.

        Replaces specific content with placeholders.
        """
        skeleton = prompt

        # Replace code blocks with placeholder
        skeleton = re.sub(
            r"```[\w]*\n.*?```",
            "{code_block}",
            skeleton,
            flags=re.DOTALL,
        )

        # Replace URLs
        skeleton = re.sub(
            r"https?://\S+",
            "{url}",
            skeleton,
        )

        # Replace file paths
        skeleton = re.sub(
            r"(/[\w./]+\.\w+)",
            "{file_path}",
            skeleton,
        )

        # Replace quoted strings (likely specific values)
        skeleton = re.sub(
            r'"[^"]{20,}"',
            '"{long_string}"',
            skeleton,
        )

        # Replace numbers
        skeleton = re.sub(
            r"\b\d{3,}\b",
            "{number}",
            skeleton,
        )

        return skeleton

    @classmethod
    def compute_similarity(cls, prompt1: str, prompt2: str) -> float:
        """Compute structural similarity between prompts."""
        # Compare section headers
        sections1 = set(cls.extract_sections(prompt1))
        sections2 = set(cls.extract_sections(prompt2))

        if not sections1 and not sections2:
            # No sections - compare by length ratio
            len1, len2 = len(prompt1), len(prompt2)
            return min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

        # Jaccard similarity on sections
        intersection = len(sections1 & sections2)
        union = len(sections1 | sections2)
        return intersection / union if union > 0 else 0.0

    @classmethod
    def classify_style(cls, prompt: str) -> str:
        """Classify the prompt style."""
        sections = cls.extract_sections(prompt)

        if len(sections) >= 3:
            return "structured"
        elif "step" in prompt.lower() or "1." in prompt:
            return "stepwise"
        elif len(prompt) < 100:
            return "concise"
        elif "```" in prompt:
            return "code_heavy"
        else:
            return "narrative"


class TemplateLearner:
    """Learns and manages prompt templates."""

    def __init__(self, templates_path: Optional[Path] = None):
        """Initialize the learner.

        Args:
            templates_path: Path to templates JSONL file
        """
        self.templates_path = templates_path or (
            Path.home() / ".ara" / "meta" / "research" / "templates.jsonl"
        )
        self.templates_path.parent.mkdir(parents=True, exist_ok=True)

        self._templates: Dict[str, PromptTemplate] = {}
        self._clusters: Dict[str, PromptCluster] = {}
        self._loaded = False
        self._next_id = 1

        self.analyzer = PromptAnalyzer()

    def _load(self, force: bool = False) -> None:
        """Load templates from disk."""
        if self._loaded and not force:
            return

        self._templates.clear()

        if self.templates_path.exists():
            try:
                with open(self.templates_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            template = PromptTemplate.from_dict(data)
                            self._templates[template.id] = template
                            # Update ID counter
                            if template.id.startswith("TPL-"):
                                try:
                                    num = int(template.id[4:8])
                                    self._next_id = max(self._next_id, num + 1)
                                except ValueError:
                                    pass
                        except Exception as e:
                            logger.warning(f"Failed to parse template: {e}")
            except Exception as e:
                logger.warning(f"Failed to load templates: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save templates to disk."""
        with open(self.templates_path, "w") as f:
            for template in self._templates.values():
                f.write(json.dumps(template.to_dict()) + "\n")

    def _generate_id(self, teacher: str, intent: str) -> str:
        """Generate a unique template ID."""
        id_str = f"TPL-{self._next_id:04d}-{teacher[:3]}-{intent[:4]}"
        self._next_id += 1
        return id_str

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID."""
        self._load()
        return self._templates.get(template_id)

    def get_templates_for(
        self,
        teacher: str,
        intent: str,
    ) -> List[PromptTemplate]:
        """Get templates for a teacher/intent pair."""
        self._load()
        return [
            t for t in self._templates.values()
            if t.teacher == teacher and t.intent == intent
        ]

    def get_best_template(
        self,
        teacher: str,
        intent: str,
        min_samples: int = 5,
    ) -> Optional[PromptTemplate]:
        """Get the best-performing template.

        Args:
            teacher: Teacher name
            intent: Intent classification
            min_samples: Minimum samples required

        Returns:
            Best template or None
        """
        candidates = [
            t for t in self.get_templates_for(teacher, intent)
            if t.sample_count >= min_samples
        ]

        if not candidates:
            return None

        # Sort by success rate, then by reward
        candidates.sort(
            key=lambda t: (t.success_rate or 0, t.avg_reward or 0),
            reverse=True,
        )

        return candidates[0]

    def learn_from_prompt(
        self,
        prompt: str,
        teacher: str,
        intent: str,
        success: bool,
        reward: float = 0.0,
    ) -> Optional[PromptTemplate]:
        """Learn from an observed prompt.

        Args:
            prompt: The prompt text
            teacher: Teacher used
            intent: Intent classification
            success: Whether it succeeded
            reward: Quality score

        Returns:
            Matched or new template
        """
        self._load()

        # Extract features
        skeleton = self.analyzer.extract_skeleton(prompt)
        sections = self.analyzer.extract_sections(prompt)
        style = self.analyzer.classify_style(prompt)

        # Find matching template
        existing = self.get_templates_for(teacher, intent)
        best_match = None
        best_similarity = 0.0

        for template in existing:
            similarity = self.analyzer.compute_similarity(
                skeleton, template.skeleton
            )
            if similarity > best_similarity and similarity > 0.7:
                best_similarity = similarity
                best_match = template

        if best_match:
            # Update existing template
            best_match.record_usage(success, reward)
            self._save()
            return best_match

        # Create new template
        template = PromptTemplate(
            id=self._generate_id(teacher, intent),
            teacher=teacher,
            intent=intent,
            name=f"{teacher}_{intent}_{style}_v1",
            skeleton=skeleton,
            example_prompt=prompt[:500],  # Truncate
            sections=sections,
            tags=[style],
        )
        template.record_usage(success, reward)

        self._templates[template.id] = template
        self._save()
        logger.info(f"Created new template: {template.id}")

        return template

    def mutate_template(
        self,
        template_id: str,
        mutation_type: str = "section_reorder",
    ) -> Optional[PromptTemplate]:
        """Create a mutation of a template.

        Args:
            template_id: Template to mutate
            mutation_type: Type of mutation

        Returns:
            New mutated template
        """
        self._load()

        parent = self._templates.get(template_id)
        if not parent:
            return None

        # Create mutation
        new_skeleton = parent.skeleton
        new_sections = parent.sections.copy()

        if mutation_type == "section_reorder" and len(new_sections) >= 2:
            # Swap two sections
            import random
            i, j = random.sample(range(len(new_sections)), 2)
            new_sections[i], new_sections[j] = new_sections[j], new_sections[i]

        elif mutation_type == "add_constraints":
            # Add a constraints section
            if "CONSTRAINTS" not in new_sections:
                new_sections.append("CONSTRAINTS")
                new_skeleton += "\n[CONSTRAINTS] {constraints}"

        elif mutation_type == "add_examples":
            # Add examples section
            if "EXAMPLES" not in new_sections:
                new_sections.append("EXAMPLES")
                new_skeleton += "\n[EXAMPLES] {examples}"

        elif mutation_type == "simplify":
            # Remove least common section
            if len(new_sections) > 2:
                new_sections = new_sections[:2]

        mutant = PromptTemplate(
            id=self._generate_id(parent.teacher, parent.intent),
            teacher=parent.teacher,
            intent=parent.intent,
            name=f"{parent.name.rsplit('_v', 1)[0]}_v{parent.version + 1}",
            skeleton=new_skeleton,
            example_prompt=parent.example_prompt,
            sections=new_sections,
            version=parent.version + 1,
            parent_id=parent.id,
            tags=parent.tags + [f"mutation:{mutation_type}"],
        )

        self._templates[mutant.id] = mutant
        self._save()
        logger.info(f"Created mutation: {mutant.id} from {parent.id}")

        return mutant

    def get_template_lineage(self, template_id: str) -> List[PromptTemplate]:
        """Get the evolution lineage of a template."""
        self._load()

        lineage = []
        current_id = template_id

        while current_id:
            template = self._templates.get(current_id)
            if not template:
                break
            lineage.append(template)
            current_id = template.parent_id

        return list(reversed(lineage))

    def get_top_templates(
        self,
        limit: int = 10,
        min_samples: int = 5,
    ) -> List[PromptTemplate]:
        """Get top-performing templates across all teachers."""
        self._load()

        candidates = [
            t for t in self._templates.values()
            if t.sample_count >= min_samples
        ]

        candidates.sort(
            key=lambda t: (t.success_rate or 0, t.avg_reward or 0),
            reverse=True,
        )

        return candidates[:limit]

    def get_templates_needing_data(
        self,
        max_samples: int = 10,
        limit: int = 10,
    ) -> List[PromptTemplate]:
        """Get templates that need more data."""
        self._load()

        candidates = [
            t for t in self._templates.values()
            if t.sample_count < max_samples
        ]

        # Prioritize templates with promising early results
        candidates.sort(
            key=lambda t: (
                t.success_rate or 0.5,  # Default to 50% if no data
                -t.sample_count,  # Prefer fewer samples
            ),
            reverse=True,
        )

        return candidates[:limit]

    def apply_template(
        self,
        template: PromptTemplate,
        variables: Dict[str, str],
    ) -> str:
        """Apply a template with variables.

        Args:
            template: The template
            variables: Variable values

        Returns:
            Filled prompt
        """
        result = template.skeleton

        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", value)

        # Remove unfilled placeholders
        result = re.sub(r"\{[a-z_]+\}", "", result)

        return result.strip()

    def get_summary(self) -> Dict[str, Any]:
        """Get learner summary."""
        self._load()

        # Group by teacher
        by_teacher: Dict[str, int] = defaultdict(int)
        by_intent: Dict[str, int] = defaultdict(int)

        for t in self._templates.values():
            by_teacher[t.teacher] += 1
            by_intent[t.intent] += 1

        # Top templates
        top = self.get_top_templates(5, min_samples=3)

        return {
            "total_templates": len(self._templates),
            "by_teacher": dict(by_teacher),
            "by_intent": dict(by_intent),
            "top_templates": [
                {
                    "id": t.id,
                    "name": t.name,
                    "success_rate": t.success_rate,
                    "samples": t.sample_count,
                }
                for t in top
            ],
        }


# =============================================================================
# Default Templates
# =============================================================================

DEFAULT_TEMPLATES = [
    {
        "id": "TPL-0001-cla-debu",
        "teacher": "claude",
        "intent": "debug_code",
        "name": "claude_debug_structured_v1",
        "skeleton": "[PROBLEM]\n{problem}\n\n[CODE]\n{code_block}\n\n[ERROR]\n{error}\n\n[CONTEXT]\n{context}",
        "sections": ["PROBLEM", "CODE", "ERROR", "CONTEXT"],
        "tags": ["structured"],
    },
    {
        "id": "TPL-0002-nov-desi",
        "teacher": "nova",
        "intent": "design_arch",
        "name": "nova_design_review_v1",
        "skeleton": "[OBJECTIVE]\n{objective}\n\n[CURRENT DESIGN]\n{design}\n\n[CONSTRAINTS]\n{constraints}\n\n[QUESTIONS]\n{questions}",
        "sections": ["OBJECTIVE", "CURRENT DESIGN", "CONSTRAINTS", "QUESTIONS"],
        "tags": ["structured"],
    },
    {
        "id": "TPL-0003-gem-rese",
        "teacher": "gemini",
        "intent": "research",
        "name": "gemini_research_exploration_v1",
        "skeleton": "Research topic: {topic}\n\nWhat I know so far:\n{context}\n\nSpecific questions:\n{questions}\n\nDepth: {depth}",
        "sections": [],
        "tags": ["narrative"],
    },
    {
        "id": "TPL-0004-cla-impl",
        "teacher": "claude",
        "intent": "implement",
        "name": "claude_implement_stepwise_v1",
        "skeleton": "[TASK]\n{task}\n\n[REQUIREMENTS]\n{requirements}\n\n[EXISTING CODE]\n{code_block}\n\n[STEPS]\n1. {step1}\n2. {step2}\n3. {step3}",
        "sections": ["TASK", "REQUIREMENTS", "EXISTING CODE", "STEPS"],
        "tags": ["structured", "stepwise"],
    },
]


def seed_default_templates(learner: TemplateLearner) -> int:
    """Seed default templates.

    Args:
        learner: Template learner

    Returns:
        Number seeded
    """
    seeded = 0
    for tpl_data in DEFAULT_TEMPLATES:
        if not learner.get_template(tpl_data["id"]):
            learner._load()
            template = PromptTemplate.from_dict(tpl_data)
            learner._templates[template.id] = template
            seeded += 1

    if seeded:
        learner._save()

    return seeded


# =============================================================================
# Convenience Functions
# =============================================================================

_default_learner: Optional[TemplateLearner] = None


def get_template_learner() -> TemplateLearner:
    """Get the default template learner."""
    global _default_learner
    if _default_learner is None:
        _default_learner = TemplateLearner()
    return _default_learner


def learn_from_prompt(
    prompt: str,
    teacher: str,
    intent: str,
    success: bool,
    reward: float = 0.0,
) -> Optional[PromptTemplate]:
    """Learn from an observed prompt."""
    return get_template_learner().learn_from_prompt(
        prompt, teacher, intent, success, reward
    )


def get_best_template(teacher: str, intent: str) -> Optional[PromptTemplate]:
    """Get the best template for a teacher/intent."""
    return get_template_learner().get_best_template(teacher, intent)
