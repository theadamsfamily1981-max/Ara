"""Skill Codegen Lite - A tiny student that extracts patterns into skills.

This is one of Ara's "students" - a small, specialized helper that:
1. Scans teacher session logs
2. Identifies repeating code structures
3. Proposes skill templates for the registry

It's not a full LLM - just a clever rule engine with pattern matching.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CodePattern:
    """A code pattern detected across sessions."""

    id: str
    language: str
    pattern_type: str  # "function", "class", "template", "snippet"

    # Structure
    skeleton: str  # Code with placeholders
    placeholders: List[str] = field(default_factory=list)

    # Context
    typical_imports: List[str] = field(default_factory=list)
    typical_context: str = ""

    # Statistics
    occurrences: int = 0
    teachers_seen: List[str] = field(default_factory=list)
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "language": self.language,
            "pattern_type": self.pattern_type,
            "skeleton": self.skeleton,
            "placeholders": self.placeholders,
            "typical_imports": self.typical_imports,
            "typical_context": self.typical_context,
            "occurrences": self.occurrences,
            "teachers_seen": self.teachers_seen,
            "success_rate": round(self.success_rate, 3),
        }


@dataclass
class SkillTemplate:
    """A proposed skill template."""

    name: str
    description: str
    language: str

    # Template
    template_code: str
    placeholders: Dict[str, str] = field(default_factory=dict)  # name -> description

    # Usage
    trigger_phrases: List[str] = field(default_factory=list)
    example_usage: str = ""

    # Metadata
    learned_from: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "template_code": self.template_code,
            "placeholders": self.placeholders,
            "trigger_phrases": self.trigger_phrases,
            "example_usage": self.example_usage,
            "learned_from": self.learned_from,
            "confidence": round(self.confidence, 3),
        }

    def apply(self, variables: Dict[str, str]) -> str:
        """Apply template with variables.

        Args:
            variables: Mapping of placeholder name to value

        Returns:
            Filled template
        """
        result = self.template_code
        for name, value in variables.items():
            result = result.replace(f"{{{name}}}", value)
        return result


class CodegenLite:
    """Tiny student for extracting code patterns."""

    # Common code patterns to look for
    PYTHON_PATTERNS = [
        # Function definition
        (
            r"def\s+(\w+)\s*\([^)]*\):\s*\n(?:\s+.*\n)+",
            "function",
        ),
        # Class definition
        (
            r"class\s+(\w+)(?:\([^)]*\))?:\s*\n(?:\s+.*\n)+",
            "class",
        ),
        # With statement
        (
            r"with\s+\w+\([^)]*\)\s+as\s+\w+:\s*\n(?:\s+.*\n)+",
            "context_manager",
        ),
        # Try-except block
        (
            r"try:\s*\n(?:\s+.*\n)+except[^:]*:\s*\n(?:\s+.*\n)+",
            "error_handling",
        ),
    ]

    def __init__(self, log_path: Optional[Path] = None):
        """Initialize the student.

        Args:
            log_path: Path to interaction logs
        """
        self.log_path = log_path or (
            Path.home() / ".ara" / "meta" / "interactions.jsonl"
        )

        self._patterns: Dict[str, CodePattern] = {}
        self._templates: List[SkillTemplate] = []
        self._next_id = 1

    def _extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """Extract code blocks from text.

        Args:
            text: Text containing code blocks

        Returns:
            List of (language, code) tuples
        """
        blocks = []

        # Match ```language\ncode\n```
        pattern = r"```(\w*)\n(.*?)```"
        for match in re.finditer(pattern, text, re.DOTALL):
            lang = match.group(1) or "unknown"
            code = match.group(2).strip()
            if code:
                blocks.append((lang, code))

        return blocks

    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison.

        - Replace specific names with placeholders
        - Remove extra whitespace
        """
        normalized = code

        # Replace string literals
        normalized = re.sub(r'"[^"]*"', '"{string}"', normalized)
        normalized = re.sub(r"'[^']*'", "'{string}'", normalized)

        # Replace numbers
        normalized = re.sub(r"\b\d+\b", "{number}", normalized)

        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()

    def _find_common_structure(
        self,
        codes: List[str],
    ) -> Optional[str]:
        """Find common structure across code snippets.

        Args:
            codes: List of code snippets

        Returns:
            Template skeleton or None
        """
        if len(codes) < 2:
            return None

        # Normalize all codes
        normalized = [self._normalize_code(c) for c in codes]

        # Find longest common subsequence (simplified)
        # For now, just check if they share the same structure
        first = normalized[0]
        all_similar = all(
            self._structural_similarity(first, n) > 0.7
            for n in normalized[1:]
        )

        if all_similar:
            return first

        return None

    def _structural_similarity(self, code1: str, code2: str) -> float:
        """Compute structural similarity between code snippets."""
        # Simple token-based similarity
        tokens1 = set(code1.split())
        tokens2 = set(code2.split())

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union

    def _extract_placeholders(self, template: str) -> List[str]:
        """Extract placeholder names from template."""
        placeholders = re.findall(r"\{(\w+)\}", template)
        return list(set(placeholders))

    def analyze_sessions(
        self,
        days: int = 30,
        min_occurrences: int = 3,
    ) -> List[CodePattern]:
        """Analyze sessions for code patterns.

        Args:
            days: Days of history
            min_occurrences: Minimum pattern occurrences

        Returns:
            Detected patterns
        """
        if not self.log_path.exists():
            return []

        # Group code blocks by normalized structure
        structure_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        try:
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)

                        # Extract code blocks from response
                        response = record.get("response_summary", "")
                        blocks = self._extract_code_blocks(response)

                        for lang, code in blocks:
                            normalized = self._normalize_code(code)
                            key = f"{lang}:{normalized[:100]}"  # Truncate for grouping

                            structure_groups[key].append({
                                "code": code,
                                "language": lang,
                                "teacher": record.get("primary_teacher", "unknown"),
                                "success": record.get("success", True),
                            })
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.warning(f"Failed to analyze sessions: {e}")
            return []

        # Convert groups to patterns
        patterns = []
        for key, group in structure_groups.items():
            if len(group) < min_occurrences:
                continue

            # Get common structure
            codes = [g["code"] for g in group]
            skeleton = self._find_common_structure(codes)
            if not skeleton:
                skeleton = codes[0]  # Use first as template

            # Collect statistics
            teachers = list(set(g["teacher"] for g in group))
            successes = sum(1 for g in group if g["success"])
            success_rate = successes / len(group)

            lang = group[0]["language"]

            pattern = CodePattern(
                id=f"CPAT-{self._next_id:04d}",
                language=lang,
                pattern_type="snippet",
                skeleton=skeleton,
                placeholders=self._extract_placeholders(skeleton),
                occurrences=len(group),
                teachers_seen=teachers,
                success_rate=success_rate,
            )

            self._patterns[pattern.id] = pattern
            patterns.append(pattern)
            self._next_id += 1

        logger.info(f"Found {len(patterns)} code patterns")
        return patterns

    def generate_templates(
        self,
        min_confidence: float = 0.6,
    ) -> List[SkillTemplate]:
        """Generate skill templates from patterns.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            Generated templates
        """
        templates = []

        for pattern in self._patterns.values():
            # Compute confidence based on occurrences and success rate
            confidence = min(0.9, pattern.occurrences * 0.1) * pattern.success_rate

            if confidence < min_confidence:
                continue

            # Generate template
            template = SkillTemplate(
                name=f"{pattern.language}_{pattern.pattern_type}_{pattern.id[-4:]}",
                description=f"Auto-generated {pattern.pattern_type} template for {pattern.language}",
                language=pattern.language,
                template_code=pattern.skeleton,
                placeholders={p: f"Value for {p}" for p in pattern.placeholders},
                trigger_phrases=[
                    f"generate {pattern.pattern_type}",
                    f"{pattern.language} {pattern.pattern_type}",
                ],
                learned_from=pattern.teachers_seen,
                confidence=confidence,
            )

            templates.append(template)
            self._templates.append(template)

        logger.info(f"Generated {len(templates)} skill templates")
        return templates

    def get_template(self, name: str) -> Optional[SkillTemplate]:
        """Get a template by name."""
        for t in self._templates:
            if t.name == name:
                return t
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get student summary."""
        return {
            "patterns_detected": len(self._patterns),
            "templates_generated": len(self._templates),
            "languages": list(set(p.language for p in self._patterns.values())),
            "top_patterns": [
                {
                    "id": p.id,
                    "language": p.language,
                    "occurrences": p.occurrences,
                    "success_rate": p.success_rate,
                }
                for p in sorted(
                    self._patterns.values(),
                    key=lambda x: x.occurrences,
                    reverse=True,
                )[:5]
            ],
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_student: Optional[CodegenLite] = None


def get_codegen_lite() -> CodegenLite:
    """Get the default codegen lite student."""
    global _default_student
    if _default_student is None:
        _default_student = CodegenLite()
    return _default_student


def analyze_and_generate_templates(days: int = 30) -> List[SkillTemplate]:
    """Analyze sessions and generate templates."""
    student = get_codegen_lite()
    student.analyze_sessions(days=days)
    return student.generate_templates()
