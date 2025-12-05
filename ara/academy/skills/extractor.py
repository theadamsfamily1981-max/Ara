"""Skill Extractor - Mine teacher sessions for learnable patterns.

This module analyzes past interactions with teachers and extracts
patterns that can become internalized skills:
1. Clusters similar sessions
2. Extracts common structures
3. Proposes new skills for the registry

Think of it as Ara's "pattern compiler" - turning teacher examples
into reusable local knowledge.
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

from .registry import (
    SkillRegistry,
    LearnedSkill,
    SkillImplementation,
    get_skill_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class SessionPattern:
    """A pattern detected across multiple teacher sessions."""

    id: str
    name: str
    description: str

    # Pattern characteristics
    teachers_involved: List[str] = field(default_factory=list)
    common_intents: List[str] = field(default_factory=list)
    common_keywords: List[str] = field(default_factory=list)

    # Structure
    structure_template: str = ""  # Abstracted template
    variable_slots: List[str] = field(default_factory=list)  # {task}, {code}, etc.

    # Evidence
    session_ids: List[str] = field(default_factory=list)
    example_count: int = 0
    success_rate: float = 0.0

    # Metadata
    detected_at: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "teachers_involved": self.teachers_involved,
            "common_intents": self.common_intents,
            "common_keywords": self.common_keywords,
            "structure_template": self.structure_template,
            "variable_slots": self.variable_slots,
            "session_ids": self.session_ids,
            "example_count": self.example_count,
            "success_rate": round(self.success_rate, 3),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class SkillProposal:
    """A proposed skill to add to the registry."""

    pattern: SessionPattern
    suggested_name: str
    suggested_description: str
    suggested_category: str
    suggested_triggers: List[str]
    implementation_hint: str
    confidence: float

    # Review status
    status: str = "pending"  # "pending", "approved", "rejected"
    reviewer_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern.to_dict(),
            "suggested_name": self.suggested_name,
            "suggested_description": self.suggested_description,
            "suggested_category": self.suggested_category,
            "suggested_triggers": self.suggested_triggers,
            "implementation_hint": self.implementation_hint,
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "reviewer_notes": self.reviewer_notes,
        }


class PatternDetector:
    """Detects patterns in teacher sessions."""

    # Common structural patterns to look for
    STRUCTURE_PATTERNS = [
        (r"\[([A-Z_]+)\]", "bracketed_sections"),
        (r"```(\w+)\n", "code_blocks"),
        (r"^\d+\.\s", "numbered_steps"),
        (r"^[-*]\s", "bullet_points"),
    ]

    @classmethod
    def extract_structure(cls, text: str) -> Dict[str, Any]:
        """Extract structural features from text."""
        features = {
            "has_code_blocks": "```" in text,
            "has_sections": bool(re.search(r"\[[A-Z_]+\]", text)),
            "has_numbered_steps": bool(re.search(r"^\d+\.", text, re.MULTILINE)),
            "has_bullets": bool(re.search(r"^[-*]\s", text, re.MULTILINE)),
            "length": len(text),
            "line_count": text.count("\n") + 1,
        }

        # Extract section names
        sections = re.findall(r"\[([A-Z_]+)\]", text)
        features["sections"] = sections

        # Extract code block languages
        code_langs = re.findall(r"```(\w+)\n", text)
        features["code_languages"] = code_langs

        return features

    @classmethod
    def extract_keywords(cls, text: str, min_length: int = 4) -> List[str]:
        """Extract significant keywords from text."""
        # Simple keyword extraction
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text.lower())

        # Filter common words and short words
        stopwords = {
            "the", "and", "for", "that", "this", "with", "from", "have",
            "will", "can", "are", "was", "been", "being", "would", "could",
            "should", "what", "when", "where", "which", "while", "your",
            "into", "than", "then", "them", "they", "their", "there",
            "these", "those", "some", "such", "only", "other", "more",
            "also", "just", "like", "make", "made", "each", "about",
        }

        keywords = [
            w for w in words
            if len(w) >= min_length and w not in stopwords
        ]

        # Count frequencies
        freq = defaultdict(int)
        for kw in keywords:
            freq[kw] += 1

        # Return top keywords
        sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_kw[:20]]

    @classmethod
    def compute_similarity(
        cls,
        features1: Dict[str, Any],
        features2: Dict[str, Any],
    ) -> float:
        """Compute structural similarity between two feature sets."""
        score = 0.0
        total = 0.0

        # Boolean features
        for key in ["has_code_blocks", "has_sections", "has_numbered_steps", "has_bullets"]:
            total += 1.0
            if features1.get(key) == features2.get(key):
                score += 1.0

        # Section overlap
        sections1 = set(features1.get("sections", []))
        sections2 = set(features2.get("sections", []))
        if sections1 or sections2:
            total += 1.0
            overlap = len(sections1 & sections2)
            union = len(sections1 | sections2)
            if union > 0:
                score += overlap / union

        # Length similarity
        len1 = features1.get("length", 0)
        len2 = features2.get("length", 0)
        if len1 > 0 and len2 > 0:
            total += 1.0
            ratio = min(len1, len2) / max(len1, len2)
            score += ratio

        return score / total if total > 0 else 0.0


class SkillExtractor:
    """Extracts learnable skills from teacher sessions."""

    def __init__(
        self,
        log_path: Optional[Path] = None,
        proposals_path: Optional[Path] = None,
    ):
        """Initialize the extractor.

        Args:
            log_path: Path to interaction logs
            proposals_path: Path to store proposals
        """
        self.log_path = log_path or (
            Path.home() / ".ara" / "meta" / "interactions.jsonl"
        )
        self.proposals_path = proposals_path or (
            Path.home() / ".ara" / "academy" / "skills" / "proposals.json"
        )
        self.proposals_path.parent.mkdir(parents=True, exist_ok=True)

        self._proposals: List[SkillProposal] = []
        self._patterns: Dict[str, SessionPattern] = {}
        self._next_id = 1

    def _load_sessions(
        self,
        min_success: bool = True,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Load teacher sessions from logs.

        Args:
            min_success: Only include successful sessions
            days: Days of history to include

        Returns:
            List of session records
        """
        sessions = []

        if not self.log_path.exists():
            return sessions

        cutoff = datetime.utcnow().timestamp() - (days * 86400)

        try:
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)

                        # Filter by success
                        if min_success and not record.get("success"):
                            continue

                        # Filter by time
                        ts_str = record.get("timestamp", "")
                        if ts_str:
                            try:
                                ts = datetime.fromisoformat(
                                    ts_str.replace("Z", "+00:00")
                                ).timestamp()
                                if ts < cutoff:
                                    continue
                            except ValueError:
                                pass

                        sessions.append(record)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.warning(f"Failed to load sessions: {e}")

        return sessions

    def _cluster_sessions(
        self,
        sessions: List[Dict[str, Any]],
        similarity_threshold: float = 0.7,
    ) -> List[List[Dict[str, Any]]]:
        """Cluster sessions by structural similarity.

        Args:
            sessions: List of sessions
            similarity_threshold: Minimum similarity to cluster

        Returns:
            List of session clusters
        """
        if not sessions:
            return []

        # Extract features for each session
        features = []
        for session in sessions:
            query = session.get("query", "")
            response = session.get("response_summary", "")
            text = f"{query}\n{response}"
            features.append(PatternDetector.extract_structure(text))

        # Simple greedy clustering
        clusters: List[List[int]] = []
        assigned = set()

        for i, feat_i in enumerate(features):
            if i in assigned:
                continue

            cluster = [i]
            assigned.add(i)

            for j, feat_j in enumerate(features):
                if j in assigned:
                    continue

                sim = PatternDetector.compute_similarity(feat_i, feat_j)
                if sim >= similarity_threshold:
                    cluster.append(j)
                    assigned.add(j)

            if len(cluster) >= 2:  # Only keep clusters with 2+ sessions
                clusters.append(cluster)

        # Convert to session clusters
        return [[sessions[i] for i in cluster] for cluster in clusters]

    def _extract_pattern(
        self,
        cluster: List[Dict[str, Any]],
    ) -> Optional[SessionPattern]:
        """Extract a pattern from a session cluster.

        Args:
            cluster: List of similar sessions

        Returns:
            Extracted pattern or None
        """
        if len(cluster) < 2:
            return None

        # Collect statistics
        teachers: Set[str] = set()
        intents: Set[str] = set()
        all_keywords: List[str] = []
        successes = 0

        for session in cluster:
            for teacher in session.get("teachers", []):
                teachers.add(teacher)
            if session.get("primary_teacher"):
                teachers.add(session["primary_teacher"])
            if session.get("user_intent"):
                intents.add(session["user_intent"])
            if session.get("success"):
                successes += 1

            # Extract keywords from queries
            query = session.get("query", "")
            all_keywords.extend(PatternDetector.extract_keywords(query))

        # Get common keywords
        keyword_freq = defaultdict(int)
        for kw in all_keywords:
            keyword_freq[kw] += 1

        common_keywords = [
            kw for kw, count in keyword_freq.items()
            if count >= len(cluster) * 0.5  # Present in 50%+ of sessions
        ]

        # Generate pattern
        pattern = SessionPattern(
            id=f"PAT-{self._next_id:04d}",
            name=f"pattern_{common_keywords[0] if common_keywords else 'unknown'}",
            description=f"Pattern from {len(cluster)} similar sessions",
            teachers_involved=list(teachers),
            common_intents=list(intents),
            common_keywords=common_keywords[:10],
            session_ids=[s.get("id", "") for s in cluster if s.get("id")],
            example_count=len(cluster),
            success_rate=successes / len(cluster) if cluster else 0.0,
            confidence=min(0.9, len(cluster) * 0.1),  # More examples = more confidence
        )

        self._next_id += 1
        return pattern

    def _pattern_to_proposal(
        self,
        pattern: SessionPattern,
    ) -> SkillProposal:
        """Convert a pattern to a skill proposal.

        Args:
            pattern: Detected pattern

        Returns:
            Skill proposal
        """
        # Infer category from keywords/intents
        category = "general"
        if any(kw in pattern.common_keywords for kw in ["code", "implement", "function", "class"]):
            category = "codegen"
        elif any(kw in pattern.common_keywords for kw in ["analyze", "review", "check", "lint"]):
            category = "analysis"
        elif any(kw in pattern.common_keywords for kw in ["shader", "render", "visual", "graphics"]):
            category = "viz"
        elif any(kw in pattern.common_keywords for kw in ["benchmark", "test", "performance"]):
            category = "benchmarking"

        # Generate suggested name
        if pattern.common_keywords:
            suggested_name = "_".join(pattern.common_keywords[:3])
        else:
            suggested_name = f"skill_from_{pattern.id}"

        # Generate description
        teachers_str = ", ".join(pattern.teachers_involved)
        suggested_description = (
            f"Learned from {teachers_str} across {pattern.example_count} sessions. "
            f"Common patterns: {', '.join(pattern.common_keywords[:5])}"
        )

        # Implementation hint
        if category == "codegen":
            impl_hint = "Template-based code generation with variable slots"
        elif category == "analysis":
            impl_hint = "Pattern matching and structured analysis"
        else:
            impl_hint = "Policy-driven with escalation to teachers for edge cases"

        return SkillProposal(
            pattern=pattern,
            suggested_name=suggested_name,
            suggested_description=suggested_description,
            suggested_category=category,
            suggested_triggers=pattern.common_keywords[:5],
            implementation_hint=impl_hint,
            confidence=pattern.confidence,
        )

    def extract_patterns(
        self,
        days: int = 30,
        min_cluster_size: int = 3,
    ) -> List[SessionPattern]:
        """Extract patterns from recent sessions.

        Args:
            days: Days of history to analyze
            min_cluster_size: Minimum sessions to form a pattern

        Returns:
            List of detected patterns
        """
        sessions = self._load_sessions(days=days)
        logger.info(f"Loaded {len(sessions)} sessions for pattern extraction")

        if len(sessions) < min_cluster_size:
            return []

        clusters = self._cluster_sessions(sessions)
        logger.info(f"Found {len(clusters)} session clusters")

        patterns = []
        for cluster in clusters:
            if len(cluster) >= min_cluster_size:
                pattern = self._extract_pattern(cluster)
                if pattern:
                    patterns.append(pattern)
                    self._patterns[pattern.id] = pattern

        logger.info(f"Extracted {len(patterns)} patterns")
        return patterns

    def generate_proposals(
        self,
        min_confidence: float = 0.5,
        min_examples: int = 3,
    ) -> List[SkillProposal]:
        """Generate skill proposals from patterns.

        Args:
            min_confidence: Minimum pattern confidence
            min_examples: Minimum examples required

        Returns:
            List of skill proposals
        """
        # Extract fresh patterns if needed
        if not self._patterns:
            self.extract_patterns()

        proposals = []
        for pattern in self._patterns.values():
            if pattern.confidence < min_confidence:
                continue
            if pattern.example_count < min_examples:
                continue

            proposal = self._pattern_to_proposal(pattern)
            proposals.append(proposal)
            self._proposals.append(proposal)

        # Save proposals
        self._save_proposals()

        logger.info(f"Generated {len(proposals)} skill proposals")
        return proposals

    def _save_proposals(self) -> None:
        """Save proposals to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "proposals": [p.to_dict() for p in self._proposals],
        }
        with open(self.proposals_path, "w") as f:
            json.dump(data, f, indent=2)

    def approve_proposal(
        self,
        proposal_idx: int,
        notes: str = "",
    ) -> Optional[LearnedSkill]:
        """Approve a proposal and register as skill.

        Args:
            proposal_idx: Index of proposal to approve
            notes: Reviewer notes

        Returns:
            Created skill or None
        """
        if proposal_idx < 0 or proposal_idx >= len(self._proposals):
            return None

        proposal = self._proposals[proposal_idx]
        proposal.status = "approved"
        proposal.reviewer_notes = notes

        # Register skill
        registry = get_skill_registry()
        skill = registry.register_skill(
            name=proposal.suggested_name,
            description=proposal.suggested_description,
            category=proposal.suggested_category,
            learned_from=proposal.pattern.teachers_involved,
            triggers=proposal.suggested_triggers,
            tags=proposal.suggested_triggers,
            examples_seen=proposal.pattern.example_count,
        )

        self._save_proposals()
        return skill

    def reject_proposal(
        self,
        proposal_idx: int,
        reason: str = "",
    ) -> bool:
        """Reject a proposal.

        Args:
            proposal_idx: Index of proposal to reject
            reason: Rejection reason

        Returns:
            True if rejected
        """
        if proposal_idx < 0 or proposal_idx >= len(self._proposals):
            return False

        proposal = self._proposals[proposal_idx]
        proposal.status = "rejected"
        proposal.reviewer_notes = reason

        self._save_proposals()
        return True

    def get_pending_proposals(self) -> List[SkillProposal]:
        """Get pending proposals."""
        return [p for p in self._proposals if p.status == "pending"]

    def get_summary(self) -> Dict[str, Any]:
        """Get extractor summary."""
        return {
            "patterns_detected": len(self._patterns),
            "total_proposals": len(self._proposals),
            "pending": len([p for p in self._proposals if p.status == "pending"]),
            "approved": len([p for p in self._proposals if p.status == "approved"]),
            "rejected": len([p for p in self._proposals if p.status == "rejected"]),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_extractor: Optional[SkillExtractor] = None


def get_skill_extractor() -> SkillExtractor:
    """Get the default skill extractor."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = SkillExtractor()
    return _default_extractor


def extract_skills_from_logs(days: int = 30) -> List[SkillProposal]:
    """Extract skill proposals from logs."""
    extractor = get_skill_extractor()
    extractor.extract_patterns(days=days)
    return extractor.generate_proposals()
