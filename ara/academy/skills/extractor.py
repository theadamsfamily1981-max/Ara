"""Skill Extractor - Mine teacher sessions for learnable patterns.

This module analyzes past interactions with teachers and extracts
patterns that can become internalized skills:
1. Clusters similar sessions
2. Extracts common structures
3. Uses the Architect to GENERALIZE (not just record)
4. Proposes new skills for the registry

ITERATION 27 UPDATE (The Sovereign):
=====================================
The extractor is no longer just a "recorder" - it now uses:
- TeleologyEngine to score alignment with Vision
- Architect to generalize patterns into robust skills
- VisionAwareInternalization to decide what's worth learning

OLD FLOW (Recorder):
    Episodes → Cluster → Save Pattern → Done

NEW FLOW (Architect):
    Episodes → Cluster → Architect Generalizes → Robust SkillSpec
                              ↓
                    Adds edge cases, safety checks, recovery
                    based on teleology classification
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
from .architect import Architect, Episode, SkillSpec, get_architect
from ara.cognition.teleology_engine import TeleologyEngine, get_teleology_engine
from ara.academy.curriculum.internalization import (
    VisionAwareInternalization,
    SkillCandidate,
    get_vision_aware_internalization,
)
from ara.academy.session_graph import (
    SessionGraph,
    SessionGraphBuilder,
    SessionStyleClassifier,
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
    """
    Extracts learnable skills from teacher sessions.

    ITERATION 27 UPDATE:
    Now uses Architect for generalization and TeleologyEngine for
    vision-aware prioritization.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        proposals_path: Optional[Path] = None,
        architect: Optional[Architect] = None,
        teleology: Optional[TeleologyEngine] = None,
    ):
        """Initialize the extractor.

        Args:
            log_path: Path to interaction logs
            proposals_path: Path to store proposals
            architect: Architect for skill generalization
            teleology: TeleologyEngine for alignment scoring
        """
        self.log_path = log_path or (
            Path.home() / ".ara" / "meta" / "interactions.jsonl"
        )
        self.proposals_path = proposals_path or (
            Path.home() / ".ara" / "academy" / "skills" / "proposals.json"
        )
        self.proposals_path.parent.mkdir(parents=True, exist_ok=True)

        # NEW: Architect and Teleology integration
        self.architect = architect or get_architect()
        self.teleology = teleology or get_teleology_engine()
        self.internalization = get_vision_aware_internalization()

        self._proposals: List[SkillProposal] = []
        self._patterns: Dict[str, SessionPattern] = {}
        self._skill_specs: Dict[str, SkillSpec] = {}  # NEW: Architect-generated specs
        self._next_id = 1

        # SessionGraph integration
        self.graph_builder = SessionGraphBuilder()
        self.style_classifier = SessionStyleClassifier(k=3)

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

    def _build_session_graphs(
        self,
        sessions: List[Dict[str, Any]],
    ) -> List[SessionGraph]:
        """
        Build SessionGraphs from session records.

        This enables graph-based pattern detection instead of just regex.
        """
        graphs = []
        for session in sessions:
            # Convert session record to transcript format
            transcript = self._session_to_transcript(session)
            session_id = session.get("id", f"session_{len(graphs)}")

            graph = self.graph_builder.build_from_transcript(session_id, transcript)
            graphs.append(graph)

        return graphs

    def _session_to_transcript(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert a session record to transcript format for graph building."""
        transcript = []

        # User query as intent
        if session.get("query"):
            transcript.append({
                "role": "user",
                "text": session["query"],
            })

        # Steps as tool calls / thoughts
        for step in session.get("steps", []):
            if step.get("tool"):
                transcript.append({
                    "role": "assistant",
                    "text": step.get("description", ""),
                    "tool": step["tool"],
                    "success": step.get("success", True),
                })
            else:
                transcript.append({
                    "role": "assistant",
                    "text": step.get("description", str(step)),
                })

        # Final response
        if session.get("response_summary"):
            transcript.append({
                "role": "assistant",
                "text": session["response_summary"],
            })

        return transcript

    def _cluster_sessions_by_graph(
        self,
        sessions: List[Dict[str, Any]],
        similarity_threshold: float = 0.7,
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster sessions by graph topology instead of regex structure.

        This is the NEW way - patterns are graph shapes, not text patterns.
        """
        if not sessions:
            return []

        # Build graphs for each session
        graphs = self._build_session_graphs(sessions)
        if not graphs:
            return []

        # Extract feature vectors
        features = [g.extract_feature_vector() for g in graphs]

        # Simple greedy clustering by feature distance
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

                # Compute similarity as 1 - normalized distance
                dist = self._euclidean_distance(feat_i, feat_j)
                max_dist = 20.0  # Rough max distance for normalization
                similarity = 1.0 - min(1.0, dist / max_dist)

                if similarity >= similarity_threshold:
                    cluster.append(j)
                    assigned.add(j)

            if len(cluster) >= 2:
                clusters.append(cluster)

        logger.info(
            f"Graph-based clustering: {len(clusters)} clusters from {len(sessions)} sessions"
        )

        # Convert to session clusters with graph info
        result = []
        for cluster_indices in clusters:
            cluster_sessions = []
            for idx in cluster_indices:
                session = sessions[idx]
                # Attach graph style for the Architect
                session["_graph_style"] = graphs[idx].classify_style()
                session["_graph_features"] = graphs[idx].extract_context_features()
                cluster_sessions.append(session)
            result.append(cluster_sessions)

        return result

    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """Calculate Euclidean distance between two feature vectors."""
        if len(a) != len(b):
            return float("inf")
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

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

    # =========================================================================
    # Iteration 27: Architect-Powered Generalization
    # =========================================================================

    def _session_to_episode(self, session: Dict[str, Any]) -> Episode:
        """Convert a session record to an Episode object for the Architect."""
        return Episode(
            id=session.get("id", f"session_{hash(str(session)) % 10000}"),
            timestamp=datetime.fromisoformat(
                session.get("timestamp", datetime.utcnow().isoformat()).replace("Z", "+00:00")
            ) if session.get("timestamp") else datetime.utcnow(),
            context=session.get("context", {}),
            query=session.get("query", ""),
            intent=session.get("user_intent", session.get("intent", "")),
            steps=session.get("steps", []),
            tools_used=session.get("tools", session.get("tools_used", [])),
            code_snippets=session.get("code_snippets", []),
            outcome=session.get("outcome", {}),
            success=session.get("success", True),
            error=session.get("error"),
            teacher=session.get("primary_teacher", session.get("teacher")),
            duration_ms=session.get("duration_ms", 0),
            tags=session.get("tags", []),
        )

    def _generalize_cluster_with_architect(
        self,
        cluster: List[Dict[str, Any]],
    ) -> Optional[SkillSpec]:
        """
        Use the Architect to generalize a cluster into a robust SkillSpec.

        This is the NEW way - instead of just recording the pattern,
        we ask the Architect to:
        1. Abstract the pattern
        2. Add edge cases (based on teleology classification)
        3. Generate safety checks
        4. Create recovery strategies

        Args:
            cluster: List of similar session records

        Returns:
            SkillSpec from Architect, or None if generalization fails
        """
        if len(cluster) < 2:
            return None

        # Convert sessions to Episodes
        episodes = [self._session_to_episode(s) for s in cluster]

        try:
            # Use Architect to generalize
            skill_spec = self.architect.generalize(episodes)

            # Store for later retrieval
            self._skill_specs[skill_spec.name] = skill_spec

            logger.info(
                f"Architect generalized {len(episodes)} episodes → "
                f"'{skill_spec.name}' ({skill_spec.classification}, "
                f"alignment={skill_spec.alignment_score:.2f})"
            )

            return skill_spec

        except Exception as e:
            logger.warning(f"Architect failed to generalize cluster: {e}")
            return None

    def _skillspec_to_proposal(self, spec: SkillSpec) -> SkillProposal:
        """Convert an Architect SkillSpec to a SkillProposal."""
        # Create a pattern for backward compatibility
        pattern = SessionPattern(
            id=f"ARCH-{self._next_id:04d}",
            name=spec.name,
            description=spec.description,
            teachers_involved=spec.teacher_sources,
            common_intents=[],
            common_keywords=list(spec.tags.keys())[:10],
            session_ids=spec.source_episodes,
            example_count=len(spec.source_episodes),
            success_rate=spec.confidence,
            confidence=spec.confidence,
        )
        self._next_id += 1

        # Implementation hint based on Architect's analysis
        impl_hints = spec.implementation_hints or ["See SkillSpec for details"]

        return SkillProposal(
            pattern=pattern,
            suggested_name=spec.name,
            suggested_description=spec.description,
            suggested_category=spec.category,
            suggested_triggers=list(spec.tags.keys())[:5],
            implementation_hint="; ".join(impl_hints[:3]),
            confidence=spec.confidence,
        )

    def _should_prioritize_with_vision(self, spec: SkillSpec) -> bool:
        """
        Check if this SkillSpec should be prioritized based on Vision.

        Sovereign/strategic skills that serve the Cathedral get priority
        even if they're rare.
        """
        # Create a SkillCandidate for the internalization scorer
        candidate = SkillCandidate(
            name=spec.name,
            description=spec.description,
            frequency_per_week=len(spec.source_episodes) / 4.0,  # Rough estimate
            success_rate=spec.confidence,
            tags=spec.tags,
        )

        result = self.internalization.compute_score(candidate)
        return result["should_internalize"]

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
        use_architect: bool = True,
    ) -> List[SkillProposal]:
        """Generate skill proposals from patterns.

        ITERATION 27 UPDATE:
        Now optionally uses the Architect for generalization, which produces
        more robust skills with edge cases, safety checks, and recovery strategies.

        Args:
            min_confidence: Minimum pattern confidence
            min_examples: Minimum examples required
            use_architect: If True, use Architect for generalization

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

    def generate_proposals_with_architect(
        self,
        days: int = 30,
        min_cluster_size: int = 2,
    ) -> List[SkillProposal]:
        """
        Generate skill proposals using the Architect for generalization.

        This is the NEW way (Iteration 27) - produces more robust skills
        that are aligned with the Vision and have proper edge case handling.

        Args:
            days: Days of history to analyze
            min_cluster_size: Minimum sessions to form a pattern

        Returns:
            List of skill proposals with Architect-generated SkillSpecs
        """
        sessions = self._load_sessions(days=days)
        logger.info(f"Loaded {len(sessions)} sessions for Architect analysis")

        if len(sessions) < min_cluster_size:
            return []

        clusters = self._cluster_sessions(sessions)
        logger.info(f"Found {len(clusters)} session clusters")

        proposals = []
        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue

            # Use Architect to generalize
            skill_spec = self._generalize_cluster_with_architect(cluster)
            if not skill_spec:
                continue

            # Check if we should prioritize this based on Vision
            should_prioritize = self._should_prioritize_with_vision(skill_spec)

            # Convert to proposal
            proposal = self._skillspec_to_proposal(skill_spec)

            # Mark priority based on Vision alignment
            if should_prioritize and skill_spec.classification in ("sovereign", "strategic"):
                proposal.reviewer_notes = (
                    f"[VISION-ALIGNED] {skill_spec.classification.upper()} skill - "
                    f"alignment={skill_spec.alignment_score:.2f}. "
                    "Prioritize even if low frequency."
                )

            proposals.append(proposal)
            self._proposals.append(proposal)

        # Sort by alignment (Vision-aligned first)
        proposals.sort(
            key=lambda p: self._skill_specs.get(p.suggested_name, SkillSpec(name="", description="")).alignment_score,
            reverse=True,
        )

        self._save_proposals()

        logger.info(
            f"Architect generated {len(proposals)} proposals "
            f"({sum(1 for p in proposals if 'VISION-ALIGNED' in p.reviewer_notes)} vision-aligned)"
        )
        return proposals

    def get_skill_spec(self, name: str) -> Optional[SkillSpec]:
        """Get the full SkillSpec for an Architect-generated skill."""
        return self._skill_specs.get(name)

    def generate_proposals_with_graphs(
        self,
        days: int = 30,
        min_cluster_size: int = 2,
    ) -> List[SkillProposal]:
        """
        Generate skill proposals using SessionGraph topology.

        This is the NEWEST approach that:
        - Uses graph-based clustering (not regex)
        - Detects retry patterns, Socratic loops, decomposition
        - Combines with Architect for skill generalization
        - Uses TeleologyEngine for vision alignment

        Patterns are GRAPH SHAPES, not just text patterns.
        """
        sessions = self._load_sessions(days=days)
        logger.info(f"Loaded {len(sessions)} sessions for graph analysis")

        if len(sessions) < min_cluster_size:
            return []

        # Use graph-based clustering
        clusters = self._cluster_sessions_by_graph(sessions)
        logger.info(f"Found {len(clusters)} graph-based clusters")

        proposals = []
        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue

            # Get graph style from first session (they should all be similar)
            graph_style = cluster[0].get("_graph_style", "unknown")
            graph_features = cluster[0].get("_graph_features", {})

            # Use Architect to generalize
            skill_spec = self._generalize_cluster_with_architect(cluster)
            if not skill_spec:
                continue

            # Check Vision alignment
            should_prioritize = self._should_prioritize_with_vision(skill_spec)

            # Convert to proposal
            proposal = self._skillspec_to_proposal(skill_spec)

            # Add graph-based insights
            proposal.reviewer_notes = f"[GRAPH STYLE: {graph_style}] "
            if graph_features.get("retry_count", 0) > 0:
                proposal.reviewer_notes += "Contains retry patterns. "
            if graph_features.get("success_rate", 0) > 0.8:
                proposal.reviewer_notes += "High success rate. "

            # Mark priority based on Vision alignment
            if should_prioritize and skill_spec.classification in ("sovereign", "strategic"):
                proposal.reviewer_notes += (
                    f"[VISION-ALIGNED] {skill_spec.classification.upper()} skill - "
                    f"alignment={skill_spec.alignment_score:.2f}. "
                )

            proposals.append(proposal)
            self._proposals.append(proposal)

        # Sort by alignment
        proposals.sort(
            key=lambda p: self._skill_specs.get(
                p.suggested_name, SkillSpec(name="", description="")
            ).alignment_score,
            reverse=True,
        )

        self._save_proposals()

        logger.info(
            f"Graph-based extraction: {len(proposals)} proposals "
            f"({sum(1 for p in proposals if 'VISION-ALIGNED' in p.reviewer_notes)} vision-aligned)"
        )
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
    """Extract skill proposals from logs (legacy method)."""
    extractor = get_skill_extractor()
    extractor.extract_patterns(days=days)
    return extractor.generate_proposals()


def extract_skills_with_architect(days: int = 30) -> List[SkillProposal]:
    """
    Extract skill proposals using the Architect (Iteration 27).

    This is the NEW preferred method that:
    - Uses TeleologyEngine for vision-aware prioritization
    - Uses Architect for pattern generalization
    - Produces robust skills with edge cases and safety checks
    """
    extractor = get_skill_extractor()
    return extractor.generate_proposals_with_architect(days=days)


def demo_iteration_27():
    """
    Demonstrate the Iteration 27 improvements.

    Shows how the new system prioritizes vision-aligned skills.
    """
    from ara.cognition.teleology_engine import TeleologyEngine

    print("=" * 70)
    print("Iteration 27: The Sovereign - Vision-Aware Skill Extraction")
    print("=" * 70)

    teleology = TeleologyEngine()

    # Example skill tags
    test_cases = [
        ("Thermal Recovery Handler", {
            "thermal": 1.0, "recovery": 0.9, "antifragility": 0.8
        }),
        ("Cache Clear Script", {
            "clear_cache": 1.0, "admin": 0.7, "cleanup": 0.5
        }),
        ("SNN Plasticity Optimizer", {
            "snn": 1.0, "plasticity": 0.9, "neuromorphic": 0.8, "fpga": 0.7
        }),
    ]

    print("\nTeleology Classification:")
    print("-" * 70)

    for name, tags in test_cases:
        alignment = teleology.alignment_score(tags)
        priority = teleology.strategic_priority(tags)
        classification = teleology.classify_skill(tags)

        print(f"\n{name}:")
        print(f"  Alignment:     {alignment:.2f}")
        print(f"  Priority:      {priority:.2f}")
        print(f"  Classification: {classification}")
        print(f"  Verdict: {'PRIORITIZE' if classification in ('sovereign', 'strategic') else 'Standard'}")

    print("\n" + "=" * 70)
    print("KEY: 'Thermal Recovery' and 'SNN Plasticity' are now SOVEREIGN-level")
    print("     skills that get prioritized even if they're rare.")
    print("     'Cache Clear' is SECRETARY-level and stays low priority.")
    print("=" * 70)
