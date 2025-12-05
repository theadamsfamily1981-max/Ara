"""Research Graph - Ara's structured long-term memory.

This is the map of Ara's mind about the project:
- Topics: Big pillars of research (neuromorphic graphics, quantum viz, etc.)
- Hypotheses: Concrete things being tested
- Threads: Active lines of investigation
- Evidence: Links to experiments and observations

Everything (logs, benchmarks, shaders, ideas) plugs into this graph.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ResearchTopic:
    """A major research topic/pillar."""

    id: str
    name: str
    description: str

    # Hierarchy
    parent_id: Optional[str] = None
    subtopic_ids: List[str] = field(default_factory=list)

    # Related items
    hypothesis_ids: List[str] = field(default_factory=list)
    thread_ids: List[str] = field(default_factory=list)
    experiment_ids: List[str] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    priority: str = "medium"  # "low", "medium", "high", "critical"
    status: str = "active"  # "active", "paused", "completed", "archived"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parent_id": self.parent_id,
            "subtopic_ids": self.subtopic_ids,
            "hypothesis_ids": self.hypothesis_ids,
            "thread_ids": self.thread_ids,
            "experiment_ids": self.experiment_ids,
            "tags": self.tags,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchTopic":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            parent_id=data.get("parent_id"),
            subtopic_ids=data.get("subtopic_ids", []),
            hypothesis_ids=data.get("hypothesis_ids", []),
            thread_ids=data.get("thread_ids", []),
            experiment_ids=data.get("experiment_ids", []),
            tags=data.get("tags", []),
            priority=data.get("priority", "medium"),
            status=data.get("status", "active"),
        )


@dataclass
class ResearchHypothesis:
    """A concrete hypothesis being tested."""

    id: str
    title: str
    statement: str

    # Status
    status: str = "active"  # "proposed", "active", "supported", "refuted", "inconclusive"

    # Evidence
    evidence_for: List[str] = field(default_factory=list)  # Experiment IDs
    evidence_against: List[str] = field(default_factory=list)
    confidence: float = 0.5  # 0 = refuted, 1 = strongly supported

    # Planning
    next_actions: List[str] = field(default_factory=list)
    blocking_questions: List[str] = field(default_factory=list)

    # Links
    topic_id: Optional[str] = None
    related_hypotheses: List[str] = field(default_factory=list)

    # Teacher opinions
    teacher_opinions: Dict[str, str] = field(default_factory=dict)  # teacher -> opinion

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update_confidence(self) -> None:
        """Update confidence based on evidence."""
        total_evidence = len(self.evidence_for) + len(self.evidence_against)
        if total_evidence == 0:
            self.confidence = 0.5
        else:
            self.confidence = len(self.evidence_for) / total_evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "statement": self.statement,
            "status": self.status,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "confidence": round(self.confidence, 3),
            "next_actions": self.next_actions,
            "blocking_questions": self.blocking_questions,
            "topic_id": self.topic_id,
            "related_hypotheses": self.related_hypotheses,
            "teacher_opinions": self.teacher_opinions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchHypothesis":
        return cls(
            id=data["id"],
            title=data["title"],
            statement=data.get("statement", ""),
            status=data.get("status", "active"),
            evidence_for=data.get("evidence_for", []),
            evidence_against=data.get("evidence_against", []),
            confidence=data.get("confidence", 0.5),
            next_actions=data.get("next_actions", []),
            blocking_questions=data.get("blocking_questions", []),
            topic_id=data.get("topic_id"),
            related_hypotheses=data.get("related_hypotheses", []),
            teacher_opinions=data.get("teacher_opinions", {}),
        )


@dataclass
class ResearchThread:
    """An active line of investigation."""

    id: str
    name: str
    goal: str

    # Progress
    status: str = "active"  # "proposed", "active", "blocked", "completed"
    progress_pct: float = 0.0

    # Links
    topic_id: Optional[str] = None
    hypothesis_ids: List[str] = field(default_factory=list)
    experiment_ids: List[str] = field(default_factory=list)

    # Planning
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    current_focus: str = ""
    blockers: List[str] = field(default_factory=list)

    # Metadata
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "goal": self.goal,
            "status": self.status,
            "progress_pct": round(self.progress_pct, 2),
            "topic_id": self.topic_id,
            "hypothesis_ids": self.hypothesis_ids,
            "experiment_ids": self.experiment_ids,
            "milestones": self.milestones,
            "current_focus": self.current_focus,
            "blockers": self.blockers,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchThread":
        return cls(
            id=data["id"],
            name=data["name"],
            goal=data.get("goal", ""),
            status=data.get("status", "active"),
            progress_pct=data.get("progress_pct", 0.0),
            topic_id=data.get("topic_id"),
            hypothesis_ids=data.get("hypothesis_ids", []),
            experiment_ids=data.get("experiment_ids", []),
            milestones=data.get("milestones", []),
            current_focus=data.get("current_focus", ""),
            blockers=data.get("blockers", []),
        )


class ResearchGraph:
    """The complete research graph."""

    def __init__(self, graph_path: Optional[Path] = None):
        """Initialize the graph.

        Args:
            graph_path: Path to graph JSON file
        """
        self.graph_path = graph_path or (
            Path.home() / ".ara" / "institute" / "research_graph.json"
        )
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)

        self._topics: Dict[str, ResearchTopic] = {}
        self._hypotheses: Dict[str, ResearchHypothesis] = {}
        self._threads: Dict[str, ResearchThread] = {}
        self._loaded = False
        self._next_id = {"topic": 1, "hypothesis": 1, "thread": 1}

    def _load(self, force: bool = False) -> None:
        """Load graph from disk."""
        if self._loaded and not force:
            return

        self._topics.clear()
        self._hypotheses.clear()
        self._threads.clear()

        if self.graph_path.exists():
            try:
                with open(self.graph_path) as f:
                    data = json.load(f)

                for topic_data in data.get("topics", []):
                    topic = ResearchTopic.from_dict(topic_data)
                    self._topics[topic.id] = topic

                for hyp_data in data.get("hypotheses", []):
                    hyp = ResearchHypothesis.from_dict(hyp_data)
                    self._hypotheses[hyp.id] = hyp

                for thread_data in data.get("threads", []):
                    thread = ResearchThread.from_dict(thread_data)
                    self._threads[thread.id] = thread

                self._next_id = data.get("next_id", self._next_id)

            except Exception as e:
                logger.warning(f"Failed to load research graph: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save graph to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "topics": [t.to_dict() for t in self._topics.values()],
            "hypotheses": [h.to_dict() for h in self._hypotheses.values()],
            "threads": [t.to_dict() for t in self._threads.values()],
            "next_id": self._next_id,
        }
        with open(self.graph_path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self, kind: str) -> str:
        """Generate a unique ID."""
        prefix = {"topic": "TOPIC", "hypothesis": "HYP", "thread": "THR"}[kind]
        id_str = f"{prefix}-{self._next_id[kind]:04d}"
        self._next_id[kind] += 1
        return id_str

    # =========================================================================
    # Topic Management
    # =========================================================================

    def add_topic(
        self,
        name: str,
        description: str,
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        priority: str = "medium",
    ) -> ResearchTopic:
        """Add a research topic."""
        self._load()

        topic = ResearchTopic(
            id=self._generate_id("topic"),
            name=name,
            description=description,
            parent_id=parent_id,
            tags=tags or [],
            priority=priority,
        )

        # Update parent
        if parent_id and parent_id in self._topics:
            self._topics[parent_id].subtopic_ids.append(topic.id)

        self._topics[topic.id] = topic
        self._save()
        logger.info(f"Added topic: {topic.id} ({topic.name})")

        return topic

    def get_topic(self, topic_id: str) -> Optional[ResearchTopic]:
        """Get a topic by ID."""
        self._load()
        return self._topics.get(topic_id)

    def get_all_topics(self) -> List[ResearchTopic]:
        """Get all topics."""
        self._load()
        return list(self._topics.values())

    def get_active_topics(self) -> List[ResearchTopic]:
        """Get active topics."""
        self._load()
        return [t for t in self._topics.values() if t.status == "active"]

    # =========================================================================
    # Hypothesis Management
    # =========================================================================

    def add_hypothesis(
        self,
        title: str,
        statement: str,
        topic_id: Optional[str] = None,
        next_actions: Optional[List[str]] = None,
    ) -> ResearchHypothesis:
        """Add a research hypothesis."""
        self._load()

        hyp = ResearchHypothesis(
            id=self._generate_id("hypothesis"),
            title=title,
            statement=statement,
            topic_id=topic_id,
            next_actions=next_actions or [],
            status="proposed",
        )

        # Link to topic
        if topic_id and topic_id in self._topics:
            self._topics[topic_id].hypothesis_ids.append(hyp.id)

        self._hypotheses[hyp.id] = hyp
        self._save()
        logger.info(f"Added hypothesis: {hyp.id} ({hyp.title})")

        return hyp

    def get_hypothesis(self, hyp_id: str) -> Optional[ResearchHypothesis]:
        """Get a hypothesis by ID."""
        self._load()
        return self._hypotheses.get(hyp_id)

    def get_all_hypotheses(self) -> List[ResearchHypothesis]:
        """Get all hypotheses."""
        self._load()
        return list(self._hypotheses.values())

    def get_active_hypotheses(self) -> List[ResearchHypothesis]:
        """Get active hypotheses."""
        self._load()
        return [h for h in self._hypotheses.values() if h.status == "active"]

    def add_evidence(
        self,
        hyp_id: str,
        experiment_id: str,
        supports: bool,
    ) -> bool:
        """Add evidence to a hypothesis."""
        self._load()

        hyp = self._hypotheses.get(hyp_id)
        if not hyp:
            return False

        if supports:
            hyp.evidence_for.append(experiment_id)
        else:
            hyp.evidence_against.append(experiment_id)

        hyp.update_confidence()
        hyp.updated_at = datetime.utcnow()

        self._save()
        return True

    def record_teacher_opinion(
        self,
        hyp_id: str,
        teacher: str,
        opinion: str,
    ) -> bool:
        """Record a teacher's opinion on a hypothesis."""
        self._load()

        hyp = self._hypotheses.get(hyp_id)
        if not hyp:
            return False

        hyp.teacher_opinions[teacher] = opinion
        hyp.updated_at = datetime.utcnow()

        self._save()
        return True

    # =========================================================================
    # Thread Management
    # =========================================================================

    def add_thread(
        self,
        name: str,
        goal: str,
        topic_id: Optional[str] = None,
        hypothesis_ids: Optional[List[str]] = None,
    ) -> ResearchThread:
        """Add a research thread."""
        self._load()

        thread = ResearchThread(
            id=self._generate_id("thread"),
            name=name,
            goal=goal,
            topic_id=topic_id,
            hypothesis_ids=hypothesis_ids or [],
        )

        # Link to topic
        if topic_id and topic_id in self._topics:
            self._topics[topic_id].thread_ids.append(thread.id)

        self._threads[thread.id] = thread
        self._save()
        logger.info(f"Added thread: {thread.id} ({thread.name})")

        return thread

    def get_thread(self, thread_id: str) -> Optional[ResearchThread]:
        """Get a thread by ID."""
        self._load()
        return self._threads.get(thread_id)

    def get_active_threads(self) -> List[ResearchThread]:
        """Get active threads."""
        self._load()
        return [t for t in self._threads.values() if t.status == "active"]

    def update_thread_progress(
        self,
        thread_id: str,
        progress_pct: float,
        current_focus: str = "",
    ) -> bool:
        """Update thread progress."""
        self._load()

        thread = self._threads.get(thread_id)
        if not thread:
            return False

        thread.progress_pct = progress_pct
        if current_focus:
            thread.current_focus = current_focus

        if progress_pct >= 100:
            thread.status = "completed"
            thread.completed_at = datetime.utcnow()

        self._save()
        return True

    # =========================================================================
    # Graph Queries
    # =========================================================================

    def get_next_actions(self) -> List[Dict[str, Any]]:
        """Get all pending next actions across hypotheses."""
        self._load()

        actions = []
        for hyp in self._hypotheses.values():
            if hyp.status != "active":
                continue
            for action in hyp.next_actions:
                actions.append({
                    "hypothesis_id": hyp.id,
                    "hypothesis_title": hyp.title,
                    "action": action,
                    "confidence": hyp.confidence,
                })

        return actions

    def get_blockers(self) -> List[Dict[str, Any]]:
        """Get all blockers across threads."""
        self._load()

        blockers = []
        for thread in self._threads.values():
            if thread.status != "blocked":
                continue
            for blocker in thread.blockers:
                blockers.append({
                    "thread_id": thread.id,
                    "thread_name": thread.name,
                    "blocker": blocker,
                })

        return blockers

    def get_summary(self) -> Dict[str, Any]:
        """Get graph summary."""
        self._load()

        return {
            "topics": {
                "total": len(self._topics),
                "active": len([t for t in self._topics.values() if t.status == "active"]),
            },
            "hypotheses": {
                "total": len(self._hypotheses),
                "active": len([h for h in self._hypotheses.values() if h.status == "active"]),
                "supported": len([h for h in self._hypotheses.values() if h.status == "supported"]),
                "refuted": len([h for h in self._hypotheses.values() if h.status == "refuted"]),
            },
            "threads": {
                "total": len(self._threads),
                "active": len([t for t in self._threads.values() if t.status == "active"]),
                "blocked": len([t for t in self._threads.values() if t.status == "blocked"]),
                "completed": len([t for t in self._threads.values() if t.status == "completed"]),
            },
            "next_actions": len(self.get_next_actions()),
            "blockers": len(self.get_blockers()),
        }

    def generate_morning_brief(self) -> str:
        """Generate a morning brief of research status."""
        self._load()

        lines = []
        lines.append("# Institute Morning Brief")
        lines.append(f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC*")
        lines.append("")

        # Active threads
        active_threads = self.get_active_threads()
        if active_threads:
            lines.append("## Active Research Threads")
            for thread in active_threads[:5]:
                lines.append(f"- **{thread.name}**: {thread.progress_pct:.0f}% complete")
                if thread.current_focus:
                    lines.append(f"  Current focus: {thread.current_focus}")
            lines.append("")

        # Active hypotheses
        active_hyps = self.get_active_hypotheses()
        if active_hyps:
            lines.append("## Hypotheses Under Investigation")
            for hyp in active_hyps[:5]:
                status = "↗" if hyp.confidence > 0.6 else "↘" if hyp.confidence < 0.4 else "→"
                lines.append(f"- {status} [{hyp.id}] {hyp.title} ({hyp.confidence:.0%} confidence)")
            lines.append("")

        # Next actions
        actions = self.get_next_actions()
        if actions:
            lines.append("## Recommended Next Actions")
            for action in actions[:5]:
                lines.append(f"- {action['action']}")
                lines.append(f"  (from hypothesis: {action['hypothesis_title']})")
            lines.append("")

        # Blockers
        blockers = self.get_blockers()
        if blockers:
            lines.append("## Blockers Requiring Attention")
            for blocker in blockers[:3]:
                lines.append(f"- {blocker['blocker']}")
                lines.append(f"  (blocking thread: {blocker['thread_name']})")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_graph: Optional[ResearchGraph] = None


def get_research_graph() -> ResearchGraph:
    """Get the default research graph."""
    global _default_graph
    if _default_graph is None:
        _default_graph = ResearchGraph()
    return _default_graph


def add_hypothesis(title: str, statement: str) -> ResearchHypothesis:
    """Add a hypothesis to the graph."""
    return get_research_graph().add_hypothesis(title, statement)


def get_morning_brief() -> str:
    """Get the morning brief."""
    return get_research_graph().generate_morning_brief()
