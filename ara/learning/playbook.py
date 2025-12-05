"""Prompt Playbook - Learn how to talk to each teacher.

For each teacher, Ara keeps a library of prompt templates that work well.
When facing a new task:
1. Embed the problem description
2. Retrieve nearest past exemplars
3. Start from highest-scoring prompt
4. Adapt it to current context

This is "student analyzing teacher and optimizing how to talk to them."
"""

from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PromptExemplar:
    """A successful prompt pattern for a specific task type.

    Stores the template + metadata about how well it worked.
    """

    exemplar_id: str = ""
    tool: str = ""                    # "claude", "nova", "gemini"
    task_type: str = ""               # "fpga_register_map", "graphics_opt", etc.
    prompt_template: str = ""         # The actual prompt pattern
    description: str = ""             # What this prompt is for

    # Performance tracking
    times_used: int = 0
    total_reward: float = 0.0
    best_score: float = 0.0
    avg_score: float = 0.5

    # Example uses
    example_contexts: List[str] = field(default_factory=list)
    example_outcomes: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_used_at: Optional[str] = None

    # Keywords for retrieval
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.exemplar_id:
            content = f"{self.tool}:{self.task_type}:{self.prompt_template[:50]}"
            self.exemplar_id = f"EX-{hashlib.md5(content.encode()).hexdigest()[:8]}"

    def update_score(self, score: float) -> None:
        """Update scores after using this exemplar."""
        self.times_used += 1
        self.total_reward += score
        self.avg_score = self.total_reward / self.times_used
        self.best_score = max(self.best_score, score)
        self.last_used_at = datetime.utcnow().isoformat()

    def add_example(self, context: str, outcome: str, max_examples: int = 5) -> None:
        """Add an example use of this prompt."""
        self.example_contexts.append(context[:200])
        self.example_outcomes.append(outcome[:200])

        # Keep bounded
        if len(self.example_contexts) > max_examples:
            self.example_contexts.pop(0)
            self.example_outcomes.pop(0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exemplar_id": self.exemplar_id,
            "tool": self.tool,
            "task_type": self.task_type,
            "prompt_template": self.prompt_template,
            "description": self.description,
            "times_used": self.times_used,
            "total_reward": self.total_reward,
            "best_score": self.best_score,
            "avg_score": self.avg_score,
            "example_contexts": self.example_contexts,
            "example_outcomes": self.example_outcomes,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PromptExemplar":
        return cls(**d)

    def match_score(self, query: str, task_type: Optional[str] = None) -> float:
        """Compute match score for a query.

        Args:
            query: The query string
            task_type: Optional task type filter

        Returns:
            Match score [0, 1]
        """
        score = 0.0
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Task type match (strong signal)
        if task_type and self.task_type == task_type:
            score += 0.4

        # Keyword match
        keyword_matches = sum(1 for kw in self.keywords if kw.lower() in query_lower)
        if self.keywords:
            score += 0.3 * (keyword_matches / len(self.keywords))

        # Description word overlap
        desc_words = set(self.description.lower().split())
        overlap = len(query_words & desc_words)
        if desc_words:
            score += 0.2 * (overlap / len(desc_words))

        # Bonus for high avg_score (prefer proven templates)
        score += 0.1 * self.avg_score

        return min(1.0, score)


class PromptPlaybook:
    """Library of prompt templates organized by tool and task type.

    Provides retrieval of best prompts for new situations.
    """

    def __init__(self, playbook_path: Optional[Path] = None):
        """Initialize the playbook.

        Args:
            playbook_path: Path to persist the playbook
        """
        self.playbook_path = playbook_path

        # Exemplars indexed by tool
        self.exemplars: Dict[str, List[PromptExemplar]] = {}

        # Load persisted playbook
        if playbook_path and playbook_path.exists():
            self._load()

    def add(self, exemplar: PromptExemplar) -> None:
        """Add an exemplar to the playbook.

        Args:
            exemplar: The exemplar to add
        """
        if exemplar.tool not in self.exemplars:
            self.exemplars[exemplar.tool] = []

        # Check for duplicate
        for ex in self.exemplars[exemplar.tool]:
            if ex.exemplar_id == exemplar.exemplar_id:
                logger.debug(f"Exemplar {exemplar.exemplar_id} already exists")
                return

        self.exemplars[exemplar.tool].append(exemplar)
        logger.info(f"Added exemplar {exemplar.exemplar_id} for {exemplar.tool}/{exemplar.task_type}")

        if self.playbook_path:
            self._save()

    def retrieve(
        self,
        tool: str,
        query: str,
        task_type: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Tuple[PromptExemplar, float]]:
        """Retrieve best matching exemplars for a query.

        Args:
            tool: Which tool to get prompts for
            query: The problem description
            task_type: Optional task type filter
            top_k: Number of exemplars to return

        Returns:
            List of (exemplar, match_score) tuples, sorted best first
        """
        if tool not in self.exemplars:
            return []

        scored = []
        for ex in self.exemplars[tool]:
            match_score = ex.match_score(query, task_type)
            scored.append((ex, match_score))

        # Sort by match score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

    def get_best(
        self,
        tool: str,
        query: str,
        task_type: Optional[str] = None,
    ) -> Optional[PromptExemplar]:
        """Get the single best exemplar.

        Args:
            tool: Which tool
            query: The problem description
            task_type: Optional task type filter

        Returns:
            Best matching exemplar or None
        """
        results = self.retrieve(tool, query, task_type, top_k=1)
        return results[0][0] if results else None

    def update_score(
        self,
        exemplar_id: str,
        score: float,
        context: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> bool:
        """Update an exemplar's score after use.

        Args:
            exemplar_id: The exemplar to update
            score: The observed reward
            context: Optional context to add as example
            outcome: Optional outcome to add as example

        Returns:
            True if found and updated
        """
        for tool_exemplars in self.exemplars.values():
            for ex in tool_exemplars:
                if ex.exemplar_id == exemplar_id:
                    ex.update_score(score)
                    if context and outcome:
                        ex.add_example(context, outcome)
                    if self.playbook_path:
                        self._save()
                    return True
        return False

    def get_by_task_type(
        self,
        task_type: str,
        tool: Optional[str] = None,
    ) -> List[PromptExemplar]:
        """Get all exemplars for a task type.

        Args:
            task_type: The task type
            tool: Optional tool filter

        Returns:
            List of exemplars
        """
        results = []
        for t, exemplars in self.exemplars.items():
            if tool and t != tool:
                continue
            for ex in exemplars:
                if ex.task_type == task_type:
                    results.append(ex)
        return results

    def get_top_performers(
        self,
        tool: Optional[str] = None,
        min_uses: int = 3,
        top_k: int = 10,
    ) -> List[PromptExemplar]:
        """Get top performing exemplars.

        Args:
            tool: Optional tool filter
            min_uses: Minimum uses to qualify
            top_k: Number to return

        Returns:
            List of exemplars sorted by avg_score
        """
        candidates = []
        for t, exemplars in self.exemplars.items():
            if tool and t != tool:
                continue
            for ex in exemplars:
                if ex.times_used >= min_uses:
                    candidates.append(ex)

        candidates.sort(key=lambda x: x.avg_score, reverse=True)
        return candidates[:top_k]

    def _save(self) -> None:
        """Save playbook to disk."""
        if not self.playbook_path:
            return

        self.playbook_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            tool: [ex.to_dict() for ex in exemplars]
            for tool, exemplars in self.exemplars.items()
        }

        with open(self.playbook_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load playbook from disk."""
        if not self.playbook_path or not self.playbook_path.exists():
            return

        with open(self.playbook_path) as f:
            data = json.load(f)

        for tool, exemplars in data.items():
            self.exemplars[tool] = [
                PromptExemplar.from_dict(ex) for ex in exemplars
            ]


# =============================================================================
# Convenience Functions
# =============================================================================

_default_playbook: Optional[PromptPlaybook] = None


def get_playbook(path: Optional[Path] = None) -> PromptPlaybook:
    """Get the default playbook."""
    global _default_playbook
    if _default_playbook is None:
        path = path or Path.home() / ".ara" / "learning" / "playbook.json"
        _default_playbook = PromptPlaybook(path)
    return _default_playbook


def retrieve_best_prompt(
    tool: str,
    query: str,
    task_type: Optional[str] = None,
) -> Optional[PromptExemplar]:
    """Retrieve the best prompt for a query.

    Args:
        tool: Which tool
        query: Problem description
        task_type: Optional task type filter

    Returns:
        Best exemplar or None
    """
    return get_playbook().get_best(tool, query, task_type)


def save_exemplar(
    tool: str,
    task_type: str,
    prompt_template: str,
    description: str,
    keywords: Optional[List[str]] = None,
    initial_score: float = 0.7,
) -> PromptExemplar:
    """Save a new prompt exemplar.

    Args:
        tool: Which tool this is for
        task_type: The task type
        prompt_template: The prompt pattern
        description: What it's for
        keywords: Keywords for retrieval
        initial_score: Initial score estimate

    Returns:
        The created exemplar
    """
    ex = PromptExemplar(
        tool=tool,
        task_type=task_type,
        prompt_template=prompt_template,
        description=description,
        keywords=keywords or [],
        avg_score=initial_score,
    )
    get_playbook().add(ex)
    return ex


# =============================================================================
# Seed Exemplars
# =============================================================================

def seed_default_exemplars() -> None:
    """Seed the playbook with default exemplars.

    These are starting points that can be refined through use.
    """
    playbook = get_playbook()

    # Claude - Code implementation
    playbook.add(PromptExemplar(
        tool="claude",
        task_type="fpga_register_map",
        prompt_template="""You are a hardware reverse-engineer. Given this register table and logic capture, infer the register map and produce a Python interface.

Register dump:
{registers}

Logic capture:
{logic}

Produce:
1. Register map as Python dataclass
2. Read/write functions
3. Any discovered protocols""",
        description="Reverse-engineer FPGA register maps from captures",
        keywords=["fpga", "register", "reverse", "hardware"],
        avg_score=0.85,
    ))

    playbook.add(PromptExemplar(
        tool="claude",
        task_type="kernel_optimization",
        prompt_template="""Optimize this kernel for throughput on the given hardware.

Current kernel:
```
{code}
```

Profiler stats:
{profiler}

Target hardware: {hardware}

Constraints:
- {constraints}

Produce optimized code with explanation of changes.""",
        description="Optimize compute kernels for specific hardware",
        keywords=["kernel", "optimize", "throughput", "performance"],
        avg_score=0.82,
    ))

    # Nova - Architecture and review
    playbook.add(PromptExemplar(
        tool="nova",
        task_type="architecture_review",
        prompt_template="""Review this architecture for {system_name}.

Current design:
{design}

Requirements:
{requirements}

Concerns:
{concerns}

Provide:
1. Trade-off analysis
2. Risk assessment
3. Alternative approaches
4. Recommended path forward""",
        description="Review system architecture for trade-offs and risks",
        keywords=["architecture", "review", "design", "trade-off"],
        avg_score=0.88,
    ))

    playbook.add(PromptExemplar(
        tool="nova",
        task_type="spec_synthesis",
        prompt_template="""Turn this idea into a concrete implementation spec.

Idea: {idea}
Context: {context}
Constraints: {constraints}

Produce a spec that includes:
1. Files/modules to touch
2. API contracts
3. Step-by-step implementation plan
4. Test plan
5. Rollback plan
6. Risk assessment""",
        description="Convert ideas into implementation specifications",
        keywords=["spec", "specification", "plan", "implementation"],
        avg_score=0.86,
    ))

    # Gemini - Research and ideation
    playbook.add(PromptExemplar(
        tool="gemini",
        task_type="ideation",
        prompt_template="""Explore approaches for: {problem}

Constraints:
{constraints}

Don't write code. Generate 3-5 conceptual approaches, each with:
- Name
- Description (2-3 sentences)
- Pros
- Cons
- Novelty level (low/medium/high)
- Related prior art

Go a little wild—filtering happens later.""",
        description="Generate creative approaches to problems",
        keywords=["ideas", "brainstorm", "approaches", "creative"],
        avg_score=0.78,
    ))

    playbook.add(PromptExemplar(
        tool="gemini",
        task_type="literature_survey",
        prompt_template="""Survey approaches for: {topic}

Focus areas:
{focus}

Provide:
1. Key papers/techniques (with references if known)
2. Trade-offs between approaches
3. What's cutting edge vs. well-established
4. Gaps in the field
5. Unexpected connections from other domains""",
        description="Survey literature and prior art",
        keywords=["survey", "literature", "research", "papers", "state of art"],
        avg_score=0.75,
    ))

    logger.info("Seeded playbook with default exemplars")
