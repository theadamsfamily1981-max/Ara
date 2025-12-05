"""Tournaments - Competitive evaluation of agents and workflows.

Ara runs tournaments to determine which agents, workflows, or configurations
perform best for specific tasks. This is systematic comparison through:
- Round-robin competitions
- Head-to-head matches
- Benchmark suites

Think of it as "natural selection" for agent configurations.

Example tournament:
  tournament_id: debug_python_showdown
  participants: ["AGENT-0001", "AGENT-0002", "workflow:claude→nova"]
  benchmark: "python_debug_suite"
  winner: "AGENT-0001" (87% success vs 72%)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkTask:
    """A task in a benchmark suite."""

    id: str
    name: str
    description: str

    # Task specification
    intent: str
    query: str
    expected_output_keywords: List[str] = field(default_factory=list)

    # Difficulty
    difficulty: str = "medium"  # "easy", "medium", "hard"
    weight: float = 1.0

    # Metadata
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "intent": self.intent,
            "query": self.query,
            "expected_output_keywords": self.expected_output_keywords,
            "difficulty": self.difficulty,
            "weight": self.weight,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkTask":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            intent=data.get("intent", "general"),
            query=data["query"],
            expected_output_keywords=data.get("expected_output_keywords", []),
            difficulty=data.get("difficulty", "medium"),
            weight=data.get("weight", 1.0),
            tags=data.get("tags", []),
        )


@dataclass
class BenchmarkSuite:
    """A suite of benchmark tasks."""

    id: str
    name: str
    description: str

    # Tasks
    tasks: List[BenchmarkTask] = field(default_factory=list)

    # Configuration
    passing_score: float = 0.7  # Minimum score to "pass"
    time_limit_sec: Optional[float] = None

    # Metadata
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tasks": [t.to_dict() for t in self.tasks],
            "passing_score": self.passing_score,
            "time_limit_sec": self.time_limit_sec,
            "version": self.version,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSuite":
        suite = cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            passing_score=data.get("passing_score", 0.7),
            time_limit_sec=data.get("time_limit_sec"),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
        )
        for task_data in data.get("tasks", []):
            suite.tasks.append(BenchmarkTask.from_dict(task_data))
        return suite


@dataclass
class MatchResult:
    """Result of a single match."""

    task_id: str
    participant_id: str
    success: bool
    reward: float = 0.0
    latency_sec: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "participant_id": self.participant_id,
            "success": self.success,
            "reward": round(self.reward, 3),
            "latency_sec": round(self.latency_sec, 2),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatchResult":
        return cls(
            task_id=data["task_id"],
            participant_id=data["participant_id"],
            success=data["success"],
            reward=data.get("reward", 0.0),
            latency_sec=data.get("latency_sec", 0.0),
            notes=data.get("notes", ""),
        )


@dataclass
class ParticipantScore:
    """Aggregated score for a tournament participant."""

    participant_id: str
    participant_type: str  # "agent", "workflow", "capsule"

    # Raw counts
    tasks_attempted: int = 0
    tasks_succeeded: int = 0
    total_reward: float = 0.0
    total_latency_sec: float = 0.0

    # Computed metrics
    @property
    def success_rate(self) -> Optional[float]:
        if self.tasks_attempted == 0:
            return None
        return self.tasks_succeeded / self.tasks_attempted

    @property
    def avg_reward(self) -> Optional[float]:
        if self.tasks_attempted == 0:
            return None
        return self.total_reward / self.tasks_attempted

    @property
    def avg_latency_sec(self) -> Optional[float]:
        if self.tasks_attempted == 0:
            return None
        return self.total_latency_sec / self.tasks_attempted

    def record_result(self, result: MatchResult) -> None:
        """Record a match result."""
        self.tasks_attempted += 1
        if result.success:
            self.tasks_succeeded += 1
        self.total_reward += result.reward
        self.total_latency_sec += result.latency_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "participant_type": self.participant_type,
            "tasks_attempted": self.tasks_attempted,
            "tasks_succeeded": self.tasks_succeeded,
            "success_rate": self.success_rate,
            "avg_reward": self.avg_reward,
            "avg_latency_sec": self.avg_latency_sec,
        }


@dataclass
class Tournament:
    """A tournament between participants."""

    id: str
    name: str
    description: str

    # What we're testing
    benchmark_id: str
    participants: List[str] = field(default_factory=list)

    # Results
    results: List[MatchResult] = field(default_factory=list)
    scores: Dict[str, ParticipantScore] = field(default_factory=dict)

    # Status
    status: str = "pending"  # "pending", "running", "completed"
    winner: Optional[str] = None
    winner_score: Optional[float] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    notes: str = ""

    def add_result(self, result: MatchResult) -> None:
        """Add a match result."""
        self.results.append(result)

        # Update scores
        if result.participant_id not in self.scores:
            # Determine type from ID
            ptype = "agent"
            if result.participant_id.startswith("SKILL-"):
                ptype = "capsule"
            elif "→" in result.participant_id or "->" in result.participant_id:
                ptype = "workflow"

            self.scores[result.participant_id] = ParticipantScore(
                participant_id=result.participant_id,
                participant_type=ptype,
            )

        self.scores[result.participant_id].record_result(result)

    def compute_winner(self) -> Optional[str]:
        """Compute the tournament winner."""
        if not self.scores:
            return None

        # Rank by success rate, then by avg reward
        ranked = sorted(
            self.scores.values(),
            key=lambda s: (s.success_rate or 0, s.avg_reward or 0),
            reverse=True,
        )

        if ranked:
            winner = ranked[0]
            self.winner = winner.participant_id
            self.winner_score = winner.success_rate
            return self.winner

        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "benchmark_id": self.benchmark_id,
            "participants": self.participants,
            "results": [r.to_dict() for r in self.results],
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "status": self.status,
            "winner": self.winner,
            "winner_score": self.winner_score,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tournament":
        tournament = cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            benchmark_id=data["benchmark_id"],
            participants=data.get("participants", []),
            status=data.get("status", "pending"),
            winner=data.get("winner"),
            winner_score=data.get("winner_score"),
            notes=data.get("notes", ""),
        )

        for result_data in data.get("results", []):
            result = MatchResult.from_dict(result_data)
            tournament.results.append(result)

        # Rebuild scores from results
        for result in tournament.results:
            if result.participant_id not in tournament.scores:
                ptype = "agent"
                if result.participant_id.startswith("SKILL-"):
                    ptype = "capsule"
                elif "→" in result.participant_id:
                    ptype = "workflow"

                tournament.scores[result.participant_id] = ParticipantScore(
                    participant_id=result.participant_id,
                    participant_type=ptype,
                )
            tournament.scores[result.participant_id].record_result(result)

        return tournament

    def format_leaderboard(self) -> str:
        """Format a leaderboard string."""
        lines = [
            f"# Tournament: {self.name}",
            f"Status: {self.status}",
            "",
            "## Leaderboard",
        ]

        # Sort by success rate
        ranked = sorted(
            self.scores.values(),
            key=lambda s: (s.success_rate or 0, s.avg_reward or 0),
            reverse=True,
        )

        for i, score in enumerate(ranked, 1):
            medal = ""
            if i == 1:
                medal = " [WINNER]"
            elif i == 2:
                medal = " [2nd]"
            elif i == 3:
                medal = " [3rd]"

            success = score.success_rate
            success_str = f"{success:.0%}" if success is not None else "N/A"
            reward = score.avg_reward
            reward_str = f"{reward:.0%}" if reward is not None else "N/A"

            lines.append(
                f"{i}. {score.participant_id}{medal}"
            )
            lines.append(
                f"   Success: {success_str} | Reward: {reward_str} | "
                f"Tasks: {score.tasks_attempted}"
            )

        return "\n".join(lines)


class TournamentManager:
    """Manages tournaments and benchmarks."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the manager.

        Args:
            data_path: Path to tournament data
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "meta" / "toolsmith" / "tournaments"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._tournaments: Dict[str, Tournament] = {}
        self._benchmarks: Dict[str, BenchmarkSuite] = {}
        self._loaded = False
        self._next_id = 1

    def _load(self, force: bool = False) -> None:
        """Load data from disk."""
        if self._loaded and not force:
            return

        self._tournaments.clear()
        self._benchmarks.clear()

        # Load tournaments
        tournaments_file = self.data_path / "tournaments.json"
        if tournaments_file.exists():
            try:
                with open(tournaments_file) as f:
                    data = json.load(f)
                for t_data in data.get("tournaments", []):
                    tournament = Tournament.from_dict(t_data)
                    self._tournaments[tournament.id] = tournament
                    # Update ID counter
                    if tournament.id.startswith("TOURN-"):
                        try:
                            num = int(tournament.id[6:10])
                            self._next_id = max(self._next_id, num + 1)
                        except ValueError:
                            pass
            except Exception as e:
                logger.warning(f"Failed to load tournaments: {e}")

        # Load benchmarks
        benchmarks_file = self.data_path / "benchmarks.json"
        if benchmarks_file.exists():
            try:
                with open(benchmarks_file) as f:
                    data = json.load(f)
                for b_data in data.get("benchmarks", []):
                    benchmark = BenchmarkSuite.from_dict(b_data)
                    self._benchmarks[benchmark.id] = benchmark
            except Exception as e:
                logger.warning(f"Failed to load benchmarks: {e}")

        self._loaded = True

    def _save_tournaments(self) -> None:
        """Save tournaments to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "tournaments": [t.to_dict() for t in self._tournaments.values()],
        }
        with open(self.data_path / "tournaments.json", "w") as f:
            json.dump(data, f, indent=2)

    def _save_benchmarks(self) -> None:
        """Save benchmarks to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "benchmarks": [b.to_dict() for b in self._benchmarks.values()],
        }
        with open(self.data_path / "benchmarks.json", "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self) -> str:
        """Generate a unique tournament ID."""
        id_str = f"TOURN-{self._next_id:04d}"
        self._next_id += 1
        return id_str

    # =========================================================================
    # Benchmark Management
    # =========================================================================

    def get_benchmark(self, benchmark_id: str) -> Optional[BenchmarkSuite]:
        """Get a benchmark by ID."""
        self._load()
        return self._benchmarks.get(benchmark_id)

    def get_all_benchmarks(self) -> List[BenchmarkSuite]:
        """Get all benchmarks."""
        self._load()
        return list(self._benchmarks.values())

    def create_benchmark(
        self,
        benchmark_id: str,
        name: str,
        description: str,
        tasks: List[Dict[str, Any]],
        passing_score: float = 0.7,
        tags: Optional[List[str]] = None,
    ) -> BenchmarkSuite:
        """Create a benchmark suite.

        Args:
            benchmark_id: Unique ID
            name: Human-readable name
            description: What this benchmark tests
            tasks: List of task specifications
            passing_score: Minimum score to pass
            tags: Categorization tags

        Returns:
            The new benchmark
        """
        self._load()

        suite = BenchmarkSuite(
            id=benchmark_id,
            name=name,
            description=description,
            passing_score=passing_score,
            tags=tags or [],
        )

        for i, task_data in enumerate(tasks):
            task = BenchmarkTask(
                id=task_data.get("id", f"{benchmark_id}_task_{i+1}"),
                name=task_data.get("name", f"Task {i+1}"),
                description=task_data.get("description", ""),
                intent=task_data.get("intent", "general"),
                query=task_data["query"],
                expected_output_keywords=task_data.get("expected_output_keywords", []),
                difficulty=task_data.get("difficulty", "medium"),
                weight=task_data.get("weight", 1.0),
            )
            suite.tasks.append(task)

        self._benchmarks[benchmark_id] = suite
        self._save_benchmarks()
        logger.info(f"Created benchmark: {benchmark_id}")

        return suite

    # =========================================================================
    # Tournament Management
    # =========================================================================

    def get_tournament(self, tournament_id: str) -> Optional[Tournament]:
        """Get a tournament by ID."""
        self._load()
        return self._tournaments.get(tournament_id)

    def get_all_tournaments(self) -> List[Tournament]:
        """Get all tournaments."""
        self._load()
        return list(self._tournaments.values())

    def create_tournament(
        self,
        name: str,
        description: str,
        benchmark_id: str,
        participants: List[str],
    ) -> Tournament:
        """Create a new tournament.

        Args:
            name: Tournament name
            description: What we're testing
            benchmark_id: Benchmark to use
            participants: List of participant IDs

        Returns:
            The new tournament
        """
        self._load()

        tournament = Tournament(
            id=self._generate_id(),
            name=name,
            description=description,
            benchmark_id=benchmark_id,
            participants=participants,
        )

        self._tournaments[tournament.id] = tournament
        self._save_tournaments()
        logger.info(f"Created tournament: {tournament.id}")

        return tournament

    def record_match_result(
        self,
        tournament_id: str,
        task_id: str,
        participant_id: str,
        success: bool,
        reward: float = 0.0,
        latency_sec: float = 0.0,
        notes: str = "",
    ) -> bool:
        """Record a match result.

        Args:
            tournament_id: Tournament ID
            task_id: Task ID
            participant_id: Participant ID
            success: Whether succeeded
            reward: Quality score
            latency_sec: Time taken
            notes: Additional notes

        Returns:
            True if recorded
        """
        self._load()

        tournament = self._tournaments.get(tournament_id)
        if not tournament:
            return False

        result = MatchResult(
            task_id=task_id,
            participant_id=participant_id,
            success=success,
            reward=reward,
            latency_sec=latency_sec,
            notes=notes,
        )

        tournament.add_result(result)
        tournament.status = "running"
        self._save_tournaments()

        return True

    def complete_tournament(self, tournament_id: str) -> Optional[str]:
        """Complete a tournament and determine winner.

        Args:
            tournament_id: Tournament ID

        Returns:
            Winner ID if completed
        """
        self._load()

        tournament = self._tournaments.get(tournament_id)
        if not tournament:
            return None

        winner = tournament.compute_winner()
        tournament.status = "completed"
        tournament.completed_at = datetime.utcnow()

        self._save_tournaments()
        logger.info(f"Completed tournament {tournament_id}: winner = {winner}")

        return winner

    def simulate_tournament(
        self,
        tournament_id: str,
    ) -> Optional[str]:
        """Simulate a tournament with random results.

        For testing and demonstration purposes.

        Args:
            tournament_id: Tournament to simulate

        Returns:
            Winner ID
        """
        self._load()

        tournament = self._tournaments.get(tournament_id)
        if not tournament:
            return None

        benchmark = self._benchmarks.get(tournament.benchmark_id)
        if not benchmark:
            return None

        # Simulate each participant on each task
        for task in benchmark.tasks:
            for participant in tournament.participants:
                # Random success based on difficulty
                base_prob = {"easy": 0.9, "medium": 0.7, "hard": 0.5}.get(
                    task.difficulty, 0.7
                )
                success = random.random() < base_prob
                reward = random.uniform(0.5, 1.0) if success else random.uniform(0, 0.3)
                latency = random.uniform(1.0, 10.0)

                self.record_match_result(
                    tournament_id=tournament_id,
                    task_id=task.id,
                    participant_id=participant,
                    success=success,
                    reward=reward,
                    latency_sec=latency,
                )

        return self.complete_tournament(tournament_id)

    def get_summary(self) -> Dict[str, Any]:
        """Get manager summary."""
        self._load()

        completed = [t for t in self._tournaments.values() if t.status == "completed"]

        return {
            "total_tournaments": len(self._tournaments),
            "completed": len(completed),
            "total_benchmarks": len(self._benchmarks),
            "recent_winners": [
                {
                    "tournament": t.name,
                    "winner": t.winner,
                    "score": t.winner_score,
                }
                for t in sorted(completed, key=lambda t: t.completed_at or datetime.min, reverse=True)[:5]
            ],
        }


# =============================================================================
# Default Benchmarks
# =============================================================================

DEFAULT_BENCHMARKS = [
    {
        "id": "debug_basics",
        "name": "Debug Basics",
        "description": "Basic debugging tasks",
        "tasks": [
            {
                "id": "debug_1",
                "name": "Fix NameError",
                "intent": "debug_code",
                "query": "Fix this Python error: NameError: name 'foo' is not defined",
                "expected_output_keywords": ["define", "variable", "foo"],
                "difficulty": "easy",
            },
            {
                "id": "debug_2",
                "name": "Fix TypeError",
                "intent": "debug_code",
                "query": "Fix: TypeError: can only concatenate str (not 'int') to str",
                "expected_output_keywords": ["str", "convert", "type"],
                "difficulty": "easy",
            },
            {
                "id": "debug_3",
                "name": "Fix IndexError",
                "intent": "debug_code",
                "query": "Fix: IndexError: list index out of range in loop",
                "expected_output_keywords": ["bounds", "length", "index"],
                "difficulty": "medium",
            },
        ],
        "tags": ["debugging", "python"],
    },
    {
        "id": "design_review",
        "name": "Design Review",
        "description": "Architecture design review tasks",
        "tasks": [
            {
                "id": "design_1",
                "name": "Review API Design",
                "intent": "design_arch",
                "query": "Review this REST API design for a user management service",
                "expected_output_keywords": ["REST", "endpoints", "authentication"],
                "difficulty": "medium",
            },
            {
                "id": "design_2",
                "name": "Database Schema",
                "intent": "design_arch",
                "query": "Design a database schema for a blog with posts, comments, and users",
                "expected_output_keywords": ["schema", "relations", "foreign key"],
                "difficulty": "medium",
            },
        ],
        "tags": ["design", "architecture"],
    },
]


def seed_default_benchmarks(manager: TournamentManager) -> int:
    """Seed default benchmarks.

    Args:
        manager: Tournament manager

    Returns:
        Number seeded
    """
    seeded = 0
    for bm_data in DEFAULT_BENCHMARKS:
        if not manager.get_benchmark(bm_data["id"]):
            manager.create_benchmark(
                benchmark_id=bm_data["id"],
                name=bm_data["name"],
                description=bm_data["description"],
                tasks=bm_data["tasks"],
                tags=bm_data.get("tags", []),
            )
            seeded += 1

    return seeded


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[TournamentManager] = None


def get_tournament_manager() -> TournamentManager:
    """Get the default tournament manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = TournamentManager()
    return _default_manager


def create_tournament(
    name: str,
    benchmark_id: str,
    participants: List[str],
) -> Tournament:
    """Create a new tournament."""
    return get_tournament_manager().create_tournament(
        name=name,
        description=f"Tournament for {benchmark_id}",
        benchmark_id=benchmark_id,
        participants=participants,
    )


def run_simulated_tournament(
    name: str,
    benchmark_id: str,
    participants: List[str],
) -> Optional[str]:
    """Create and simulate a tournament."""
    manager = get_tournament_manager()
    tournament = manager.create_tournament(
        name=name,
        description=f"Simulated tournament for {benchmark_id}",
        benchmark_id=benchmark_id,
        participants=participants,
    )
    return manager.simulate_tournament(tournament.id)
