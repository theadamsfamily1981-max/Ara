"""
Swarm Patterns - Job Execution Templates
========================================

Patterns define how jobs flow through agent layers.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import random
import yaml
from pathlib import Path

from .schema import AgentLayer, RiskLevel


@dataclass
class PatternStep:
    """A single step in a pattern."""
    layer: AgentLayer
    specialty: str  # e.g., "code", "research", "review"
    description: str
    required: bool = True
    timeout_ms: int = 30000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer.value,
            "specialty": self.specialty,
            "description": self.description,
            "required": self.required,
            "timeout_ms": self.timeout_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatternStep":
        return cls(
            layer=AgentLayer(d["layer"]),
            specialty=d["specialty"],
            description=d["description"],
            required=d.get("required", True),
            timeout_ms=d.get("timeout_ms", 30000),
        )


@dataclass
class Pattern:
    """A job execution pattern."""
    pattern_id: str
    job_types: List[str]  # Which job types this pattern handles
    risk_levels: List[RiskLevel]  # Which risk levels it's appropriate for
    steps: List[PatternStep]
    description: str = ""
    deprecated: bool = False

    # Stats (updated by optimization loop)
    win_rate: float = 0.5
    avg_cost: float = 100.0
    avg_latency_ms: float = 1000.0
    total_runs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "job_types": self.job_types,
            "risk_levels": [r.value for r in self.risk_levels],
            "steps": [s.to_dict() for s in self.steps],
            "description": self.description,
            "deprecated": self.deprecated,
            "win_rate": self.win_rate,
            "avg_cost": self.avg_cost,
            "avg_latency_ms": self.avg_latency_ms,
            "total_runs": self.total_runs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pattern":
        return cls(
            pattern_id=d["pattern_id"],
            job_types=d["job_types"],
            risk_levels=[RiskLevel(r) for r in d["risk_levels"]],
            steps=[PatternStep.from_dict(s) for s in d["steps"]],
            description=d.get("description", ""),
            deprecated=d.get("deprecated", False),
            win_rate=d.get("win_rate", 0.5),
            avg_cost=d.get("avg_cost", 100.0),
            avg_latency_ms=d.get("avg_latency_ms", 1000.0),
            total_runs=d.get("total_runs", 0),
        )

    @property
    def score(self) -> float:
        """Score for pattern selection (higher = better)."""
        if self.deprecated:
            return 0.0
        # win_rate / cost, with smoothing
        return self.win_rate / (self.avg_cost + 1.0)

    @property
    def max_layer(self) -> AgentLayer:
        """Highest layer used in this pattern."""
        if not self.steps:
            return AgentLayer.L0_REFLEX
        return max(s.layer for s in self.steps)


class PatternRegistry:
    """Registry of available patterns."""

    def __init__(self, config_path: Optional[str] = None):
        self.patterns: Dict[str, Pattern] = {}
        self.config_path = config_path

        if config_path and Path(config_path).exists():
            self._load_from_yaml(config_path)
        else:
            self._load_defaults()

    def _load_defaults(self):
        """Load default patterns."""
        defaults = [
            # Simple 2-step pattern for low-risk tasks
            Pattern(
                pattern_id="P_simple_2step",
                job_types=["format", "lint", "simple_query"],
                risk_levels=[RiskLevel.LOW],
                steps=[
                    PatternStep(AgentLayer.L0_REFLEX, "general", "Execute simple task"),
                    PatternStep(AgentLayer.L1_SPECIALIST, "review", "Quick sanity check"),
                ],
                description="Fast 2-step for trivial tasks",
            ),
            # Standard 3-step for code tasks
            Pattern(
                pattern_id="P_code_3step",
                job_types=["code_refactor", "code_review", "bug_fix"],
                risk_levels=[RiskLevel.LOW, RiskLevel.MEDIUM],
                steps=[
                    PatternStep(AgentLayer.L0_REFLEX, "code", "Initial code generation"),
                    PatternStep(AgentLayer.L1_SPECIALIST, "staticcheck", "Static analysis"),
                    PatternStep(AgentLayer.L2_PLANNER, "review", "Plan validation"),
                ],
                description="Standard code workflow",
            ),
            # Research pattern
            Pattern(
                pattern_id="P_research_3step",
                job_types=["research", "analysis", "summarize"],
                risk_levels=[RiskLevel.LOW, RiskLevel.MEDIUM],
                steps=[
                    PatternStep(AgentLayer.L1_SPECIALIST, "research", "Gather information"),
                    PatternStep(AgentLayer.L1_SPECIALIST, "analyze", "Analyze findings"),
                    PatternStep(AgentLayer.L2_PLANNER, "synthesize", "Synthesize report"),
                ],
                description="Research and analysis workflow",
            ),
            # High-risk pattern requiring governor approval
            Pattern(
                pattern_id="P_high_risk_4step",
                job_types=["deploy", "database_migration", "financial"],
                risk_levels=[RiskLevel.HIGH],
                steps=[
                    PatternStep(AgentLayer.L1_SPECIALIST, "prepare", "Prepare action"),
                    PatternStep(AgentLayer.L2_PLANNER, "plan", "Create execution plan"),
                    PatternStep(AgentLayer.L2_PLANNER, "validate", "Validate safety"),
                    PatternStep(AgentLayer.L3_GOVERNOR, "approve", "Final approval"),
                ],
                description="High-risk tasks requiring governor sign-off",
            ),
        ]

        for p in defaults:
            self.patterns[p.pattern_id] = p

    def _load_from_yaml(self, path: str):
        """Load patterns from YAML config."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        for p_data in data.get("patterns", []):
            pattern = Pattern.from_dict(p_data)
            self.patterns[pattern.pattern_id] = pattern

    def save_to_yaml(self, path: str):
        """Save patterns to YAML config."""
        data = {
            "patterns": [p.to_dict() for p in self.patterns.values()]
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def get(self, pattern_id: str) -> Optional[Pattern]:
        """Get pattern by ID."""
        return self.patterns.get(pattern_id)

    def find_patterns(self, job_type: str, risk: RiskLevel) -> List[Pattern]:
        """Find patterns suitable for a job type and risk level."""
        candidates = []
        for p in self.patterns.values():
            if p.deprecated:
                continue
            if job_type in p.job_types and risk in p.risk_levels:
                candidates.append(p)
        return sorted(candidates, key=lambda p: p.score, reverse=True)

    def select(self, job_type: str, risk: RiskLevel, epsilon: float = 0.1) -> Optional[Pattern]:
        """Select a pattern using epsilon-greedy strategy."""
        candidates = self.find_patterns(job_type, risk)

        if not candidates:
            # Fall back to any pattern that handles this risk level
            candidates = [p for p in self.patterns.values()
                          if not p.deprecated and risk in p.risk_levels]

        if not candidates:
            return None

        # Epsilon-greedy: usually best, sometimes explore
        if random.random() < epsilon:
            return random.choice(candidates)
        else:
            return candidates[0]  # Best by score

    def update_stats(self, pattern_id: str, won: bool, cost: float, latency_ms: float):
        """Update pattern statistics after a job run."""
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            return

        # Exponential moving average
        alpha = 0.1
        pattern.win_rate = (1 - alpha) * pattern.win_rate + alpha * (1.0 if won else 0.0)
        pattern.avg_cost = (1 - alpha) * pattern.avg_cost + alpha * cost
        pattern.avg_latency_ms = (1 - alpha) * pattern.avg_latency_ms + alpha * latency_ms
        pattern.total_runs += 1

    def deprecate(self, pattern_id: str, threshold: float = 0.3, min_runs: int = 10):
        """Deprecate a pattern if it underperforms."""
        pattern = self.patterns.get(pattern_id)
        if pattern and pattern.total_runs >= min_runs and pattern.win_rate < threshold:
            pattern.deprecated = True
            return True
        return False


# =============================================================================
# SINGLETON AND CONVENIENCE
# =============================================================================

_registry: Optional[PatternRegistry] = None


def get_registry() -> PatternRegistry:
    """Get the global pattern registry."""
    global _registry
    if _registry is None:
        _registry = PatternRegistry()
    return _registry


def select_pattern(job_type: str, risk: RiskLevel, epsilon: float = 0.1) -> Optional[Pattern]:
    """Select a pattern for a job."""
    return get_registry().select(job_type, risk, epsilon)
