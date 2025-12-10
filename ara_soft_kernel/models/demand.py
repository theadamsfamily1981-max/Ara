"""
Demand Profile
==============

Represents what the user needs and their current state.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class UserMode(str, Enum):
    """User activity modes."""
    DEEP_WORK = "deep_work"
    NORMAL = "normal"
    RELAX = "relax"
    URGENT = "urgent"


class PrivacyTier(str, Enum):
    """Privacy level for data processing."""
    LOCAL_FIRST = "local_first"
    HIVE_OK = "hive_ok"
    LAB_OK = "lab_ok"


@dataclass
class UserState:
    """Current state of the user."""
    task: str = "unknown"
    task_description: str = ""
    mode: UserMode = UserMode.NORMAL

    # Pheromone-derived values (0-1)
    interrupt_cost: float = 0.5
    attention_world: float = 0.5
    attention_self: float = 0.5
    helpfulness_pred: float = 0.5
    safety_risk: float = 0.0

    trust_level: float = 0.5
    energy_level: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "task_description": self.task_description,
            "mode": self.mode.value,
            "interrupt_cost": self.interrupt_cost,
            "attention_world": self.attention_world,
            "attention_self": self.attention_self,
            "helpfulness_pred": self.helpfulness_pred,
            "safety_risk": self.safety_risk,
            "trust_level": self.trust_level,
            "energy_level": self.energy_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserState:
        mode = data.get("mode", "normal")
        if isinstance(mode, str):
            mode = UserMode(mode)
        return cls(
            task=data.get("task", "unknown"),
            task_description=data.get("task_description", ""),
            mode=mode,
            interrupt_cost=data.get("interrupt_cost", 0.5),
            attention_world=data.get("attention_world", 0.5),
            attention_self=data.get("attention_self", 0.5),
            helpfulness_pred=data.get("helpfulness_pred", 0.5),
            safety_risk=data.get("safety_risk", 0.0),
            trust_level=data.get("trust_level", 0.5),
            energy_level=data.get("energy_level", 0.5),
        )

    def update_from_pheromones(self, pheromones: Dict[str, float]) -> None:
        """Update state from pheromone values."""
        self.interrupt_cost = pheromones.get("INTERRUPT_COST", self.interrupt_cost)
        self.attention_world = pheromones.get("ATTN_WORLD", self.attention_world)
        self.attention_self = pheromones.get("ATTN_SELF", self.attention_self)
        self.helpfulness_pred = pheromones.get("HELPFULNESS_PRED", self.helpfulness_pred)
        self.safety_risk = pheromones.get("SAFETY_RISK", self.safety_risk)

        # Derive mode from pheromones
        if self.interrupt_cost > 0.7:
            self.mode = UserMode.DEEP_WORK
        elif self.safety_risk > 0.5:
            self.mode = UserMode.URGENT
        elif self.attention_world > 0.7:
            self.mode = UserMode.RELAX
        else:
            self.mode = UserMode.NORMAL


@dataclass
class Goal:
    """A user goal that drives agent orchestration."""
    id: str
    description: str
    priority: float = 0.5          # 0-1
    deadline: Optional[float] = None  # Unix timestamp
    time_horizon_s: Optional[float] = None
    required_capabilities: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "deadline": self.deadline,
            "time_horizon_s": self.time_horizon_s,
            "required_capabilities": self.required_capabilities,
            "subtasks": self.subtasks,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Goal:
        return cls(
            id=data["id"],
            description=data["description"],
            priority=data.get("priority", 0.5),
            deadline=data.get("deadline"),
            time_horizon_s=data.get("time_horizon_s"),
            required_capabilities=data.get("required_capabilities", []),
            subtasks=data.get("subtasks", []),
            active=data.get("active", True),
        )


@dataclass
class Constraints:
    """User-specified constraints on system behavior."""
    latency_tolerance_ms: float = 100.0
    privacy_tier: PrivacyTier = PrivacyTier.LOCAL_FIRST
    max_visual_distraction: float = 0.5  # 0 = nothing, 1 = full AR
    prefer_voice: bool = False
    prefer_visual: bool = True
    max_cost_per_hour_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_tolerance_ms": self.latency_tolerance_ms,
            "privacy_tier": self.privacy_tier.value,
            "max_visual_distraction": self.max_visual_distraction,
            "prefer_voice": self.prefer_voice,
            "prefer_visual": self.prefer_visual,
            "max_cost_per_hour_usd": self.max_cost_per_hour_usd,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Constraints:
        privacy = data.get("privacy_tier", "local_first")
        if isinstance(privacy, str):
            privacy = PrivacyTier(privacy)
        return cls(
            latency_tolerance_ms=data.get("latency_tolerance_ms", 100.0),
            privacy_tier=privacy,
            max_visual_distraction=data.get("max_visual_distraction", 0.5),
            prefer_voice=data.get("prefer_voice", False),
            prefer_visual=data.get("prefer_visual", True),
            max_cost_per_hour_usd=data.get("max_cost_per_hour_usd"),
        )


@dataclass
class DemandProfile:
    """Complete demand profile of user needs."""
    timestamp: float = field(default_factory=time.time)
    user_state: UserState = field(default_factory=UserState)
    goals: List[Goal] = field(default_factory=list)
    constraints: Constraints = field(default_factory=Constraints)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "user_state": self.user_state.to_dict(),
            "goals": [g.to_dict() for g in self.goals],
            "constraints": self.constraints.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DemandProfile:
        return cls(
            timestamp=data.get("timestamp", time.time()),
            user_state=UserState.from_dict(data.get("user_state", {})),
            goals=[Goal.from_dict(g) for g in data.get("goals", [])],
            constraints=Constraints.from_dict(data.get("constraints", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> DemandProfile:
        return cls.from_dict(json.loads(json_str))

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals sorted by priority."""
        active = [g for g in self.goals if g.active]
        return sorted(active, key=lambda g: g.priority, reverse=True)

    def get_highest_priority_goal(self) -> Optional[Goal]:
        """Get the highest priority active goal."""
        active = self.get_active_goals()
        return active[0] if active else None

    def add_goal(self, goal: Goal) -> None:
        """Add a new goal."""
        self.goals.append(goal)
        self.timestamp = time.time()

    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal by ID."""
        for i, g in enumerate(self.goals):
            if g.id == goal_id:
                self.goals.pop(i)
                self.timestamp = time.time()
                return True
        return False

    def is_deep_work(self) -> bool:
        """Check if user is in deep work mode."""
        return self.user_state.mode == UserMode.DEEP_WORK

    def is_high_safety_risk(self) -> bool:
        """Check if safety risk is elevated."""
        return self.user_state.safety_risk > 0.5

    def allows_hive(self) -> bool:
        """Check if privacy settings allow hive access."""
        return self.constraints.privacy_tier in (PrivacyTier.HIVE_OK, PrivacyTier.LAB_OK)
