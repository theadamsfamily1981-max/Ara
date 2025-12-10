#!/usr/bin/env python3
"""
Ara Daily Planner - Day-Level Planning
========================================

Plans a full day of interactions as a timeline of blocks.
Considers:
- Time-of-day patterns (morning, afternoon, evening)
- User's likely activities and needs
- Resource budgets and constraints
- Project deadlines and priorities
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .session import SessionPlanner, SessionPlan, PlannedBlock, BlockPriority


@dataclass
class TimeSlot:
    """A time slot in the daily plan."""
    start_hour: int
    end_hour: int
    label: str
    default_goals: List[str]
    energy_expectation: float  # Expected user energy level


@dataclass
class DailyPlan:
    """A complete daily plan."""
    date: str
    time_slots: List[TimeSlot]
    sessions: Dict[str, SessionPlan] = field(default_factory=dict)

    # Budgets
    daily_gpu_budget_s: float = 14400   # 4 hours
    daily_power_budget_wh: float = 2000
    daily_cost_budget: float = 10.0

    # Usage tracking
    gpu_used_s: float = 0.0
    power_used_wh: float = 0.0
    cost_used: float = 0.0

    def get_current_slot(self) -> Optional[TimeSlot]:
        """Get the current time slot."""
        hour = datetime.now().hour
        for slot in self.time_slots:
            if slot.start_hour <= hour < slot.end_hour:
                return slot
        return None

    def get_budget_status(self) -> Dict[str, float]:
        """Get current budget utilization."""
        return {
            "gpu_remaining_pct": 1 - self.gpu_used_s / self.daily_gpu_budget_s,
            "power_remaining_pct": 1 - self.power_used_wh / self.daily_power_budget_wh,
            "cost_remaining_pct": 1 - self.cost_used / self.daily_cost_budget,
        }


class DailyPlanner:
    """
    Plans the full day for Ara.

    The daily planner:
    1. Divides the day into time slots
    2. Associates default goals with each slot
    3. Creates session plans as needed
    4. Manages daily resource budgets
    """

    def __init__(self):
        self.session_planner = SessionPlanner()
        self.default_slots = self._create_default_slots()

    def _create_default_slots(self) -> List[TimeSlot]:
        """Create default time slots for a typical day."""
        return [
            TimeSlot(
                start_hour=6, end_hour=9,
                label="morning_start",
                default_goals=["morning_briefing", "plan_day"],
                energy_expectation=0.6
            ),
            TimeSlot(
                start_hour=9, end_hour=12,
                label="morning_work",
                default_goals=["advance_project", "focused_work"],
                energy_expectation=0.8
            ),
            TimeSlot(
                start_hour=12, end_hour=14,
                label="midday",
                default_goals=["light_interaction", "quick_check"],
                energy_expectation=0.5
            ),
            TimeSlot(
                start_hour=14, end_hour=17,
                label="afternoon_work",
                default_goals=["advance_project", "creative_work"],
                energy_expectation=0.7
            ),
            TimeSlot(
                start_hour=17, end_hour=20,
                label="evening",
                default_goals=["decompress_user", "review_day"],
                energy_expectation=0.4
            ),
            TimeSlot(
                start_hour=20, end_hour=23,
                label="night",
                default_goals=["relax", "light_creative"],
                energy_expectation=0.3
            ),
            TimeSlot(
                start_hour=23, end_hour=6,
                label="late_night",
                default_goals=["minimal_interaction"],
                energy_expectation=0.2
            ),
        ]

    def create_daily_plan(self,
                          custom_goals: Dict[str, List[str]] = None,
                          constraints: Dict[str, float] = None) -> DailyPlan:
        """
        Create a plan for the day.

        Args:
            custom_goals: Override goals per time slot label
            constraints: Budget constraints

        Returns:
            DailyPlan for today
        """
        today = datetime.now().strftime("%Y-%m-%d")
        slots = self.default_slots.copy()

        # Apply custom goals
        if custom_goals:
            for slot in slots:
                if slot.label in custom_goals:
                    slot.default_goals = custom_goals[slot.label]

        plan = DailyPlan(
            date=today,
            time_slots=slots,
        )

        # Apply constraints
        if constraints:
            plan.daily_gpu_budget_s = constraints.get("gpu_budget_s", plan.daily_gpu_budget_s)
            plan.daily_power_budget_wh = constraints.get("power_budget_wh", plan.daily_power_budget_wh)
            plan.daily_cost_budget = constraints.get("cost_budget", plan.daily_cost_budget)

        return plan

    def get_session_for_now(self, daily_plan: DailyPlan,
                            override_goals: List[str] = None) -> SessionPlan:
        """
        Get or create a session plan for the current time.

        Args:
            daily_plan: The daily plan
            override_goals: Optional override for goals

        Returns:
            SessionPlan for current time slot
        """
        slot = daily_plan.get_current_slot()
        if not slot:
            slot = self.default_slots[-1]  # Default to late night

        slot_key = f"{daily_plan.date}_{slot.label}"

        # Check if we already have a session for this slot
        if slot_key in daily_plan.sessions:
            return daily_plan.sessions[slot_key]

        # Create new session
        goals = override_goals or slot.default_goals

        # Calculate remaining budget for this session
        budget_status = daily_plan.get_budget_status()
        remaining_hours = 24 - datetime.now().hour

        # Allocate portion of remaining budget
        session_constraints = {
            "max_gpu_seconds": daily_plan.daily_gpu_budget_s * budget_status["gpu_remaining_pct"] / max(remaining_hours / 4, 1),
            "max_cost": daily_plan.daily_cost_budget * budget_status["cost_remaining_pct"] / max(remaining_hours / 4, 1),
            "max_duration_s": (slot.end_hour - slot.start_hour) * 3600,
        }

        session = self.session_planner.create_plan(
            goals=goals,
            context={
                "time_of_day": slot.label,
                "energy": slot.energy_expectation,
            },
            constraints=session_constraints,
        )

        daily_plan.sessions[slot_key] = session
        return session

    def update_daily_usage(self, daily_plan: DailyPlan,
                           gpu_seconds: float = 0,
                           power_wh: float = 0,
                           cost: float = 0):
        """Update daily resource usage."""
        daily_plan.gpu_used_s += gpu_seconds
        daily_plan.power_used_wh += power_wh
        daily_plan.cost_used += cost

    def get_daily_summary(self, daily_plan: DailyPlan) -> Dict[str, Any]:
        """Get summary of the day's activity."""
        total_sessions = len(daily_plan.sessions)
        total_blocks = sum(len(s.blocks) for s in daily_plan.sessions.values())
        completed_blocks = sum(
            sum(1 for b in s.blocks if b.state.value == "completed")
            for s in daily_plan.sessions.values()
        )
        total_reward = sum(s.total_reward for s in daily_plan.sessions.values())

        return {
            "date": daily_plan.date,
            "sessions": total_sessions,
            "blocks_total": total_blocks,
            "blocks_completed": completed_blocks,
            "total_reward": total_reward,
            "budget": daily_plan.get_budget_status(),
            "usage": {
                "gpu_seconds": daily_plan.gpu_used_s,
                "power_wh": daily_plan.power_used_wh,
                "cost_usd": daily_plan.cost_used,
            }
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_daily_planner: Optional[DailyPlanner] = None
_current_daily_plan: Optional[DailyPlan] = None


def get_daily_planner() -> DailyPlanner:
    """Get singleton daily planner."""
    global _daily_planner
    if _daily_planner is None:
        _daily_planner = DailyPlanner()
    return _daily_planner


def get_todays_plan() -> DailyPlan:
    """Get or create today's plan."""
    global _current_daily_plan
    today = datetime.now().strftime("%Y-%m-%d")

    if _current_daily_plan is None or _current_daily_plan.date != today:
        planner = get_daily_planner()
        _current_daily_plan = planner.create_daily_plan()

    return _current_daily_plan


def get_current_session() -> SessionPlan:
    """Get session for current time."""
    planner = get_daily_planner()
    daily_plan = get_todays_plan()
    return planner.get_session_for_now(daily_plan)
