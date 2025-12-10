#!/usr/bin/env python3
"""
Ara Reward System - Cost-Aware Interaction Rewards
====================================================

Computes rewards for interactions based on:
- r_user: User satisfaction and engagement
- r_progress: Project/task advancement
- r_cost: Resource consumption (negative)

Total reward: R = r_user + r_progress - λ * r_cost

This enables cost-aware optimization of Ara's behavior.
"""

import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path

from .signals import InteractionFeedback


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    # Weights
    w_user: float = 1.0          # Weight for user satisfaction
    w_progress: float = 0.5      # Weight for project progress
    w_cost: float = 0.3          # Weight for cost (lambda)

    # Cost model
    gpu_cost_per_second: float = 0.001    # $/second of GPU
    power_cost_per_kwh: float = 0.15      # $/kWh
    opportunity_cost_factor: float = 0.1  # Cost of Ara's time

    # Constraints
    max_gpu_seconds_daily: float = 14400   # 4 hours
    max_power_kwh_daily: float = 2.0
    max_cost_daily: float = 10.0


@dataclass
class InteractionReward:
    """Computed reward for an interaction."""
    interaction_id: str

    # Component rewards (0-1 scale)
    r_user: float = 0.0          # User satisfaction/engagement
    r_progress: float = 0.0      # Task/project progress
    r_cost: float = 0.0          # Normalized cost (0-1)

    # Raw costs
    gpu_seconds: float = 0.0
    power_wh: float = 0.0
    compute_cost_usd: float = 0.0

    # Computed total
    total_reward: float = 0.0

    # Context
    timestamp: float = field(default_factory=time.time)
    voice_params: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "interaction_id": self.interaction_id,
            "r_user": self.r_user,
            "r_progress": self.r_progress,
            "r_cost": self.r_cost,
            "total_reward": self.total_reward,
            "gpu_seconds": self.gpu_seconds,
            "power_wh": self.power_wh,
            "compute_cost_usd": self.compute_cost_usd,
            "voice_params": self.voice_params,
            "context": self.context,
            "timestamp": self.timestamp,
        }


class RewardTracker:
    """
    Tracks rewards across interactions and computes optimization signals.

    Usage:
        tracker = RewardTracker()

        # After each interaction:
        reward = tracker.compute_reward(
            feedback=feedback,
            gpu_seconds=10,
            power_wh=5,
            progress_indicators={"scene_added": True}
        )

        # Get optimization signals:
        stats = tracker.get_reward_stats()
        trends = tracker.get_param_reward_trends()
    """

    def __init__(self, config: RewardConfig = None, storage_path: str = None):
        self.config = config or RewardConfig()
        self.storage_path = Path(storage_path) if storage_path else None

        self.rewards: List[InteractionReward] = []
        self.daily_costs = {
            "gpu_seconds": 0.0,
            "power_wh": 0.0,
            "cost_usd": 0.0,
            "day": time.strftime("%Y-%m-%d"),
        }

        if self.storage_path and self.storage_path.exists():
            self._load_history()

    def _reset_daily_if_needed(self):
        """Reset daily counters if it's a new day."""
        today = time.strftime("%Y-%m-%d")
        if self.daily_costs["day"] != today:
            self.daily_costs = {
                "gpu_seconds": 0.0,
                "power_wh": 0.0,
                "cost_usd": 0.0,
                "day": today,
            }

    def compute_reward(self,
                       feedback: InteractionFeedback,
                       gpu_seconds: float = 0.0,
                       power_wh: float = 0.0,
                       progress_indicators: Dict[str, Any] = None) -> InteractionReward:
        """
        Compute reward for an interaction.

        Args:
            feedback: Feedback signals from the interaction
            gpu_seconds: GPU time consumed
            power_wh: Power consumed in Wh
            progress_indicators: Dict of progress signals (e.g., {"scene_added": True})

        Returns:
            InteractionReward with all components
        """
        self._reset_daily_if_needed()

        # r_user: Weighted combination of feedback scores
        r_user = (
            0.3 * feedback.understanding_score +
            0.3 * feedback.engagement_score +
            0.3 * feedback.satisfaction_score +
            0.1 * feedback.task_success_score
        )

        # r_progress: Based on progress indicators
        progress_indicators = progress_indicators or {}
        r_progress = 0.0

        # Each indicator contributes to progress
        progress_weights = {
            "scene_added": 0.3,
            "script_written": 0.3,
            "asset_created": 0.2,
            "task_completed": 0.2,
            "video_rendered": 0.4,
            "episode_advanced": 0.5,
        }

        for indicator, value in progress_indicators.items():
            weight = progress_weights.get(indicator, 0.1)
            if isinstance(value, bool):
                r_progress += weight if value else 0
            else:
                r_progress += weight * float(value)

        r_progress = min(1.0, r_progress)

        # r_cost: Normalized cost (higher = more expensive)
        compute_cost = (
            gpu_seconds * self.config.gpu_cost_per_second +
            (power_wh / 1000) * self.config.power_cost_per_kwh
        )

        # Add opportunity cost (time Ara spent)
        compute_cost += feedback.duration_s * self.config.opportunity_cost_factor / 3600

        # Normalize cost to 0-1 (based on typical interaction cost ~$0.01-0.10)
        r_cost = min(1.0, compute_cost / 0.10)

        # Update daily tracking
        self.daily_costs["gpu_seconds"] += gpu_seconds
        self.daily_costs["power_wh"] += power_wh
        self.daily_costs["cost_usd"] += compute_cost

        # Compute total reward
        total_reward = (
            self.config.w_user * r_user +
            self.config.w_progress * r_progress -
            self.config.w_cost * r_cost
        )

        reward = InteractionReward(
            interaction_id=feedback.interaction_id,
            r_user=r_user,
            r_progress=r_progress,
            r_cost=r_cost,
            gpu_seconds=gpu_seconds,
            power_wh=power_wh,
            compute_cost_usd=compute_cost,
            total_reward=total_reward,
            voice_params=feedback.voice_params,
            context={
                "scores": {
                    "understanding": feedback.understanding_score,
                    "engagement": feedback.engagement_score,
                    "satisfaction": feedback.satisfaction_score,
                    "task_success": feedback.task_success_score,
                },
                "progress": progress_indicators,
                "duration_s": feedback.duration_s,
            }
        )

        self.rewards.append(reward)
        self._save_if_configured()

        return reward

    def get_reward_stats(self, n_recent: int = 50) -> Dict[str, Any]:
        """Get statistics on recent rewards."""
        recent = self.rewards[-n_recent:] if self.rewards else []

        if not recent:
            return {
                "count": 0,
                "avg_total": 0,
                "avg_r_user": 0,
                "avg_r_progress": 0,
                "avg_r_cost": 0,
            }

        return {
            "count": len(recent),
            "avg_total": np.mean([r.total_reward for r in recent]),
            "avg_r_user": np.mean([r.r_user for r in recent]),
            "avg_r_progress": np.mean([r.r_progress for r in recent]),
            "avg_r_cost": np.mean([r.r_cost for r in recent]),
            "total_gpu_seconds": sum(r.gpu_seconds for r in recent),
            "total_cost_usd": sum(r.compute_cost_usd for r in recent),
        }

    def get_daily_budget_status(self) -> Dict[str, Any]:
        """Get current daily budget usage."""
        self._reset_daily_if_needed()

        return {
            "date": self.daily_costs["day"],
            "gpu_seconds_used": self.daily_costs["gpu_seconds"],
            "gpu_seconds_limit": self.config.max_gpu_seconds_daily,
            "gpu_budget_remaining": 1 - self.daily_costs["gpu_seconds"] / self.config.max_gpu_seconds_daily,
            "power_wh_used": self.daily_costs["power_wh"],
            "cost_usd_used": self.daily_costs["cost_usd"],
            "cost_limit": self.config.max_cost_daily,
            "cost_budget_remaining": 1 - self.daily_costs["cost_usd"] / self.config.max_cost_daily,
        }

    def get_param_reward_trends(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze how voice/visual params correlate with rewards.

        Returns correlations between params and reward components.
        """
        if len(self.rewards) < 10:
            return {}

        # Collect param names
        param_names = set()
        for r in self.rewards:
            param_names.update(r.voice_params.keys())

        trends = {}
        for param in param_names:
            param_values = []
            rewards = {"total": [], "r_user": [], "r_progress": []}

            for r in self.rewards:
                if param in r.voice_params:
                    param_values.append(r.voice_params[param])
                    rewards["total"].append(r.total_reward)
                    rewards["r_user"].append(r.r_user)
                    rewards["r_progress"].append(r.r_progress)

            if len(param_values) >= 5:
                trends[param] = {}
                for reward_name, reward_values in rewards.items():
                    if np.std(param_values) > 0 and np.std(reward_values) > 0:
                        corr = np.corrcoef(param_values, reward_values)[0, 1]
                        trends[param][reward_name] = corr if not np.isnan(corr) else 0

        return trends

    def should_reduce_costs(self) -> bool:
        """Check if we should reduce costs based on budget."""
        status = self.get_daily_budget_status()
        return (
            status["gpu_budget_remaining"] < 0.2 or
            status["cost_budget_remaining"] < 0.2
        )

    def _save_if_configured(self):
        """Save rewards to storage if configured."""
        if self.storage_path:
            data = {
                "rewards": [r.to_dict() for r in self.rewards[-1000:]],
                "daily_costs": self.daily_costs,
            }
            self.storage_path.write_text(json.dumps(data, indent=2))

    def _load_history(self):
        """Load reward history from storage."""
        try:
            data = json.loads(self.storage_path.read_text())
            # Reconstruct rewards (simplified)
            self.daily_costs = data.get("daily_costs", self.daily_costs)
        except Exception:
            pass
