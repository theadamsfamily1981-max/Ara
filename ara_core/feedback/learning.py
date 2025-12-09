#!/usr/bin/env python3
"""
Ara Parameter Learning - Contextual Optimization
==================================================

Learns optimal voice/visual parameters from feedback over time.
Uses contextual bandits approach: different contexts (time, mood, topic)
may have different optimal parameters.

Key insight: Beauty and communication are *personal* - what works
is cached in memory and refined over time.
"""

import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from .signals import InteractionFeedback
from .rewards import InteractionReward


@dataclass
class ContextualParams:
    """Optimal parameters for a specific context."""
    context_key: str  # e.g., "evening_tired", "morning_work", "creative"

    # Voice parameters
    tone: str = "neutral"
    warmth: float = 0.5
    pace: float = 1.0
    energy: float = 0.5

    # Visual parameters (placeholder for avatar/video)
    visual_style: str = "default"
    brightness: float = 0.5
    animation_speed: float = 1.0

    # Learning stats
    n_samples: int = 0
    avg_reward: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def to_voice_params(self) -> Dict[str, float]:
        """Get voice params dict."""
        return {
            "warmth": self.warmth,
            "pace": self.pace,
            "energy": self.energy,
        }

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LearningConfig:
    """Configuration for parameter learning."""
    # Exploration
    exploration_rate: float = 0.15      # Probability of trying new params
    exploration_decay: float = 0.995    # Decay per interaction

    # Learning rate
    learning_rate: float = 0.1          # How fast to update params
    min_samples: int = 3                # Min samples before trusting context

    # Parameter bounds
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "warmth": (0.2, 0.95),
        "pace": (0.7, 1.3),
        "energy": (0.2, 0.9),
        "brightness": (0.3, 0.8),
        "animation_speed": (0.8, 1.2),
    })


class ParameterLearner:
    """
    Learns optimal parameters per context using feedback.

    Approach:
    1. Extract context from interaction (time, emotion, topic)
    2. Look up or initialize params for that context
    3. Occasionally explore (try slightly different params)
    4. Update params based on reward

    This creates a personalized "style memory" that adapts
    to what works for the specific user in specific situations.
    """

    def __init__(self, config: LearningConfig = None, storage_path: str = None):
        self.config = config or LearningConfig()
        self.storage_path = Path(storage_path) if storage_path else None

        # Context -> ContextualParams mapping
        self.context_params: Dict[str, ContextualParams] = {}

        # Default params (used when context is unknown)
        self.default_params = ContextualParams(
            context_key="default",
            warmth=0.6,
            pace=1.0,
            energy=0.5,
        )

        # Current exploration rate
        self.exploration_rate = self.config.exploration_rate

        # History for analysis
        self.param_history: List[Dict] = []

        if self.storage_path and self.storage_path.exists():
            self._load()

    def extract_context_key(self,
                            time_of_day: str = None,
                            emotion: str = None,
                            topic: str = None,
                            energy_level: float = None) -> str:
        """
        Extract a context key from interaction context.

        Examples:
        - "evening_tired" -> slower pace, warmer tone
        - "morning_work" -> crisper, more energetic
        - "creative_excited" -> more expressive
        """
        parts = []

        # Time context
        if time_of_day:
            parts.append(time_of_day)
        else:
            hour = time.localtime().tm_hour
            if hour < 6:
                parts.append("night")
            elif hour < 12:
                parts.append("morning")
            elif hour < 18:
                parts.append("afternoon")
            else:
                parts.append("evening")

        # Emotion/energy context
        if emotion:
            # Simplify emotion to broad category
            calm_emotions = ["calm", "peaceful", "relaxed", "tender", "intimate"]
            energetic_emotions = ["excited", "happy", "joyful", "uplifting", "powerful"]
            focused_emotions = ["neutral", "confident", "anticipation"]

            if emotion.lower() in calm_emotions:
                parts.append("calm")
            elif emotion.lower() in energetic_emotions:
                parts.append("energetic")
            elif emotion.lower() in focused_emotions:
                parts.append("focused")
            else:
                parts.append("mixed")

        # Topic context (simplified)
        if topic:
            if any(t in topic.lower() for t in ["work", "task", "code", "technical"]):
                parts.append("work")
            elif any(t in topic.lower() for t in ["creative", "art", "show", "music"]):
                parts.append("creative")
            elif any(t in topic.lower() for t in ["relax", "chill", "decompress"]):
                parts.append("relax")

        return "_".join(parts) if parts else "default"

    def get_params(self, context_key: str = None, **context_kwargs) -> ContextualParams:
        """
        Get parameters for a context, with optional exploration.

        Args:
            context_key: Explicit context key, or None to extract from kwargs
            **context_kwargs: Context info (time_of_day, emotion, topic, etc.)

        Returns:
            ContextualParams to use for this interaction
        """
        if context_key is None:
            context_key = self.extract_context_key(**context_kwargs)

        # Get or create params for this context
        if context_key in self.context_params:
            base_params = self.context_params[context_key]
        else:
            # Initialize from default
            base_params = ContextualParams(
                context_key=context_key,
                warmth=self.default_params.warmth,
                pace=self.default_params.pace,
                energy=self.default_params.energy,
            )
            self.context_params[context_key] = base_params

        # Maybe explore
        if np.random.random() < self.exploration_rate:
            return self._explore(base_params)

        return base_params

    def _explore(self, base_params: ContextualParams) -> ContextualParams:
        """Create an exploratory variant of params."""
        explored = ContextualParams(
            context_key=base_params.context_key + "_explore",
            tone=base_params.tone,
            warmth=base_params.warmth,
            pace=base_params.pace,
            energy=base_params.energy,
            visual_style=base_params.visual_style,
            brightness=base_params.brightness,
            animation_speed=base_params.animation_speed,
        )

        # Perturb one or more parameters
        params_to_perturb = ["warmth", "pace", "energy"]
        for param in params_to_perturb:
            if np.random.random() < 0.5:
                current = getattr(explored, param)
                bounds = self.config.param_bounds.get(param, (0, 1))

                # Random perturbation
                delta = np.random.normal(0, 0.1)
                new_val = np.clip(current + delta, bounds[0], bounds[1])
                setattr(explored, param, new_val)

        return explored

    def update(self, context_key: str, used_params: ContextualParams,
               reward: InteractionReward):
        """
        Update parameters based on observed reward.

        Uses exponential moving average to smoothly update params
        toward values that yield higher rewards.
        """
        # Skip exploration-specific updates for the main context
        base_key = context_key.replace("_explore", "")

        if base_key not in self.context_params:
            self.context_params[base_key] = ContextualParams(context_key=base_key)

        params = self.context_params[base_key]
        lr = self.config.learning_rate

        # Update running average reward
        params.n_samples += 1
        params.avg_reward = (
            (1 - lr) * params.avg_reward +
            lr * reward.total_reward
        )

        # If this was a good interaction, move params toward what we used
        if reward.total_reward > params.avg_reward:
            # Move toward used params
            blend = lr * (reward.total_reward - params.avg_reward + 0.5)
            blend = np.clip(blend, 0, 0.3)

            params.warmth = (1 - blend) * params.warmth + blend * used_params.warmth
            params.pace = (1 - blend) * params.pace + blend * used_params.pace
            params.energy = (1 - blend) * params.energy + blend * used_params.energy

            # Clamp to bounds
            for param_name in ["warmth", "pace", "energy"]:
                bounds = self.config.param_bounds.get(param_name, (0, 1))
                val = getattr(params, param_name)
                setattr(params, param_name, np.clip(val, bounds[0], bounds[1]))

        params.last_updated = time.time()

        # Decay exploration rate
        self.exploration_rate *= self.config.exploration_decay
        self.exploration_rate = max(0.05, self.exploration_rate)

        # Record history
        self.param_history.append({
            "context": base_key,
            "params": used_params.to_voice_params(),
            "reward": reward.total_reward,
            "timestamp": time.time(),
        })

        self._save_if_configured()

    def get_learned_insights(self) -> Dict[str, Any]:
        """Get summary of what has been learned."""
        insights = {
            "contexts_learned": len(self.context_params),
            "total_samples": sum(p.n_samples for p in self.context_params.values()),
            "exploration_rate": self.exploration_rate,
            "top_contexts": [],
        }

        # Find best-performing contexts
        sorted_contexts = sorted(
            self.context_params.values(),
            key=lambda p: p.avg_reward if p.n_samples >= self.config.min_samples else -1,
            reverse=True
        )

        for ctx in sorted_contexts[:5]:
            if ctx.n_samples >= self.config.min_samples:
                insights["top_contexts"].append({
                    "context": ctx.context_key,
                    "avg_reward": ctx.avg_reward,
                    "n_samples": ctx.n_samples,
                    "params": ctx.to_voice_params(),
                })

        return insights

    def get_recommendation(self, context_key: str) -> str:
        """Get a human-readable recommendation for a context."""
        if context_key not in self.context_params:
            return f"No data yet for '{context_key}' context."

        params = self.context_params[context_key]
        if params.n_samples < self.config.min_samples:
            return f"Still learning '{context_key}' ({params.n_samples} samples so far)."

        # Generate recommendation
        rec = f"For '{context_key}' context:\n"

        if params.warmth > 0.7:
            rec += "  - Use warmer, softer tone\n"
        elif params.warmth < 0.4:
            rec += "  - Use crisper, more neutral tone\n"

        if params.pace < 0.9:
            rec += "  - Speak slower\n"
        elif params.pace > 1.1:
            rec += "  - Speak faster\n"

        if params.energy > 0.6:
            rec += "  - More energetic delivery\n"
        elif params.energy < 0.4:
            rec += "  - Calmer, lower energy\n"

        rec += f"  (based on {params.n_samples} interactions, avg reward: {params.avg_reward:.2f})"

        return rec

    def _save_if_configured(self):
        """Save learned params to storage."""
        if self.storage_path:
            data = {
                "context_params": {k: v.to_dict() for k, v in self.context_params.items()},
                "default_params": self.default_params.to_dict(),
                "exploration_rate": self.exploration_rate,
                "history_count": len(self.param_history),
            }
            self.storage_path.write_text(json.dumps(data, indent=2))

    def _load(self):
        """Load learned params from storage."""
        try:
            data = json.loads(self.storage_path.read_text())

            for key, pdata in data.get("context_params", {}).items():
                self.context_params[key] = ContextualParams(**pdata)

            if "default_params" in data:
                self.default_params = ContextualParams(**data["default_params"])

            self.exploration_rate = data.get("exploration_rate", self.config.exploration_rate)

        except Exception as e:
            print(f"[learner] Failed to load: {e}")
