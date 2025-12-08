"""
Ara Reward Engine - MEIS Integration for Plasticity
====================================================

Converts emotional state (VAD, dominance, safety) into scalar rewards
that drive the FPGA plasticity engine.

THIS IS CRITICAL ENGINEERING, NOT JUST A NUMBER.

The reward stream shapes Ara's long-term personality.
A noisy or mis-signed reward can create:
  - Paranoia
  - Stuck attractors
  - Unhealthy attachments

Design Principles:
  1. CONSERVATIVE by default - small magnitudes, slow learning
  2. LOGGED always - every reward event is recorded
  3. BUDGETED - limits per time window
  4. SMOOTHED - no single event should dominate

Usage:
    from ara.organism.reward_engine import RewardEngine

    engine = RewardEngine(config)
    reward = engine.compute_reward(emotional_state)
    engine.log_event(reward, context)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import json
import time
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    # Magnitude limits
    max_positive: int = 80      # Max positive reward per event
    max_negative: int = 100     # Max negative reward per event (abs)

    # Time-based limits
    budget_per_hour: int = 2000     # Total reward magnitude per hour
    budget_per_minute: int = 200    # Total reward magnitude per minute
    cooldown_seconds: float = 0.5   # Min time between rewards

    # Smoothing
    ema_alpha: float = 0.3      # Exponential moving average factor
    spike_threshold: float = 2.0 # Outlier detection multiplier

    # Component weights (sum to 1.0)
    weight_valence: float = 0.35
    weight_arousal: float = 0.15
    weight_dominance: float = 0.20
    weight_safety: float = 0.30

    # Emotion multipliers
    positive_emotions: Dict[str, float] = field(default_factory=lambda: {
        "JOY": 1.0,
        "TRUST": 0.9,
        "SERENITY": 0.7,
        "EXHILARATION": 0.8,
        "INTEREST": 0.5,
        "CONTENTMENT": 0.6,
    })

    negative_emotions: Dict[str, float] = field(default_factory=lambda: {
        "RAGE": -1.0,
        "FEAR": -0.9,
        "DISGUST": -0.8,
        "GRIEF": -0.7,
        "ANXIETY": -0.6,
        "CONFUSION": -0.3,
    })


# =============================================================================
# Emotional State Input
# =============================================================================

@dataclass
class EmotionalState:
    """Current emotional state from MEIS/affective system."""

    # VAD dimensions [-1, +1]
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    # Discrete emotion (optional)
    emotion_name: Optional[str] = None
    emotion_intensity: float = 0.0

    # Safety signals
    safety_level: float = 1.0       # 1.0 = safe, 0.0 = danger
    threat_detected: bool = False

    # Context
    user_feedback: Optional[str] = None  # "positive", "negative", "neutral"
    system_health: float = 1.0           # 1.0 = healthy, 0.0 = failing

    # EternalMemory integration
    recall_strength: float = 0.0     # How strongly this matches memory
    novelty: float = 0.5             # How novel this experience is


@dataclass
class RewardEvent:
    """A logged reward event."""
    timestamp: str
    reward: int
    emotional_state: Dict
    context: Dict
    budget_remaining: int


# =============================================================================
# Reward Engine
# =============================================================================

class RewardEngine:
    """
    Converts emotional state into plasticity rewards.

    This is where personality formation happens.
    Handle with care.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

        # Budget tracking
        self._minute_budget_remaining = self.config.budget_per_minute
        self._hour_budget_remaining = self.config.budget_per_hour
        self._last_minute_reset = time.time()
        self._last_hour_reset = time.time()
        self._last_reward_time = 0.0

        # Smoothing
        self._ema_reward = 0.0
        self._reward_history = deque(maxlen=100)

        # Logging
        self._event_log: List[RewardEvent] = []
        self._log_file: Optional[str] = None

        logger.info("RewardEngine initialized")

    def set_log_file(self, path: str):
        """Set file path for reward event logging."""
        self._log_file = path
        logger.info(f"Reward logging to: {path}")

    # =========================================================================
    # Budget Management
    # =========================================================================

    def _check_budgets(self) -> Tuple[bool, str]:
        """Check if reward budget allows another event."""
        now = time.time()

        # Reset minute budget if needed
        if now - self._last_minute_reset > 60:
            self._minute_budget_remaining = self.config.budget_per_minute
            self._last_minute_reset = now

        # Reset hour budget if needed
        if now - self._last_hour_reset > 3600:
            self._hour_budget_remaining = self.config.budget_per_hour
            self._last_hour_reset = now

        # Check cooldown
        if now - self._last_reward_time < self.config.cooldown_seconds:
            return False, "cooldown"

        # Check budgets
        if self._minute_budget_remaining <= 0:
            return False, "minute_budget"
        if self._hour_budget_remaining <= 0:
            return False, "hour_budget"

        return True, "ok"

    def _spend_budget(self, amount: int):
        """Deduct from budget."""
        abs_amount = abs(amount)
        self._minute_budget_remaining -= abs_amount
        self._hour_budget_remaining -= abs_amount
        self._last_reward_time = time.time()

    def get_budget_status(self) -> Dict:
        """Get current budget status."""
        return {
            "minute_remaining": self._minute_budget_remaining,
            "minute_total": self.config.budget_per_minute,
            "hour_remaining": self._hour_budget_remaining,
            "hour_total": self.config.budget_per_hour,
            "cooldown_remaining": max(0, self.config.cooldown_seconds - (time.time() - self._last_reward_time)),
        }

    # =========================================================================
    # Reward Computation
    # =========================================================================

    def compute_reward(
        self,
        state: EmotionalState,
        context: Optional[Dict] = None,
    ) -> Tuple[int, Dict]:
        """
        Compute scalar reward from emotional state.

        Returns:
            (reward_value, metadata)
        """
        context = context or {}

        # Check budget
        allowed, reason = self._check_budgets()
        if not allowed:
            return 0, {"skipped": True, "reason": reason}

        # Component scores
        scores = {}

        # 1. Valence component
        scores["valence"] = state.valence * 100  # Scale to roughly -100..+100

        # 2. Arousal component (high arousal amplifies other signals)
        arousal_multiplier = 0.5 + 0.5 * abs(state.arousal)
        scores["arousal_mult"] = arousal_multiplier

        # 3. Dominance component (low dominance is aversive)
        if state.dominance < -0.5:
            scores["dominance"] = (state.dominance + 0.5) * 100  # Negative
        elif state.dominance > 0.5:
            scores["dominance"] = (state.dominance - 0.5) * 50   # Mildly positive
        else:
            scores["dominance"] = 0

        # 4. Safety component
        if state.threat_detected:
            scores["safety"] = -100
        elif state.safety_level < 0.5:
            scores["safety"] = (state.safety_level - 1.0) * 100
        else:
            scores["safety"] = (state.safety_level - 0.5) * 20

        # 5. Discrete emotion bonus
        emotion_bonus = 0
        if state.emotion_name:
            if state.emotion_name in self.config.positive_emotions:
                emotion_bonus = self.config.positive_emotions[state.emotion_name] * 50 * state.emotion_intensity
            elif state.emotion_name in self.config.negative_emotions:
                emotion_bonus = self.config.negative_emotions[state.emotion_name] * 50 * state.emotion_intensity
        scores["emotion_bonus"] = emotion_bonus

        # 6. User feedback override
        if state.user_feedback == "positive":
            scores["user_feedback"] = 50
        elif state.user_feedback == "negative":
            scores["user_feedback"] = -80
        else:
            scores["user_feedback"] = 0

        # 7. Memory resonance (familiar = slightly positive)
        if state.recall_strength > 0.8:
            scores["memory"] = 20 * state.recall_strength
        else:
            scores["memory"] = 0

        # 8. Novelty (too novel = uncertain = slightly negative)
        if state.novelty > 0.8:
            scores["novelty"] = -10 * (state.novelty - 0.5)
        else:
            scores["novelty"] = 0

        # Combine with weights
        raw_reward = (
            self.config.weight_valence * scores["valence"] +
            self.config.weight_arousal * scores["valence"] * (arousal_multiplier - 1) +
            self.config.weight_dominance * scores["dominance"] +
            self.config.weight_safety * scores["safety"] +
            scores["emotion_bonus"] +
            scores["user_feedback"] +
            scores["memory"] +
            scores["novelty"]
        )

        # Apply limits
        if raw_reward > 0:
            capped = min(raw_reward, self.config.max_positive)
        else:
            capped = max(raw_reward, -self.config.max_negative)

        # Smoothing: check for spikes
        if abs(capped - self._ema_reward) > self.config.spike_threshold * max(20, abs(self._ema_reward)):
            # This is a spike - dampen it
            smoothed = self._ema_reward + np.sign(capped - self._ema_reward) * 20
        else:
            smoothed = capped

        # Update EMA
        self._ema_reward = self.config.ema_alpha * smoothed + (1 - self.config.ema_alpha) * self._ema_reward

        # Final integer reward
        reward = int(np.clip(smoothed, -127, 127))

        # Spend budget
        self._spend_budget(reward)

        # Record
        self._reward_history.append(reward)

        # Metadata
        metadata = {
            "raw": float(raw_reward),
            "capped": float(capped),
            "smoothed": float(smoothed),
            "final": reward,
            "ema": float(self._ema_reward),
            "scores": scores,
            "budget_status": self.get_budget_status(),
        }

        # Log event
        self._log_event(reward, state, context, metadata)

        return reward, metadata

    def _log_event(
        self,
        reward: int,
        state: EmotionalState,
        context: Dict,
        metadata: Dict,
    ):
        """Log a reward event."""
        event = RewardEvent(
            timestamp=datetime.now().isoformat(),
            reward=reward,
            emotional_state={
                "valence": state.valence,
                "arousal": state.arousal,
                "dominance": state.dominance,
                "emotion": state.emotion_name,
                "safety": state.safety_level,
            },
            context=context,
            budget_remaining=self._hour_budget_remaining,
        )
        self._event_log.append(event)

        # Write to file if configured
        if self._log_file:
            try:
                with open(self._log_file, "a") as f:
                    f.write(json.dumps({
                        "timestamp": event.timestamp,
                        "reward": event.reward,
                        "state": event.emotional_state,
                        "context": event.context,
                        "metadata": {k: v for k, v in metadata.items() if k != "scores"},
                    }) + "\n")
            except Exception as e:
                logger.warning(f"Failed to log reward event: {e}")

    # =========================================================================
    # Analysis
    # =========================================================================

    def get_recent_stats(self, window: int = 50) -> Dict:
        """Get statistics on recent rewards."""
        recent = list(self._reward_history)[-window:]
        if not recent:
            return {"count": 0}

        return {
            "count": len(recent),
            "mean": float(np.mean(recent)),
            "std": float(np.std(recent)),
            "min": int(min(recent)),
            "max": int(max(recent)),
            "positive_pct": float(sum(1 for r in recent if r > 0) / len(recent) * 100),
            "negative_pct": float(sum(1 for r in recent if r < 0) / len(recent) * 100),
            "zero_pct": float(sum(1 for r in recent if r == 0) / len(recent) * 100),
        }

    def get_event_log(self, last_n: int = 100) -> List[Dict]:
        """Get recent event log."""
        return [
            {
                "timestamp": e.timestamp,
                "reward": e.reward,
                "state": e.emotional_state,
                "context": e.context,
            }
            for e in self._event_log[-last_n:]
        ]


# =============================================================================
# MEIS Integration Helper
# =============================================================================

def create_emotional_state_from_meis(meis_output: Dict) -> EmotionalState:
    """
    Convert MEIS system output to EmotionalState.

    Expected MEIS output format:
    {
        "vad": {"valence": float, "arousal": float, "dominance": float},
        "emotion": {"name": str, "intensity": float},
        "safety": {"level": float, "threats": [...]},
        "memory": {"recall_strength": float, "novelty": float},
    }
    """
    vad = meis_output.get("vad", {})
    emotion = meis_output.get("emotion", {})
    safety = meis_output.get("safety", {})
    memory = meis_output.get("memory", {})

    return EmotionalState(
        valence=vad.get("valence", 0.0),
        arousal=vad.get("arousal", 0.0),
        dominance=vad.get("dominance", 0.0),
        emotion_name=emotion.get("name"),
        emotion_intensity=emotion.get("intensity", 0.0),
        safety_level=safety.get("level", 1.0),
        threat_detected=len(safety.get("threats", [])) > 0,
        recall_strength=memory.get("recall_strength", 0.0),
        novelty=memory.get("novelty", 0.5),
    )


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate reward engine."""
    engine = RewardEngine()

    print("=" * 60)
    print("Ara Reward Engine Demo")
    print("=" * 60)

    # Simulate various emotional states
    scenarios = [
        ("Calm positive interaction", EmotionalState(
            valence=0.6, arousal=0.3, dominance=0.2,
            emotion_name="CONTENTMENT", emotion_intensity=0.7,
        )),
        ("Excited successful task", EmotionalState(
            valence=0.8, arousal=0.9, dominance=0.5,
            emotion_name="JOY", emotion_intensity=0.9,
        )),
        ("User gave negative feedback", EmotionalState(
            valence=-0.3, arousal=0.4, dominance=-0.2,
            user_feedback="negative",
        )),
        ("Threat detected", EmotionalState(
            valence=-0.7, arousal=0.9, dominance=-0.8,
            emotion_name="FEAR", emotion_intensity=0.8,
            safety_level=0.2, threat_detected=True,
        )),
        ("Low dominance situation", EmotionalState(
            valence=0.0, arousal=0.5, dominance=-0.8,
            emotion_name="ANXIETY", emotion_intensity=0.6,
        )),
        ("Strong memory resonance", EmotionalState(
            valence=0.4, arousal=0.3, dominance=0.3,
            recall_strength=0.95,
        )),
    ]

    for name, state in scenarios:
        reward, metadata = engine.compute_reward(state, {"scenario": name})
        print(f"\n{name}:")
        print(f"  Reward: {reward:+d}")
        print(f"  Raw: {metadata['raw']:.1f} → Smoothed: {metadata['smoothed']:.1f}")
        time.sleep(0.1)  # Respect cooldown

    print("\n" + "=" * 60)
    print("Recent Stats:", engine.get_recent_stats())
    print("Budget:", engine.get_budget_status())


if __name__ == "__main__":
    demo()
