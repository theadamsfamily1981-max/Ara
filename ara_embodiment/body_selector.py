"""
Body Selector
=============

Picks the appropriate avatar based on:
- Device class (AR, VR, phone, desktop, speaker)
- Context (work, social, focus, rest)
- User preferences
- Environment state
- Energy/formality level

Can be overridden with fixed personas for specific situations.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum

from ara_embodiment.body_registry import (
    BodyRegistry,
    AvatarDefinition,
    AvatarCapability,
    DeviceClass,
)


class ContextType(str, Enum):
    """High-level context types."""
    WORK = "work"            # Focused productivity
    SOCIAL = "social"        # Casual conversation
    CREATIVE = "creative"    # Art, writing, brainstorming
    LEARNING = "learning"    # Study, tutorial
    REST = "rest"            # Relaxation, low energy
    EXERCISE = "exercise"    # Physical activity
    TRANSIT = "transit"      # Moving between places
    UNKNOWN = "unknown"


@dataclass
class DeviceContext:
    """Current device and its capabilities."""
    device_class: DeviceClass
    screen_width: int = 0
    screen_height: int = 0
    has_camera: bool = False
    has_depth: bool = False
    has_microphone: bool = True
    has_speaker: bool = True
    gpu_available: bool = False
    battery_level: float = 1.0  # 0-1
    network_quality: float = 1.0  # 0-1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeviceContext:
        return cls(
            device_class=DeviceClass(data.get("device_class", "desktop")),
            screen_width=data.get("screen_width", 0),
            screen_height=data.get("screen_height", 0),
            has_camera=data.get("has_camera", False),
            has_depth=data.get("has_depth", False),
            has_microphone=data.get("has_microphone", True),
            has_speaker=data.get("has_speaker", True),
            gpu_available=data.get("gpu_available", False),
            battery_level=data.get("battery_level", 1.0),
            network_quality=data.get("network_quality", 1.0),
        )


@dataclass
class SelectionCriteria:
    """Criteria for avatar selection."""
    # Required
    device: DeviceContext

    # Context
    context_type: ContextType = ContextType.UNKNOWN
    formality: float = 0.5  # 0 = casual, 1 = formal
    energy: float = 0.5     # 0 = calm, 1 = energetic

    # Preferences
    preferred_style: Optional[str] = None  # "realistic", "stylized", "abstract"
    required_capabilities: Set[AvatarCapability] = field(default_factory=set)
    excluded_capabilities: Set[AvatarCapability] = field(default_factory=set)
    preferred_tags: List[str] = field(default_factory=list)

    # Overrides
    force_avatar_id: Optional[str] = None  # Hard override

    # Resource constraints
    max_fps_budget: int = 60
    low_power_mode: bool = False


@dataclass
class SelectionResult:
    """Result of avatar selection."""
    avatar: AvatarDefinition
    confidence: float  # 0-1, how good is this match
    reason: str        # Why this avatar was chosen
    alternatives: List[str] = field(default_factory=list)  # Other avatar IDs considered


class BodySelector:
    """
    Selects appropriate avatar based on context and preferences.

    Thread-safe: selection logic is stateless, preferences protected by lock.
    """

    def __init__(self, registry: BodyRegistry):
        self._lock = threading.RLock()
        self._registry = registry

        # User preferences (learned over time)
        self._device_preferences: Dict[DeviceClass, str] = {}
        self._context_preferences: Dict[ContextType, str] = {}
        self._selection_history: List[Dict[str, Any]] = []

        # Override rules
        self._override_rules: List[Callable[[SelectionCriteria], Optional[str]]] = []

    def select(self, criteria: SelectionCriteria) -> SelectionResult:
        """Select the best avatar for the given criteria."""

        # Check for hard override
        if criteria.force_avatar_id:
            avatar = self._registry.get(criteria.force_avatar_id)
            if avatar:
                return SelectionResult(
                    avatar=avatar,
                    confidence=1.0,
                    reason="forced_override",
                )

        # Check override rules
        for rule in self._override_rules:
            override_id = rule(criteria)
            if override_id:
                avatar = self._registry.get(override_id)
                if avatar:
                    return SelectionResult(
                        avatar=avatar,
                        confidence=0.95,
                        reason="rule_override",
                    )

        # Get candidates for this device
        candidates = self._registry.find_for_device(criteria.device.device_class)
        if not candidates:
            # Fallback to default
            default = self._registry.get_default()
            if default:
                return SelectionResult(
                    avatar=default,
                    confidence=0.3,
                    reason="no_device_match_fallback",
                )
            raise ValueError(f"No avatars available for {criteria.device.device_class}")

        # Score each candidate
        scored = []
        for avatar in candidates:
            score, reasons = self._score_avatar(avatar, criteria)
            scored.append((avatar, score, reasons))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        best_avatar, best_score, best_reasons = scored[0]
        alternatives = [a.avatar_id for a, _, _ in scored[1:4]]

        # Record selection for learning
        self._record_selection(criteria, best_avatar.avatar_id)

        return SelectionResult(
            avatar=best_avatar,
            confidence=min(1.0, best_score / 100.0),
            reason="; ".join(best_reasons),
            alternatives=alternatives,
        )

    def _score_avatar(
        self,
        avatar: AvatarDefinition,
        criteria: SelectionCriteria,
    ) -> tuple[float, List[str]]:
        """Score an avatar against criteria. Returns (score, reasons)."""
        score = 50.0  # Base score
        reasons = []

        # Required capabilities check
        for cap in criteria.required_capabilities:
            if not avatar.has_capability(cap):
                return 0.0, ["missing_required_capability"]

        # Excluded capabilities check
        for cap in criteria.excluded_capabilities:
            if avatar.has_capability(cap):
                return 0.0, ["has_excluded_capability"]

        # Style preference
        if criteria.preferred_style:
            if avatar.style == criteria.preferred_style:
                score += 15
                reasons.append("style_match")
            else:
                score -= 5

        # Formality match
        formality_diff = abs(avatar.formality - criteria.formality)
        score += (1 - formality_diff) * 10
        if formality_diff < 0.2:
            reasons.append("formality_match")

        # Energy match
        energy_diff = abs(avatar.energy_default - criteria.energy)
        score += (1 - energy_diff) * 10
        if energy_diff < 0.2:
            reasons.append("energy_match")

        # Tag bonuses
        for tag in criteria.preferred_tags:
            if tag in avatar.tags:
                score += 5
                reasons.append(f"tag:{tag}")

        # Context-specific bonuses
        if criteria.context_type == ContextType.WORK:
            if avatar.formality > 0.5:
                score += 10
                reasons.append("work_formal")
        elif criteria.context_type == ContextType.REST:
            if "low-stim" in avatar.tags or "minimal" in avatar.tags:
                score += 15
                reasons.append("rest_minimal")
        elif criteria.context_type == ContextType.CREATIVE:
            if avatar.gestures.expressiveness > 0.5:
                score += 10
                reasons.append("creative_expressive")

        # Resource constraints
        if criteria.low_power_mode:
            if avatar.gpu_required:
                score -= 20
            if avatar.min_fps < 30:
                score += 10
                reasons.append("low_power_friendly")

        # Battery considerations
        if criteria.device.battery_level < 0.2:
            if avatar.gpu_required:
                score -= 15

        # User preference from history
        with self._lock:
            if criteria.device.device_class in self._device_preferences:
                if self._device_preferences[criteria.device.device_class] == avatar.avatar_id:
                    score += 20
                    reasons.append("user_device_preference")

            if criteria.context_type in self._context_preferences:
                if self._context_preferences[criteria.context_type] == avatar.avatar_id:
                    score += 15
                    reasons.append("user_context_preference")

        return score, reasons

    def _record_selection(self, criteria: SelectionCriteria, avatar_id: str) -> None:
        """Record a selection for preference learning."""
        with self._lock:
            self._selection_history.append({
                "timestamp": time.time(),
                "device": criteria.device.device_class.value,
                "context": criteria.context_type.value,
                "avatar_id": avatar_id,
            })

            # Keep history bounded
            if len(self._selection_history) > 1000:
                self._selection_history = self._selection_history[-500:]

    def set_device_preference(self, device: DeviceClass, avatar_id: str) -> None:
        """Set preferred avatar for a device class."""
        with self._lock:
            self._device_preferences[device] = avatar_id

    def set_context_preference(self, context: ContextType, avatar_id: str) -> None:
        """Set preferred avatar for a context type."""
        with self._lock:
            self._context_preferences[context] = avatar_id

    def add_override_rule(
        self,
        rule: Callable[[SelectionCriteria], Optional[str]],
    ) -> None:
        """Add a custom override rule. Returns avatar_id or None."""
        with self._lock:
            self._override_rules.append(rule)

    def clear_preferences(self) -> None:
        """Clear all learned preferences."""
        with self._lock:
            self._device_preferences.clear()
            self._context_preferences.clear()
            self._selection_history.clear()

    def learn_from_feedback(self, avatar_id: str, positive: bool) -> None:
        """Update preferences based on user feedback."""
        if not self._selection_history:
            return

        with self._lock:
            # Get most recent selection
            recent = self._selection_history[-1]

            if positive:
                # Reinforce this choice
                device = DeviceClass(recent["device"])
                context = ContextType(recent["context"])

                # Only set preference if it was the selected avatar
                if recent["avatar_id"] == avatar_id:
                    self._device_preferences[device] = avatar_id
                    self._context_preferences[context] = avatar_id

    def get_stats(self) -> Dict[str, Any]:
        """Get selector statistics."""
        with self._lock:
            return {
                "device_preferences": {
                    k.value: v for k, v in self._device_preferences.items()
                },
                "context_preferences": {
                    k.value: v for k, v in self._context_preferences.items()
                },
                "selection_history_size": len(self._selection_history),
                "override_rules": len(self._override_rules),
            }
