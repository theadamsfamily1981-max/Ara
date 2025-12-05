"""User Predictor - Anticipating user needs.

This module uses the user model to predict:
- What the user is likely to ask next
- What information they might need
- What tools they'll want to use
- When they'll be most active

The goal is proactive assistance - preparing for needs
before they're explicitly expressed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import Counter

from .model import UserModel, get_user_model

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A prediction about user behavior or needs."""

    prediction_type: str  # "next_action", "tool_needed", "topic_interest"
    description: str
    confidence: float  # 0-1

    # Details
    suggested_action: Optional[str] = None
    suggested_tools: List[str] = field(default_factory=list)
    relevant_context: Dict[str, Any] = field(default_factory=dict)

    # Timing
    predicted_for: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_type": self.prediction_type,
            "description": self.description,
            "confidence": round(self.confidence, 2),
            "suggested_action": self.suggested_action,
            "suggested_tools": self.suggested_tools,
            "relevant_context": self.relevant_context,
            "predicted_for": self.predicted_for.isoformat() if self.predicted_for else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }


@dataclass
class Suggestion:
    """A proactive suggestion for the user."""

    suggestion_type: str  # "shortcut", "reminder", "recommendation"
    message: str
    priority: str  # "low", "medium", "high"

    # Action
    action_label: Optional[str] = None
    action_command: Optional[str] = None

    # Context
    reason: str = ""
    based_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion_type": self.suggestion_type,
            "message": self.message,
            "priority": self.priority,
            "action_label": self.action_label,
            "action_command": self.action_command,
            "reason": self.reason,
            "based_on": self.based_on,
        }


class UserPredictor:
    """Predicts user needs and behaviors."""

    def __init__(self, user_model: Optional[UserModel] = None):
        """Initialize the predictor.

        Args:
            user_model: User model to use
        """
        self._model = user_model or get_user_model()

        # Recent request tracking
        self._recent_requests: List[str] = []
        self._recent_tools: List[str] = []
        self._recent_domains: List[str] = []

    def record_request(
        self,
        request_text: str,
        tools_used: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> None:
        """Record a user request for prediction learning.

        Args:
            request_text: The request text
            tools_used: Tools that were used
            domain: Domain of the request
        """
        self._recent_requests.append(request_text)
        if len(self._recent_requests) > 100:
            self._recent_requests = self._recent_requests[-100:]

        if tools_used:
            self._recent_tools.extend(tools_used)
            if len(self._recent_tools) > 100:
                self._recent_tools = self._recent_tools[-100:]

        if domain:
            self._recent_domains.append(domain)
            if len(self._recent_domains) > 100:
                self._recent_domains = self._recent_domains[-100:]

    def predict_next_action(self) -> Prediction:
        """Predict what the user is likely to do next.

        Returns:
            Prediction about next action
        """
        # Analyze recent patterns
        if len(self._recent_requests) < 3:
            return Prediction(
                prediction_type="next_action",
                description="Not enough data to predict",
                confidence=0.1,
            )

        # Look for common patterns
        profile = self._model.get_profile()
        patterns = profile.work_patterns

        # Find most common recent domain
        if self._recent_domains:
            domain_counts = Counter(self._recent_domains[-10:])
            likely_domain = domain_counts.most_common(1)[0][0]

            return Prediction(
                prediction_type="next_action",
                description=f"Likely to continue working on {likely_domain}",
                confidence=min(domain_counts[likely_domain] / 10, 0.8),
                relevant_context={"domain": likely_domain},
            )

        return Prediction(
            prediction_type="next_action",
            description="No clear pattern detected",
            confidence=0.2,
        )

    def predict_tools_needed(self) -> Prediction:
        """Predict what tools the user is likely to need.

        Returns:
            Prediction about needed tools
        """
        if len(self._recent_tools) < 5:
            return Prediction(
                prediction_type="tool_needed",
                description="Not enough tool usage data",
                confidence=0.1,
                suggested_tools=[],
            )

        # Find most frequently used tools recently
        tool_counts = Counter(self._recent_tools[-20:])
        common_tools = [tool for tool, _ in tool_counts.most_common(3)]

        confidence = min(len(self._recent_tools) / 50, 0.7)

        return Prediction(
            prediction_type="tool_needed",
            description=f"Likely to use: {', '.join(common_tools)}",
            confidence=confidence,
            suggested_tools=common_tools,
        )

    def predict_active_times(self) -> Prediction:
        """Predict when the user will be active.

        Returns:
            Prediction about active times
        """
        patterns = self._model.detect_patterns()

        # Find daily pattern
        daily_pattern = None
        for pattern in patterns:
            if pattern.pattern_type == "daily":
                daily_pattern = pattern
                break

        if daily_pattern and daily_pattern.typical_hours:
            hours = daily_pattern.typical_hours
            return Prediction(
                prediction_type="active_times",
                description=f"Usually active during hours: {hours}",
                confidence=daily_pattern.confidence,
                relevant_context={"peak_hours": hours},
            )

        return Prediction(
            prediction_type="active_times",
            description="No clear activity pattern detected",
            confidence=0.2,
        )

    def generate_suggestions(self) -> List[Suggestion]:
        """Generate proactive suggestions for the user.

        Returns:
            List of suggestions
        """
        suggestions = []
        profile = self._model.get_profile()

        # Check expertise-based suggestions
        for area in profile.expertise_areas:
            if area.level == "novice" and area.interactions > 5:
                suggestions.append(Suggestion(
                    suggestion_type="recommendation",
                    message=f"Would you like tips for {area.domain}? You've asked several questions in this area.",
                    priority="low",
                    reason="Detected as learning area",
                    based_on=[f"{area.questions_asked} questions asked"],
                ))

        # Check for common tool shortcuts
        if self._recent_tools:
            tool_counts = Counter(self._recent_tools[-20:])
            top_tool = tool_counts.most_common(1)
            if top_tool and top_tool[0][1] >= 5:
                tool_name = top_tool[0][0]
                suggestions.append(Suggestion(
                    suggestion_type="shortcut",
                    message=f"You frequently use {tool_name}. Consider creating an alias.",
                    priority="low",
                    action_label=f"Create {tool_name} shortcut",
                    reason="Frequent tool usage detected",
                    based_on=[f"Used {top_tool[0][1]} times recently"],
                ))

        # Check communication preferences
        prefs = self._model.get_preferences_by_category("communication")
        if not prefs:
            suggestions.append(Suggestion(
                suggestion_type="recommendation",
                message="Would you like to set your communication preferences? (verbosity, formality, etc.)",
                priority="medium",
                action_label="Set preferences",
                reason="No communication preferences set",
            ))

        return suggestions

    def should_offer_help(self, context: str = "") -> bool:
        """Determine if Ara should proactively offer help.

        Args:
            context: Current context

        Returns:
            True if should offer help
        """
        profile = self._model.get_profile()

        # Don't offer help too frequently
        if profile.total_interactions < 3:
            return False

        # Check if user prefers explanations
        if profile.prefers_explanations:
            return True

        # Check expertise - offer help more to novices
        for area in profile.expertise_areas:
            if context.lower() in area.domain.lower():
                if area.level == "novice":
                    return True
                elif area.level == "expert":
                    return False

        return profile.prefers_explanations

    def get_response_style(self) -> Dict[str, Any]:
        """Get recommended response style for the user.

        Returns:
            Style recommendations
        """
        profile = self._model.get_profile()
        style = self._model.get_communication_style()

        return {
            "verbosity": style["verbosity"],
            "formality": style["formality"],
            "include_examples": profile.prefers_examples,
            "include_explanations": profile.prefers_explanations,
            "technical_level": self._estimate_technical_level(),
        }

    def _estimate_technical_level(self) -> str:
        """Estimate user's technical level."""
        profile = self._model.get_profile()

        expert_count = len([
            a for a in profile.expertise_areas
            if a.level == "expert"
        ])
        intermediate_count = len([
            a for a in profile.expertise_areas
            if a.level == "intermediate"
        ])

        if expert_count >= 2:
            return "advanced"
        elif intermediate_count >= 2 or expert_count >= 1:
            return "intermediate"
        else:
            return "beginner"

    def get_summary(self) -> Dict[str, Any]:
        """Get predictor summary."""
        return {
            "recent_requests": len(self._recent_requests),
            "recent_tools": len(self._recent_tools),
            "recent_domains": len(set(self._recent_domains)),
            "next_action": self.predict_next_action().to_dict(),
            "response_style": self.get_response_style(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_predictor: Optional[UserPredictor] = None


def get_user_predictor() -> UserPredictor:
    """Get the default user predictor."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = UserPredictor()
    return _default_predictor


def predict_next_action() -> Prediction:
    """Predict user's next action."""
    return get_user_predictor().predict_next_action()


def get_suggestions() -> List[Suggestion]:
    """Get proactive suggestions."""
    return get_user_predictor().generate_suggestions()


def get_response_style() -> Dict[str, Any]:
    """Get recommended response style."""
    return get_user_predictor().get_response_style()
