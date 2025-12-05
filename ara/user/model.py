"""User Model - Understanding and predicting user behavior.

This module builds and maintains a model of the user:
- Preferences and working patterns
- Common requests and workflows
- Communication style preferences
- Expertise areas and knowledge gaps

The goal is not surveillance but anticipation - helping Ara
provide better assistance by understanding context.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class UserPreference:
    """A learned user preference."""

    category: str  # "communication", "workflow", "tools", "timing"
    name: str
    value: Any

    # Confidence
    confidence: float = 0.5  # 0-1
    evidence_count: int = 0

    # Timestamps
    first_observed: datetime = field(default_factory=datetime.utcnow)
    last_observed: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "value": self.value,
            "confidence": round(self.confidence, 2),
            "evidence_count": self.evidence_count,
            "first_observed": self.first_observed.isoformat(),
            "last_observed": self.last_observed.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreference":
        return cls(
            category=data["category"],
            name=data["name"],
            value=data["value"],
            confidence=data.get("confidence", 0.5),
            evidence_count=data.get("evidence_count", 0),
        )


@dataclass
class WorkPattern:
    """A detected work pattern."""

    pattern_type: str  # "daily", "weekly", "project", "tool_usage"
    description: str

    # Timing
    typical_hours: List[int] = field(default_factory=list)  # 0-23
    typical_days: List[int] = field(default_factory=list)   # 0-6 (Mon-Sun)

    # Frequency
    occurrences: int = 0
    last_occurrence: Optional[datetime] = None

    # Confidence
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "typical_hours": self.typical_hours,
            "typical_days": self.typical_days,
            "occurrences": self.occurrences,
            "last_occurrence": self.last_occurrence.isoformat() if self.last_occurrence else None,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class ExpertiseArea:
    """An area of user expertise or interest."""

    domain: str
    level: str  # "novice", "intermediate", "expert"

    # Evidence
    interactions: int = 0
    successful_tasks: int = 0
    questions_asked: int = 0

    # Topics
    subtopics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "level": self.level,
            "interactions": self.interactions,
            "successful_tasks": self.successful_tasks,
            "questions_asked": self.questions_asked,
            "subtopics": self.subtopics,
        }


@dataclass
class UserProfile:
    """Complete user profile."""

    user_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Preferences
    preferences: List[UserPreference] = field(default_factory=list)

    # Patterns
    work_patterns: List[WorkPattern] = field(default_factory=list)

    # Expertise
    expertise_areas: List[ExpertiseArea] = field(default_factory=list)

    # Statistics
    total_sessions: int = 0
    total_interactions: int = 0
    avg_session_duration_min: float = 0.0

    # Communication style
    verbosity_preference: str = "medium"  # "terse", "medium", "verbose"
    formality_level: str = "casual"       # "formal", "casual", "technical"
    prefers_examples: bool = True
    prefers_explanations: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "preferences": [p.to_dict() for p in self.preferences],
            "work_patterns": [w.to_dict() for w in self.work_patterns],
            "expertise_areas": [e.to_dict() for e in self.expertise_areas],
            "total_sessions": self.total_sessions,
            "total_interactions": self.total_interactions,
            "avg_session_duration_min": round(self.avg_session_duration_min, 1),
            "verbosity_preference": self.verbosity_preference,
            "formality_level": self.formality_level,
            "prefers_examples": self.prefers_examples,
            "prefers_explanations": self.prefers_explanations,
        }


class UserModel:
    """Builds and maintains a model of the user."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the user model.

        Args:
            data_path: Path to user data
        """
        self.data_path = data_path or (
            Path.home() / ".ara" / "user"
        )
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._profile: Optional[UserProfile] = None
        self._loaded = False

        # Activity tracking
        self._hour_activity: Dict[int, int] = defaultdict(int)
        self._day_activity: Dict[int, int] = defaultdict(int)
        self._domain_activity: Dict[str, int] = defaultdict(int)

    def _load(self, force: bool = False) -> None:
        """Load user profile from disk."""
        if self._loaded and not force:
            return

        profile_file = self.data_path / "profile.json"
        if profile_file.exists():
            try:
                with open(profile_file) as f:
                    data = json.load(f)
                self._profile = UserProfile(
                    user_id=data.get("user_id", "default"),
                    total_sessions=data.get("total_sessions", 0),
                    total_interactions=data.get("total_interactions", 0),
                    verbosity_preference=data.get("verbosity_preference", "medium"),
                    formality_level=data.get("formality_level", "casual"),
                )
                # Load preferences
                for p_data in data.get("preferences", []):
                    self._profile.preferences.append(UserPreference.from_dict(p_data))
            except Exception as e:
                logger.warning(f"Failed to load user profile: {e}")

        if not self._profile:
            self._profile = UserProfile(user_id="default")

        self._loaded = True

    def _save(self) -> None:
        """Save user profile to disk."""
        if not self._profile:
            return

        self._profile.updated_at = datetime.utcnow()
        with open(self.data_path / "profile.json", "w") as f:
            json.dump(self._profile.to_dict(), f, indent=2)

    def get_profile(self) -> UserProfile:
        """Get the user profile."""
        self._load()
        return self._profile

    # =========================================================================
    # Preference Learning
    # =========================================================================

    def record_preference(
        self,
        category: str,
        name: str,
        value: Any,
        confidence_boost: float = 0.1,
    ) -> UserPreference:
        """Record a user preference observation.

        Args:
            category: Preference category
            name: Preference name
            value: Preference value
            confidence_boost: How much to increase confidence

        Returns:
            Updated preference
        """
        self._load()

        # Find existing preference
        existing = None
        for pref in self._profile.preferences:
            if pref.category == category and pref.name == name:
                existing = pref
                break

        if existing:
            # Update if same value, or reduce confidence if different
            if existing.value == value:
                existing.confidence = min(1.0, existing.confidence + confidence_boost)
                existing.evidence_count += 1
            else:
                existing.confidence *= 0.8  # Reduce confidence
                if existing.confidence < 0.3:
                    # Value has changed significantly
                    existing.value = value
                    existing.confidence = 0.5
            existing.last_observed = datetime.utcnow()
            pref = existing
        else:
            pref = UserPreference(
                category=category,
                name=name,
                value=value,
                confidence=0.5,
                evidence_count=1,
            )
            self._profile.preferences.append(pref)

        self._save()
        return pref

    def get_preference(
        self,
        category: str,
        name: str,
        default: Any = None,
    ) -> Any:
        """Get a user preference value.

        Args:
            category: Preference category
            name: Preference name
            default: Default value

        Returns:
            Preference value or default
        """
        self._load()

        for pref in self._profile.preferences:
            if pref.category == category and pref.name == name:
                if pref.confidence > 0.3:  # Only return if confident enough
                    return pref.value
        return default

    def get_preferences_by_category(self, category: str) -> List[UserPreference]:
        """Get all preferences in a category."""
        self._load()
        return [p for p in self._profile.preferences if p.category == category]

    # =========================================================================
    # Pattern Detection
    # =========================================================================

    def record_activity(
        self,
        domain: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record user activity for pattern detection.

        Args:
            domain: Activity domain
            timestamp: When the activity occurred
        """
        self._load()

        ts = timestamp or datetime.utcnow()
        hour = ts.hour
        day = ts.weekday()

        self._hour_activity[hour] += 1
        self._day_activity[day] += 1

        if domain:
            self._domain_activity[domain] += 1

        self._profile.total_interactions += 1
        self._save()

    def detect_patterns(self) -> List[WorkPattern]:
        """Analyze activity data and detect patterns.

        Returns:
            Detected patterns
        """
        patterns = []

        # Peak hours
        if self._hour_activity:
            total_activity = sum(self._hour_activity.values())
            if total_activity > 10:
                peak_hours = sorted(
                    self._hour_activity.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

                patterns.append(WorkPattern(
                    pattern_type="daily",
                    description=f"Most active hours: {[h for h, _ in peak_hours]}",
                    typical_hours=[h for h, _ in peak_hours],
                    occurrences=total_activity,
                    confidence=min(total_activity / 100, 0.9),
                ))

        # Peak days
        if self._day_activity:
            total_activity = sum(self._day_activity.values())
            if total_activity > 5:
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                peak_days = sorted(
                    self._day_activity.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

                patterns.append(WorkPattern(
                    pattern_type="weekly",
                    description=f"Most active days: {[day_names[d] for d, _ in peak_days]}",
                    typical_days=[d for d, _ in peak_days],
                    occurrences=total_activity,
                    confidence=min(total_activity / 50, 0.9),
                ))

        # Domain patterns
        if self._domain_activity:
            total = sum(self._domain_activity.values())
            for domain, count in self._domain_activity.items():
                if count >= 3:
                    patterns.append(WorkPattern(
                        pattern_type="tool_usage",
                        description=f"Frequent work in {domain}",
                        occurrences=count,
                        confidence=min(count / total * 2, 0.9),
                    ))

        self._profile.work_patterns = patterns
        self._save()
        return patterns

    # =========================================================================
    # Expertise Tracking
    # =========================================================================

    def record_expertise_signal(
        self,
        domain: str,
        signal_type: str,  # "success", "question", "interaction"
    ) -> ExpertiseArea:
        """Record a signal about user expertise.

        Args:
            domain: Domain area
            signal_type: Type of signal

        Returns:
            Updated expertise area
        """
        self._load()

        # Find or create expertise area
        existing = None
        for area in self._profile.expertise_areas:
            if area.domain == domain:
                existing = area
                break

        if not existing:
            existing = ExpertiseArea(domain=domain, level="novice")
            self._profile.expertise_areas.append(existing)

        # Update based on signal
        existing.interactions += 1
        if signal_type == "success":
            existing.successful_tasks += 1
        elif signal_type == "question":
            existing.questions_asked += 1

        # Update level estimate
        success_rate = (
            existing.successful_tasks / existing.interactions
            if existing.interactions > 0 else 0
        )
        question_rate = (
            existing.questions_asked / existing.interactions
            if existing.interactions > 0 else 0
        )

        if existing.interactions >= 10:
            if success_rate > 0.8 and question_rate < 0.2:
                existing.level = "expert"
            elif success_rate > 0.5 or existing.interactions >= 20:
                existing.level = "intermediate"

        self._save()
        return existing

    def get_expertise_level(self, domain: str) -> str:
        """Get user's expertise level in a domain.

        Args:
            domain: Domain to check

        Returns:
            Expertise level
        """
        self._load()

        for area in self._profile.expertise_areas:
            if area.domain == domain:
                return area.level
        return "unknown"

    # =========================================================================
    # Communication Style
    # =========================================================================

    def set_communication_style(
        self,
        verbosity: Optional[str] = None,
        formality: Optional[str] = None,
        prefers_examples: Optional[bool] = None,
        prefers_explanations: Optional[bool] = None,
    ) -> None:
        """Set communication style preferences.

        Args:
            verbosity: "terse", "medium", or "verbose"
            formality: "formal", "casual", or "technical"
            prefers_examples: Whether user prefers examples
            prefers_explanations: Whether user prefers explanations
        """
        self._load()

        if verbosity:
            self._profile.verbosity_preference = verbosity
        if formality:
            self._profile.formality_level = formality
        if prefers_examples is not None:
            self._profile.prefers_examples = prefers_examples
        if prefers_explanations is not None:
            self._profile.prefers_explanations = prefers_explanations

        self._save()

    def get_communication_style(self) -> Dict[str, Any]:
        """Get communication style preferences."""
        self._load()
        return {
            "verbosity": self._profile.verbosity_preference,
            "formality": self._profile.formality_level,
            "prefers_examples": self._profile.prefers_examples,
            "prefers_explanations": self._profile.prefers_explanations,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get user model summary."""
        self._load()
        return {
            "user_id": self._profile.user_id,
            "total_sessions": self._profile.total_sessions,
            "total_interactions": self._profile.total_interactions,
            "preferences_count": len(self._profile.preferences),
            "patterns_detected": len(self._profile.work_patterns),
            "expertise_areas": len(self._profile.expertise_areas),
            "communication_style": self.get_communication_style(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_model: Optional[UserModel] = None


def get_user_model() -> UserModel:
    """Get the default user model."""
    global _default_model
    if _default_model is None:
        _default_model = UserModel()
    return _default_model


def get_user_preference(category: str, name: str, default: Any = None) -> Any:
    """Get a user preference."""
    return get_user_model().get_preference(category, name, default)


def record_user_activity(domain: Optional[str] = None) -> None:
    """Record user activity."""
    get_user_model().record_activity(domain)


def get_user_expertise(domain: str) -> str:
    """Get user expertise level."""
    return get_user_model().get_expertise_level(domain)
