"""Shadow Profiles - Statistical profiles of each teacher's performance.

Tracks per-teacher, per-intent statistics:
- Success rate
- Average latency
- Response characteristics
- Failure modes

Example profile entry:
{
  "teacher": "claude",
  "intent": "debug_code",
  "features": {
    "input_len": 2048,
    "lang": "python",
    "has_stacktrace": true,
    "has_tests": false
  },
  "response_shape": {
    "uses_steps": true,
    "edits_code": true,
    "hallucination_risk": 0.12
  },
  "reward": 0.91,
  "latency_sec": 17.4
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TeacherFeatures:
    """Features extracted from an interaction for profiling."""

    input_len: int = 0
    language: str = ""  # "python", "rust", "javascript", etc.
    has_stacktrace: bool = False
    has_tests: bool = False
    has_code_blocks: int = 0
    complexity: str = "medium"  # "simple", "medium", "complex"
    domain: str = "general"  # "code", "hardware", "design", etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_len": self.input_len,
            "language": self.language,
            "has_stacktrace": self.has_stacktrace,
            "has_tests": self.has_tests,
            "has_code_blocks": self.has_code_blocks,
            "complexity": self.complexity,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherFeatures":
        return cls(
            input_len=data.get("input_len", 0),
            language=data.get("language", ""),
            has_stacktrace=data.get("has_stacktrace", False),
            has_tests=data.get("has_tests", False),
            has_code_blocks=data.get("has_code_blocks", 0),
            complexity=data.get("complexity", "medium"),
            domain=data.get("domain", "general"),
        )

    @classmethod
    def extract_from_query(cls, query: str) -> "TeacherFeatures":
        """Extract features from a query string."""
        features = cls()
        features.input_len = len(query)

        query_lower = query.lower()

        # Detect language
        lang_keywords = {
            "python": ["python", ".py", "pip", "pytest"],
            "rust": ["rust", "cargo", ".rs", "rustc"],
            "javascript": ["javascript", "node", ".js", "npm"],
            "typescript": ["typescript", ".ts", "tsc"],
            "go": [" go ", "golang", ".go"],
            "c++": ["c++", ".cpp", ".hpp"],
            "verilog": ["verilog", ".v", "fpga", "hdl"],
        }
        for lang, keywords in lang_keywords.items():
            if any(kw in query_lower for kw in keywords):
                features.language = lang
                break

        # Detect stacktrace
        features.has_stacktrace = any(x in query_lower for x in [
            "traceback", "error:", "exception", "at line", "stack trace"
        ])

        # Detect tests
        features.has_tests = any(x in query_lower for x in [
            "test", "assert", "expect", "should"
        ])

        # Count code blocks
        features.has_code_blocks = query.count("```")

        # Estimate complexity
        if features.input_len < 500:
            features.complexity = "simple"
        elif features.input_len < 2000:
            features.complexity = "medium"
        else:
            features.complexity = "complex"

        # Detect domain
        if any(x in query_lower for x in ["fpga", "verilog", "hardware", "circuit", "pcb"]):
            features.domain = "hardware"
        elif any(x in query_lower for x in ["architecture", "design", "system", "module"]):
            features.domain = "design"
        elif any(x in query_lower for x in ["code", "function", "class", "bug", "error"]):
            features.domain = "code"
        else:
            features.domain = "general"

        return features


@dataclass
class ResponseShape:
    """Characteristics of a teacher's typical response."""

    uses_steps: bool = False  # Does it use step-by-step structure?
    edits_code: bool = False  # Does it provide code edits?
    provides_explanation: bool = True
    avg_response_len: float = 0.0
    hallucination_risk: float = 0.1  # Estimated risk [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uses_steps": self.uses_steps,
            "edits_code": self.edits_code,
            "provides_explanation": self.provides_explanation,
            "avg_response_len": self.avg_response_len,
            "hallucination_risk": self.hallucination_risk,
        }


@dataclass
class ShadowProfile:
    """Statistical profile of a teacher's performance.

    Tracks success rate, latency, and response characteristics
    for a specific (teacher, intent) combination.
    """

    teacher: str
    intent: str

    # Performance stats
    sample_count: int = 0
    success_count: int = 0
    total_reward: float = 0.0
    total_latency_sec: float = 0.0

    # Response characteristics
    response_shape: ResponseShape = field(default_factory=ResponseShape)

    # Feature-specific stats
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        if self.sample_count == 0:
            return 0.5  # Prior
        return self.success_count / self.sample_count

    @property
    def avg_reward(self) -> float:
        if self.sample_count == 0:
            return 0.5  # Prior
        return self.total_reward / self.sample_count

    @property
    def avg_latency_sec(self) -> float:
        if self.sample_count == 0:
            return 15.0  # Prior
        return self.total_latency_sec / self.sample_count

    def update(
        self,
        reward: float,
        latency_sec: float,
        success: bool,
        features: Optional[TeacherFeatures] = None,
    ) -> None:
        """Update profile with a new observation."""
        self.sample_count += 1
        if success:
            self.success_count += 1
        self.total_reward += reward
        self.total_latency_sec += latency_sec
        self.last_updated = datetime.utcnow()

        # Update feature-specific stats
        if features:
            self._update_feature_stats(features, reward)

    def _update_feature_stats(self, features: TeacherFeatures, reward: float) -> None:
        """Update feature-conditional statistics."""
        # Track reward by complexity
        key = f"complexity:{features.complexity}"
        if key not in self.feature_stats:
            self.feature_stats[key] = {"count": 0, "total_reward": 0}
        self.feature_stats[key]["count"] += 1
        self.feature_stats[key]["total_reward"] += reward

        # Track reward by language
        if features.language:
            key = f"lang:{features.language}"
            if key not in self.feature_stats:
                self.feature_stats[key] = {"count": 0, "total_reward": 0}
            self.feature_stats[key]["count"] += 1
            self.feature_stats[key]["total_reward"] += reward

        # Track reward by domain
        key = f"domain:{features.domain}"
        if key not in self.feature_stats:
            self.feature_stats[key] = {"count": 0, "total_reward": 0}
        self.feature_stats[key]["count"] += 1
        self.feature_stats[key]["total_reward"] += reward

    def get_feature_reward(self, feature_key: str) -> Optional[float]:
        """Get average reward for a feature key."""
        if feature_key not in self.feature_stats:
            return None
        stats = self.feature_stats[feature_key]
        if stats["count"] == 0:
            return None
        return stats["total_reward"] / stats["count"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher": self.teacher,
            "intent": self.intent,
            "sample_count": self.sample_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "avg_reward": self.avg_reward,
            "avg_latency_sec": self.avg_latency_sec,
            "response_shape": self.response_shape.to_dict(),
            "feature_stats": self.feature_stats,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadowProfile":
        profile = cls(
            teacher=data["teacher"],
            intent=data["intent"],
            sample_count=data.get("sample_count", 0),
            success_count=data.get("success_count", 0),
            total_reward=data.get("avg_reward", 0.5) * data.get("sample_count", 0),
            total_latency_sec=data.get("avg_latency_sec", 15.0) * data.get("sample_count", 0),
            feature_stats=data.get("feature_stats", {}),
        )
        return profile


class ProfileManager:
    """Manages shadow profiles for all teachers.

    Profiles are stored as JSONL in ~/.ara/meta/shadow/profiles.jsonl
    """

    def __init__(self, profiles_path: Optional[Path] = None):
        """Initialize the profile manager.

        Args:
            profiles_path: Path to profiles JSONL file
        """
        self.profiles_path = profiles_path or (
            Path.home() / ".ara" / "meta" / "shadow" / "profiles.jsonl"
        )
        self.profiles_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory cache: {(teacher, intent): ShadowProfile}
        self._profiles: Dict[Tuple[str, str], ShadowProfile] = {}
        self._loaded = False

    def _load(self, force: bool = False) -> None:
        """Load profiles from disk."""
        if self._loaded and not force:
            return

        self._profiles.clear()

        if self.profiles_path.exists():
            try:
                with open(self.profiles_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            profile = ShadowProfile.from_dict(data)
                            key = (profile.teacher, profile.intent)
                            self._profiles[key] = profile
                        except Exception as e:
                            logger.warning(f"Failed to parse profile line: {e}")
            except Exception as e:
                logger.warning(f"Failed to load profiles: {e}")

        self._loaded = True
        logger.info(f"Loaded {len(self._profiles)} shadow profiles")

    def _save(self) -> None:
        """Save all profiles to disk."""
        with open(self.profiles_path, "w") as f:
            for profile in self._profiles.values():
                f.write(json.dumps(profile.to_dict(), default=str) + "\n")

    def get_profile(self, teacher: str, intent: str) -> ShadowProfile:
        """Get or create a profile for a teacher+intent.

        Args:
            teacher: Teacher name
            intent: Intent classification

        Returns:
            The shadow profile
        """
        self._load()
        key = (teacher, intent)

        if key not in self._profiles:
            self._profiles[key] = ShadowProfile(teacher=teacher, intent=intent)

        return self._profiles[key]

    def update_profile(
        self,
        teacher: str,
        intent: str,
        reward: float,
        latency_sec: float,
        success: bool,
        features: Optional[TeacherFeatures] = None,
    ) -> ShadowProfile:
        """Update a profile with a new observation.

        Args:
            teacher: Teacher name
            intent: Intent classification
            reward: Outcome quality [0, 1]
            latency_sec: Response latency
            success: Whether it succeeded
            features: Optional extracted features

        Returns:
            Updated profile
        """
        profile = self.get_profile(teacher, intent)
        profile.update(reward, latency_sec, success, features)
        self._save()
        return profile

    def get_all_profiles(self) -> List[ShadowProfile]:
        """Get all profiles."""
        self._load()
        return list(self._profiles.values())

    def get_profiles_for_teacher(self, teacher: str) -> List[ShadowProfile]:
        """Get all profiles for a teacher."""
        self._load()
        return [p for p in self._profiles.values() if p.teacher == teacher]

    def get_profiles_for_intent(self, intent: str) -> List[ShadowProfile]:
        """Get all profiles for an intent."""
        self._load()
        return [p for p in self._profiles.values() if p.intent == intent]

    def get_best_teacher_for_intent(self, intent: str) -> Optional[str]:
        """Get the best performing teacher for an intent.

        Args:
            intent: The intent

        Returns:
            Best teacher name or None
        """
        profiles = self.get_profiles_for_intent(intent)
        if not profiles:
            return None

        # Require minimum samples
        valid = [p for p in profiles if p.sample_count >= 3]
        if not valid:
            return None

        best = max(valid, key=lambda p: p.avg_reward)
        return best.teacher

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all profiles."""
        self._load()

        teachers = set(p.teacher for p in self._profiles.values())
        intents = set(p.intent for p in self._profiles.values())

        total_samples = sum(p.sample_count for p in self._profiles.values())

        return {
            "total_profiles": len(self._profiles),
            "total_samples": total_samples,
            "teachers": list(teachers),
            "intents": list(intents),
            "profiles": [p.to_dict() for p in self._profiles.values()],
        }


# =============================================================================
# Default Profiles (Priors)
# =============================================================================

DEFAULT_PROFILES = {
    ("claude", "debug_code"): {"avg_reward": 0.85, "avg_latency_sec": 15.0},
    ("claude", "implement"): {"avg_reward": 0.82, "avg_latency_sec": 20.0},
    ("claude", "refactor"): {"avg_reward": 0.80, "avg_latency_sec": 18.0},
    ("nova", "design_arch"): {"avg_reward": 0.80, "avg_latency_sec": 12.0},
    ("nova", "review"): {"avg_reward": 0.78, "avg_latency_sec": 10.0},
    ("gemini", "research"): {"avg_reward": 0.75, "avg_latency_sec": 8.0},
    ("gemini", "ideate"): {"avg_reward": 0.72, "avg_latency_sec": 10.0},
}


def seed_default_profiles(manager: ProfileManager) -> int:
    """Seed profiles with default priors.

    Args:
        manager: Profile manager

    Returns:
        Number of profiles seeded
    """
    seeded = 0
    for (teacher, intent), defaults in DEFAULT_PROFILES.items():
        profile = manager.get_profile(teacher, intent)
        if profile.sample_count == 0:
            # Set priors
            profile.sample_count = 5  # Pseudo-count
            profile.success_count = int(5 * defaults["avg_reward"])
            profile.total_reward = 5 * defaults["avg_reward"]
            profile.total_latency_sec = 5 * defaults["avg_latency_sec"]
            seeded += 1
    if seeded:
        manager._save()
    return seeded


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[ProfileManager] = None


def get_profile_manager(path: Optional[Path] = None) -> ProfileManager:
    """Get the default profile manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ProfileManager(profiles_path=path)
    return _default_manager


def update_profile(
    teacher: str,
    intent: str,
    reward: float,
    latency_sec: float,
    success: bool = True,
    features: Optional[TeacherFeatures] = None,
) -> ShadowProfile:
    """Update a teacher's profile.

    Args:
        teacher: Teacher name
        intent: Intent classification
        reward: Outcome quality
        latency_sec: Response latency
        success: Whether it succeeded
        features: Optional features

    Returns:
        Updated profile
    """
    return get_profile_manager().update_profile(
        teacher, intent, reward, latency_sec, success, features
    )


def get_teacher_profile(teacher: str, intent: str) -> ShadowProfile:
    """Get a teacher's profile."""
    return get_profile_manager().get_profile(teacher, intent)
