"""Shadow Predictor - Predict outcomes before making real API calls.

Given (intent, features, teacher), predict:
- expected_reward: How good will the result be?
- expected_latency_sec: How long will it take?
- confidence: How confident are we in this prediction?

The predictor uses shadow profiles as a lightweight model.
Future: Could be a trained ML model (lightgbm, small neural net).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

from .profiles import (
    ProfileManager,
    ShadowProfile,
    TeacherFeatures,
    get_profile_manager,
)

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A prediction of what will happen if we call a teacher."""

    teacher: str
    intent: str

    # Predicted outcomes
    expected_reward: float  # [0, 1]
    expected_latency_sec: float
    success_probability: float  # [0, 1]

    # Confidence in prediction
    confidence: float  # [0, 1] - based on sample count
    sample_count: int  # How many observations this is based on

    # Feature-adjusted predictions
    feature_adjusted: bool = False
    adjustment_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher": self.teacher,
            "intent": self.intent,
            "expected_reward": round(self.expected_reward, 3),
            "expected_latency_sec": round(self.expected_latency_sec, 1),
            "success_probability": round(self.success_probability, 3),
            "confidence": round(self.confidence, 3),
            "sample_count": self.sample_count,
            "feature_adjusted": self.feature_adjusted,
            "adjustment_reason": self.adjustment_reason,
        }

    @property
    def expected_value(self) -> float:
        """Expected value = reward * success_probability."""
        return self.expected_reward * self.success_probability


class ShadowPredictor:
    """Predicts outcomes using shadow profiles.

    This is a simple statistical predictor that uses historical
    performance to estimate future outcomes.
    """

    def __init__(self, profile_manager: Optional[ProfileManager] = None):
        """Initialize the predictor.

        Args:
            profile_manager: Profile manager to use
        """
        self.profile_manager = profile_manager or get_profile_manager()

        # Priors for teachers with no data
        self.default_reward = 0.6
        self.default_latency = 15.0
        self.default_success = 0.7

        # Confidence scaling
        self.confidence_samples_full = 20  # Full confidence after this many samples

    def predict(
        self,
        teacher: str,
        intent: str,
        features: Optional[TeacherFeatures] = None,
    ) -> Prediction:
        """Predict outcome for a teacher+intent combination.

        Args:
            teacher: Teacher name
            intent: Intent classification
            features: Optional features for adjustment

        Returns:
            Prediction object
        """
        profile = self.profile_manager.get_profile(teacher, intent)

        # Base prediction from profile
        if profile.sample_count == 0:
            # No data - use defaults
            return Prediction(
                teacher=teacher,
                intent=intent,
                expected_reward=self.default_reward,
                expected_latency_sec=self.default_latency,
                success_probability=self.default_success,
                confidence=0.1,
                sample_count=0,
            )

        # Compute base prediction
        expected_reward = profile.avg_reward
        expected_latency = profile.avg_latency_sec
        success_prob = profile.success_rate

        # Confidence based on sample count
        confidence = min(1.0, profile.sample_count / self.confidence_samples_full)

        prediction = Prediction(
            teacher=teacher,
            intent=intent,
            expected_reward=expected_reward,
            expected_latency_sec=expected_latency,
            success_probability=success_prob,
            confidence=confidence,
            sample_count=profile.sample_count,
        )

        # Feature-based adjustment
        if features:
            prediction = self._adjust_for_features(prediction, profile, features)

        return prediction

    def _adjust_for_features(
        self,
        prediction: Prediction,
        profile: ShadowProfile,
        features: TeacherFeatures,
    ) -> Prediction:
        """Adjust prediction based on features."""
        adjustments = []

        # Adjust for complexity
        complexity_key = f"complexity:{features.complexity}"
        complexity_reward = profile.get_feature_reward(complexity_key)
        if complexity_reward is not None:
            delta = complexity_reward - prediction.expected_reward
            if abs(delta) > 0.05:
                prediction.expected_reward += delta * 0.5  # Partial adjustment
                adjustments.append(f"{features.complexity} complexity")

        # Adjust for language
        if features.language:
            lang_key = f"lang:{features.language}"
            lang_reward = profile.get_feature_reward(lang_key)
            if lang_reward is not None:
                delta = lang_reward - prediction.expected_reward
                if abs(delta) > 0.05:
                    prediction.expected_reward += delta * 0.5
                    adjustments.append(f"{features.language} language")

        # Adjust for domain
        domain_key = f"domain:{features.domain}"
        domain_reward = profile.get_feature_reward(domain_key)
        if domain_reward is not None:
            delta = domain_reward - prediction.expected_reward
            if abs(delta) > 0.05:
                prediction.expected_reward += delta * 0.5
                adjustments.append(f"{features.domain} domain")

        # Adjust latency for complexity
        if features.complexity == "complex":
            prediction.expected_latency_sec *= 1.3
            adjustments.append("complex → slower")
        elif features.complexity == "simple":
            prediction.expected_latency_sec *= 0.8
            adjustments.append("simple → faster")

        # Clamp reward
        prediction.expected_reward = max(0.0, min(1.0, prediction.expected_reward))

        if adjustments:
            prediction.feature_adjusted = True
            prediction.adjustment_reason = ", ".join(adjustments)

        return prediction

    def predict_workflow(
        self,
        teachers: List[str],
        intent: str,
        features: Optional[TeacherFeatures] = None,
    ) -> Dict[str, Any]:
        """Predict outcome for a multi-teacher workflow.

        Args:
            teachers: List of teachers in sequence
            intent: Intent classification
            features: Optional features

        Returns:
            Workflow prediction
        """
        if not teachers:
            return {
                "expected_reward": 0.0,
                "expected_latency_sec": 0.0,
                "confidence": 0.0,
                "teacher_predictions": [],
            }

        predictions = [
            self.predict(teacher, intent, features)
            for teacher in teachers
        ]

        # Combine predictions
        # Reward: assume each teacher can improve on the previous
        # Latency: additive
        # Confidence: geometric mean

        # Simple model: final reward is weighted average, with later teachers
        # having diminishing marginal contribution
        weights = [1.0]
        for i in range(1, len(predictions)):
            weights.append(0.3)  # Secondary teachers contribute 30% weight

        total_weight = sum(weights)
        combined_reward = sum(
            w * p.expected_reward for w, p in zip(weights, predictions)
        ) / total_weight

        # Add a small bonus for multi-teacher workflows (consensus benefit)
        if len(predictions) > 1:
            combined_reward = min(1.0, combined_reward * 1.05)

        total_latency = sum(p.expected_latency_sec for p in predictions)

        # Confidence is product of individual confidences
        combined_confidence = 1.0
        for p in predictions:
            combined_confidence *= p.confidence
        combined_confidence = combined_confidence ** (1 / len(predictions))  # Geometric mean

        return {
            "expected_reward": round(combined_reward, 3),
            "expected_latency_sec": round(total_latency, 1),
            "confidence": round(combined_confidence, 3),
            "teacher_predictions": [p.to_dict() for p in predictions],
        }

    def rank_teachers(
        self,
        intent: str,
        features: Optional[TeacherFeatures] = None,
        available_teachers: Optional[List[str]] = None,
    ) -> List[Tuple[str, Prediction]]:
        """Rank teachers by expected reward for an intent.

        Args:
            intent: Intent classification
            features: Optional features
            available_teachers: List of teachers to consider

        Returns:
            List of (teacher, prediction) sorted by expected reward
        """
        if available_teachers is None:
            available_teachers = ["claude", "nova", "gemini"]

        predictions = [
            (teacher, self.predict(teacher, intent, features))
            for teacher in available_teachers
        ]

        # Sort by expected value (reward * success probability)
        predictions.sort(key=lambda x: x[1].expected_value, reverse=True)

        return predictions

    def compare_workflows(
        self,
        workflows: List[List[str]],
        intent: str,
        features: Optional[TeacherFeatures] = None,
        alpha_latency: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """Compare multiple workflows and rank them.

        Args:
            workflows: List of teacher sequences
            intent: Intent classification
            features: Optional features
            alpha_latency: Penalty weight for latency

        Returns:
            Ranked list of workflow predictions
        """
        results = []

        for teachers in workflows:
            pred = self.predict_workflow(teachers, intent, features)

            # Compute utility: reward - alpha * latency
            utility = pred["expected_reward"] - alpha_latency * pred["expected_latency_sec"]

            results.append({
                "teachers": teachers,
                "utility": round(utility, 3),
                **pred,
            })

        # Sort by utility
        results.sort(key=lambda x: x["utility"], reverse=True)

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

_default_predictor: Optional[ShadowPredictor] = None


def get_predictor(profile_manager: Optional[ProfileManager] = None) -> ShadowPredictor:
    """Get the default predictor."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = ShadowPredictor(profile_manager=profile_manager)
    return _default_predictor


def predict_outcome(
    teacher: str,
    intent: str,
    features: Optional[TeacherFeatures] = None,
) -> Prediction:
    """Predict outcome for a teacher call.

    Args:
        teacher: Teacher name
        intent: Intent classification
        features: Optional features

    Returns:
        Prediction
    """
    return get_predictor().predict(teacher, intent, features)


def rank_teachers_for_intent(
    intent: str,
    features: Optional[TeacherFeatures] = None,
) -> List[Tuple[str, Prediction]]:
    """Rank teachers for an intent.

    Args:
        intent: Intent classification
        features: Optional features

    Returns:
        Ranked list of (teacher, prediction)
    """
    return get_predictor().rank_teachers(intent, features)
