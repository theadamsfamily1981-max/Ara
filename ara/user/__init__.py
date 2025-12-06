"""Ara User Model - Understanding and predicting user behavior.

This module builds and maintains a model of the user:
- model: Core user profile and preferences
- predictor: Anticipating user needs
- mind_reader: Real-time user state estimation
- antipatterns: Friction detection and pattern mining

The goal is proactive, personalized assistance - not surveillance,
but anticipation based on observed patterns.
"""

from .model import (
    UserPreference,
    WorkPattern,
    ExpertiseArea,
    UserProfile,
    UserModel,
    get_user_model,
    get_user_preference,
    record_user_activity,
    get_user_expertise,
)

from .predictor import (
    Prediction,
    Suggestion,
    UserPredictor,
    get_user_predictor,
    predict_next_action,
    get_suggestions,
    get_response_style,
)

from .mind_reader import (
    UserState,
    MindReader,
    get_mind_reader,
    update_user_state,
    get_user_state,
)

from .antipatterns import (
    FrictionType,
    FrictionPoint,
    AntiPatternDetector,
    get_antipattern_detector,
    scan_for_friction,
)

__all__ = [
    # Model
    "UserPreference",
    "WorkPattern",
    "ExpertiseArea",
    "UserProfile",
    "UserModel",
    "get_user_model",
    "get_user_preference",
    "record_user_activity",
    "get_user_expertise",
    # Predictor
    "Prediction",
    "Suggestion",
    "UserPredictor",
    "get_user_predictor",
    "predict_next_action",
    "get_suggestions",
    "get_response_style",
    # MindReader (real-time state)
    "UserState",
    "MindReader",
    "get_mind_reader",
    "update_user_state",
    "get_user_state",
    # AntiPatterns (friction detection)
    "FrictionType",
    "FrictionPoint",
    "AntiPatternDetector",
    "get_antipattern_detector",
    "scan_for_friction",
]
