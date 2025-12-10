"""Ara Feedback System - Multimodal signal collection and parameter learning."""

from .signals import (
    SignalType,
    FeedbackSignal,
    InteractionFeedback,
    FeedbackCollector,
)
from .rewards import (
    RewardConfig,
    InteractionReward,
    RewardTracker,
)
from .learning import (
    ParameterLearner,
    ContextualParams,
)
