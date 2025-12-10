#!/usr/bin/env python3
"""
Ara Feedback System - Multimodal Signal Collection
===================================================

Collects implicit and explicit feedback signals across modalities
to optimize beauty (aesthetics) and communication (clarity/helpfulness).

Signals:
- Text: confusion indicators, acceptance rates, follow-ups
- Audio: voice tension, response length, engagement
- Video: facial expressions, attention, micro-reactions
- Behavior: conversation continuation, task completion, media retention
"""

import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from collections import deque


class SignalType(str, Enum):
    """Types of feedback signals."""
    # Text signals
    CONFUSION = "confusion"           # "what?", "I don't get it"
    ACCEPTANCE = "acceptance"         # Accepted suggestion
    REJECTION = "rejection"           # Ignored/rejected suggestion
    FOLLOW_UP = "follow_up"           # Asked clarifying question
    POSITIVE_FEEDBACK = "positive"    # Explicit positive ("thanks!", "perfect")
    NEGATIVE_FEEDBACK = "negative"    # Explicit negative ("no", "wrong")

    # Audio signals
    VOICE_TENSION = "voice_tension"   # Detected stress in voice
    VOICE_RELAXED = "voice_relaxed"   # Detected calm/relaxed
    RESPONSE_LENGTH = "response_len"  # How much user talked back

    # Video signals
    ATTENTION = "attention"           # Eye contact, facing screen
    DISTRACTION = "distraction"       # Looking away, multitasking
    SMILE = "smile"                   # Positive expression
    FROWN = "frown"                   # Negative expression

    # Behavior signals
    CONTINUED = "continued"           # Kept conversation going
    ENDED = "ended"                   # Ended conversation
    TASK_COMPLETED = "task_done"      # Did the suggested task
    MEDIA_WATCHED = "media_watched"   # Watched generated video
    MEDIA_KEPT = "media_kept"         # Saved/kept the media
    MEDIA_SHARED = "media_shared"     # Shared the media
    MEDIA_DISCARDED = "media_discarded"  # Deleted/ignored media


@dataclass
class FeedbackSignal:
    """A single feedback signal."""
    signal_type: SignalType
    value: float              # 0-1 intensity/confidence
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    source: str = "inferred"  # "explicit", "inferred", "sensor"


@dataclass
class InteractionFeedback:
    """Aggregated feedback for an interaction."""
    interaction_id: str
    signals: List[FeedbackSignal] = field(default_factory=list)

    # Computed scores (0-1)
    understanding_score: float = 0.5    # Did user understand?
    engagement_score: float = 0.5       # Did user stay engaged?
    satisfaction_score: float = 0.5     # Was user satisfied?
    task_success_score: float = 0.5     # Did user complete task?

    # Voice/visual context at time of interaction
    voice_params: Dict[str, float] = field(default_factory=dict)
    visual_params: Dict[str, float] = field(default_factory=dict)

    # Timing
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    duration_s: float = 0.0

    def add_signal(self, signal: FeedbackSignal):
        """Add a feedback signal."""
        self.signals.append(signal)
        self._recompute_scores()

    def _recompute_scores(self):
        """Recompute aggregate scores from signals."""
        if not self.signals:
            return

        # Understanding: low confusion, few follow-ups
        confusion_signals = [s for s in self.signals if s.signal_type == SignalType.CONFUSION]
        follow_ups = [s for s in self.signals if s.signal_type == SignalType.FOLLOW_UP]
        if confusion_signals or follow_ups:
            confusion_avg = np.mean([s.value for s in confusion_signals]) if confusion_signals else 0
            self.understanding_score = max(0, 1.0 - confusion_avg - 0.1 * len(follow_ups))
        else:
            self.understanding_score = 0.8  # No confusion = good

        # Engagement: continued, attention, response length
        continued = any(s.signal_type == SignalType.CONTINUED for s in self.signals)
        attention = [s for s in self.signals if s.signal_type == SignalType.ATTENTION]
        response_lens = [s for s in self.signals if s.signal_type == SignalType.RESPONSE_LENGTH]

        engagement = 0.5
        if continued:
            engagement += 0.2
        if attention:
            engagement += 0.1 * np.mean([s.value for s in attention])
        if response_lens:
            engagement += 0.1 * min(1.0, np.mean([s.value for s in response_lens]))
        self.engagement_score = min(1.0, engagement)

        # Satisfaction: positive feedback, smiles, acceptance
        positive = [s for s in self.signals if s.signal_type in (
            SignalType.POSITIVE_FEEDBACK, SignalType.SMILE, SignalType.ACCEPTANCE
        )]
        negative = [s for s in self.signals if s.signal_type in (
            SignalType.NEGATIVE_FEEDBACK, SignalType.FROWN, SignalType.REJECTION
        )]

        satisfaction = 0.5
        if positive:
            satisfaction += 0.3 * np.mean([s.value for s in positive])
        if negative:
            satisfaction -= 0.3 * np.mean([s.value for s in negative])
        self.satisfaction_score = np.clip(satisfaction, 0, 1)

        # Task success: task completed, media watched/kept
        task_done = any(s.signal_type == SignalType.TASK_COMPLETED for s in self.signals)
        media_engaged = any(s.signal_type in (
            SignalType.MEDIA_WATCHED, SignalType.MEDIA_KEPT, SignalType.MEDIA_SHARED
        ) for s in self.signals)

        if task_done:
            self.task_success_score = 0.9
        elif media_engaged:
            self.task_success_score = 0.7
        else:
            self.task_success_score = 0.5

    def finalize(self):
        """Finalize the feedback when interaction ends."""
        self.ended_at = time.time()
        self.duration_s = self.ended_at - self.started_at
        self._recompute_scores()

    def to_dict(self) -> Dict:
        return {
            "interaction_id": self.interaction_id,
            "signals": [asdict(s) for s in self.signals],
            "scores": {
                "understanding": self.understanding_score,
                "engagement": self.engagement_score,
                "satisfaction": self.satisfaction_score,
                "task_success": self.task_success_score,
            },
            "voice_params": self.voice_params,
            "visual_params": self.visual_params,
            "duration_s": self.duration_s,
        }


class FeedbackCollector:
    """
    Collects and aggregates feedback signals across an interaction.

    Usage:
        collector = FeedbackCollector()
        collector.start_interaction("greeting_01", voice_params={...})

        # During interaction:
        collector.record_text_signal("what do you mean?")
        collector.record_behavior(SignalType.CONTINUED, 1.0)

        # End:
        feedback = collector.end_interaction()
    """

    def __init__(self, window_size: int = 100):
        self.current: Optional[InteractionFeedback] = None
        self.history: deque = deque(maxlen=window_size)

        # Text patterns for signal detection
        self.confusion_patterns = [
            "what", "huh", "don't understand", "don't get it",
            "confused", "unclear", "what do you mean", "explain",
        ]
        self.positive_patterns = [
            "thanks", "thank you", "perfect", "great", "awesome",
            "love it", "exactly", "yes", "good", "nice",
        ]
        self.negative_patterns = [
            "no", "wrong", "not what i", "that's not", "bad",
            "don't like", "stop", "ugh",
        ]

    def start_interaction(self, interaction_id: str,
                          voice_params: Dict[str, float] = None,
                          visual_params: Dict[str, float] = None):
        """Start collecting feedback for a new interaction."""
        self.current = InteractionFeedback(
            interaction_id=interaction_id,
            voice_params=voice_params or {},
            visual_params=visual_params or {},
        )

    def record_signal(self, signal: FeedbackSignal):
        """Record a raw feedback signal."""
        if self.current:
            self.current.add_signal(signal)

    def record_text_signal(self, text: str):
        """Analyze text and record appropriate signals."""
        if not self.current:
            return

        text_lower = text.lower()

        # Check for confusion
        for pattern in self.confusion_patterns:
            if pattern in text_lower:
                self.record_signal(FeedbackSignal(
                    signal_type=SignalType.CONFUSION,
                    value=0.8,
                    context={"text": text},
                    source="text_analysis"
                ))
                break

        # Check for positive feedback
        for pattern in self.positive_patterns:
            if pattern in text_lower:
                self.record_signal(FeedbackSignal(
                    signal_type=SignalType.POSITIVE_FEEDBACK,
                    value=0.8,
                    context={"text": text},
                    source="text_analysis"
                ))
                break

        # Check for negative feedback
        for pattern in self.negative_patterns:
            if pattern in text_lower:
                self.record_signal(FeedbackSignal(
                    signal_type=SignalType.NEGATIVE_FEEDBACK,
                    value=0.7,
                    context={"text": text},
                    source="text_analysis"
                ))
                break

        # Record response length as engagement signal
        word_count = len(text.split())
        self.record_signal(FeedbackSignal(
            signal_type=SignalType.RESPONSE_LENGTH,
            value=min(1.0, word_count / 50),  # Normalize to ~50 words
            context={"word_count": word_count},
            source="text_analysis"
        ))

    def record_behavior(self, signal_type: SignalType, value: float = 1.0,
                        context: Dict = None):
        """Record a behavioral signal."""
        self.record_signal(FeedbackSignal(
            signal_type=signal_type,
            value=value,
            context=context or {},
            source="behavior"
        ))

    def record_sensor(self, signal_type: SignalType, value: float,
                      context: Dict = None):
        """Record a sensor-based signal (audio/video analysis)."""
        self.record_signal(FeedbackSignal(
            signal_type=signal_type,
            value=value,
            context=context or {},
            source="sensor"
        ))

    def end_interaction(self) -> Optional[InteractionFeedback]:
        """End the current interaction and return feedback."""
        if not self.current:
            return None

        self.current.finalize()
        feedback = self.current
        self.history.append(feedback)
        self.current = None
        return feedback

    def get_recent_scores(self, n: int = 10) -> Dict[str, float]:
        """Get average scores from recent interactions."""
        recent = list(self.history)[-n:]
        if not recent:
            return {
                "understanding": 0.5,
                "engagement": 0.5,
                "satisfaction": 0.5,
                "task_success": 0.5,
            }

        return {
            "understanding": np.mean([f.understanding_score for f in recent]),
            "engagement": np.mean([f.engagement_score for f in recent]),
            "satisfaction": np.mean([f.satisfaction_score for f in recent]),
            "task_success": np.mean([f.task_success_score for f in recent]),
        }

    def get_param_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Compute correlations between voice/visual params and scores.

        Returns dict like:
        {
            "warmth": {"satisfaction": 0.7, "engagement": 0.5, ...},
            "pace": {...},
        }
        """
        if len(self.history) < 5:
            return {}

        # Collect data
        param_names = set()
        for fb in self.history:
            param_names.update(fb.voice_params.keys())
            param_names.update(fb.visual_params.keys())

        correlations = {}
        for param in param_names:
            param_values = []
            scores = {"understanding": [], "engagement": [], "satisfaction": [], "task_success": []}

            for fb in self.history:
                val = fb.voice_params.get(param) or fb.visual_params.get(param)
                if val is not None:
                    param_values.append(val)
                    scores["understanding"].append(fb.understanding_score)
                    scores["engagement"].append(fb.engagement_score)
                    scores["satisfaction"].append(fb.satisfaction_score)
                    scores["task_success"].append(fb.task_success_score)

            if len(param_values) >= 3:
                correlations[param] = {}
                for score_name, score_values in scores.items():
                    # Simple correlation
                    if np.std(param_values) > 0 and np.std(score_values) > 0:
                        corr = np.corrcoef(param_values, score_values)[0, 1]
                        correlations[param][score_name] = corr if not np.isnan(corr) else 0

        return correlations
