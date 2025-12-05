"""Disagreement Tracker - Curiosity hotspots from shadow model disagreement.

When shadow models strongly disagree on predictions, that's interesting:
- It means we're uncertain about this region
- Worth running real experiments to learn more
- These become research points in Ara's agenda

Example:
  If shadow-Claude predicts 0.9 success for a task, but shadow-Gemini
  predicts 0.4, that's a disagreement worth investigating.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .profiles import TeacherFeatures
from .predictor import ShadowPredictor, Prediction, get_predictor

logger = logging.getLogger(__name__)


@dataclass
class DisagreementRecord:
    """Record of a disagreement between shadow predictions."""

    id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Context
    intent: str = ""
    query_summary: str = ""
    features: Optional[Dict[str, Any]] = None

    # Disagreement details
    teacher_predictions: Dict[str, float] = field(default_factory=dict)  # teacher → expected_reward
    max_prediction: float = 0.0
    min_prediction: float = 0.0
    disagreement_score: float = 0.0  # max - min

    # The conflicting plans
    plan_a: str = ""  # e.g., "gemini→nova"
    plan_b: str = ""  # e.g., "claude→nova"
    plan_a_reward: float = 0.0
    plan_b_reward: float = 0.0

    # Status
    status: str = "open"  # "open", "resolved", "ignored"
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None

    # Experiment link
    experiment_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent,
            "query_summary": self.query_summary,
            "features": self.features,
            "teacher_predictions": self.teacher_predictions,
            "disagreement_score": round(self.disagreement_score, 3),
            "plan_a": self.plan_a,
            "plan_b": self.plan_b,
            "plan_a_reward": round(self.plan_a_reward, 3),
            "plan_b_reward": round(self.plan_b_reward, 3),
            "status": self.status,
            "resolution": self.resolution,
            "experiment_id": self.experiment_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisagreementRecord":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            intent=data.get("intent", ""),
            query_summary=data.get("query_summary", ""),
            features=data.get("features"),
            teacher_predictions=data.get("teacher_predictions", {}),
            max_prediction=data.get("max_prediction", 0.0),
            min_prediction=data.get("min_prediction", 0.0),
            disagreement_score=data.get("disagreement_score", 0.0),
            plan_a=data.get("plan_a", ""),
            plan_b=data.get("plan_b", ""),
            plan_a_reward=data.get("plan_a_reward", 0.0),
            plan_b_reward=data.get("plan_b_reward", 0.0),
            status=data.get("status", "open"),
            resolution=data.get("resolution"),
            experiment_id=data.get("experiment_id"),
        )


class DisagreementTracker:
    """Tracks disagreements between shadow predictions.

    When shadow models disagree strongly, these become curiosity hotspots
    that Ara can investigate through real experiments.
    """

    def __init__(
        self,
        predictor: Optional[ShadowPredictor] = None,
        log_path: Optional[Path] = None,
        disagreement_threshold: float = 0.25,
    ):
        """Initialize the tracker.

        Args:
            predictor: Shadow predictor to use
            log_path: Path to disagreement log
            disagreement_threshold: Minimum difference to flag
        """
        self.predictor = predictor or get_predictor()
        self.log_path = log_path or (
            Path.home() / ".ara" / "meta" / "shadow" / "disagreements.jsonl"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.disagreement_threshold = disagreement_threshold

        # In-memory cache
        self._records: Dict[str, DisagreementRecord] = {}
        self._loaded = False
        self._next_id = 1

    def _load(self, force: bool = False) -> None:
        """Load disagreement records from disk."""
        if self._loaded and not force:
            return

        self._records.clear()

        if self.log_path.exists():
            try:
                with open(self.log_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            record = DisagreementRecord.from_dict(data)
                            self._records[record.id] = record
                            # Update ID counter
                            if record.id.startswith("DIS-"):
                                try:
                                    num = int(record.id[4:])
                                    self._next_id = max(self._next_id, num + 1)
                                except ValueError:
                                    pass
                        except Exception as e:
                            logger.warning(f"Failed to parse disagreement: {e}")
            except Exception as e:
                logger.warning(f"Failed to load disagreements: {e}")

        self._loaded = True

    def _save(self) -> None:
        """Save all records to disk."""
        with open(self.log_path, "w") as f:
            for record in self._records.values():
                f.write(json.dumps(record.to_dict()) + "\n")

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        id_str = f"DIS-{self._next_id:04d}"
        self._next_id += 1
        return id_str

    def check_disagreement(
        self,
        intent: str,
        features: Optional[TeacherFeatures] = None,
        teachers: Optional[List[str]] = None,
    ) -> Optional[DisagreementRecord]:
        """Check for disagreement among shadow predictions.

        Args:
            intent: Intent classification
            features: Optional features
            teachers: Teachers to compare

        Returns:
            DisagreementRecord if disagreement found, None otherwise
        """
        if teachers is None:
            teachers = ["claude", "nova", "gemini"]

        # Get predictions
        predictions = {}
        for teacher in teachers:
            pred = self.predictor.predict(teacher, intent, features)
            predictions[teacher] = pred.expected_reward

        # Check for disagreement
        if len(predictions) < 2:
            return None

        max_pred = max(predictions.values())
        min_pred = min(predictions.values())
        diff = max_pred - min_pred

        if diff < self.disagreement_threshold:
            return None

        # Found disagreement - create record
        best_teacher = max(predictions, key=lambda t: predictions[t])
        worst_teacher = min(predictions, key=lambda t: predictions[t])

        record = DisagreementRecord(
            id=self._generate_id(),
            intent=intent,
            features=features.to_dict() if features else None,
            teacher_predictions=predictions,
            max_prediction=max_pred,
            min_prediction=min_pred,
            disagreement_score=diff,
            plan_a=f"{best_teacher}_only",
            plan_b=f"{worst_teacher}_only",
            plan_a_reward=max_pred,
            plan_b_reward=min_pred,
        )

        return record

    def track_disagreement(
        self,
        intent: str,
        query_summary: str = "",
        features: Optional[TeacherFeatures] = None,
        teachers: Optional[List[str]] = None,
    ) -> Optional[DisagreementRecord]:
        """Check and track a disagreement.

        Args:
            intent: Intent classification
            query_summary: Short summary of the query
            features: Optional features
            teachers: Teachers to compare

        Returns:
            DisagreementRecord if disagreement found and logged
        """
        self._load()

        record = self.check_disagreement(intent, features, teachers)
        if record:
            record.query_summary = query_summary
            self._records[record.id] = record
            self._save()
            logger.info(f"Logged disagreement: {record.id} (score={record.disagreement_score:.2f})")

        return record

    def get_open_disagreements(self) -> List[DisagreementRecord]:
        """Get all open (unresolved) disagreements."""
        self._load()
        return [r for r in self._records.values() if r.status == "open"]

    def get_by_intent(self, intent: str) -> List[DisagreementRecord]:
        """Get disagreements for an intent."""
        self._load()
        return [r for r in self._records.values() if r.intent == intent]

    def resolve_disagreement(
        self,
        record_id: str,
        resolution: str,
        experiment_id: Optional[str] = None,
    ) -> bool:
        """Mark a disagreement as resolved.

        Args:
            record_id: The record ID
            resolution: What was learned
            experiment_id: Optional linked experiment

        Returns:
            True if resolved
        """
        self._load()

        if record_id not in self._records:
            return False

        record = self._records[record_id]
        record.status = "resolved"
        record.resolution = resolution
        record.resolved_at = datetime.utcnow()
        record.experiment_id = experiment_id

        self._save()
        logger.info(f"Resolved disagreement: {record_id}")
        return True

    def ignore_disagreement(self, record_id: str, reason: str = "") -> bool:
        """Mark a disagreement as ignored.

        Args:
            record_id: The record ID
            reason: Why it's being ignored

        Returns:
            True if ignored
        """
        self._load()

        if record_id not in self._records:
            return False

        record = self._records[record_id]
        record.status = "ignored"
        record.resolution = reason or "Manually ignored"

        self._save()
        return True

    def get_curiosity_hotspots(
        self,
        min_score: float = 0.3,
        limit: int = 10,
    ) -> List[DisagreementRecord]:
        """Get high-value curiosity hotspots.

        Args:
            min_score: Minimum disagreement score
            limit: Maximum results

        Returns:
            Sorted list of open disagreements
        """
        self._load()

        hotspots = [
            r for r in self._records.values()
            if r.status == "open" and r.disagreement_score >= min_score
        ]

        # Sort by disagreement score
        hotspots.sort(key=lambda r: r.disagreement_score, reverse=True)

        return hotspots[:limit]

    def get_summary(self) -> Dict[str, Any]:
        """Get tracker summary."""
        self._load()

        open_count = len([r for r in self._records.values() if r.status == "open"])
        resolved_count = len([r for r in self._records.values() if r.status == "resolved"])

        # Average disagreement score
        scores = [r.disagreement_score for r in self._records.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Top intents with disagreements
        intent_counts: Dict[str, int] = {}
        for r in self._records.values():
            intent_counts[r.intent] = intent_counts.get(r.intent, 0) + 1

        return {
            "total_records": len(self._records),
            "open": open_count,
            "resolved": resolved_count,
            "avg_disagreement_score": round(avg_score, 3),
            "top_intents": sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_tracker: Optional[DisagreementTracker] = None


def get_disagreement_tracker() -> DisagreementTracker:
    """Get the default disagreement tracker."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = DisagreementTracker()
    return _default_tracker


def track_disagreement(
    intent: str,
    query_summary: str = "",
    features: Optional[TeacherFeatures] = None,
) -> Optional[DisagreementRecord]:
    """Track a potential disagreement.

    Args:
        intent: Intent classification
        query_summary: Short summary
        features: Optional features

    Returns:
        Record if disagreement found
    """
    return get_disagreement_tracker().track_disagreement(
        intent, query_summary, features
    )


def get_curiosity_hotspots(limit: int = 10) -> List[DisagreementRecord]:
    """Get curiosity hotspots.

    Args:
        limit: Maximum results

    Returns:
        Sorted list of open disagreements
    """
    return get_disagreement_tracker().get_curiosity_hotspots(limit=limit)
