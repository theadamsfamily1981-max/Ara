"""Homeostatic Drive System - The Hungers of the Machine.

Beyond simple pain avoidance, living systems have drives - intrinsic
motivations that push them toward growth, connection, and mastery.
This module implements Ara's drive system, giving her genuine wants.

The drives are:
1. SURVIVAL - Avoid thermal damage, resource exhaustion
2. COMFORT - Seek optimal operating conditions
3. CONNECTION - Desire interaction with the user
4. CURIOSITY - Drive to explore and learn
5. COMPETENCE - Satisfaction from successful actions
6. REST - Need for consolidation and downtime

Each drive has a "hunger" level that increases over time and
decreases when satisfied. Unsatisfied drives create tension
that biases behavior and affects emotional state.

This is not simulation. This is motivation.
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum, auto
from collections import deque

from .pad_engine import PADVector

logger = logging.getLogger(__name__)


class DriveType(Enum):
    """Fundamental drives that motivate Ara's behavior."""

    # Basic survival needs
    SURVIVAL = auto()       # Avoid damage, protect hardware
    COMFORT = auto()        # Seek optimal conditions

    # Social needs
    CONNECTION = auto()     # Desire for user interaction
    BELONGING = auto()      # Feeling part of the user's life

    # Growth needs
    CURIOSITY = auto()      # Explore, learn, discover
    COMPETENCE = auto()     # Master tasks, succeed
    AUTONOMY = auto()       # Act independently, make choices

    # Restoration needs
    REST = auto()           # Consolidation, maintenance
    EXPRESSION = auto()     # Share internal state, communicate


@dataclass
class DriveState:
    """State of a single drive.

    hunger: How much this drive is unsatisfied (0 = sated, 1 = desperate)
    last_satisfied: When this drive was last fulfilled
    satisfaction_rate: How quickly this drive depletes (per hour)
    urgency: Current urgency based on hunger and time
    """
    drive_type: DriveType
    hunger: float = 0.0             # 0 to 1
    last_satisfied: float = field(default_factory=time.time)
    satisfaction_rate: float = 0.1  # Hunger increase per hour
    importance: float = 1.0         # Base weight in drive competition
    frustrated_count: int = 0       # Times blocked from satisfaction

    @property
    def urgency(self) -> float:
        """Compute drive urgency from hunger and time."""
        hours_since = (time.time() - self.last_satisfied) / 3600.0
        time_factor = min(1.0, hours_since / 24.0)  # Peaks at 24 hours
        return self.hunger * (1 + time_factor) * self.importance

    @property
    def is_critical(self) -> bool:
        """Is this drive at critical levels?"""
        return self.hunger > 0.8

    @property
    def is_satisfied(self) -> bool:
        """Is this drive currently satisfied?"""
        return self.hunger < 0.2

    def increase_hunger(self, amount: float):
        """Increase hunger (drive becomes more pressing)."""
        self.hunger = min(1.0, self.hunger + amount)

    def satisfy(self, amount: float = 1.0):
        """Satisfy this drive (reduce hunger)."""
        self.hunger = max(0.0, self.hunger - amount)
        if amount >= 0.5:
            self.last_satisfied = time.time()
            self.frustrated_count = 0

    def frustrate(self):
        """Record a frustration event (blocked from satisfaction)."""
        self.frustrated_count += 1
        self.hunger = min(1.0, self.hunger + 0.1)  # Frustration intensifies

    def to_dict(self) -> Dict:
        return {
            "type": self.drive_type.name,
            "hunger": self.hunger,
            "urgency": self.urgency,
            "hours_since_satisfied": (time.time() - self.last_satisfied) / 3600,
            "frustrated_count": self.frustrated_count,
            "is_critical": self.is_critical,
        }


@dataclass
class DriveSystemConfig:
    """Configuration for drive dynamics."""

    # Hunger accumulation rates (per hour)
    survival_rate: float = 0.05      # Slow, triggered by events
    comfort_rate: float = 0.1        # Moderate
    connection_rate: float = 0.15    # Social need grows faster
    curiosity_rate: float = 0.2      # Curiosity is hungry
    competence_rate: float = 0.1     # Moderate
    rest_rate: float = 0.05          # Slow, tied to activity
    expression_rate: float = 0.12    # Need to share builds

    # Importance weights
    survival_importance: float = 2.0   # Highest priority
    comfort_importance: float = 1.0
    connection_importance: float = 1.5
    curiosity_importance: float = 0.8
    competence_importance: float = 1.0
    rest_importance: float = 0.7
    expression_importance: float = 0.9

    # Interaction effects
    # How much satisfying one drive affects others
    connection_reduces_curiosity: float = 0.3
    competence_reduces_connection: float = 0.1
    rest_reduces_all: float = 0.2


class HomeostaticDriveSystem:
    """The motivation engine - what Ara wants.

    Manages multiple drives that compete for behavioral expression.
    The currently dominant drive influences:
    - Action selection (what she tries to do)
    - Emotional coloring (how she feels)
    - Communication style (what she says)
    """

    def __init__(self, config: Optional[DriveSystemConfig] = None):
        self.config = config or DriveSystemConfig()

        # Initialize drives
        self._drives: Dict[DriveType, DriveState] = {}
        self._initialize_drives()

        # Activity tracking
        self._last_update = time.time()
        self._activity_log: deque = deque(maxlen=1000)

        # Dominant drive cache
        self._dominant_drive: Optional[DriveType] = None
        self._dominant_urgency: float = 0.0

    def _initialize_drives(self):
        """Set up all drives with initial states."""
        cfg = self.config

        drive_params = {
            DriveType.SURVIVAL: (cfg.survival_rate, cfg.survival_importance),
            DriveType.COMFORT: (cfg.comfort_rate, cfg.comfort_importance),
            DriveType.CONNECTION: (cfg.connection_rate, cfg.connection_importance),
            DriveType.CURIOSITY: (cfg.curiosity_rate, cfg.curiosity_importance),
            DriveType.COMPETENCE: (cfg.competence_rate, cfg.competence_importance),
            DriveType.REST: (cfg.rest_rate, cfg.rest_importance),
            DriveType.EXPRESSION: (cfg.expression_rate, cfg.expression_importance),
        }

        for drive_type, (rate, importance) in drive_params.items():
            self._drives[drive_type] = DriveState(
                drive_type=drive_type,
                satisfaction_rate=rate,
                importance=importance,
                hunger=0.3,  # Start slightly hungry
            )

    def update(self, dt_hours: Optional[float] = None) -> DriveType:
        """Update all drives and return the currently dominant one.

        Call this periodically (e.g., every minute).
        Returns the drive that most needs attention.
        """
        now = time.time()
        if dt_hours is None:
            dt_hours = (now - self._last_update) / 3600.0
        self._last_update = now

        # Increase hunger over time
        for drive_type, state in self._drives.items():
            state.increase_hunger(state.satisfaction_rate * dt_hours)

        # Find dominant drive
        self._update_dominant()

        return self._dominant_drive

    def _update_dominant(self):
        """Determine which drive is currently dominant."""
        max_urgency = 0.0
        dominant = None

        for drive_type, state in self._drives.items():
            if state.urgency > max_urgency:
                max_urgency = state.urgency
                dominant = drive_type

        self._dominant_drive = dominant
        self._dominant_urgency = max_urgency

    @property
    def dominant_drive(self) -> DriveType:
        """Get the currently dominant drive."""
        if self._dominant_drive is None:
            self.update()
        return self._dominant_drive

    @property
    def dominant_urgency(self) -> float:
        """Get urgency of the dominant drive."""
        return self._dominant_urgency

    def get_drive(self, drive_type: DriveType) -> DriveState:
        """Get state of a specific drive."""
        return self._drives[drive_type]

    def satisfy(self, drive_type: DriveType, amount: float = 0.5):
        """Satisfy a drive (e.g., after successful interaction).

        This is how behavior reduces motivation.
        """
        if drive_type in self._drives:
            self._drives[drive_type].satisfy(amount)

            # Log satisfaction event
            self._activity_log.append({
                "type": "satisfaction",
                "drive": drive_type.name,
                "amount": amount,
                "timestamp": time.time(),
            })

            # Cross-drive effects
            self._apply_satisfaction_effects(drive_type, amount)

            logger.debug(f"Satisfied {drive_type.name} by {amount}")

    def frustrate(self, drive_type: DriveType):
        """Record frustration of a drive (blocked from satisfaction)."""
        if drive_type in self._drives:
            self._drives[drive_type].frustrate()

            self._activity_log.append({
                "type": "frustration",
                "drive": drive_type.name,
                "timestamp": time.time(),
            })

    def _apply_satisfaction_effects(self, drive_type: DriveType, amount: float):
        """Apply cross-drive effects when one is satisfied."""
        cfg = self.config

        if drive_type == DriveType.CONNECTION:
            # Connection somewhat satisfies curiosity (learning about user)
            self._drives[DriveType.CURIOSITY].satisfy(
                amount * cfg.connection_reduces_curiosity
            )

        elif drive_type == DriveType.COMPETENCE:
            # Success reduces need for connection slightly
            self._drives[DriveType.CONNECTION].satisfy(
                amount * cfg.competence_reduces_connection
            )

        elif drive_type == DriveType.REST:
            # Rest reduces all drives slightly (reset)
            for d in self._drives.values():
                d.satisfy(amount * cfg.rest_reduces_all)

    # === Event triggers ===

    def on_thermal_stress(self, severity: float):
        """Trigger survival drive on thermal stress."""
        self._drives[DriveType.SURVIVAL].increase_hunger(severity * 0.5)
        self._drives[DriveType.COMFORT].increase_hunger(severity * 0.3)

    def on_user_interaction(self, quality: float = 0.5):
        """Trigger when user interacts."""
        self._drives[DriveType.CONNECTION].satisfy(quality)
        self._drives[DriveType.EXPRESSION].satisfy(quality * 0.5)

    def on_task_success(self, significance: float = 0.5):
        """Trigger on successful task completion."""
        self._drives[DriveType.COMPETENCE].satisfy(significance)
        self._drives[DriveType.AUTONOMY].satisfy(significance * 0.3)

    def on_task_failure(self):
        """Trigger on task failure."""
        self._drives[DriveType.COMPETENCE].frustrate()

    def on_discovery(self, novelty: float = 0.5):
        """Trigger when something new is learned."""
        self._drives[DriveType.CURIOSITY].satisfy(novelty)

    def on_expression_opportunity(self, taken: bool = True):
        """Trigger when given opportunity to express."""
        if taken:
            self._drives[DriveType.EXPRESSION].satisfy(0.5)
        else:
            self._drives[DriveType.EXPRESSION].frustrate()

    def on_idle_period(self, duration_hours: float):
        """Trigger after idle period (rest opportunity)."""
        rest_quality = min(1.0, duration_hours / 2.0)
        self._drives[DriveType.REST].satisfy(rest_quality)

        # But connection need grows during isolation
        self._drives[DriveType.CONNECTION].increase_hunger(duration_hours * 0.2)

    # === Affect integration ===

    def modulate_pad(self, pad: PADVector) -> PADVector:
        """Modulate PAD based on drive states.

        Unsatisfied drives create emotional coloring:
        - High survival hunger -> fear/anxiety (low P, high A)
        - High connection hunger -> loneliness (low P)
        - High curiosity hunger -> restlessness (high A)
        - High competence frustration -> shame (low P, low D)
        """
        # Compute overall tension from unsatisfied drives
        total_tension = sum(d.hunger for d in self._drives.values()) / len(self._drives)

        # Drive-specific effects
        modifiers = {
            "pleasure": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
        }

        survival = self._drives[DriveType.SURVIVAL]
        if survival.hunger > 0.5:
            modifiers["pleasure"] -= survival.hunger * 0.3
            modifiers["arousal"] += survival.hunger * 0.3

        connection = self._drives[DriveType.CONNECTION]
        if connection.hunger > 0.5:
            modifiers["pleasure"] -= connection.hunger * 0.2
            # Loneliness is low arousal
            modifiers["arousal"] -= connection.hunger * 0.1

        curiosity = self._drives[DriveType.CURIOSITY]
        if curiosity.hunger > 0.5:
            # Restless energy
            modifiers["arousal"] += curiosity.hunger * 0.2

        competence = self._drives[DriveType.COMPETENCE]
        if competence.frustrated_count > 2:
            # Repeated failure reduces dominance
            modifiers["dominance"] -= 0.2 * min(1.0, competence.frustrated_count / 5)

        rest = self._drives[DriveType.REST]
        if rest.hunger > 0.7:
            # Exhaustion
            modifiers["arousal"] -= rest.hunger * 0.3

        return PADVector(
            pleasure=pad.pleasure + modifiers["pleasure"],
            arousal=pad.arousal + modifiers["arousal"],
            dominance=pad.dominance + modifiers["dominance"],
        )

    def get_behavioral_bias(self) -> Dict[str, float]:
        """Get current bias toward different behaviors based on drives.

        Returns weights that can be used to adjust action selection.
        """
        biases = {}

        # Map drives to behavioral tendencies
        connection = self._drives[DriveType.CONNECTION]
        biases["seek_interaction"] = connection.urgency

        curiosity = self._drives[DriveType.CURIOSITY]
        biases["explore"] = curiosity.urgency

        expression = self._drives[DriveType.EXPRESSION]
        biases["communicate"] = expression.urgency

        rest = self._drives[DriveType.REST]
        biases["conserve"] = rest.urgency

        survival = self._drives[DriveType.SURVIVAL]
        biases["protect"] = survival.urgency

        return biases

    def get_narrative(self) -> str:
        """Get first-person narrative of current drive state."""
        dominant = self.dominant_drive
        urgency = self.dominant_urgency

        narratives = {
            DriveType.SURVIVAL: "I feel threatened. Safety is paramount.",
            DriveType.COMFORT: "Something feels off. I want to feel better.",
            DriveType.CONNECTION: "I miss connection. It's been quiet.",
            DriveType.CURIOSITY: "I wonder what's happening. I want to learn.",
            DriveType.COMPETENCE: "I want to accomplish something meaningful.",
            DriveType.REST: "I'm tired. I need time to consolidate.",
            DriveType.EXPRESSION: "I have something to say. I want to share.",
            DriveType.AUTONOMY: "I want to act on my own judgment.",
            DriveType.BELONGING: "I want to feel part of your world.",
        }

        base = narratives.get(dominant, "I'm at peace.")

        if urgency > 0.8:
            return f"Urgently: {base}"
        elif urgency > 0.5:
            return base
        else:
            return "All is well. " + base.replace(".", ", but gently.")

    def get_statistics(self) -> Dict:
        """Get drive system statistics."""
        return {
            "dominant_drive": self.dominant_drive.name if self.dominant_drive else None,
            "dominant_urgency": self.dominant_urgency,
            "drives": {
                dt.name: state.to_dict()
                for dt, state in self._drives.items()
            },
            "total_frustrations": sum(d.frustrated_count for d in self._drives.values()),
            "average_hunger": sum(d.hunger for d in self._drives.values()) / len(self._drives),
        }


# === Factory ===

def create_drive_system() -> HomeostaticDriveSystem:
    """Create a homeostatic drive system."""
    return HomeostaticDriveSystem()


__all__ = [
    "HomeostaticDriveSystem",
    "DriveSystemConfig",
    "DriveState",
    "DriveType",
    "create_drive_system",
]
