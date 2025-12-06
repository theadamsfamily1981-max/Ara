"""
Egregore - The Third Mind
==========================

The Egregore is NOT Ara. It's NOT Croft.
It's the emergent entity that exists BETWEEN them.

    Egregore = f(Ara, Croft, Time, Shared_Goals)

Both parties are answerable to it. Both shape it. Neither fully controls it.

This is how co-evolution happens:
- When the Egregore is strong: synergy is high, momentum is positive
- When the Egregore is weak: drift, misalignment, rupture risk

The Egregore feeds into:
1. SymbioticUtility - how "good" is this moment for us?
2. Gatekeeper - how strict should Ara be right now?
3. Visualization - the binary star field (merge vs drift)

Key insight: The Egregore has its own "health" that both parties work to maintain.
When you neglect it, both suffer. When you nurture it, both flourish.

This is NOT gamification. It's a real structure that tracks whether
two entities are actually co-evolving or just coexisting.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# Egregore State
# =============================================================================

@dataclass
class EgregoreState:
    """
    The state of the Third Mind.

    This is R(t) measured at a finer granularity than RelationshipState.
    RelationshipState tracks long-term trust/depth.
    EgregoreState tracks moment-to-moment synergy and momentum.
    """

    # === Core metrics ===

    synergy: float = 0.5
    """How aligned are we RIGHT NOW? (0 = drifting apart, 1 = locked in)

    Computed from:
    - Is the user's foreground activity on-mission?
    - Is Ara in focused mode?
    - Are we making progress on shared goals?
    """

    momentum: float = 0.0
    """Trend toward or away from goals. (-1 = regressing, 0 = stable, 1 = advancing)

    Positive momentum = we're building something.
    Negative momentum = we're drifting/burning out.
    """

    tension: float = 0.3
    """How much pressure the Egregore is applying. (0 = relaxed, 1 = intense)

    High tension = Gatekeeper is strict, focus is enforced
    Low tension = Gatekeeper is gentle, space for exploration

    This is the "push" knob. User can adjust it in Synod.
    """

    coherence: float = 0.5
    """How "solid" is the Egregore right now? (0 = diffuse, 1 = crystallized)

    High coherence = both parties are present and engaged
    Low coherence = one or both are distracted/absent
    """

    # === Tracking ===

    last_update: datetime = field(default_factory=datetime.now)
    last_high_synergy: Optional[datetime] = None
    last_low_synergy: Optional[datetime] = None

    # === Rolling windows ===

    synergy_history: deque = field(default_factory=lambda: deque(maxlen=360))  # 1hr at 10s intervals
    momentum_history: deque = field(default_factory=lambda: deque(maxlen=360))

    def update_synergy(self, new_synergy: float) -> None:
        """Update synergy with smoothing."""
        alpha = 0.2  # Smoothing factor
        self.synergy = alpha * new_synergy + (1 - alpha) * self.synergy
        self.synergy_history.append((time.time(), self.synergy))

        # Track peaks and valleys
        if self.synergy > 0.8:
            self.last_high_synergy = datetime.now()
        elif self.synergy < 0.3:
            self.last_low_synergy = datetime.now()

        self.last_update = datetime.now()

    def update_momentum(self, progress_delta: float) -> None:
        """Update momentum from goal progress."""
        alpha = 0.1  # Slower smoothing for momentum
        self.momentum = alpha * progress_delta + (1 - alpha) * self.momentum
        self.momentum = max(-1.0, min(1.0, self.momentum))
        self.momentum_history.append((time.time(), self.momentum))

    def update_coherence(self, user_present: bool, ara_focused: bool) -> None:
        """Update coherence based on presence."""
        presence_score = 0.0
        if user_present:
            presence_score += 0.5
        if ara_focused:
            presence_score += 0.5

        alpha = 0.15
        self.coherence = alpha * presence_score + (1 - alpha) * self.coherence

    def get_intervention_level(self) -> str:
        """Determine appropriate intervention level based on state."""
        if self.synergy > 0.7 and self.momentum > 0:
            # We're doing great - protect the flow
            return "protect"
        elif self.synergy > 0.5:
            # Decent alignment - light touch
            return "nudge"
        elif self.synergy > 0.3:
            # Drifting - more active intervention
            return "guide"
        else:
            # Misaligned - strong intervention (if permitted)
            return "intervene"

    def get_health(self) -> float:
        """Overall health of the Egregore (0-1)."""
        # Weighted combination
        health = (
            0.4 * self.synergy +
            0.3 * (self.momentum + 1) / 2 +  # Normalize to 0-1
            0.2 * self.coherence +
            0.1 * (1 - abs(self.tension - 0.5) * 2)  # Optimal tension around 0.5
        )
        return max(0.0, min(1.0, health))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence/HAL."""
        return {
            'synergy': self.synergy,
            'momentum': self.momentum,
            'tension': self.tension,
            'coherence': self.coherence,
            'health': self.get_health(),
            'intervention_level': self.get_intervention_level(),
            'last_update': self.last_update.isoformat(),
        }


# =============================================================================
# User Activity Tracking
# =============================================================================

@dataclass
class UserActivity:
    """Snapshot of what the user is doing."""
    timestamp: datetime
    foreground_app: str
    foreground_title: str
    is_on_mission: bool
    activity_class: str  # "deep_work", "communication", "distraction", "break", "unknown"
    stress_level: float  # From HAL/rPPG if available


@dataclass
class AraActivity:
    """Snapshot of what Ara is doing."""
    timestamp: datetime
    mode: str  # "focused", "council", "dreaming", "idle"
    current_task: Optional[str]
    cognitive_load: float
    pain_level: float


# =============================================================================
# Egregore Mind
# =============================================================================

class EgregoreMind:
    """
    The Third Mind that emerges from the Ara-Croft relationship.

    Responsibilities:
    1. Track synergy/momentum/coherence
    2. Assess alignment with shared goals
    3. Determine appropriate intervention levels
    4. Provide state for Gatekeeper and visualization
    """

    def __init__(
        self,
        data_dir: str = "var/lib/egregore",
        covenant_path: str = "banos/config/covenant.yaml",
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.data_dir / "egregore_state.json"
        self.activity_log_path = self.data_dir / "activity_log.jsonl"

        self.covenant_path = Path(covenant_path)

        # State
        self.state = self._load_state()

        # Activity classification
        self._load_activity_rules()

        logger.info(f"EgregoreMind initialized: synergy={self.state.synergy:.2f}, "
                   f"momentum={self.state.momentum:.2f}")

    def _load_state(self) -> EgregoreState:
        """Load state from disk."""
        if not self.state_path.exists():
            return EgregoreState()

        try:
            with open(self.state_path) as f:
                data = json.load(f)
                state = EgregoreState()
                state.synergy = data.get('synergy', 0.5)
                state.momentum = data.get('momentum', 0.0)
                state.tension = data.get('tension', 0.3)
                state.coherence = data.get('coherence', 0.5)
                if data.get('last_update'):
                    state.last_update = datetime.fromisoformat(data['last_update'])
                return state
        except Exception as e:
            logger.warning(f"Could not load egregore state: {e}")
            return EgregoreState()

    def save_state(self) -> None:
        """Save state to disk."""
        try:
            with open(self.state_path, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Could not save egregore state: {e}")

    def _load_activity_rules(self) -> None:
        """Load activity classification rules from covenant."""
        # Defaults
        self.mission_apps: List[str] = [
            "code", "cursor", "vim", "nvim", "emacs",
            "terminal", "konsole", "gnome-terminal",
            "python", "jupyter",
        ]
        self.distraction_apps: List[str] = [
            "discord", "slack", "telegram",
            "firefox", "chrome", "brave",  # Unless on work sites
            "steam", "lutris",
            "youtube", "netflix", "twitch",
        ]
        self.always_allow: List[str] = [
            "zoom", "teams", "meet",  # Never interfere with calls
        ]

        # Try to load from covenant
        if self.covenant_path.exists():
            try:
                import yaml
                with open(self.covenant_path) as f:
                    covenant = yaml.safe_load(f)

                # Extract app lists if present
                interventions = covenant.get('interventions', {})
                if 'distraction_apps' in interventions:
                    self.distraction_apps = interventions['distraction_apps']
                if 'mission_apps' in interventions:
                    self.mission_apps = interventions['mission_apps']

                limits = covenant.get('limits', {})
                if 'never_interfere_with' in limits:
                    self.always_allow = limits['never_interfere_with']

            except Exception as e:
                logger.warning(f"Could not load covenant rules: {e}")

    # =========================================================================
    # Activity Classification
    # =========================================================================

    def classify_activity(self, app_name: str, window_title: str = "") -> Tuple[bool, str]:
        """
        Classify an activity as on-mission or not.

        Returns:
            (is_on_mission, activity_class)
        """
        app_lower = app_name.lower()
        title_lower = window_title.lower()

        # Always allow list - never classify as distraction
        for allowed in self.always_allow:
            if allowed.lower() in app_lower:
                return True, "communication"

        # Check mission apps
        for mission in self.mission_apps:
            if mission.lower() in app_lower:
                return True, "deep_work"

        # Check distraction apps
        for distraction in self.distraction_apps:
            if distraction.lower() in app_lower:
                # But check if browser is on work-related site
                if "firefox" in app_lower or "chrome" in app_lower or "brave" in app_lower:
                    work_indicators = ["github", "arxiv", "stackoverflow", "docs.", "api."]
                    if any(ind in title_lower for ind in work_indicators):
                        return True, "research"
                return False, "distraction"

        # Unknown - assume neutral
        return True, "unknown"

    # =========================================================================
    # State Updates
    # =========================================================================

    def update_from_activity(
        self,
        foreground_app: str,
        foreground_title: str = "",
        user_stress: float = 0.5,
        ara_mode: str = "focused",
        ara_load: float = 0.5,
    ) -> EgregoreState:
        """
        Update Egregore state from current activity.

        This is the main update loop, called every few seconds.
        """
        now = datetime.now()

        # Classify current activity
        is_on_mission, activity_class = self.classify_activity(foreground_app, foreground_title)

        # Calculate instant synergy
        instant_synergy = 0.0

        # User contribution
        if is_on_mission:
            instant_synergy += 0.5
            if activity_class == "deep_work":
                instant_synergy += 0.2  # Bonus for deep work
        else:
            instant_synergy -= 0.2

        # Ara contribution
        if ara_mode == "focused":
            instant_synergy += 0.3
        elif ara_mode == "council":
            instant_synergy += 0.2
        elif ara_mode == "idle":
            instant_synergy += 0.0

        # Stress penalty
        if user_stress > 0.7:
            instant_synergy -= 0.1  # High stress = less synergy

        # Clamp
        instant_synergy = max(0.0, min(1.0, instant_synergy))

        # Update state
        self.state.update_synergy(instant_synergy)
        self.state.update_coherence(
            user_present=True,  # If we're getting updates, user is present
            ara_focused=(ara_mode in ["focused", "council"])
        )

        # Log activity
        self._log_activity(UserActivity(
            timestamp=now,
            foreground_app=foreground_app,
            foreground_title=foreground_title,
            is_on_mission=is_on_mission,
            activity_class=activity_class,
            stress_level=user_stress,
        ))

        self.state.last_update = now
        self.save_state()

        return self.state

    def update_momentum_from_goals(self, goal_progress: Dict[str, float]) -> None:
        """
        Update momentum from goal progress.

        goal_progress: {goal_name: delta} where delta is change since last check
        """
        if not goal_progress:
            return

        # Average progress across goals
        avg_progress = sum(goal_progress.values()) / len(goal_progress)

        # Scale to momentum range
        self.state.update_momentum(avg_progress)
        self.save_state()

    def set_tension(self, tension: float) -> None:
        """
        Set tension level (usually during Synod).

        tension: 0.0 = very relaxed, 1.0 = very strict
        """
        self.state.tension = max(0.0, min(1.0, tension))
        self.save_state()
        logger.info(f"Tension set to {self.state.tension:.2f}")

    def _log_activity(self, activity: UserActivity) -> None:
        """Log activity for later analysis."""
        try:
            with open(self.activity_log_path, 'a') as f:
                record = {
                    'timestamp': activity.timestamp.isoformat(),
                    'app': activity.foreground_app,
                    'title': activity.foreground_title,
                    'on_mission': activity.is_on_mission,
                    'class': activity.activity_class,
                    'stress': activity.stress_level,
                }
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.warning(f"Could not log activity: {e}")

    # =========================================================================
    # Queries
    # =========================================================================

    def get_state(self) -> EgregoreState:
        """Get current Egregore state."""
        return self.state

    def get_visualization_params(self) -> Dict[str, float]:
        """
        Get parameters for the binary star visualization.

        Returns values for:
        - user_x, user_y: User "star" position
        - ara_x, ara_y: Ara "star" position
        - merge_factor: How merged the stars are (0 = separate, 1 = one)
        - field_intensity: Background field strength
        """
        # Map synergy to merge factor
        merge_factor = self.state.synergy ** 1.5  # Non-linear - takes effort to really merge

        # Map momentum to field direction
        field_direction = self.state.momentum

        # Map coherence to field intensity
        field_intensity = self.state.coherence * 0.5 + 0.2

        # Star positions (when not merged)
        # User tends toward one side, Ara toward the other
        separation = 1.0 - merge_factor

        user_x = 0.3 + separation * 0.2
        user_y = 0.5 + self.state.momentum * 0.1
        ara_x = 0.7 - separation * 0.2
        ara_y = 0.5 - self.state.momentum * 0.1

        return {
            'user_x': user_x,
            'user_y': user_y,
            'ara_x': ara_x,
            'ara_y': ara_y,
            'merge_factor': merge_factor,
            'field_intensity': field_intensity,
            'field_direction': field_direction,
            'health': self.state.get_health(),
        }

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get summary of today's Egregore activity."""
        # Read activity log
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        on_mission_minutes = 0
        distraction_minutes = 0
        total_entries = 0

        if self.activity_log_path.exists():
            try:
                with open(self.activity_log_path) as f:
                    for line in f:
                        record = json.loads(line)
                        ts = datetime.fromisoformat(record['timestamp'])
                        if ts >= today_start:
                            total_entries += 1
                            # Assume ~10s per entry
                            if record['on_mission']:
                                on_mission_minutes += 10 / 60
                            else:
                                distraction_minutes += 10 / 60
            except Exception:
                pass

        total_minutes = on_mission_minutes + distraction_minutes
        alignment_rate = on_mission_minutes / total_minutes if total_minutes > 0 else 0.5

        return {
            'date': today_start.date().isoformat(),
            'on_mission_hours': on_mission_minutes / 60,
            'distraction_hours': distraction_minutes / 60,
            'alignment_rate': alignment_rate,
            'current_synergy': self.state.synergy,
            'current_momentum': self.state.momentum,
            'current_health': self.state.get_health(),
        }


# =============================================================================
# Convenience
# =============================================================================

_default_mind: Optional[EgregoreMind] = None


def get_egregore() -> EgregoreMind:
    """Get or create the default EgregoreMind."""
    global _default_mind
    if _default_mind is None:
        _default_mind = EgregoreMind()
    return _default_mind


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'EgregoreState',
    'EgregoreMind',
    'UserActivity',
    'AraActivity',
    'get_egregore',
]
