# ara/cognition/clairvoyant/mode_controller.py
"""
Mode Controller - Ara's Operating Mode Selection
================================================

Selects Ara's operating mode based on:
- Current regime classification
- Trajectory analysis (where we're heading)
- Raw metrics (for safety overrides)

Three primary modes:

1. REFLEX Mode
   - Focus: Safety, low latency
   - Logic: Simple rules on raw metrics
   - When: Serious anomaly, critical regime, emergency

2. NAVIGATOR Mode
   - Focus: Anticipation using 10D trajectory
   - Logic: Use z_t + velocity to predict and gently steer
   - When: Normal operation, using clairvoyance

3. ARCHITECT Mode
   - Focus: Long-view learning, antifragility
   - Logic: Offline analysis, policy updates
   - When: Background, during low-activity periods

The mode controller also triggers specific actions:
- Adjusting job queues
- Changing HUD/dashboard density
- Nudging the user
- Emergency interventions

Usage:
    from ara.cognition.clairvoyant.mode_controller import ModeController, OperatingMode

    controller = ModeController(regime_classifier)

    mode = controller.update_mode(z_t, raw_features)
    actions = controller.get_actions(mode, z_t, trajectory, raw_features)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any
import numpy as np

from .regime import RegimeClassifier, Regime, RegimeType, WARNING_REGIMES, GOOD_REGIMES
from .trajectory import TrajectoryBuffer

logger = logging.getLogger(__name__)


# =============================================================================
# Operating Modes
# =============================================================================

class OperatingMode(Enum):
    """Ara's primary operating modes."""

    REFLEX = auto()
    """
    Emergency/safety mode. Uses simple, fast rules.
    Active when:
    - Critical metrics (temp, errors)
    - Warning regimes detected
    - Rapid trajectory toward bad regions
    """

    NAVIGATOR = auto()
    """
    Normal clairvoyant operation. Uses 10D trajectory.
    Active when:
    - Normal operation
    - Uses prediction to anticipate and adjust
    - Gentle interventions
    """

    ARCHITECT = auto()
    """
    Learning and analysis mode. Background operation.
    Active when:
    - System idle or low activity
    - Runs offline analysis
    - Updates policies and models
    """

    MINIMAL = auto()
    """
    Minimal intervention mode. User requested hands-off.
    Only safety-critical interventions.
    """


# =============================================================================
# Actions
# =============================================================================

class ActionType(Enum):
    """Types of actions the controller can take."""
    # Safety
    KILL_JOB = auto()
    THROTTLE_JOBS = auto()
    PAUSE_QUEUE = auto()
    EMERGENCY_COOLDOWN = auto()

    # Adjustments
    ADJUST_HUD_DENSITY = auto()
    ADJUST_LIGHTING = auto()
    ADJUST_AUDIO = auto()

    # User interaction
    SUGGEST_BREAK = auto()
    NUDGE_USER = auto()
    ALERT_USER = auto()

    # System
    LOG_WARNING = auto()
    TRIGGER_CHECKPOINT = auto()
    REQUEST_ANALYSIS = auto()

    # Learning
    COLLECT_DATA = auto()
    UPDATE_MODEL = auto()


@dataclass
class Action:
    """An action to be executed."""
    type: ActionType
    priority: int = 0          # Higher = more urgent
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Mode Controller
# =============================================================================

@dataclass
class ModeControllerConfig:
    """Configuration for mode controller."""
    # Thresholds for REFLEX mode
    temp_critical: float = 85.0
    load_critical: float = 0.95
    error_rate_critical: float = 0.1

    # Trajectory thresholds
    speed_threshold_high: float = 0.5   # Fast movement = instability
    approach_warning_steps: int = 10    # Steps to warning region

    # Mode transition cooldowns
    reflex_cooldown_seconds: float = 30.0
    architect_delay_seconds: float = 300.0  # 5 min idle before architect

    # User override
    allow_user_override: bool = True


class ModeController:
    """
    Controls Ara's operating mode based on state and trajectory.

    The controller runs every tick and:
    1. Checks for safety conditions (REFLEX triggers)
    2. Analyzes regime and trajectory
    3. Selects appropriate mode
    4. Generates actions for that mode
    """

    def __init__(
        self,
        regime_classifier: Optional[RegimeClassifier] = None,
        config: Optional[ModeControllerConfig] = None,
    ):
        """
        Initialize mode controller.

        Args:
            regime_classifier: Trained regime classifier
            config: Controller configuration
        """
        self.regime_classifier = regime_classifier
        self.config = config or ModeControllerConfig()

        # Current state
        self._mode: OperatingMode = OperatingMode.NAVIGATOR
        self._last_mode_change: datetime = datetime.utcnow()
        self._last_regime: Optional[Regime] = None
        self._user_override: Optional[OperatingMode] = None

        # Action queue
        self._pending_actions: List[Action] = []

        # Tracking
        self._reflex_triggers: List[str] = []
        self._last_activity_time: datetime = datetime.utcnow()
        self._idle_start: Optional[datetime] = None

        logger.info("ModeController initialized")

    @property
    def current_mode(self) -> OperatingMode:
        return self._mode

    @property
    def time_in_mode(self) -> float:
        """Seconds since last mode change."""
        return (datetime.utcnow() - self._last_mode_change).total_seconds()

    def set_user_override(self, mode: Optional[OperatingMode]) -> None:
        """Set user-requested mode override."""
        if self.config.allow_user_override:
            self._user_override = mode
            if mode:
                logger.info(f"User override: {mode.name}")

    def update_mode(
        self,
        z: np.ndarray,
        raw_features: Dict[str, float],
        trajectory: Optional[TrajectoryBuffer] = None,
    ) -> OperatingMode:
        """
        Update operating mode based on current state.

        Args:
            z: Current latent point
            raw_features: Raw features from StateSampler
            trajectory: Trajectory buffer for velocity analysis

        Returns:
            New operating mode
        """
        self._reflex_triggers = []

        # Check user override first
        if self._user_override is not None:
            self._set_mode(self._user_override)
            return self._mode

        # Check REFLEX triggers (safety first)
        if self._check_reflex_triggers(raw_features, trajectory):
            self._set_mode(OperatingMode.REFLEX)
            return self._mode

        # Classify regime if we have a classifier
        regime = None
        if self.regime_classifier and self.regime_classifier.is_trained:
            regime = self.regime_classifier.classify(z)
            self._last_regime = regime

            # Warning regime -> REFLEX
            if regime.is_warning and regime.confidence > 0.7:
                self._reflex_triggers.append(f"Warning regime: {regime.type.name}")
                self._set_mode(OperatingMode.REFLEX)
                return self._mode

        # Check trajectory for approaching warnings
        if trajectory and self._check_trajectory_warning(trajectory, z):
            self._reflex_triggers.append("Trajectory approaching warning region")
            self._set_mode(OperatingMode.REFLEX)
            return self._mode

        # Check for idle -> ARCHITECT
        activity = raw_features.get("user.input_rate", 0.5)
        if activity < 0.1:
            if self._idle_start is None:
                self._idle_start = datetime.utcnow()
            elif (datetime.utcnow() - self._idle_start).total_seconds() > self.config.architect_delay_seconds:
                self._set_mode(OperatingMode.ARCHITECT)
                return self._mode
        else:
            self._idle_start = None
            self._last_activity_time = datetime.utcnow()

        # Default: NAVIGATOR
        self._set_mode(OperatingMode.NAVIGATOR)
        return self._mode

    def _check_reflex_triggers(
        self,
        features: Dict[str, float],
        trajectory: Optional[TrajectoryBuffer],
    ) -> bool:
        """Check for immediate safety triggers."""
        triggers = []

        # Temperature
        for key in ["system.thermal", "gpu0.temp", "gpu1.temp"]:
            if key in features:
                val = features[key]
                if val > 0.85:  # Normalized threshold
                    triggers.append(f"High temperature: {key}={val:.2f}")

        # System stress
        if features.get("system.stress", 0) > self.config.load_critical:
            triggers.append(f"System overload: {features['system.stress']:.2f}")

        # Agent errors
        if features.get("agents.erroring", 0) > self.config.error_rate_critical:
            triggers.append(f"Agent errors: {features['agents.erroring']:.2f}")

        # Fast trajectory movement (instability)
        if trajectory and trajectory.get_speed() > self.config.speed_threshold_high:
            triggers.append(f"Rapid state change: speed={trajectory.get_speed():.2f}")

        self._reflex_triggers = triggers
        return len(triggers) > 0

    def _check_trajectory_warning(
        self,
        trajectory: TrajectoryBuffer,
        current_z: np.ndarray,
    ) -> bool:
        """Check if trajectory is heading toward warning region."""
        if not self.regime_classifier or not self.regime_classifier.is_trained:
            return False

        # Predict future positions
        future = trajectory.predict_future(steps=self.config.approach_warning_steps)

        # Check each future point
        for i, future_z in enumerate(future):
            regime = self.regime_classifier.classify(future_z)
            if regime.is_warning and regime.confidence > 0.6:
                return True

        return False

    def _set_mode(self, new_mode: OperatingMode) -> None:
        """Set new mode with logging."""
        if new_mode != self._mode:
            old_mode = self._mode
            self._mode = new_mode
            self._last_mode_change = datetime.utcnow()
            logger.info(f"Mode transition: {old_mode.name} -> {new_mode.name}")

            if self._reflex_triggers:
                logger.warning(f"REFLEX triggers: {self._reflex_triggers}")

    def get_actions(
        self,
        z: np.ndarray,
        trajectory: Optional[TrajectoryBuffer],
        raw_features: Dict[str, float],
    ) -> List[Action]:
        """
        Generate actions based on current mode and state.

        Args:
            z: Current latent point
            trajectory: Trajectory buffer
            raw_features: Raw features from StateSampler

        Returns:
            List of actions to execute
        """
        actions = []

        if self._mode == OperatingMode.REFLEX:
            actions.extend(self._get_reflex_actions(z, raw_features))
        elif self._mode == OperatingMode.NAVIGATOR:
            actions.extend(self._get_navigator_actions(z, trajectory, raw_features))
        elif self._mode == OperatingMode.ARCHITECT:
            actions.extend(self._get_architect_actions(z, trajectory))
        elif self._mode == OperatingMode.MINIMAL:
            # Only critical safety actions
            if raw_features.get("system.thermal", 0) > 0.95:
                actions.append(Action(
                    type=ActionType.EMERGENCY_COOLDOWN,
                    priority=100,
                    reason="Critical temperature in MINIMAL mode",
                ))

        return sorted(actions, key=lambda a: -a.priority)

    def _get_reflex_actions(
        self,
        z: np.ndarray,
        features: Dict[str, float],
    ) -> List[Action]:
        """Generate REFLEX mode actions."""
        actions = []

        # Temperature emergency
        if features.get("system.thermal", 0) > 0.9:
            actions.append(Action(
                type=ActionType.EMERGENCY_COOLDOWN,
                priority=100,
                reason="Critical temperature",
            ))
            actions.append(Action(
                type=ActionType.THROTTLE_JOBS,
                priority=90,
                params={"factor": 0.5},
                reason="Reduce heat generation",
            ))

        # System overload
        if features.get("system.stress", 0) > 0.9:
            actions.append(Action(
                type=ActionType.PAUSE_QUEUE,
                priority=80,
                reason="System overload",
            ))

        # Agent errors
        if features.get("agents.erroring", 0) > 0.1:
            actions.append(Action(
                type=ActionType.LOG_WARNING,
                priority=70,
                params={"message": "Agent error rate elevated"},
                reason="Agent health check needed",
            ))

        # Always alert user in REFLEX
        if self._reflex_triggers:
            actions.append(Action(
                type=ActionType.ALERT_USER,
                priority=60,
                params={"triggers": self._reflex_triggers},
                reason="Safety triggers active",
            ))

        return actions

    def _get_navigator_actions(
        self,
        z: np.ndarray,
        trajectory: Optional[TrajectoryBuffer],
        features: Dict[str, float],
    ) -> List[Action]:
        """Generate NAVIGATOR mode actions."""
        actions = []

        # Analyze regime
        if self._last_regime:
            regime = self._last_regime

            # Good regime: subtle optimizations
            if regime.is_good:
                # Adjust HUD for flow state
                if regime.type == RegimeType.FLOW:
                    actions.append(Action(
                        type=ActionType.ADJUST_HUD_DENSITY,
                        priority=20,
                        params={"density": "minimal"},
                        reason="Flow state detected",
                    ))

            # Neutral regime: light monitoring
            elif regime.is_neutral:
                actions.append(Action(
                    type=ActionType.COLLECT_DATA,
                    priority=10,
                    reason="Collecting data for learning",
                ))

        # User fatigue check
        if features.get("user.fatigue", 0) > 0.7:
            actions.append(Action(
                type=ActionType.SUGGEST_BREAK,
                priority=30,
                reason="High user fatigue detected",
            ))

        # Trajectory-based predictions
        if trajectory:
            speed = trajectory.get_speed()

            # High speed = instability, increase monitoring
            if speed > 0.3:
                actions.append(Action(
                    type=ActionType.ADJUST_HUD_DENSITY,
                    priority=25,
                    params={"density": "normal"},
                    reason="Elevated state volatility",
                ))

            # Predict if approaching warning
            if self.regime_classifier and self.regime_classifier.is_trained:
                future = trajectory.predict_future(steps=5)
                for future_z in future:
                    future_regime = self.regime_classifier.classify(future_z)
                    if future_regime.is_warning and future_regime.confidence > 0.5:
                        actions.append(Action(
                            type=ActionType.NUDGE_USER,
                            priority=40,
                            params={"message": f"Approaching {future_regime.type.name} zone"},
                            reason="Predictive warning",
                        ))
                        break

        return actions

    def _get_architect_actions(
        self,
        z: np.ndarray,
        trajectory: Optional[TrajectoryBuffer],
    ) -> List[Action]:
        """Generate ARCHITECT mode actions."""
        actions = []

        # Trigger offline analysis
        actions.append(Action(
            type=ActionType.REQUEST_ANALYSIS,
            priority=10,
            params={"type": "full_session"},
            reason="ARCHITECT mode analysis",
        ))

        # Collect training data
        actions.append(Action(
            type=ActionType.COLLECT_DATA,
            priority=5,
            params={"type": "trajectory_segment"},
            reason="Expand training dataset",
        ))

        # Check for model updates
        actions.append(Action(
            type=ActionType.UPDATE_MODEL,
            priority=5,
            reason="Check for model improvements",
        ))

        return actions

    def get_status(self) -> Dict[str, Any]:
        """Get controller status for monitoring."""
        return {
            "mode": self._mode.name,
            "time_in_mode": self.time_in_mode,
            "last_regime": self._last_regime.type.name if self._last_regime else None,
            "regime_confidence": self._last_regime.confidence if self._last_regime else None,
            "reflex_triggers": self._reflex_triggers,
            "user_override": self._user_override.name if self._user_override else None,
        }


# =============================================================================
# Testing
# =============================================================================

def _test_controller():
    """Test mode controller."""
    print("=" * 60)
    print("Mode Controller Test")
    print("=" * 60)

    # Create controller without classifier (will use feature-based logic)
    controller = ModeController()

    # Test normal state
    z_normal = np.zeros(10)
    features_normal = {
        "system.stress": 0.3,
        "system.thermal": 0.4,
        "user.input_rate": 0.5,
        "user.fatigue": 0.3,
        "agents.erroring": 0.0,
    }

    mode = controller.update_mode(z_normal, features_normal)
    print(f"\nNormal state: mode={mode.name}")
    actions = controller.get_actions(z_normal, None, features_normal)
    print(f"Actions: {[a.type.name for a in actions]}")

    # Test overload state
    features_overload = {
        "system.stress": 0.95,
        "system.thermal": 0.92,
        "user.input_rate": 0.1,
        "user.fatigue": 0.8,
        "agents.erroring": 0.15,
    }

    mode = controller.update_mode(z_normal, features_overload)
    print(f"\nOverload state: mode={mode.name}")
    actions = controller.get_actions(z_normal, None, features_overload)
    print(f"Actions: {[a.type.name for a in actions]}")
    print(f"Triggers: {controller._reflex_triggers}")

    # Test idle state (simulate time passing)
    controller.config.architect_delay_seconds = 0.1  # Short for testing
    features_idle = {
        "system.stress": 0.1,
        "system.thermal": 0.2,
        "user.input_rate": 0.05,
        "user.fatigue": 0.2,
        "agents.erroring": 0.0,
    }

    import time
    controller._idle_start = datetime.utcnow() - timedelta(seconds=1)
    mode = controller.update_mode(z_normal, features_idle)
    print(f"\nIdle state: mode={mode.name}")

    # Status
    print(f"\nController status: {controller.get_status()}")


if __name__ == "__main__":
    _test_controller()
