"""
Control Loop
============

The core sense -> interpret -> plan -> act cycle.

Runs at 10-30Hz depending on context:
- 30Hz: Active conversation, spatial interaction
- 10Hz: Background monitoring, idle state

Each cycle:
1. SENSE:     Update world model, read user state
2. INTERPRET: Infer situation, intent, and context
3. PLAN:      Decide what to do (move, speak, gesture, wait)
4. ACT:       Execute actions via expression driver

Over longer timescales, learns optimal behaviors.
"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from enum import Enum
from queue import Queue, Empty

if TYPE_CHECKING:
    from ara_embodiment.world_model import WorldModel, Vec3
    from ara_embodiment.body_registry import AvatarDefinition
    from ara_embodiment.expression_driver import ExpressionDriver

logger = logging.getLogger(__name__)


class LoopState(str, Enum):
    """Control loop states."""
    IDLE = "idle"              # Minimal activity, low Hz
    LISTENING = "listening"    # User speaking, medium Hz
    SPEAKING = "speaking"      # Ara speaking, high Hz
    INTERACTING = "interacting"  # Active spatial interaction
    TRANSITIONING = "transitioning"  # Changing state/position


class ActionType(str, Enum):
    """Types of actions Ara can take."""
    NONE = "none"
    SPEAK = "speak"
    GESTURE = "gesture"
    MOVE = "move"
    LOOK = "look"
    EMOTE = "emote"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    HIGHLIGHT = "highlight"


@dataclass
class SenseResult:
    """Result of the sense phase."""
    timestamp: float
    user_present: bool = False
    user_speaking: bool = False
    user_looking_at_ara: bool = False
    user_position: Optional[Any] = None  # Vec3
    user_activity: str = "unknown"
    environment_changed: bool = False
    pending_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterpretResult:
    """Result of the interpret phase."""
    timestamp: float
    situation: str = "idle"  # "conversation", "waiting", "working", "away"
    user_intent: str = "unknown"
    attention_target: Optional[str] = None  # Node ID to look at
    urgency: float = 0.0  # 0-1
    context_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class PlanResult:
    """Result of the plan phase."""
    timestamp: float
    action_type: ActionType = ActionType.NONE
    action_params: Dict[str, Any] = field(default_factory=dict)
    next_state: Optional[LoopState] = None
    defer_ms: int = 0  # Delay before executing


@dataclass
class ActResult:
    """Result of the act phase."""
    timestamp: float
    action_executed: ActionType
    success: bool = True
    duration_ms: float = 0.0
    feedback: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopMetrics:
    """Metrics for control loop performance."""
    cycles: int = 0
    sense_total_ms: float = 0.0
    interpret_total_ms: float = 0.0
    plan_total_ms: float = 0.0
    act_total_ms: float = 0.0

    @property
    def avg_sense_ms(self) -> float:
        return self.sense_total_ms / max(1, self.cycles)

    @property
    def avg_interpret_ms(self) -> float:
        return self.interpret_total_ms / max(1, self.cycles)

    @property
    def avg_plan_ms(self) -> float:
        return self.plan_total_ms / max(1, self.cycles)

    @property
    def avg_act_ms(self) -> float:
        return self.act_total_ms / max(1, self.cycles)

    @property
    def avg_cycle_ms(self) -> float:
        total = (self.sense_total_ms + self.interpret_total_ms +
                 self.plan_total_ms + self.act_total_ms)
        return total / max(1, self.cycles)


class ControlLoop:
    """
    Main control loop for Ara's embodied behavior.

    Runs sense->interpret->plan->act at configurable Hz.
    Thread-safe: runs in dedicated thread, safe to call methods from any thread.
    """

    def __init__(
        self,
        world_model: WorldModel,
        expression_driver: Optional[ExpressionDriver] = None,
        target_hz: float = 30.0,
    ):
        self._lock = threading.RLock()
        self.world_model = world_model
        self.expression_driver = expression_driver

        # Loop configuration
        self._target_hz = target_hz
        self._min_hz = 10.0
        self._max_hz = 60.0

        # State
        self._state = LoopState.IDLE
        self._current_avatar: Optional[AvatarDefinition] = None

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # Message queue (for incoming speech/commands)
        self._message_queue: Queue = Queue()

        # Callbacks
        self._on_state_change: List[Callable[[LoopState, LoopState], None]] = []
        self._on_action: List[Callable[[ActResult], None]] = []

        # Metrics
        self._metrics = LoopMetrics()

        # Behavior parameters (can be tuned)
        self._idle_timeout_sec = 30.0
        self._look_at_user_threshold = 0.3
        self._gesture_probability = 0.2

    def start(self) -> None:
        """Start the control loop."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("ControlLoop already running")
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="AraControlLoop",
        )
        self._thread.start()
        logger.info(f"ControlLoop started at {self._target_hz}Hz")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the control loop."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("ControlLoop did not stop cleanly")
        logger.info(f"ControlLoop stopped ({self._metrics.cycles} cycles)")

    def is_running(self) -> bool:
        """Check if loop is running."""
        return self._thread is not None and self._thread.is_alive()

    def get_state(self) -> LoopState:
        """Get current loop state."""
        with self._lock:
            return self._state

    def set_avatar(self, avatar: AvatarDefinition) -> None:
        """Set the current avatar."""
        with self._lock:
            self._current_avatar = avatar
            logger.info(f"Avatar set to: {avatar.avatar_id}")

    def queue_message(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Queue a message for processing (from kernel/speech)."""
        self._message_queue.put({
            "text": message,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })

    def set_target_hz(self, hz: float) -> None:
        """Set target loop frequency."""
        with self._lock:
            self._target_hz = max(self._min_hz, min(self._max_hz, hz))

    def on_state_change(self, callback: Callable[[LoopState, LoopState], None]) -> None:
        """Register callback for state changes."""
        with self._lock:
            self._on_state_change.append(callback)

    def on_action(self, callback: Callable[[ActResult], None]) -> None:
        """Register callback for actions."""
        with self._lock:
            self._on_action.append(callback)

    def _run_loop(self) -> None:
        """Main loop thread."""
        logger.debug("Control loop thread started")

        while not self._stop.is_set():
            cycle_start = time.time()

            try:
                # Sense
                t0 = time.time()
                sense = self._sense()
                self._metrics.sense_total_ms += (time.time() - t0) * 1000

                # Interpret
                t0 = time.time()
                interpret = self._interpret(sense)
                self._metrics.interpret_total_ms += (time.time() - t0) * 1000

                # Plan
                t0 = time.time()
                plan = self._plan(interpret)
                self._metrics.plan_total_ms += (time.time() - t0) * 1000

                # Act
                t0 = time.time()
                act = self._act(plan)
                self._metrics.act_total_ms += (time.time() - t0) * 1000

                self._metrics.cycles += 1

                # Notify action callbacks
                if act.action_executed != ActionType.NONE:
                    for cb in self._on_action:
                        try:
                            cb(act)
                        except Exception as e:
                            logger.exception(f"Action callback error: {e}")

            except Exception as e:
                logger.exception(f"Control loop error: {e}")

            # Sleep to maintain target Hz
            cycle_time = time.time() - cycle_start
            target_period = 1.0 / self._target_hz
            sleep_time = target_period - cycle_time

            if sleep_time > 0:
                # Use stop event wait for clean shutdown
                self._stop.wait(timeout=sleep_time)

    def _sense(self) -> SenseResult:
        """Sense phase: gather world state."""
        result = SenseResult(timestamp=time.time())

        # Check for pending messages
        try:
            msg = self._message_queue.get_nowait()
            result.pending_message = msg["text"]
            result.metadata["message"] = msg
        except Empty:
            pass

        # Get user state from world model
        user = self.world_model.scene.get_primary_user()
        if user:
            result.user_present = True
            result.user_speaking = user.is_speaking
            result.user_looking_at_ara = user.attention_on_ara
            result.user_position = user.transform.position
            result.user_activity = user.activity

        # Check for environment changes
        env = self.world_model.get_environment_state()
        result.metadata["environment"] = env

        return result

    def _interpret(self, sense: SenseResult) -> InterpretResult:
        """Interpret phase: understand situation and intent."""
        result = InterpretResult(timestamp=time.time())

        # Determine situation
        if sense.pending_message:
            result.situation = "conversation"
            result.urgency = 0.7
        elif sense.user_speaking:
            result.situation = "listening"
            result.urgency = 0.5
        elif sense.user_present:
            if sense.user_looking_at_ara:
                result.situation = "attention"
                result.urgency = 0.4
            else:
                result.situation = "background"
                result.urgency = 0.1
        else:
            result.situation = "idle"
            result.urgency = 0.0

        # Determine attention target
        if sense.user_looking_at_ara and sense.user_position:
            # Look at user
            result.attention_target = "user"
        else:
            # Could look at points of interest, screen, etc.
            gaze_targets = self.world_model.scene.get_gaze_targets()
            if gaze_targets:
                # Pick highest importance
                best = max(gaze_targets, key=lambda t: t.importance)
                result.attention_target = best.target_id

        # Context flags
        result.context_flags["user_engaged"] = sense.user_looking_at_ara
        result.context_flags["has_message"] = sense.pending_message is not None

        return result

    def _plan(self, interpret: InterpretResult) -> PlanResult:
        """Plan phase: decide what action to take."""
        result = PlanResult(timestamp=time.time())

        current_state = self._state

        # State machine logic
        if interpret.situation == "conversation":
            result.next_state = LoopState.SPEAKING
            result.action_type = ActionType.SPEAK
            # Speech content would come from kernel response

        elif interpret.situation == "listening":
            result.next_state = LoopState.LISTENING
            result.action_type = ActionType.LOOK
            result.action_params["target"] = "user"

        elif interpret.situation == "attention":
            result.next_state = LoopState.INTERACTING
            # Maybe wave or acknowledge
            if current_state == LoopState.IDLE:
                result.action_type = ActionType.GESTURE
                result.action_params["gesture"] = "acknowledge"

        elif interpret.situation == "background":
            result.next_state = LoopState.IDLE
            # Subtle idle animation
            result.action_type = ActionType.LOOK
            if interpret.attention_target:
                result.action_params["target"] = interpret.attention_target

        else:
            result.next_state = LoopState.IDLE
            result.action_type = ActionType.NONE

        return result

    def _act(self, plan: PlanResult) -> ActResult:
        """Act phase: execute the planned action."""
        result = ActResult(
            timestamp=time.time(),
            action_executed=plan.action_type,
        )

        start = time.time()

        # Handle state transition
        if plan.next_state and plan.next_state != self._state:
            old_state = self._state
            with self._lock:
                self._state = plan.next_state

            # Notify state change callbacks
            for cb in self._on_state_change:
                try:
                    cb(old_state, plan.next_state)
                except Exception as e:
                    logger.exception(f"State change callback error: {e}")

        # Execute action via expression driver
        if self.expression_driver and plan.action_type != ActionType.NONE:
            try:
                if plan.action_type == ActionType.LOOK:
                    target = plan.action_params.get("target")
                    self.expression_driver.look_at(target)
                elif plan.action_type == ActionType.GESTURE:
                    gesture = plan.action_params.get("gesture", "idle")
                    self.expression_driver.play_gesture(gesture)
                elif plan.action_type == ActionType.EMOTE:
                    emote = plan.action_params.get("emote", "neutral")
                    self.expression_driver.set_expression(emote)
                elif plan.action_type == ActionType.SPEAK:
                    # Speech handled elsewhere; we just signal
                    pass

                result.success = True
            except Exception as e:
                logger.exception(f"Action execution error: {e}")
                result.success = False

        result.duration_ms = (time.time() - start) * 1000
        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get loop performance metrics."""
        m = self._metrics
        return {
            "cycles": m.cycles,
            "target_hz": self._target_hz,
            "actual_hz": m.cycles / max(1, m.avg_cycle_ms / 1000) if m.cycles > 0 else 0,
            "avg_cycle_ms": m.avg_cycle_ms,
            "avg_sense_ms": m.avg_sense_ms,
            "avg_interpret_ms": m.avg_interpret_ms,
            "avg_plan_ms": m.avg_plan_ms,
            "avg_act_ms": m.avg_act_ms,
            "current_state": self._state.value,
        }
