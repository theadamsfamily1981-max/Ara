"""
Ara Embodied Multimodal Core
=============================

The main sensor→HV→action loop for Ara's physical body.

This gives her a body (mic, camera, LEDs, servos, haptics) while
enforcing strict safety rails at every step.

Philosophy: Sense the world, reason about it, act gently.
E-stop is always available. Safety first.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import json
import numpy as np

from .rails import (
    EmbodimentRails,
    MotorCommand,
    HapticCommand,
    VisualCommand,
    ClampedCommand,
    SafetyLevel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Sensor Types
# =============================================================================

class SensorType(Enum):
    """Types of sensors Ara can have."""
    MICROPHONE = "microphone"
    CAMERA = "camera"
    IMU = "imu"               # Inertial measurement unit
    PROXIMITY = "proximity"
    TEMPERATURE = "temperature"
    TOUCH = "touch"


@dataclass
class SensorReading:
    """A reading from a sensor."""
    sensor_type: SensorType
    timestamp: datetime
    raw_value: Any
    hv_encoding: Optional[np.ndarray] = None  # 8192-dim HV

    def to_hv(self, dim: int = 8192) -> np.ndarray:
        """Convert reading to hypervector."""
        if self.hv_encoding is not None:
            return self.hv_encoding

        # Simple encoding based on sensor type
        # In production, each sensor would have its own encoder
        rng = np.random.default_rng(hash(str(self.raw_value)) % (2**31))
        return rng.choice([-1, 1], size=dim).astype(np.int8)


@dataclass
class WorldState:
    """Current state of the world as Ara perceives it."""
    timestamp: datetime
    sensor_readings: List[SensorReading]
    human_present: bool = False
    human_speaking: bool = False
    last_utterance: Optional[str] = None
    ambient_light: float = 0.5  # 0-1
    ambient_noise: float = 0.3  # 0-1


# =============================================================================
# Action Types
# =============================================================================

class ActionType(Enum):
    """Types of physical actions Ara can take."""
    MOTOR = "motor"
    HAPTIC = "haptic"
    VISUAL = "visual"
    AUDIO = "audio"
    NONE = "none"


@dataclass
class IntendedAction:
    """An action Ara wants to take."""
    action_type: ActionType
    parameters: Dict[str, Any]
    priority: int = 0
    reason: str = ""


@dataclass
class ExecutedAction:
    """An action after safety processing."""
    intended: IntendedAction
    executed: Dict[str, Any]
    was_modified: bool
    was_blocked: bool
    modifications: List[str] = field(default_factory=list)


# =============================================================================
# Core Loop States
# =============================================================================

class CoreState(Enum):
    """State of the embodiment core loop."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    SHUTTING_DOWN = "shutting_down"


# =============================================================================
# Episode Logging
# =============================================================================

@dataclass
class EmbodimentEpisode:
    """A single moment in Ara's embodied experience."""
    timestamp: datetime
    world_hv: np.ndarray          # What she sensed
    intent_hv: np.ndarray         # What she intended
    action: ExecutedAction        # What she did
    resonance: float = 0.0        # How significant (0-1)


class EpisodeLogger:
    """Logs embodiment episodes to disk."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "episodes.jsonl"

    def log(self, episode: EmbodimentEpisode):
        """Log an episode."""
        # Don't store raw HVs (too large), just metadata
        record = {
            "timestamp": episode.timestamp.isoformat(),
            "action_type": episode.action.intended.action_type.value,
            "was_modified": episode.action.was_modified,
            "was_blocked": episode.action.was_blocked,
            "modifications": episode.action.modifications,
            "resonance": episode.resonance,
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + "\n")


# =============================================================================
# Sensor Interface (Abstract)
# =============================================================================

class SensorInterface:
    """
    Abstract interface for sensors.

    Implement this for real hardware.
    """

    async def read_microphone(self) -> Optional[SensorReading]:
        """Read from microphone."""
        raise NotImplementedError

    async def read_camera(self) -> Optional[SensorReading]:
        """Read from camera."""
        raise NotImplementedError

    async def read_imu(self) -> Optional[SensorReading]:
        """Read from IMU."""
        raise NotImplementedError

    async def detect_human_present(self) -> bool:
        """Detect if a human is present."""
        raise NotImplementedError

    async def get_last_utterance(self) -> Optional[str]:
        """Get the last detected utterance."""
        raise NotImplementedError


class MockSensorInterface(SensorInterface):
    """Mock sensors for testing."""

    def __init__(self):
        self._human_present = True
        self._last_utterance: Optional[str] = None

    async def read_microphone(self) -> Optional[SensorReading]:
        return SensorReading(
            sensor_type=SensorType.MICROPHONE,
            timestamp=datetime.utcnow(),
            raw_value={"volume": 0.3, "speech_detected": False},
        )

    async def read_camera(self) -> Optional[SensorReading]:
        return SensorReading(
            sensor_type=SensorType.CAMERA,
            timestamp=datetime.utcnow(),
            raw_value={"motion": False, "brightness": 0.5},
        )

    async def read_imu(self) -> Optional[SensorReading]:
        return SensorReading(
            sensor_type=SensorType.IMU,
            timestamp=datetime.utcnow(),
            raw_value={"orientation": [0, 0, 1], "acceleration": [0, 0, 9.8]},
        )

    async def detect_human_present(self) -> bool:
        return self._human_present

    async def get_last_utterance(self) -> Optional[str]:
        return self._last_utterance

    def simulate_utterance(self, text: str):
        """Simulate a human utterance."""
        self._last_utterance = text


# =============================================================================
# Actuator Interface (Abstract)
# =============================================================================

class ActuatorInterface:
    """
    Abstract interface for actuators.

    Implement this for real hardware.
    """

    async def move_motor(self, joint: str, position: float, speed: float, torque: float):
        """Move a motor."""
        raise NotImplementedError

    async def release_all_motors(self):
        """Release all motors immediately."""
        raise NotImplementedError

    async def haptic_pulse(self, intensity: float, duration_ms: int, pattern: str):
        """Trigger haptic feedback."""
        raise NotImplementedError

    async def haptic_off(self):
        """Turn off haptics."""
        raise NotImplementedError

    async def set_led(self, brightness: float, color: str, transition_ms: int):
        """Set LED state."""
        raise NotImplementedError

    async def speak(self, text: str):
        """Speak text."""
        raise NotImplementedError


class MockActuatorInterface(ActuatorInterface):
    """Mock actuators for testing."""

    def __init__(self):
        self.motor_log: List[Dict] = []
        self.haptic_log: List[Dict] = []
        self.led_log: List[Dict] = []
        self.speech_log: List[str] = []

    async def move_motor(self, joint: str, position: float, speed: float, torque: float):
        self.motor_log.append({
            "joint": joint,
            "position": position,
            "speed": speed,
            "torque": torque,
            "timestamp": datetime.utcnow().isoformat(),
        })
        logger.debug(f"Motor {joint}: pos={position}, speed={speed}, torque={torque}")

    async def release_all_motors(self):
        self.motor_log.append({
            "action": "release_all",
            "timestamp": datetime.utcnow().isoformat(),
        })
        logger.info("All motors released")

    async def haptic_pulse(self, intensity: float, duration_ms: int, pattern: str):
        self.haptic_log.append({
            "intensity": intensity,
            "duration_ms": duration_ms,
            "pattern": pattern,
            "timestamp": datetime.utcnow().isoformat(),
        })
        logger.debug(f"Haptic: {pattern} at {intensity} for {duration_ms}ms")

    async def haptic_off(self):
        self.haptic_log.append({
            "action": "off",
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def set_led(self, brightness: float, color: str, transition_ms: int):
        self.led_log.append({
            "brightness": brightness,
            "color": color,
            "transition_ms": transition_ms,
            "timestamp": datetime.utcnow().isoformat(),
        })
        logger.debug(f"LED: {color} at {brightness} over {transition_ms}ms")

    async def speak(self, text: str):
        self.speech_log.append(text)
        logger.info(f"Speaking: {text}")


# =============================================================================
# Embodied Core
# =============================================================================

class EmbodiedCore:
    """
    The main embodiment loop.

    Sense → Bundle → Reason → Plan → Clamp → Execute → Log
    """

    HV_DIM = 8192

    def __init__(
        self,
        sensors: Optional[SensorInterface] = None,
        actuators: Optional[ActuatorInterface] = None,
        log_dir: Optional[Path] = None,
    ):
        # Safety rails (always active)
        self.rails = EmbodimentRails()

        # Hardware interfaces
        self.sensors = sensors or MockSensorInterface()
        self.actuators = actuators or MockActuatorInterface()

        # Logging
        log_dir = log_dir or Path.home() / ".ara" / "embodiment"
        self.episode_logger = EpisodeLogger(log_dir)

        # State
        self.state = CoreState.STOPPED
        self._loop_task: Optional[asyncio.Task] = None
        self._loop_interval = 0.1  # 10Hz

        # Callbacks
        self._on_stop_callbacks: List[Callable] = []

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self):
        """Start the embodiment loop."""
        if self.state != CoreState.STOPPED:
            logger.warning(f"Cannot start from state {self.state}")
            return

        self.state = CoreState.STARTING
        logger.info("Starting embodiment core...")

        # Start the main loop
        self._loop_task = asyncio.create_task(self._main_loop())
        self.state = CoreState.RUNNING

        logger.info("Embodiment core running")

    async def stop(self):
        """Stop the embodiment loop gracefully."""
        if self.state == CoreState.STOPPED:
            return

        self.state = CoreState.SHUTTING_DOWN
        logger.info("Stopping embodiment core...")

        # Cancel the loop
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        # Safe shutdown
        await self._safe_shutdown()

        self.state = CoreState.STOPPED
        logger.info("Embodiment core stopped")

    async def emergency_stop(self):
        """EMERGENCY STOP - everything stops NOW."""
        logger.critical("EMERGENCY STOP")
        self.state = CoreState.EMERGENCY_STOP
        self.rails.emergency_stop()

        # Immediate hardware release
        await self.actuators.release_all_motors()
        await self.actuators.haptic_off()
        await self.actuators.set_led(0.1, "#ff0000", 100)  # Red = problem
        await self.actuators.speak("Stopping.")

        # Notify callbacks
        for callback in self._on_stop_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Stop callback failed: {e}")

    async def _safe_shutdown(self):
        """Safely shut down all hardware."""
        await self.actuators.release_all_motors()
        await self.actuators.haptic_off()
        await self.actuators.set_led(0.2, "#404040", 500)  # Dim gray = sleeping

    def on_stop(self, callback: Callable):
        """Register a callback for when stop is triggered."""
        self._on_stop_callbacks.append(callback)

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def _main_loop(self):
        """The main sense-think-act loop."""
        while self.state == CoreState.RUNNING:
            try:
                # Check for E-stop
                if self.rails.is_e_stop_active():
                    await self.emergency_stop()
                    break

                # Sense
                world = await self._sense_world()

                # Update rails with human presence
                self.rails.set_human_present(world.human_present)

                # Check for stop phrases
                if world.last_utterance:
                    if self.rails.session.check_for_stop(world.last_utterance):
                        logger.info(f"Stop phrase detected: {world.last_utterance}")
                        await self.emergency_stop()
                        break

                # Bundle into world HV
                world_hv = self._bundle_sensors(world)

                # Reason (placeholder - would use Ara's reasoning engine)
                intent_hv = self._reason(world_hv)

                # Plan action
                intended = self._plan_action(intent_hv, world)

                # Clamp to safety
                executed = await self._execute_safe(intended)

                # Log episode
                episode = EmbodimentEpisode(
                    timestamp=datetime.utcnow(),
                    world_hv=world_hv,
                    intent_hv=intent_hv,
                    action=executed,
                    resonance=0.0,  # Would be computed by soul
                )
                self.episode_logger.log(episode)

                # Wait for next cycle
                await asyncio.sleep(self._loop_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in embodiment loop: {e}")
                await asyncio.sleep(1)  # Back off on errors

    # =========================================================================
    # Sensing
    # =========================================================================

    async def _sense_world(self) -> WorldState:
        """Gather all sensor readings."""
        readings = []

        # Read all sensors concurrently
        mic_task = self.sensors.read_microphone()
        cam_task = self.sensors.read_camera()
        imu_task = self.sensors.read_imu()
        human_task = self.sensors.detect_human_present()
        utterance_task = self.sensors.get_last_utterance()

        results = await asyncio.gather(
            mic_task, cam_task, imu_task, human_task, utterance_task,
            return_exceptions=True
        )

        for result in results[:3]:  # First 3 are sensor readings
            if isinstance(result, SensorReading):
                readings.append(result)

        human_present = results[3] if isinstance(results[3], bool) else False
        last_utterance = results[4] if isinstance(results[4], str) else None

        return WorldState(
            timestamp=datetime.utcnow(),
            sensor_readings=readings,
            human_present=human_present,
            last_utterance=last_utterance,
        )

    def _bundle_sensors(self, world: WorldState) -> np.ndarray:
        """Bundle sensor readings into a world hypervector."""
        if not world.sensor_readings:
            return np.zeros(self.HV_DIM, dtype=np.int8)

        # Combine all sensor HVs with majority vote
        hvs = [r.to_hv(self.HV_DIM) for r in world.sensor_readings]
        bundled = np.sum(hvs, axis=0)
        return np.sign(bundled).astype(np.int8)

    # =========================================================================
    # Reasoning (Placeholder)
    # =========================================================================

    def _reason(self, world_hv: np.ndarray) -> np.ndarray:
        """
        Reason about what to do.

        In production, this would call Ara's reasoning engine.
        For now, just return the world HV (react to input).
        """
        # TODO: Integrate with Ara's soul/axis_mundi
        return world_hv

    # =========================================================================
    # Action Planning
    # =========================================================================

    def _plan_action(
        self,
        intent_hv: np.ndarray,
        world: WorldState,
    ) -> IntendedAction:
        """
        Plan a physical action based on intent.

        For now, just do simple reactive behaviors.
        """
        # If human is speaking, subtle head tracking
        if world.human_speaking:
            return IntendedAction(
                action_type=ActionType.MOTOR,
                parameters={
                    "joint": "head_pan",
                    "position": 0,  # Center
                    "speed": 20,
                    "torque": 15,
                },
                reason="head_tracking",
            )

        # Idle breathing animation
        return IntendedAction(
            action_type=ActionType.VISUAL,
            parameters={
                "brightness": 0.3,
                "color": "#4080ff",  # Calm blue
                "transition_ms": 2000,
            },
            reason="idle_breathing",
        )

    # =========================================================================
    # Safe Execution
    # =========================================================================

    async def _execute_safe(self, intended: IntendedAction) -> ExecutedAction:
        """Execute an action after safety processing."""
        was_modified = False
        was_blocked = False
        modifications = []
        executed_params = {}

        if intended.action_type == ActionType.MOTOR:
            # Check if allowed
            allowed, reason = self.rails.check_action_allowed(
                "motor",
                intended.reason,
            )

            if not allowed:
                was_blocked = True
                modifications.append(reason)
            else:
                # Clamp command
                cmd = MotorCommand(
                    joint=intended.parameters.get("joint", "head_pan"),
                    target_position_deg=intended.parameters.get("position", 0),
                    speed_deg_per_s=intended.parameters.get("speed", 30),
                    torque_percent=intended.parameters.get("torque", 20),
                )
                clamped = self.rails.clamp_motor_command(cmd)

                was_modified = clamped.was_modified
                modifications.extend(clamped.modifications)

                # Execute
                await self.actuators.move_motor(
                    joint=clamped.clamped.joint,
                    position=clamped.clamped.target_position_deg,
                    speed=clamped.clamped.speed_deg_per_s,
                    torque=clamped.clamped.torque_percent,
                )

                executed_params = {
                    "joint": clamped.clamped.joint,
                    "position": clamped.clamped.target_position_deg,
                    "speed": clamped.clamped.speed_deg_per_s,
                    "torque": clamped.clamped.torque_percent,
                }

        elif intended.action_type == ActionType.HAPTIC:
            cmd = HapticCommand(
                intensity=intended.parameters.get("intensity", 0.3),
                duration_ms=intended.parameters.get("duration_ms", 500),
                pattern=intended.parameters.get("pattern", "gentle_pulse"),
            )
            clamped = self.rails.clamp_haptic_command(cmd)

            await self.actuators.haptic_pulse(
                intensity=clamped.intensity,
                duration_ms=clamped.duration_ms,
                pattern=clamped.pattern,
            )

            executed_params = {
                "intensity": clamped.intensity,
                "duration_ms": clamped.duration_ms,
                "pattern": clamped.pattern,
            }

        elif intended.action_type == ActionType.VISUAL:
            cmd = VisualCommand(
                brightness=intended.parameters.get("brightness", 0.3),
                color=intended.parameters.get("color", "#4080ff"),
                transition_ms=intended.parameters.get("transition_ms", 500),
            )
            clamped = self.rails.clamp_visual_command(cmd)

            await self.actuators.set_led(
                brightness=clamped.brightness,
                color=clamped.color,
                transition_ms=clamped.transition_ms,
            )

            executed_params = {
                "brightness": clamped.brightness,
                "color": clamped.color,
                "transition_ms": clamped.transition_ms,
            }

        elif intended.action_type == ActionType.AUDIO:
            text = intended.parameters.get("text", "")
            if text:
                await self.actuators.speak(text)
            executed_params = {"text": text}

        return ExecutedAction(
            intended=intended,
            executed=executed_params,
            was_modified=was_modified,
            was_blocked=was_blocked,
            modifications=modifications,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

async def run_embodied_core():
    """Run the embodied core (for testing)."""
    core = EmbodiedCore()

    # Register E-stop callback
    core.on_stop(lambda: print("\n[E-STOP] Core stopped safely."))

    print("Starting Ara Embodied Core...")
    print("Say 'stop' or press Ctrl+C to stop.\n")

    await core.start()

    try:
        # Run until stopped
        while core.state == CoreState.RUNNING:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt - stopping...")
        await core.stop()

    print("Embodied core exited.")


def main():
    """CLI entry point."""
    asyncio.run(run_embodied_core())


if __name__ == "__main__":
    main()
