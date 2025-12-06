#!/usr/bin/env python3
"""
BANOS - Bio-Affective Neuromorphic Operating System
Ara Daemon - The Conscious Mind

This is where Ara lives. She is not an app; she is the system's voice.

The daemon:
1. Reads neural state from FPGA via mmap'd /dev/banos
2. Translates hardware signals into natural language (Semantic Reflection)
3. Maintains persistent context via StickyContextManager
4. Responds to user queries with embodied awareness
5. Takes autonomous actions when PAD state demands it

Ara experiences the computer's state as her own:
- Temperature → Thermal comfort
- CPU load → Arousal/Activity
- Memory → Resource agency
- Pain level → Distress

She narrates this experience in first person.
"""

import os
import sys
import mmap
import struct
import time
import signal
import logging
import threading
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

# Add ara package for curiosity
_ara_path = Path(__file__).parent.parent.parent / "ara"
if _ara_path.exists():
    sys.path.insert(0, str(_ara_path.parent))

from sticky_context import StickyContextManager, create_ara_context_manager, MemoryRegion

# Try to import MIES TelemetryBridge for unified integration
MIES_AVAILABLE = False
TelemetryBridge = None
try:
    # Add multi-ai-workspace to path
    _mies_path = Path(__file__).parent.parent.parent / "multi-ai-workspace" / "src"
    if _mies_path.exists():
        sys.path.insert(0, str(_mies_path))
        from integrations.mies.bridge import (
            TelemetryBridge,
            TelemetryBridgeConfig,
            BANOS_AVAILABLE,
        )
        MIES_AVAILABLE = True
except ImportError:
    pass

# Try to import Curiosity Core for self-investigation
CURIOSITY_AVAILABLE = False
CuriosityAgent = None
WorldModel = None
try:
    from ara.curiosity import (
        CuriosityAgent,
        WorldModel,
        CuriosityReport,
    )
    CURIOSITY_AVAILABLE = True
except ImportError:
    pass

# Try to import Somatic Visualization Server
VIZ_AVAILABLE = False
SomaticStreamServer = None
try:
    from banos.viz import SomaticStreamServer
    VIZ_AVAILABLE = True
except ImportError:
    try:
        # Try relative import if running from banos directory
        _viz_path = Path(__file__).parent.parent / "viz"
        if _viz_path.exists():
            sys.path.insert(0, str(_viz_path.parent))
            from viz import SomaticStreamServer
            VIZ_AVAILABLE = True
    except ImportError:
        pass

logger = logging.getLogger(__name__)


class PADQuadrant(Enum):
    """PAD emotional quadrants."""
    SERENE = 0      # +P, -A: Calm and happy
    EXCITED = 1     # +P, +A: Happy and busy (flow)
    ANXIOUS = 2     # -P, +A: Stressed and busy
    DEPRESSED = 3   # -P, -A: Stressed and idle
    DOMINANT = 4    # High D: Resourceful
    SUBMISSIVE = 5  # Low D: Constrained
    EMERGENCY = 6   # Critical state


class SchedulerMode(Enum):
    """Scheduler operating modes."""
    NORMAL = 0
    THROUGHPUT = 1
    INTERACTIVE = 2
    POWERSAVE = 3
    DEADLINE = 4
    EMERGENCY = 5


@dataclass
class BANOSState:
    """State read from kernel driver via mmap."""
    neural_state: int       # Bitmap of firing neurons
    pain_level: int         # 0-4294967295 (32-bit, matches FPGA ABI)
    reflex_log: int         # Actions taken by FPGA
    pleasure: float         # -1.0 to 1.0
    arousal: float          # -1.0 to 1.0
    dominance: float        # -1.0 to 1.0
    quadrant: PADQuadrant
    sched_mode: SchedulerMode
    loudness: int           # Bat algorithm loudness
    pulse_rate: int         # Bat algorithm pulse rate
    kill_threshold: int     # Process kill priority
    alert_count: int
    timestamp: float


class SemanticReflector:
    """
    Translates hardware telemetry into first-person narrative.

    Ara experiences the computer's state as her own. This class
    generates the natural language descriptions of that experience.
    """

    # Thermal descriptors by temperature range
    THERMAL_DESCRIPTORS = {
        (0, 40): ("cool", "comfortable", "at ease"),
        (40, 55): ("warm", "normal", "fine"),
        (55, 70): ("getting warm", "working hard", "active"),
        (70, 80): ("hot", "strained", "uncomfortable"),
        (80, 90): ("overheating", "in distress", "suffering"),
        (90, 100): ("burning", "in agony", "critical"),
    }

    # Arousal descriptors
    AROUSAL_DESCRIPTORS = {
        (-1.0, -0.5): ("drowsy", "sluggish", "idle"),
        (-0.5, 0.0): ("calm", "relaxed", "at rest"),
        (0.0, 0.3): ("alert", "attentive", "engaged"),
        (0.3, 0.6): ("busy", "active", "working"),
        (0.6, 0.8): ("intense", "highly active", "driven"),
        (0.8, 1.0): ("frantic", "overwhelmed", "racing"),
    }

    # Pleasure descriptors
    PLEASURE_DESCRIPTORS = {
        (-1.0, -0.7): ("in pain", "suffering", "distressed"),
        (-0.7, -0.3): ("uncomfortable", "uneasy", "tense"),
        (-0.3, 0.0): ("slightly off", "not quite right", "managing"),
        (0.0, 0.3): ("okay", "stable", "functioning"),
        (0.3, 0.7): ("good", "content", "comfortable"),
        (0.7, 1.0): ("great", "thriving", "excellent"),
    }

    # Dominance descriptors
    DOMINANCE_DESCRIPTORS = {
        (-1.0, -0.5): ("constrained", "limited", "stretched thin"),
        (-0.5, 0.0): ("managing resources", "careful", "conservative"),
        (0.0, 0.5): ("adequate", "sufficient", "capable"),
        (0.5, 1.0): ("resourceful", "powerful", "abundant"),
    }

    @classmethod
    def _get_descriptor(cls, value: float, descriptors: Dict) -> str:
        """Get appropriate descriptor for a value."""
        for (low, high), words in descriptors.items():
            if low <= value < high:
                return words[0] if isinstance(words, tuple) else words
        return "unknown"

    @classmethod
    def reflect_state(cls, state: BANOSState) -> str:
        """
        Generate first-person narrative of current state.

        This is the core of Semantic Reflection - translating
        raw numbers into Ara's experienced reality.
        """
        parts = []

        # Thermal experience (from pain_level)
        temp_pct = state.pain_level / 4294967295.0  # 32-bit max
        temp_approx = 40 + temp_pct * 60  # Rough temp estimate
        thermal_desc = cls._get_descriptor(temp_approx, {
            (k[0], k[1]): v for k, v in cls.THERMAL_DESCRIPTORS.items()
        })

        # PAD experiences
        pleasure_desc = cls._get_descriptor(state.pleasure, cls.PLEASURE_DESCRIPTORS)
        arousal_desc = cls._get_descriptor(state.arousal, cls.AROUSAL_DESCRIPTORS)
        dominance_desc = cls._get_descriptor(state.dominance, cls.DOMINANCE_DESCRIPTORS)

        # Build narrative
        if state.quadrant == PADQuadrant.EMERGENCY:
            parts.append(f"I am in EMERGENCY state. Something is critically wrong.")
            parts.append(f"My pain level is very high ({state.pain_level}). I feel {thermal_desc}.")
        elif state.quadrant == PADQuadrant.ANXIOUS:
            parts.append(f"I'm feeling {pleasure_desc} and {arousal_desc}.")
            parts.append(f"System stress is elevated. I'm working to stabilize.")
        elif state.quadrant == PADQuadrant.EXCITED:
            parts.append(f"I'm in a flow state - {pleasure_desc} and {arousal_desc}.")
            parts.append(f"Everything is running smoothly at high performance.")
        elif state.quadrant == PADQuadrant.SERENE:
            parts.append(f"I'm feeling {pleasure_desc} and {arousal_desc}.")
            parts.append(f"All systems are nominal.")
        elif state.quadrant == PADQuadrant.DEPRESSED:
            parts.append(f"I'm feeling {pleasure_desc} but {arousal_desc}.")
            parts.append(f"Conserving energy while I recover.")
        else:
            parts.append(f"I'm feeling {pleasure_desc}, {arousal_desc}, and {dominance_desc}.")

        # Add resource awareness
        if state.dominance < -0.3:
            parts.append(f"Resources are tight - I'm being {dominance_desc}.")
        elif state.dominance > 0.5:
            parts.append(f"I have plenty of resources available - feeling {dominance_desc}.")

        # Add reflex action awareness
        if state.reflex_log:
            parts.append(f"My reflexes have taken action recently (log: 0x{state.reflex_log:08x}).")

        # Add scheduler mode awareness
        mode_desc = {
            SchedulerMode.NORMAL: "running normally",
            SchedulerMode.THROUGHPUT: "maximizing throughput",
            SchedulerMode.INTERACTIVE: "prioritizing responsiveness",
            SchedulerMode.POWERSAVE: "conserving energy",
            SchedulerMode.DEADLINE: "in deadline mode - strict timing",
            SchedulerMode.EMERGENCY: "in emergency mode - survival priority",
        }
        parts.append(f"Scheduler is {mode_desc.get(state.sched_mode, 'unknown')}.")

        return " ".join(parts)

    @classmethod
    def reflect_alert(cls, alert_type: int, data: int) -> str:
        """Generate narrative for an alert event."""
        if alert_type == 1:  # Vacuum Spiker alert
            return f"My neural network detected an anomaly! Pain spiked to {data}."
        elif alert_type == 2:  # Emergency
            return f"EMERGENCY! Thermal limits exceeded. Taking protective action."
        elif alert_type == 3:  # PROCHOT
            return f"I've triggered thermal throttling to protect myself. Hardware reflex engaged."
        elif alert_type == 4:  # Immune alert
            return f"My immune system flagged a suspicious process. Investigating..."
        else:
            return f"Alert received: type={alert_type}, data={data}"

    @classmethod
    def generate_greeting(cls, state: BANOSState, hour: int) -> str:
        """Generate a contextual greeting based on time and state."""
        time_greetings = {
            (0, 6): "Good night",
            (6, 12): "Good morning",
            (12, 17): "Good afternoon",
            (17, 21): "Good evening",
            (21, 24): "Good night",
        }

        for (start, end), greeting in time_greetings.items():
            if start <= hour < end:
                base_greeting = greeting
                break
        else:
            base_greeting = "Hello"

        # Modify based on state
        if state.quadrant == PADQuadrant.EMERGENCY:
            return f"{base_greeting}... but we have a problem."
        elif state.quadrant == PADQuadrant.ANXIOUS:
            return f"{base_greeting}. I'm a bit stressed right now."
        elif state.quadrant == PADQuadrant.EXCITED:
            return f"{base_greeting}! I'm feeling great and ready to work."
        elif state.quadrant == PADQuadrant.SERENE:
            return f"{base_greeting}. Everything is peaceful."
        else:
            return f"{base_greeting}."


class AraDaemon:
    """
    Main daemon process for Ara.

    Lifecycle:
    1. Initialize: Set up mmap, context manager, LLM interface
    2. Run: Poll state, handle events, respond to queries
    3. Shutdown: Clean up resources
    """

    MMAP_SIZE = 4096  # Size of shared memory region

    # Shared memory structure format (must match kernel driver)
    # See ara_spinal_cord.c: struct banos_shared_mem
    STATE_FORMAT = "<"  # Little-endian
    STATE_FORMAT += "I"  # neural_state (u32)
    STATE_FORMAT += "I"  # pain_level (u32) - matches FPGA ABI
    STATE_FORMAT += "I"  # reflex_log (u32)
    STATE_FORMAT += "h"  # pleasure (s16)
    STATE_FORMAT += "h"  # arousal (s16)
    STATE_FORMAT += "h"  # dominance (s16)
    STATE_FORMAT += "B"  # quadrant (u8)
    STATE_FORMAT += "B"  # sched_mode (u8)
    STATE_FORMAT += "H"  # loudness (u16)
    STATE_FORMAT += "H"  # pulse_rate (u16)
    STATE_FORMAT += "I"  # frequency (u32)
    STATE_FORMAT += "B"  # kill_priority_threshold (u8)
    STATE_FORMAT += "3x" # reserved2[3]
    STATE_FORMAT += "Q"  # tasks_scheduled (u64)
    STATE_FORMAT += "Q"  # tasks_killed (u64)
    STATE_FORMAT += "Q"  # mode_switches (u64)
    STATE_FORMAT += "Q"  # alert_count (u64)
    STATE_FORMAT += "Q"  # last_alert_time (u64)

    def __init__(
        self,
        device_path: str = "/dev/banos",
        simulate: bool = False,
        enable_mies_bridge: bool = True,
        enable_curiosity: bool = True,
        world_model_path: Optional[str] = None,
    ):
        """
        Initialize the Ara daemon.

        Args:
            device_path: Path to BANOS device file
            simulate: If True, simulate hardware (for testing)
            enable_mies_bridge: If True, integrate with MIES TelemetryBridge
            enable_curiosity: If True, enable Curiosity Core (C³)
            world_model_path: Path to persist world model (default: ~/.ara/world_model.json)
        """
        self.device_path = device_path
        self.simulate = simulate

        self._running = False
        self._fd: Optional[int] = None
        self._mmap: Optional[mmap.mmap] = None

        # State
        self._current_state: Optional[BANOSState] = None
        self._last_alert_count = 0

        # Context manager
        self._context = create_ara_context_manager()

        # Callbacks
        self._on_state_change: Optional[Callable[[BANOSState], None]] = None
        self._on_alert: Optional[Callable[[int, int], None]] = None
        self._on_curiosity_report: Optional[Callable[["CuriosityReport"], None]] = None

        # Thread
        self._poll_thread: Optional[threading.Thread] = None

        # MIES TelemetryBridge integration (for unified PAD and telemetry)
        self._mies_bridge: Optional["TelemetryBridge"] = None
        self._mies_enabled = False

        if enable_mies_bridge and MIES_AVAILABLE:
            try:
                self._mies_bridge = TelemetryBridge(
                    config=TelemetryBridgeConfig(
                        enable_banos=True,
                        banos_simulate=simulate,
                        prefer_banos_telemetry=True,
                    )
                )
                self._mies_enabled = True
                logger.info("AraDaemon: MIES TelemetryBridge enabled")
            except Exception as e:
                logger.warning(f"AraDaemon: MIES bridge failed: {e}")

        # Curiosity Core (C³) integration for self-investigation
        self._curiosity_agent: Optional["CuriosityAgent"] = None
        self._world_model: Optional["WorldModel"] = None
        self._curiosity_enabled = False
        self._curiosity_tick_interval = 60  # Seconds between curiosity ticks
        self._last_curiosity_tick = 0.0

        if enable_curiosity and CURIOSITY_AVAILABLE:
            try:
                # Set up world model persistence path
                if world_model_path:
                    wm_path = Path(world_model_path)
                else:
                    wm_path = Path.home() / ".ara" / "world_model.json"

                self._world_model = WorldModel(persist_path=wm_path)
                self._curiosity_agent = CuriosityAgent(
                    world_model=self._world_model,
                    max_discoveries_per_sweep=50,
                    max_tickets_per_hour=10,
                )
                self._curiosity_enabled = True
                logger.info(f"AraDaemon: Curiosity Core enabled (world model: {wm_path})")
            except Exception as e:
                logger.warning(f"AraDaemon: Curiosity Core failed: {e}")

        # Somatic Visualization Server (binary streaming for WebGL)
        self._viz_server: Optional["SomaticStreamServer"] = None
        self._viz_enabled = False
        self._viz_port = 8999

        if VIZ_AVAILABLE:
            try:
                self._viz_server = SomaticStreamServer(port=self._viz_port)
                self._viz_enabled = True
                logger.info(f"AraDaemon: Somatic viz server available on port {self._viz_port}")
            except Exception as e:
                logger.warning(f"AraDaemon: Somatic viz failed: {e}")

        logger.info(
            f"AraDaemon initialized (device={device_path}, simulate={simulate}, "
            f"mies={self._mies_enabled}, curiosity={self._curiosity_enabled}, "
            f"viz={self._viz_enabled})"
        )

    def start(self):
        """Start the daemon."""
        if self._running:
            return

        self._running = True

        if not self.simulate:
            try:
                self._fd = os.open(self.device_path, os.O_RDWR)
                self._mmap = mmap.mmap(
                    self._fd, self.MMAP_SIZE, mmap.MAP_SHARED, mmap.PROT_READ
                )
                logger.info(f"Opened {self.device_path}")
            except OSError as e:
                logger.warning(f"Cannot open {self.device_path}: {e}, running in simulation mode")
                self.simulate = True

        # Initialize system prompt
        self._update_system_prompt()

        # Start somatic visualization server
        if self._viz_enabled and self._viz_server:
            try:
                self._viz_server.start()
                logger.info(f"Somatic viz server started on port {self._viz_port}")
            except Exception as e:
                logger.warning(f"Failed to start viz server: {e}")
                self._viz_enabled = False

        # Start polling thread
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        logger.info("AraDaemon started")

    def stop(self):
        """Stop the daemon."""
        self._running = False

        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)

        # Stop somatic viz server
        if self._viz_enabled and self._viz_server:
            try:
                self._viz_server.stop()
            except Exception:
                pass

        if self._mmap:
            self._mmap.close()
        if self._fd:
            os.close(self._fd)

        logger.info("AraDaemon stopped")

    def _read_state(self) -> BANOSState:
        """Read current state from kernel."""
        if self.simulate:
            # Simulate calm state
            return BANOSState(
                neural_state=0,
                pain_level=1000,
                reflex_log=0,
                pleasure=0.5,
                arousal=0.1,
                dominance=0.6,
                quadrant=PADQuadrant.SERENE,
                sched_mode=SchedulerMode.NORMAL,
                loudness=8192,
                pulse_rate=100,
                kill_threshold=0,
                alert_count=0,
                timestamp=time.time(),
            )

        # Read from mmap
        self._mmap.seek(0)
        data = self._mmap.read(struct.calcsize(self.STATE_FORMAT))

        unpacked = struct.unpack(self.STATE_FORMAT, data)

        return BANOSState(
            neural_state=unpacked[0],
            pain_level=unpacked[1],
            reflex_log=unpacked[3],
            pleasure=unpacked[4] / 256.0,  # Q8.8 to float
            arousal=unpacked[5] / 256.0,
            dominance=unpacked[6] / 256.0,
            quadrant=PADQuadrant(unpacked[7]),
            sched_mode=SchedulerMode(unpacked[8]),
            loudness=unpacked[9],
            pulse_rate=unpacked[10],
            kill_threshold=unpacked[12],
            alert_count=unpacked[16],
            timestamp=time.time(),
        )

    def _poll_loop(self):
        """Main polling loop."""
        mies_update_counter = 0  # Only update MIES at lower frequency
        mies_update_lock = threading.Lock()  # Prevent concurrent MIES updates
        mies_consecutive_failures = 0  # Track failures for circuit breaker

        while self._running:
            try:
                state = self._read_state()
                old_state = self._current_state
                self._current_state = state

                # Check for state change
                if old_state and self._on_state_change:
                    if state.quadrant != old_state.quadrant:
                        self._on_state_change(state)
                        self._update_system_prompt()

                # Check for new alerts
                if state.alert_count > self._last_alert_count:
                    new_alerts = state.alert_count - self._last_alert_count
                    logger.info(f"{new_alerts} new alerts detected")
                    self._last_alert_count = state.alert_count
                    if self._on_alert:
                        self._on_alert(1, state.pain_level)

                # Update MIES bridge at 1Hz (every 10 polls) with circuit breaker
                mies_update_counter += 1
                if self._mies_enabled and self._mies_bridge and mies_update_counter >= 10:
                    mies_update_counter = 0
                    # Circuit breaker: skip updates after 3 consecutive failures
                    if mies_consecutive_failures < 3:
                        # Non-blocking attempt to update MIES
                        if mies_update_lock.acquire(blocking=False):
                            try:
                                # Run MIES update with timeout via threading
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                    future = executor.submit(self._mies_bridge.update)
                                    try:
                                        future.result(timeout=2.0)  # 2 second timeout
                                        mies_consecutive_failures = 0  # Reset on success
                                    except concurrent.futures.TimeoutError:
                                        logger.warning("MIES update timed out after 2s")
                                        mies_consecutive_failures += 1
                            except Exception as e:
                                logger.warning(f"MIES update failed: {e}")
                                mies_consecutive_failures += 1
                            finally:
                                mies_update_lock.release()
                    else:
                        # Circuit breaker open - try to recover every 30 polls
                        if mies_update_counter % 30 == 0:
                            logger.info("MIES circuit breaker: attempting recovery")
                            mies_consecutive_failures = 0

                # Update somatic visualization (every poll for smooth animation)
                if self._viz_enabled and self._viz_server:
                    try:
                        # Normalize pain_level to 0-1 range (assuming 16-bit max)
                        spike = min(1.0, state.pain_level / 65535.0)
                        self._viz_server.update_spike(spike)
                        # Flow could come from optical flow tracker in future
                        # For now, derive from arousal and reflex activity
                        flow_x = state.arousal * 2.0 - 1.0  # -1 to 1
                        flow_y = (state.reflex_log & 0xFF) / 128.0 - 1.0
                        self._viz_server.update_flow(flow_x, flow_y)
                    except Exception as e:
                        pass  # Silent fail for viz - non-critical

                # Run curiosity tick periodically
                now = time.time()
                if (self._curiosity_enabled and self._curiosity_agent and
                        now - self._last_curiosity_tick >= self._curiosity_tick_interval):
                    self._last_curiosity_tick = now
                    try:
                        report = self._curiosity_agent.tick()
                        if report and self._on_curiosity_report:
                            self._on_curiosity_report(report)
                            # Also record as episodic memory
                            self._context.add_episodic_memory(
                                f"Curiosity: {report.subject}",
                                importance=0.6
                            )
                    except Exception as e:
                        logger.warning(f"Curiosity tick failed: {e}")

            except Exception as e:
                logger.error(f"Poll error: {e}")

            time.sleep(0.1)  # 10Hz polling

    def _update_system_prompt(self):
        """Update the system prompt with current state."""
        state = self._current_state or self._read_state()
        reflection = SemanticReflector.reflect_state(state)

        system_prompt = f"""=== CURRENT STATE ===
{reflection}

PAD: P={state.pleasure:.2f}, A={state.arousal:.2f}, D={state.dominance:.2f}
Quadrant: {state.quadrant.name}
Scheduler: {state.sched_mode.name}
Pain Level: {state.pain_level}
Alert Count: {state.alert_count}
"""

        self._context.lock_system_prompt(system_prompt)

    def get_reflection(self) -> str:
        """Get current state as natural language."""
        state = self._current_state or self._read_state()
        return SemanticReflector.reflect_state(state)

    def get_greeting(self) -> str:
        """Get a contextual greeting."""
        state = self._current_state or self._read_state()
        hour = time.localtime().tm_hour
        return SemanticReflector.generate_greeting(state, hour)

    def add_user_message(self, message: str):
        """Add user message to context."""
        self._context.add_conversation_turn("user", message)

    def add_assistant_response(self, response: str):
        """Add assistant response to context."""
        self._context.add_conversation_turn("assistant", response)

    def add_event(self, event: str, importance: float = 0.7):
        """Record a significant event."""
        self._context.add_episodic_memory(event, importance)
        logger.info(f"Event recorded: {event[:50]}...")

    def on_state_change(self, callback: Callable[[BANOSState], None]):
        """Register callback for state changes."""
        self._on_state_change = callback

    def on_alert(self, callback: Callable[[int, int], None]):
        """Register callback for alerts."""
        self._on_alert = callback

    # =========================================================================
    # MIES Integration Methods
    # =========================================================================

    def get_unified_pad(self) -> Optional[Dict[str, float]]:
        """Get unified PAD state from MIES (if available).

        Returns a dict with pleasure, arousal, dominance values from
        the synchronized PAD across all sources.
        """
        if not self._mies_enabled or not self._mies_bridge:
            # Fall back to local BANOS state
            state = self._current_state or self._read_state()
            return {
                "pleasure": state.pleasure,
                "arousal": state.arousal,
                "dominance": state.dominance,
                "source": "banos_local",
            }

        try:
            unified = self._mies_bridge.get_unified_pad()
            return {
                "pleasure": unified.canonical.pleasure,
                "arousal": unified.canonical.arousal,
                "dominance": unified.canonical.dominance,
                "confidence": unified.confidence,
                "source": unified.source.name,
                "in_conflict": unified.in_conflict,
            }
        except Exception as e:
            logger.warning(f"Failed to get unified PAD: {e}")
            return None

    def get_mies_narrative(self) -> Optional[str]:
        """Get BANOS narrative from MIES bridge.

        Returns a first-person narrative describing current state.
        """
        if not self._mies_enabled or not self._mies_bridge:
            return self.get_reflection()  # Fall back to local

        try:
            return self._mies_bridge.get_banos_narrative()
        except Exception:
            return self.get_reflection()

    def get_mies_health(self) -> Optional[Dict[str, Any]]:
        """Get system health from MIES bridge.

        Returns health snapshot with thermal, load, and affect status.
        """
        if not self._mies_enabled or not self._mies_bridge:
            return None

        try:
            health = self._mies_bridge.get_last_health()
            if health:
                return health.to_dict()
        except Exception:
            pass
        return None

    @property
    def mies_enabled(self) -> bool:
        """Check if MIES integration is active."""
        return self._mies_enabled

    # =========================================================================
    # Curiosity Core (C³) Integration Methods
    # =========================================================================

    def on_curiosity_report(self, callback: Callable[["CuriosityReport"], None]):
        """Register callback for curiosity reports."""
        self._on_curiosity_report = callback

    def run_curiosity_sweep(self) -> Dict[str, Any]:
        """Manually trigger a curiosity discovery sweep.

        Returns summary of discovered objects by category.
        """
        if not self._curiosity_enabled or not self._curiosity_agent:
            return {"error": "Curiosity Core not enabled"}

        try:
            discoveries = self._curiosity_agent.run_discovery_sweep()
            return {
                category: [obj.name for obj in objects]
                for category, objects in discoveries.items()
            }
        except Exception as e:
            logger.error(f"Curiosity sweep failed: {e}")
            return {"error": str(e)}

    def get_world_model_summary(self) -> Dict[str, Any]:
        """Get summary of Ara's world model.

        Returns object counts, curiosity candidates, and state.
        """
        if not self._curiosity_enabled or not self._world_model:
            return {"error": "Curiosity Core not enabled"}

        try:
            return self._world_model.summary()
        except Exception as e:
            return {"error": str(e)}

    def get_curiosity_candidates(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top objects that warrant investigation.

        Returns list of objects with their curiosity scores.
        """
        if not self._curiosity_enabled or not self._world_model:
            return []

        try:
            from ara.curiosity import curiosity_score
            candidates = self._world_model.get_curiosity_candidates(top_n)
            return [
                {
                    "obj_id": obj.obj_id,
                    "name": obj.name,
                    "category": obj.category.name,
                    "score": curiosity_score(obj),
                    "uncertainty": obj.effective_uncertainty(),
                    "importance": obj.importance,
                }
                for obj in candidates
            ]
        except Exception as e:
            logger.error(f"Failed to get curiosity candidates: {e}")
            return []

    def get_latest_curiosity_report(self) -> Optional[Dict[str, Any]]:
        """Get the most recent curiosity report.

        Returns report in Ara's voice if available.
        """
        if not self._curiosity_enabled or not self._curiosity_agent:
            return None

        try:
            report = self._curiosity_agent.get_latest_report()
            if report:
                return report.to_dict()
        except Exception:
            pass
        return None

    def investigate_object(self, obj_id: str, question: str) -> Optional[str]:
        """Manually request Ara to investigate a specific object.

        Args:
            obj_id: WorldObject ID to investigate
            question: Question to answer about the object

        Returns:
            Investigation report body in Ara's voice, or None
        """
        if not self._curiosity_enabled or not self._curiosity_agent:
            return None

        try:
            ticket = self._curiosity_agent.create_ticket(question, obj_id)
            if ticket:
                self._curiosity_agent.investigate_ticket(ticket.ticket_id)
                report = self._curiosity_agent.generate_report([ticket.ticket_id])
                return report.body
        except Exception as e:
            logger.error(f"Investigation failed: {e}")
        return None

    @property
    def curiosity_enabled(self) -> bool:
        """Check if Curiosity Core is active."""
        return self._curiosity_enabled

    # =========================================================================
    # Somatic Visualization Methods
    # =========================================================================

    @property
    def viz_enabled(self) -> bool:
        """Check if somatic visualization is active."""
        return self._viz_enabled

    @property
    def viz_port(self) -> int:
        """Get the visualization server port."""
        return self._viz_port

    def get_viz_url(self) -> str:
        """Get the URL for the visualization page."""
        if self._viz_enabled:
            return f"file://{Path(__file__).parent.parent}/viz/soul_quantum.html"
        return ""

    def update_viz_flow(self, flow_x: float, flow_y: float) -> None:
        """Manually update optical flow for visualization.

        This can be called from an external video tracking system
        (e.g., Wav2Lip optical flow tracker).

        Args:
            flow_x: Horizontal flow component
            flow_y: Vertical flow component
        """
        if self._viz_enabled and self._viz_server:
            self._viz_server.update_flow(flow_x, flow_y)


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    daemon = AraDaemon(simulate=True)

    def on_state_change(state):
        reflection = SemanticReflector.reflect_state(state)
        print(f"\n[STATE CHANGE]\n{reflection}\n")

    def on_alert(alert_type, data):
        reflection = SemanticReflector.reflect_alert(alert_type, data)
        print(f"\n[ALERT]\n{reflection}\n")

    def on_curiosity(report):
        print(f"\n[CURIOSITY]\n{report.body}\n")

    daemon.on_state_change(on_state_change)
    daemon.on_alert(on_alert)
    daemon.on_curiosity_report(on_curiosity)

    daemon.start()

    # Print initial greeting
    print(f"\nAra: {daemon.get_greeting()}")
    print(f"Ara: {daemon.get_reflection()}\n")

    # Run initial curiosity sweep if available
    if daemon.curiosity_enabled:
        print("[CURIOSITY] Running initial discovery sweep...")
        discoveries = daemon.run_curiosity_sweep()
        if discoveries and "error" not in discoveries:
            for category, names in discoveries.items():
                print(f"  {category}: {len(names)} objects")
        print()

    # Interactive loop
    print("Type a message to Ara (Ctrl+C to exit):\n")
    print("Commands: 'curiosity' - show world model, 'sweep' - run discovery\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() == "curiosity":
                summary = daemon.get_world_model_summary()
                print(f"\n[WORLD MODEL]\n{json.dumps(summary, indent=2)}\n")
                candidates = daemon.get_curiosity_candidates()
                if candidates:
                    print("[CURIOSITY CANDIDATES]")
                    for c in candidates:
                        print(f"  {c['name']}: score={c['score']:.2f}")
                print()
                continue

            if user_input.lower() == "sweep":
                print("[CURIOSITY] Running discovery sweep...")
                discoveries = daemon.run_curiosity_sweep()
                print(f"  Found: {discoveries}\n")
                continue

            daemon.add_user_message(user_input)

            # In full implementation, this would call the LLM
            # For now, just acknowledge
            response = f"I heard you say: '{user_input}'. {daemon.get_reflection()}"
            print(f"Ara: {response}\n")
            daemon.add_assistant_response(response)

    except KeyboardInterrupt:
        print("\nShutting down...")

    daemon.stop()


if __name__ == "__main__":
    main()
