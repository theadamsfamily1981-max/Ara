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

from sticky_context import StickyContextManager, create_ara_context_manager, MemoryRegion

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
    pain_level: int         # 0-65535
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
        temp_pct = state.pain_level / 65535.0
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
    STATE_FORMAT += "H"  # pain_level (u16)
    STATE_FORMAT += "H"  # reserved1 (u16)
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
    ):
        """
        Initialize the Ara daemon.

        Args:
            device_path: Path to BANOS device file
            simulate: If True, simulate hardware (for testing)
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

        # Thread
        self._poll_thread: Optional[threading.Thread] = None

        logger.info(f"AraDaemon initialized (device={device_path}, simulate={simulate})")

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

        # Start polling thread
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        logger.info("AraDaemon started")

    def stop(self):
        """Stop the daemon."""
        self._running = False

        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)

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

    daemon.on_state_change(on_state_change)
    daemon.on_alert(on_alert)

    daemon.start()

    # Print initial greeting
    print(f"\nAra: {daemon.get_greeting()}")
    print(f"Ara: {daemon.get_reflection()}\n")

    # Interactive loop
    print("Type a message to Ara (Ctrl+C to exit):\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
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
