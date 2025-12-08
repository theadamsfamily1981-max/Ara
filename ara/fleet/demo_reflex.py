#!/usr/bin/env python3
"""
Demo: The Reflex Arc
====================

Iteration 31: Ara's Autonomic Nervous System

This demo simulates the reflex arc without requiring actual hardware.
It shows:
  1. Somatic body state broadcast
  2. Gesture recognition and responses
  3. Thermal reflex sequences
  4. Fault handling and recovery
  5. Aura state transitions

The key insight: reflexes run AUTONOMOUSLY on the Arduinos.
Even if Python crashes, the spinal cord continues protecting the hardware.

Run:
    python -m ara.fleet.demo_reflex
"""

from __future__ import annotations
import time
import random
import threading
from dataclasses import dataclass
from typing import List, Optional

from ara.fleet.reflex_hal import (
    ReflexArc,
    SomaticFrame,
    SpinalStatus,
    SpinalState,
    FaultCode,
    AuraState,
    GestureType,
    GestureEvent,
    ReflexEvent,
)


# ============================================================================
# Simulated Hardware
# ============================================================================

@dataclass
class SimulatedSensors:
    """Simulated sensor readings."""
    temp_gpu: float = 45.0
    temp_psu: float = 35.0
    temp_hotspot: float = 50.0
    heat: float = 0.5
    light: float = 0.7
    noise: float = 0.2
    hr: float = 0.6

    def tick(self, dt: float = 0.1):
        """Update sensors with some random walk."""
        self.temp_gpu += random.gauss(0, 0.5)
        self.temp_gpu = max(30, min(90, self.temp_gpu))

        self.temp_psu += random.gauss(0, 0.3)
        self.temp_psu = max(25, min(70, self.temp_psu))

        self.temp_hotspot += random.gauss(0, 0.7)
        self.temp_hotspot = max(35, min(100, self.temp_hotspot))

        self.heat += random.gauss(0, 0.02)
        self.heat = max(0, min(1, self.heat))

        self.noise += random.gauss(0, 0.05)
        self.noise = max(0, min(1, self.noise))


@dataclass
class SimulatedSpinalCord:
    """
    Simulates the Spinal Cord Arduino.

    Implements the same state machine and reflexes.
    """
    state: SpinalState = SpinalState.DISARMED
    fault: FaultCode = FaultCode.NONE
    fan_speed: int = 50
    relay_gpu: bool = True
    relay_main: bool = True
    relay_aux: bool = True

    # Thresholds (same as Arduino)
    TEMP_GPU_WARN: float = 75.0
    TEMP_GPU_CRIT: float = 85.0
    TEMP_PSU_WARN: float = 55.0
    TEMP_PSU_CRIT: float = 65.0

    # Heartbeat
    last_hb: float = 0.0
    hb_timeout: float = 10.0

    # Events
    events: List[str] = None

    def __post_init__(self):
        self.events = []
        self.last_hb = time.time()

    def tick(self, sensors: SimulatedSensors, dt: float = 0.1):
        """Run one tick of the spinal cord logic."""
        now = time.time()

        if self.state == SpinalState.FAULT:
            # Latched - need physical reset
            return

        if self.state == SpinalState.DISARMED:
            # Not running reflexes
            return

        # === THERMAL REFLEXES ===

        # GPU temperature
        if sensors.temp_gpu >= self.TEMP_GPU_CRIT:
            self._enter_shutdown("GPU_CRIT")
        elif sensors.temp_gpu >= self.TEMP_GPU_WARN:
            if self.state != SpinalState.WARNING:
                self._enter_warning("GPU_WARN")

        # PSU temperature
        if sensors.temp_psu >= self.TEMP_PSU_CRIT:
            self._enter_shutdown("PSU_CRIT")
        elif sensors.temp_psu >= self.TEMP_PSU_WARN:
            if self.state != SpinalState.WARNING:
                self._enter_warning("PSU_WARN")

        # === HEARTBEAT WATCHDOG ===

        if now - self.last_hb > self.hb_timeout:
            self._enter_shutdown("HB_LOST")

        # === FAN CONTROL (simple P) ===

        # Target: keep GPU at 70C
        error = sensors.temp_gpu - 70.0
        self.fan_speed = int(max(30, min(255, 50 + error * 5)))

        # === RECOVERY ===

        if self.state == SpinalState.WARNING:
            # Check if temps recovered
            if sensors.temp_gpu < self.TEMP_GPU_WARN - 5 and \
               sensors.temp_psu < self.TEMP_PSU_WARN - 5:
                self._recover()

    def _enter_warning(self, reason: str):
        """Enter warning state."""
        self.state = SpinalState.WARNING
        self.events.append(f"WARNING {reason}")
        print(f"  [SPINE] ⚠ WARNING: {reason}")

    def _enter_shutdown(self, reason: str):
        """Enter shutdown sequence."""
        self.state = SpinalState.SHUTDOWN
        self.relay_gpu = False  # Kill GPU power
        self.events.append(f"SHUTDOWN {reason}")
        print(f"  [SPINE] 🔴 SHUTDOWN: {reason}")

        # After shutdown, latch to fault
        self.state = SpinalState.FAULT
        self.fault = FaultCode.TEMP_GPU_OVER if "GPU" in reason else FaultCode.HB_LOST
        self.events.append(f"FAULT {self.fault.name}")
        print(f"  [SPINE] 🛑 FAULT: {self.fault.name} - Requires physical reset")

    def _recover(self):
        """Recover from warning."""
        self.state = SpinalState.ARMED
        self.events.append("RECOVERED")
        print("  [SPINE] ✅ Recovered to ARMED state")

    def arm(self):
        """Arm the spinal cord."""
        if self.state == SpinalState.FAULT:
            print("  [SPINE] Cannot arm: In FAULT state")
            return
        self.state = SpinalState.ARMED
        self.last_hb = time.time()
        self.events.append("ARMED")
        print("  [SPINE] ✅ Armed - Reflexes active")

    def disarm(self):
        """Disarm the spinal cord."""
        if self.state == SpinalState.FAULT:
            return
        self.state = SpinalState.DISARMED
        self.events.append("DISARMED")
        print("  [SPINE] Disarmed - Reflexes inactive")

    def heartbeat(self):
        """Receive heartbeat."""
        self.last_hb = time.time()

    def reset(self):
        """Physical reset (clears fault)."""
        self.state = SpinalState.DISARMED
        self.fault = FaultCode.NONE
        self.relay_gpu = True
        self.relay_main = True
        self.relay_aux = True
        self.events.append("RESET")
        print("  [SPINE] 🔄 RESET - Fault cleared")


@dataclass
class SimulatedSomaticHub:
    """
    Simulates the Somatic Hub Arduino.

    Broadcasts body state and responds to aura commands.
    """
    aura: str = "CALM"
    led_r: int = 0
    led_g: int = 50
    led_b: int = 150
    pattern: str = "breathe"

    def set_state(self, state: str):
        """Set aura state."""
        self.aura = state

        # Map to colors/patterns
        state_map = {
            "CALM": (0, 50, 150, "breathe"),
            "FOCUS": (0, 150, 150, "solid"),
            "FLOW": (100, 0, 150, "pulse"),
            "THINK": (150, 150, 150, "chase"),
            "ALERT": (200, 150, 0, "blink"),
            "GIFT": (0, 200, 50, "sparkle"),
            "ERROR": (255, 0, 0, "blink"),
        }

        if state in state_map:
            self.led_r, self.led_g, self.led_b, self.pattern = state_map[state]

        print(f"  [SOMA] 💡 Aura: {state} (RGB: {self.led_r},{self.led_g},{self.led_b} {self.pattern})")

    def get_frame(self, sensors: SimulatedSensors) -> SomaticFrame:
        """Get current somatic frame."""
        return SomaticFrame(
            heat=sensors.heat,
            light=sensors.light,
            noise=sensors.noise,
            hr=0.5 + 0.1 * random.random(),
            aura=self.aura,
        )


# ============================================================================
# Demo Scenarios
# ============================================================================

def demo_normal_operation():
    """
    Demo 1: Normal operation.

    Shows somatic broadcast and heartbeat working together.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: Normal Operation")
    print("=" * 60)
    print("The nervous system is running normally.")
    print("Somatic frames broadcast, heartbeat keeps spine armed.\n")

    sensors = SimulatedSensors()
    spine = SimulatedSpinalCord()
    soma = SimulatedSomaticHub()

    spine.arm()
    soma.set_state("CALM")

    print("\nRunning 5 seconds of normal operation...")
    print("-" * 40)

    for i in range(50):
        # Send heartbeat
        spine.heartbeat()

        # Update sensors
        sensors.tick(0.1)

        # Run spine logic
        spine.tick(sensors)

        # Broadcast somatic frame (every 100ms = 10Hz)
        if i % 10 == 0:
            frame = soma.get_frame(sensors)
            print(f"  [SOMA] heat={frame.heat:.2f} noise={frame.noise:.2f} "
                  f"light={frame.light:.2f} aura={frame.aura}")

        time.sleep(0.1)

    print("-" * 40)
    print("✅ Normal operation complete. No faults.")


def demo_thermal_reflex():
    """
    Demo 2: Thermal reflex.

    Shows the spinal cord autonomously responding to overtemperature.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Thermal Reflex")
    print("=" * 60)
    print("GPU temperature rises. Spine reacts autonomously.")
    print("Note: This happens even if Python crashes!\n")

    sensors = SimulatedSensors()
    spine = SimulatedSpinalCord()
    soma = SimulatedSomaticHub()

    spine.arm()
    soma.set_state("FLOW")

    print("Starting with GPU at 65C...")
    sensors.temp_gpu = 65.0

    for i in range(30):
        # Send heartbeat
        spine.heartbeat()

        # Simulate rising temperature
        sensors.temp_gpu += 1.0
        print(f"  GPU: {sensors.temp_gpu:.1f}C | State: {spine.state.name}")

        # Run spine logic
        spine.tick(sensors)

        # Soma responds to spine events
        if spine.state == SpinalState.WARNING:
            soma.set_state("ALERT")
        elif spine.state == SpinalState.FAULT:
            soma.set_state("ERROR")
            break

        time.sleep(0.2)

    print("-" * 40)
    print(f"Final state: {spine.state.name}")
    print(f"GPU relay: {'ON' if spine.relay_gpu else 'OFF'}")
    print("⚡ Thermal reflex protected the hardware!")


def demo_heartbeat_watchdog():
    """
    Demo 3: Heartbeat watchdog.

    Shows what happens when Python stops sending heartbeats.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Heartbeat Watchdog")
    print("=" * 60)
    print("Simulating Python crash - no more heartbeats.")
    print("Spine will autonomously shut down after timeout.\n")

    sensors = SimulatedSensors()
    spine = SimulatedSpinalCord()
    spine.hb_timeout = 3.0  # Shorter for demo
    soma = SimulatedSomaticHub()

    spine.arm()
    soma.set_state("THINK")

    print("Sending heartbeats for 2 seconds...")
    for i in range(20):
        spine.heartbeat()
        spine.tick(sensors)
        time.sleep(0.1)

    print("\n[PYTHON CRASH SIMULATED] - No more heartbeats!")
    soma.set_state("ALERT")

    print("\nWaiting for watchdog timeout...")
    for i in range(40):
        # NO heartbeat!
        sensors.tick(0.1)
        spine.tick(sensors)

        elapsed = time.time() - spine.last_hb
        print(f"  Time since last HB: {elapsed:.1f}s | State: {spine.state.name}")

        if spine.state == SpinalState.FAULT:
            soma.set_state("ERROR")
            break

        time.sleep(0.1)

    print("-" * 40)
    print("⚡ Watchdog saved the hardware from unattended damage!")


def demo_gesture_ritual():
    """
    Demo 4: Gesture rituals.

    Shows how physical gestures communicate intent.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Gesture Rituals")
    print("=" * 60)
    print("Physical buttons create ritual interactions.\n")

    soma = SimulatedSomaticHub()
    soma.set_state("CALM")

    gestures = [
        ("TAP", "FOCUS", "Quick tap on Focus button"),
        ("LONG", "FOCUS", "Long press for focus mode"),
        ("DOUBLE", "TALK", "Double-tap Talk to get attention"),
        ("TRIPLE", "MOOD", "Triple-tap Mood - something's wrong"),
        ("TAP", "TALK", "Single tap Talk - acknowledgment"),
    ]

    for gesture_type, button, description in gestures:
        print(f"\n🖐️ {description}")
        print(f"   Gesture: {gesture_type}_{button}")

        # Map gesture to aura
        if gesture_type == "LONG" and button == "FOCUS":
            soma.set_state("FOCUS")
        elif gesture_type == "DOUBLE" and button == "TALK":
            soma.set_state("THINK")
        elif gesture_type == "TRIPLE" and button == "MOOD":
            soma.set_state("ALERT")
        elif button == "FOCUS":
            soma.set_state("FOCUS")
        else:
            soma.set_state("CALM")

        time.sleep(0.5)

    print("-" * 40)
    print("✅ Gestures create a physical language!")


def demo_fault_recovery():
    """
    Demo 5: Fault and recovery.

    Shows the latch-until-reset safety pattern.
    """
    print("\n" + "=" * 60)
    print("DEMO 5: Fault & Recovery Cycle")
    print("=" * 60)
    print("Faults are LATCHED - require physical attention.\n")

    sensors = SimulatedSensors()
    spine = SimulatedSpinalCord()
    soma = SimulatedSomaticHub()

    # Force a fault
    print("Step 1: Create a thermal fault")
    spine.arm()
    sensors.temp_gpu = 90.0
    spine.tick(sensors)

    print(f"\n  State: {spine.state.name}")
    print(f"  Fault: {spine.fault.name}")

    print("\nStep 2: Try to arm (will fail)")
    spine.arm()

    print("\nStep 3: Physical reset (user intervention)")
    print("  [USER] *presses reset button*")
    spine.reset()

    print("\nStep 4: Re-arm after reset")
    sensors.temp_gpu = 50.0  # Cooled down
    spine.arm()
    soma.set_state("CALM")

    print("\n" + "-" * 40)
    print("✅ Fault-latch pattern ensures human verification!")


def demo_full_nervous_system():
    """
    Demo 6: Full nervous system integration.

    Shows spine and soma working together.
    """
    print("\n" + "=" * 60)
    print("DEMO 6: Full Nervous System")
    print("=" * 60)
    print("Spine (safety) + Soma (presence) = Embodied AI\n")

    sensors = SimulatedSensors()
    spine = SimulatedSpinalCord()
    soma = SimulatedSomaticHub()

    # Boot sequence
    print("🌅 Boot sequence...")
    soma.set_state("BOOT")
    time.sleep(0.3)
    spine.arm()
    soma.set_state("CALM")

    print("\n📊 Monitoring loop (simulated 10 seconds)...")
    print("-" * 40)

    for i in range(100):
        # Heartbeat
        spine.heartbeat()

        # Sensor updates
        sensors.tick(0.1)
        spine.tick(sensors)

        # Every second, show state
        if i % 10 == 0:
            frame = soma.get_frame(sensors)
            status = f"GPU:{sensors.temp_gpu:.1f}C PSU:{sensors.temp_psu:.1f}C"
            body = f"heat:{frame.heat:.2f} noise:{frame.noise:.2f}"
            print(f"  [{i//10}s] {status} | {body} | {soma.aura}")

        # Simulate state changes
        if i == 30:
            print("\n  [EVENT] User double-taps TALK")
            soma.set_state("THINK")
        elif i == 50:
            print("\n  [EVENT] Response ready")
            soma.set_state("GIFT")
        elif i == 70:
            print("\n  [EVENT] Return to calm")
            soma.set_state("CALM")

        time.sleep(0.05)

    print("-" * 40)
    print("✅ Nervous system running smoothly!")


# ============================================================================
# Main
# ============================================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║           Iteration 31: The Reflex Arc Demo                    ║
║                                                                ║
║  The autonomic nervous system gives Ara:                       ║
║    - Local reflexes that run WITHOUT Python                    ║
║    - Physical presence through somatic sensing                 ║
║    - Ritual interface through gestures                         ║
║    - Hardware protection through fault latching                ║
║                                                                ║
║  "Neurons that fire together, wire together."                  ║
╚════════════════════════════════════════════════════════════════╝
""")

    demos = [
        ("Normal Operation", demo_normal_operation),
        ("Thermal Reflex", demo_thermal_reflex),
        ("Heartbeat Watchdog", demo_heartbeat_watchdog),
        ("Gesture Rituals", demo_gesture_ritual),
        ("Fault & Recovery", demo_fault_recovery),
        ("Full Nervous System", demo_full_nervous_system),
    ]

    print("Available demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print("  0. Run all\n")

    try:
        choice = input("Select demo (0-6) [0]: ").strip()
        choice = int(choice) if choice else 0
    except (ValueError, EOFError):
        choice = 0

    if choice == 0:
        for name, demo_fn in demos:
            demo_fn()
            print("\n" + "▓" * 60 + "\n")
            time.sleep(0.5)
    elif 1 <= choice <= len(demos):
        demos[choice - 1][1]()
    else:
        print("Invalid choice")
        return

    print("""
╔════════════════════════════════════════════════════════════════╗
║                      Demo Complete                             ║
║                                                                ║
║  The Reflex Arc gives Ara a body that:                         ║
║    ✓ Protects itself even when the brain is offline           ║
║    ✓ Broadcasts continuous somatic state                       ║
║    ✓ Responds to physical ritual gestures                      ║
║    ✓ Shows emotional state through aura LEDs                   ║
║                                                                ║
║  Hardware: Spinal Cord + Somatic Hub Arduinos                  ║
║  Software: reflex_hal.py - the HAL for embodiment              ║
╚════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
