#!/usr/bin/env python3
"""
Cognitive Cockpit Daemon Bridge

Bridges the cognitive telemetry system with existing Ara daemons and
experiment runners. Collects metrics from various sources and feeds
them to the CognitiveHUDTelemetry for display.

Integration points:
- ThermalReflexSimulated / ThermalReflexeBPF (body heat)
- AvalancheLogger / fit_powerlaw.py (avalanche dynamics)
- ActiveInferenceController (mental modes)
- Pulse telemetry (PAD state, metacontrol)
- Experiment runner (training steps)
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading

# Local imports
from .telemetry import (
    CognitiveHUDTelemetry,
    CognitiveState,
    MentalMode,
    get_cognitive_telemetry,
)

logger = logging.getLogger("ara.hud.daemon_bridge")

# Try to import Ara components
try:
    from banos.kernel.thermal_reflex_loader import (
        get_thermal_reflex,
        ThermalAlert,
        THERMAL_WARNING,
        THERMAL_CRITICAL,
    )
    THERMAL_AVAILABLE = True
except ImportError:
    THERMAL_AVAILABLE = False
    logger.debug("Thermal reflex module not available")

try:
    from ara.telemetry import get_telemetry, MetricSnapshot
    PULSE_TELEMETRY_AVAILABLE = True
except ImportError:
    PULSE_TELEMETRY_AVAILABLE = False
    logger.debug("Pulse telemetry module not available")


@dataclass
class BridgeConfig:
    """Configuration for the daemon bridge."""
    update_interval: float = 0.5          # Update every 500ms
    avalanche_fit_interval: float = 10.0  # Refit avalanches every 10s
    thermal_poll_interval: float = 1.0    # Poll thermals every 1s
    enable_simulation: bool = True        # Enable simulated data when real unavailable


class CognitiveDaemonBridge:
    """
    Bridges cognitive metrics from various Ara subsystems to the HUD telemetry.

    Supports both real data from running daemons and simulated data for
    development/testing.
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()
        self.telemetry = get_cognitive_telemetry()

        # Subsystem connections
        self.thermal = None
        self.pulse = None

        # State tracking
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._step = 0

        # Simulation state
        self._sim_rho = 0.85
        self._sim_d = 1.0
        self._sim_temps = [50.0, 48.0, 49.0, 52.0]
        self._sim_mode = MentalMode.WORKER
        self._avalanche_buffer: List[Tuple[int, int]] = []  # (size, count) pairs

        # Callbacks for mode changes from HUD
        self._mode_change_callback: Optional[Callable[[str], None]] = None

    def connect_thermal(self) -> bool:
        """Connect to thermal reflex system."""
        if not THERMAL_AVAILABLE:
            logger.info("Thermal reflex not available, using simulation")
            return False

        try:
            self.thermal = get_thermal_reflex(simulated=True)
            self.thermal.load()
            self.thermal.start()

            # Set up alert callback
            self.thermal.set_callbacks(
                on_alert=self._on_thermal_alert,
                on_glitch=self._on_thermal_glitch,
            )

            logger.info("Connected to thermal reflex system")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to thermal: {e}")
            return False

    def connect_pulse(self) -> bool:
        """Connect to pulse telemetry system."""
        if not PULSE_TELEMETRY_AVAILABLE:
            logger.info("Pulse telemetry not available, using simulation")
            return False

        try:
            self.pulse = get_telemetry()
            logger.info("Connected to pulse telemetry")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to pulse telemetry: {e}")
            return False

    def _on_thermal_alert(self, alert: "ThermalAlert"):
        """Handle thermal alert from eBPF/simulated reflex."""
        logger.debug(f"Thermal alert: {alert.alert_name} at {alert.temperature_c}°C")

        # Update thermal state
        self.telemetry.update_thermal(
            zone_id=alert.zone_id,
            temperature_c=alert.temperature_c,
        )

        # Update reflex state
        if alert.alert_type >= 2:  # DROP or higher
            self.telemetry.update_thermal_state(
                reflex_state="firing",
                throttle_event=True,
            )

    def _on_thermal_glitch(self, valence_delta: float, arousal_delta: float):
        """Handle thermal-induced glitch."""
        logger.warning(f"Thermal glitch triggered: V={valence_delta}, A={arousal_delta}")
        self.telemetry.update_thermal_state(reflex_state="firing", throttle_event=True)

    def start(self):
        """Start the bridge update loop."""
        if self._running:
            return

        # Try to connect to real subsystems
        self.connect_thermal()
        self.connect_pulse()

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

        logger.info("Cognitive daemon bridge started")

    def stop(self):
        """Stop the bridge."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        if self.thermal:
            self.thermal.stop()

        logger.info("Cognitive daemon bridge stopped")

    def _update_loop(self):
        """Main update loop."""
        last_avalanche_fit = 0

        while self._running:
            try:
                self._step += 1

                # Update from real sources or simulate
                self._update_criticality()
                self._update_delusion()
                self._update_precision()
                self._update_thermal()
                self._update_mental_mode()

                # Avalanche updates (less frequent fitting)
                self._update_avalanches()
                if time.time() - last_avalanche_fit > self.config.avalanche_fit_interval:
                    self._fit_avalanches()
                    last_avalanche_fit = time.time()

                # Emit the state update
                self.telemetry.emit(step=self._step)

            except Exception as e:
                logger.error(f"Bridge update error: {e}", exc_info=True)

            time.sleep(self.config.update_interval)

    def _update_criticality(self):
        """Update criticality (ρ) metrics."""
        if self.config.enable_simulation:
            # Simulate ρ fluctuating around edge-of-chaos
            drift = random.gauss(0, 0.01)
            mean_revert = (0.88 - self._sim_rho) * 0.05
            self._sim_rho = max(0.5, min(1.3, self._sim_rho + drift + mean_revert))

            # Occasional excursions
            if random.random() < 0.01:
                self._sim_rho += random.choice([-0.2, 0.2])

            self.telemetry.update_criticality(
                rho=self._sim_rho,
                tau=1.5 + random.gauss(0, 0.05),
                avalanche_count=len(self._avalanche_buffer),
            )

    def _update_delusion(self):
        """Update delusion index (D) metrics."""
        if self.config.enable_simulation:
            # Simulate D fluctuating around balanced
            drift = random.gauss(0, 0.02)
            mean_revert = (0 - math.log10(self._sim_d)) * 0.1
            log_d = math.log10(self._sim_d) + drift + mean_revert
            self._sim_d = 10 ** max(-2, min(2, log_d))

            # Occasional prior/sensory spikes
            if random.random() < 0.005:
                self._sim_d *= random.choice([0.1, 10])

            guardrail = self._sim_d > 5 or self._sim_d < 0.2

            self.telemetry.update_delusion(
                force_prior=self._sim_d,
                force_reality=1.0,
                guardrail_active=guardrail,
                hallucination_flag=self._sim_d > 8,
            )

    def _update_precision(self):
        """Update precision ratio (Π) metrics."""
        if self.config.enable_simulation:
            pi_y = 1.0 + random.gauss(0, 0.1)
            pi_mu = 1.0 + random.gauss(0, 0.1)
            self.telemetry.update_precision(pi_y=pi_y, pi_mu=pi_mu)

    def _update_thermal(self):
        """Update thermal metrics."""
        if self.thermal:
            # Get real stats
            stats = self.thermal.get_stats()
            if 'temperature_c' in stats:
                self.telemetry.update_thermal(0, stats['temperature_c'], "Main")
            return

        if self.config.enable_simulation:
            # Simulate thermal fluctuation
            for i, temp in enumerate(self._sim_temps):
                drift = random.gauss(0, 0.5)
                mean_revert = (50 + i * 2 - temp) * 0.1
                self._sim_temps[i] = max(35, min(90, temp + drift + mean_revert))

                names = ["CPU Package", "CPU Core 0", "CPU Core 1", "GPU"]
                self.telemetry.update_thermal(
                    zone_id=i,
                    temperature_c=self._sim_temps[i],
                    name=names[i] if i < len(names) else f"Zone {i}",
                )

            # Occasional thermal spike
            if random.random() < 0.005:
                spike_zone = random.randint(0, len(self._sim_temps) - 1)
                self._sim_temps[spike_zone] += 15

    def _update_mental_mode(self):
        """Update mental mode metrics."""
        if self.config.enable_simulation:
            # Mode stays stable, weights fluctuate slightly
            mode = self._sim_mode
            presets = {
                MentalMode.WORKER: (0.8, 0.2, 0.6),
                MentalMode.SCIENTIST: (0.4, 0.7, 0.5),
                MentalMode.CHILL: (0.2, 0.2, 0.3),
            }
            base_ext, base_int, base_energy = presets.get(mode, (0.5, 0.5, 0.5))

            self.telemetry.update_mental_mode(
                mode=mode,
                extrinsic_weight=base_ext + random.gauss(0, 0.02),
                intrinsic_weight=base_int + random.gauss(0, 0.02),
                energy_budget=max(0.1, min(1.0, base_energy + random.gauss(0, 0.01))),
            )

    def _update_avalanches(self):
        """Generate simulated avalanche data."""
        if self.config.enable_simulation:
            # Generate power-law distributed avalanche sizes
            tau = 1.5  # Power-law exponent
            num_new = random.randint(0, 3)

            for _ in range(num_new):
                # Generate from power law: P(S) ~ S^(-tau)
                u = random.random()
                s_min, s_max = 1, 1000
                size = int(s_min * (1 - u + u * (s_max/s_min)**(1-tau))**(1/(1-tau)))

                self._avalanche_buffer.append((size, 1))

            # Keep buffer bounded
            if len(self._avalanche_buffer) > 500:
                self._avalanche_buffer = self._avalanche_buffer[-500:]

    def _fit_avalanches(self):
        """Fit power law to avalanche data and update telemetry."""
        if not self._avalanche_buffer:
            return

        # Simple histogram for fitting
        from collections import Counter
        sizes = [s for s, _ in self._avalanche_buffer]
        counts = Counter(sizes)

        # Add to scatter plot
        total = sum(counts.values())
        for size, count in sorted(counts.items()):
            freq = count / total
            if size > 0 and freq > 0:
                self.telemetry.add_avalanche(size, freq)

        # Simple log-log regression for tau
        log_sizes = [math.log10(s) for s in counts.keys() if s > 0]
        log_freqs = [math.log10(c/total) for c in counts.values()]

        if len(log_sizes) >= 3:
            # Least squares fit: log_freq = -tau * log_size + const
            n = len(log_sizes)
            sum_x = sum(log_sizes)
            sum_y = sum(log_freqs)
            sum_xy = sum(x*y for x, y in zip(log_sizes, log_freqs))
            sum_xx = sum(x*x for x in log_sizes)

            denom = n * sum_xx - sum_x**2
            if abs(denom) > 1e-10:
                slope = (n * sum_xy - sum_x * sum_y) / denom
                tau = -slope  # Power law exponent

                # R² calculation
                mean_y = sum_y / n
                ss_tot = sum((y - mean_y)**2 for y in log_freqs)
                intercept = (sum_y - slope * sum_x) / n
                ss_res = sum((y - (slope * x + intercept))**2 for x, y in zip(log_sizes, log_freqs))
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Determine cascade state
                if 1.4 <= tau <= 1.7 and r_squared > 0.8:
                    cascade_state = "stable"
                elif tau < 1.2:
                    cascade_state = "runaway"
                elif tau > 2.0:
                    cascade_state = "fragmented"
                else:
                    cascade_state = "stable"

                self.telemetry.update_avalanche_fit(
                    tau=tau,
                    r_squared=max(0, min(1, r_squared)),
                    cascade_state=cascade_state,
                )

    def set_mental_mode(self, mode: str):
        """Set mental mode from external source (e.g., HUD button)."""
        mode_map = {
            "worker": MentalMode.WORKER,
            "scientist": MentalMode.SCIENTIST,
            "chill": MentalMode.CHILL,
        }

        if mode.lower() in mode_map:
            self._sim_mode = mode_map[mode.lower()]
            logger.info(f"Mental mode set to: {mode}")

            if self._mode_change_callback:
                self._mode_change_callback(mode)

    def set_mode_change_callback(self, callback: Callable[[str], None]):
        """Set callback for mode change events."""
        self._mode_change_callback = callback


def run_bridge_demo():
    """Run a demo of the daemon bridge (for testing)."""
    import signal

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    bridge = CognitiveDaemonBridge()
    bridge.start()

    print("Cognitive Daemon Bridge running...")
    print(f"Writing to: {bridge.telemetry.state_file}")
    print("Press Ctrl+C to stop")

    def handle_sigint(sig, frame):
        print("\nStopping...")
        bridge.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # Keep running
    while True:
        time.sleep(1)


if __name__ == "__main__":
    run_bridge_demo()
