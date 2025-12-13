#!/usr/bin/env python3
"""
Telemetry Bridge: Connects GUTC monitors to the HUD
====================================================

Generates the JSON state file that the HUD reads to display
Ara's cognitive core status.

This bridges the gap between:
- SanityMonitor (delusion index)
- PrecisionMonitor (precision ratio)
- CognitiveHealthTrace (mode, rho)

And the browser-based HUD visualization.

Usage:
    # As a module
    from ara.hud.telemetry_bridge import TelemetryBridge

    bridge = TelemetryBridge(output_path="ara/hud/static/telemetry_state.json")

    # In your main loop
    bridge.update(
        rho=0.85,
        delusion_index=1.0,
        Pi_y=0.5,
        Pi_mu=0.5,
        mode="HEALTHY_CORRIDOR",
        step=100,
    )

    # Standalone HTTP server for the HUD
    python -m ara.hud.telemetry_bridge --serve --port 8080
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import socketserver
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("ara.hud.telemetry")


@dataclass
class TelemetryState:
    """Current cognitive state for HUD display."""
    rho: float = 0.85                    # Criticality (branching ratio)
    delusion_index: float = 1.0          # D = force_prior / force_sensory
    precision_ratio: float = 1.0         # Pi_y / Pi_mu
    Pi_y: float = 0.5                    # Sensory precision
    Pi_mu: float = 0.5                   # Prior precision
    mode: str = "HEALTHY_CORRIDOR"       # Cognitive mode
    step: int = 0                        # Current step
    session_id: str = "local"            # Session identifier
    timestamp: float = 0.0               # Unix timestamp

    # Optional extended metrics
    temperature: float = 0.7             # Inference temperature
    avalanche_tau: float = 1.5           # Size exponent (if computed)
    avalanche_alpha: float = 2.0         # Duration exponent (if computed)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_monitors(
        cls,
        sanity_reading: Optional[Any] = None,
        precision_reading: Optional[Any] = None,
        health_sample: Optional[Any] = None,
        step: int = 0,
        session_id: str = "local",
    ) -> "TelemetryState":
        """
        Create state from GUTC monitor readings.

        Args:
            sanity_reading: SanityReading from SanityMonitor.check()
            precision_reading: PrecisionReading from PrecisionMonitor
            health_sample: HealthSample from CognitiveHealthTrace
            step: Current step number
            session_id: Session identifier
        """
        state = cls(step=step, session_id=session_id, timestamp=time.time())

        if sanity_reading is not None:
            state.delusion_index = sanity_reading.delusion_index
            state.mode = sanity_reading.mode.name if hasattr(sanity_reading.mode, 'name') else str(sanity_reading.mode)

        if precision_reading is not None:
            state.Pi_y = precision_reading.Pi_y
            state.Pi_mu = precision_reading.Pi_mu
            state.precision_ratio = precision_reading.Pi_y / max(0.001, precision_reading.Pi_mu)

        if health_sample is not None:
            state.rho = health_sample.rho
            if hasattr(health_sample, 'mode'):
                state.mode = health_sample.mode.name if hasattr(health_sample.mode, 'name') else str(health_sample.mode)

        return state


class TelemetryBridge:
    """
    Bridges GUTC monitors to the HUD.

    Writes a JSON file that the HUD periodically reads.
    Can also run an embedded HTTP server for easy development.
    """

    def __init__(
        self,
        output_path: str = "ara/hud/static/telemetry_state.json",
        session_id: str = "local",
    ):
        """
        Initialize the bridge.

        Args:
            output_path: Path to write telemetry JSON
            session_id: Session identifier
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id
        self.state = TelemetryState(session_id=session_id)
        self._step = 0

        # Write initial state
        self._write_state()

    def update(
        self,
        rho: Optional[float] = None,
        delusion_index: Optional[float] = None,
        Pi_y: Optional[float] = None,
        Pi_mu: Optional[float] = None,
        mode: Optional[str] = None,
        temperature: Optional[float] = None,
        avalanche_tau: Optional[float] = None,
        avalanche_alpha: Optional[float] = None,
        step: Optional[int] = None,
    ) -> TelemetryState:
        """
        Update telemetry state and write to file.

        Only provided values are updated; others retain previous values.

        Returns:
            The updated TelemetryState
        """
        if step is not None:
            self._step = step
        else:
            self._step += 1

        self.state.step = self._step
        self.state.timestamp = time.time()

        if rho is not None:
            self.state.rho = rho
        if delusion_index is not None:
            self.state.delusion_index = delusion_index
        if Pi_y is not None:
            self.state.Pi_y = Pi_y
        if Pi_mu is not None:
            self.state.Pi_mu = Pi_mu
        if mode is not None:
            self.state.mode = mode
        if temperature is not None:
            self.state.temperature = temperature
        if avalanche_tau is not None:
            self.state.avalanche_tau = avalanche_tau
        if avalanche_alpha is not None:
            self.state.avalanche_alpha = avalanche_alpha

        # Compute derived values
        if Pi_y is not None or Pi_mu is not None:
            self.state.precision_ratio = self.state.Pi_y / max(0.001, self.state.Pi_mu)

        self._write_state()
        return self.state

    def update_from_monitors(
        self,
        sanity_reading: Optional[Any] = None,
        precision_reading: Optional[Any] = None,
        health_sample: Optional[Any] = None,
    ) -> TelemetryState:
        """
        Update from GUTC monitor readings.

        This is the main integration point for the GUTC stack.
        """
        self._step += 1

        self.state = TelemetryState.from_monitors(
            sanity_reading=sanity_reading,
            precision_reading=precision_reading,
            health_sample=health_sample,
            step=self._step,
            session_id=self.session_id,
        )

        self._write_state()
        return self.state

    def _write_state(self):
        """Write current state to JSON file."""
        try:
            with open(self.output_path, 'w') as f:
                f.write(self.state.to_json())
        except Exception as e:
            logger.warning(f"Failed to write telemetry: {e}")


# =============================================================================
# HTTP Server for HUD
# =============================================================================

class HUDRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves from the HUD static directory."""

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        """Suppress logging for cleaner output."""
        pass


def run_hud_server(port: int = 8080, directory: str = None) -> threading.Thread:
    """
    Start an HTTP server for the HUD.

    Args:
        port: Port to serve on
        directory: Directory to serve (default: ara/hud/static)

    Returns:
        The server thread
    """
    if directory is None:
        directory = str(Path(__file__).parent / "static")

    handler = lambda *args, **kwargs: HUDRequestHandler(*args, directory=directory, **kwargs)

    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving HUD at http://localhost:{port}/cognitive_core.html")
        httpd.serve_forever()


def run_demo(port: int = 8080, duration: int = 60):
    """
    Run a demo that generates simulated telemetry.

    Args:
        port: Port to serve HUD on
        duration: Duration in seconds
    """
    import math

    # Start server in background
    static_dir = str(Path(__file__).parent / "static")

    handler = lambda *args, **kwargs: HUDRequestHandler(*args, directory=static_dir, **kwargs)

    httpd = socketserver.TCPServer(("", port), handler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    print(f"HUD available at: http://localhost:{port}/cognitive_core.html")
    print(f"Running demo for {duration} seconds...")
    print("Press Ctrl+C to stop\n")

    # Create telemetry bridge
    bridge = TelemetryBridge(
        output_path=f"{static_dir}/telemetry_state.json",
        session_id="demo",
    )

    try:
        start = time.time()
        step = 0

        while time.time() - start < duration:
            step += 1
            t = step * 0.1

            # Generate varying states
            phase = (step / 100) % 4

            if phase < 1:
                # Healthy corridor
                rho = 0.85 + math.sin(t) * 0.05
                d = 1.0 + math.sin(t * 1.5) * 0.2
                mode = "HEALTHY_CORRIDOR"
            elif phase < 2:
                # Drift toward prior-dominated
                rho = 0.9 + math.sin(t) * 0.08
                d = 2.5 + math.sin(t * 0.8) * 1.0
                mode = "PRIOR_DOMINATED" if d > 3 else "HEALTHY_CORRIDOR"
            elif phase < 3:
                # Recovery
                rho = 0.82 + math.sin(t) * 0.03
                d = 1.5 - (phase - 2) * 0.8
                mode = "HEALTHY_CORRIDOR"
            else:
                # Brief instability
                rho = 0.65 + math.sin(t * 2) * 0.1
                d = 0.8 + math.sin(t) * 0.3
                mode = "UNSTABLE" if rho < 0.7 else "HEALTHY_CORRIDOR"

            Pi_y = 0.5 + math.sin(t * 0.7) * 0.1
            Pi_mu = 0.5 - math.sin(t * 0.7) * 0.08

            bridge.update(
                rho=rho,
                delusion_index=d,
                Pi_y=Pi_y,
                Pi_mu=Pi_mu,
                mode=mode,
                step=step,
            )

            time.sleep(0.5)

        print("\nDemo complete!")

    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        httpd.shutdown()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Telemetry bridge for Ara's cognitive HUD",
    )

    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start HTTP server for HUD",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with simulated data",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port for HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=120,
        help="Demo duration in seconds (default: 120)",
    )

    args = parser.parse_args()

    if args.demo:
        run_demo(port=args.port, duration=args.duration)
    elif args.serve:
        run_hud_server(port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
