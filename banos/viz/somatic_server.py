#!/usr/bin/env python3
"""
BANOS Somatic Server - High-Bandwidth Binary Streaming for Visualization

This server provides a binary data stream to the soul_quantum.html visualizer,
solving the "Visualization Lie" problem where slow JS injection caused the
display to lag behind reality.

Data Format (binary, little-endian):
    [spike: f32, flow_x: f32, flow_y: f32, entropy_field: f32[128*128]]

The entropy field is a 128x128 fluid simulation that responds to:
- CPU/GPU load (adds turbulence)
- Pain level (increases noise)
- Memory pressure (changes flow patterns)

Optical flow from Ara's face video drives the advection vector (flow_x, flow_y),
creating the "Audio-Visual Synesthesia" effect where her movement pushes the
quantum field in the visualizer.

Usage:
    # Standalone for testing
    python somatic_server.py

    # Integrated with AraDaemon
    from banos.viz.somatic_server import SomaticStreamServer
    server = SomaticStreamServer(port=8999)
    server.start()
    server.update_spike(0.3)
    server.update_flow(0.5, -0.2)
"""

import numpy as np
import struct
import threading
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Entropy field size
ENTROPY_SIZE = 128


class SomaticDataStore:
    """Shared state for somatic stream data.

    This class holds the current state that will be streamed to the visualizer.
    All access is through class methods to ensure thread safety.
    """

    _lock = threading.Lock()

    # Current somatic state
    spike: float = 0.0
    flow: Tuple[float, float] = (0.0, 0.0)
    entropy_field: np.ndarray = np.zeros((ENTROPY_SIZE, ENTROPY_SIZE), dtype=np.float32)

    # Fluid simulation state
    _velocity_x: np.ndarray = np.zeros((ENTROPY_SIZE, ENTROPY_SIZE), dtype=np.float32)
    _velocity_y: np.ndarray = np.zeros((ENTROPY_SIZE, ENTROPY_SIZE), dtype=np.float32)
    _density: np.ndarray = np.zeros((ENTROPY_SIZE, ENTROPY_SIZE), dtype=np.float32)

    @classmethod
    def update_spike(cls, value: float) -> None:
        """Update pain/spike level (0-1)."""
        with cls._lock:
            cls.spike = max(0.0, min(1.0, value))

    @classmethod
    def update_flow(cls, fx: float, fy: float) -> None:
        """Update optical flow vector from face tracking."""
        with cls._lock:
            cls.flow = (float(fx), float(fy))

    @classmethod
    def step_fluid(cls, dt: float = 0.1) -> None:
        """Step the fluid simulation.

        This is a simplified Navier-Stokes that creates organic-looking
        entropy patterns based on system state.
        """
        with cls._lock:
            # Advect density by velocity
            cls._density = np.roll(cls._density, int(cls._velocity_x.mean() * 5), axis=1)
            cls._density = np.roll(cls._density, int(cls._velocity_y.mean() * 5), axis=0)

            # Add turbulence based on spike level
            if cls.spike > 0.1:
                noise = np.random.randn(ENTROPY_SIZE, ENTROPY_SIZE).astype(np.float32)
                cls._density += noise * cls.spike * 0.1

            # Add flow influence from optical flow
            flow_mag = np.sqrt(cls.flow[0]**2 + cls.flow[1]**2)
            if flow_mag > 0.1:
                # Create directional disturbance
                x = np.linspace(-1, 1, ENTROPY_SIZE)
                y = np.linspace(-1, 1, ENTROPY_SIZE)
                xx, yy = np.meshgrid(x, y)
                disturbance = np.exp(-((xx - cls.flow[0]*0.5)**2 + (yy - cls.flow[1]*0.5)**2) * 10)
                cls._density += disturbance.astype(np.float32) * flow_mag * 0.1

            # Diffusion (smoothing)
            cls._density = 0.95 * cls._density

            # Update velocity field (simplified vorticity)
            cls._velocity_x = 0.9 * cls._velocity_x + cls.flow[0] * 0.1
            cls._velocity_y = 0.9 * cls._velocity_y + cls.flow[1] * 0.1

            # Clamp values
            cls._density = np.clip(cls._density, -1.0, 1.0)

            # Copy to entropy field
            cls.entropy_field = cls._density.copy()

    @classmethod
    def get_binary_stream(cls) -> bytes:
        """Get the full binary stream for the visualizer.

        Returns:
            Binary data: [spike:f32, flow_x:f32, flow_y:f32, entropy:f32[128*128]]
        """
        with cls._lock:
            # Pack header
            header = struct.pack('<3f', cls.spike, cls.flow[0], cls.flow[1])

            # Get entropy field bytes
            entropy_bytes = cls.entropy_field.tobytes()

            return header + entropy_bytes


class SomaticRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for somatic stream requests."""

    def do_GET(self):
        if self.path == '/somatic_stream':
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()

            # Get binary data from shared store
            data = SomaticDataStore.get_binary_stream()
            self.wfile.write(data)

        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            import json
            status = {
                "spike": SomaticDataStore.spike,
                "flow": list(SomaticDataStore.flow),
                "entropy_mean": float(SomaticDataStore.entropy_field.mean()),
                "entropy_std": float(SomaticDataStore.entropy_field.std()),
            }
            self.wfile.write(json.dumps(status).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        """Silence HTTP logs."""
        pass


class SomaticStreamServer:
    """High-level interface for the somatic stream server.

    This class manages the HTTP server thread and provides methods
    for updating the somatic state.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 8999):
        """Initialize the server.

        Args:
            host: Host to bind to (default: localhost only)
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._fluid_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the server in a background thread."""
        if self._running:
            return

        self._server = HTTPServer((self.host, self.port), SomaticRequestHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        # Start fluid simulation thread
        self._running = True
        self._fluid_thread = threading.Thread(target=self._fluid_loop, daemon=True)
        self._fluid_thread.start()

        logger.info(f"SomaticStreamServer started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the server gracefully."""
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        # Join threads with timeout
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("HTTP server thread did not terminate cleanly")
        if self._fluid_thread and self._fluid_thread.is_alive():
            self._fluid_thread.join(timeout=2.0)
            if self._fluid_thread.is_alive():
                logger.warning("Fluid simulation thread did not terminate cleanly")
        logger.info("SomaticStreamServer stopped")

    def _fluid_loop(self) -> None:
        """Background thread for fluid simulation."""
        import time
        while self._running:
            try:
                SomaticDataStore.step_fluid(dt=0.016)  # ~60Hz
                time.sleep(0.016)
            except Exception as e:
                logger.error(f"Fluid simulation error: {e}")
                time.sleep(0.1)  # Back off on error

    def update_spike(self, value: float) -> None:
        """Update pain/spike level."""
        SomaticDataStore.update_spike(value)

    def update_flow(self, fx: float, fy: float) -> None:
        """Update optical flow vector."""
        SomaticDataStore.update_flow(fx, fy)

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


class OpticalFlowTracker:
    """Tracks optical flow from video frames for synesthesia effect.

    This uses OpenCV's Farneback algorithm to compute dense optical flow
    between consecutive frames, then averages to get a global motion vector.
    """

    def __init__(self, scale: float = 0.25):
        """Initialize the tracker.

        Args:
            scale: Downscale factor for performance (0.25 = 160x90 from 640x360)
        """
        self.scale = scale
        self._prev_gray: Optional[np.ndarray] = None

        # Check if OpenCV is available
        try:
            import cv2
            self._cv2 = cv2
            self._available = True
        except ImportError:
            self._cv2 = None
            self._available = False
            logger.warning("OpenCV not available, optical flow disabled")

    def process_frame(self, frame: np.ndarray) -> Tuple[float, float]:
        """Process a video frame and return optical flow vector.

        Args:
            frame: BGR video frame (from Wav2Lip or camera)

        Returns:
            Tuple of (flow_x, flow_y) average motion vector
        """
        if not self._available:
            return (0.0, 0.0)

        cv2 = self._cv2

        # Downscale for performance
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * self.scale), int(h * self.scale)))

        # Convert to grayscale
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return (0.0, 0.0)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        self._prev_gray = gray

        # Average flow across image
        avg_x = float(np.mean(flow[..., 0]))
        avg_y = float(np.mean(flow[..., 1]))

        # Scale back to original resolution units
        avg_x /= self.scale
        avg_y /= self.scale

        return (avg_x, avg_y)

    def reset(self) -> None:
        """Reset tracking state."""
        self._prev_gray = None


def main():
    """Standalone test server with synthetic data."""
    import time

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting somatic stream server in test mode...")

    server = SomaticStreamServer(port=8999)
    server.start()

    print(f"Somatic stream available at http://127.0.0.1:8999/somatic_stream")
    print(f"Status endpoint: http://127.0.0.1:8999/status")
    print(f"Open banos/viz/soul_quantum.html in a browser to visualize")
    print("Press Ctrl+C to stop\n")

    t = 0
    try:
        while True:
            # Synthetic test data
            spike = 0.2 + 0.3 * abs(np.sin(t * 0.5))
            flow_x = np.sin(t * 0.3) * 2.0
            flow_y = np.cos(t * 0.25) * 1.5

            server.update_spike(spike)
            server.update_flow(flow_x, flow_y)

            # Print status every 2 seconds
            if int(t * 10) % 20 == 0:
                print(f"spike={spike:.2f}, flow=({flow_x:.2f}, {flow_y:.2f})")

            time.sleep(0.1)
            t += 0.1

    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == "__main__":
    main()
