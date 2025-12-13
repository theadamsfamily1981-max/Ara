#!/usr/bin/env python3
"""
Cognitive Cockpit Demo Runner

Starts the daemon bridge (which generates simulated telemetry)
and opens the cockpit in a web browser.

Usage:
    python -m hud.cognitive_cockpit.run_demo
    # or
    ./hud/cognitive_cockpit/run_demo.py
"""

import http.server
import logging
import os
import signal
import socketserver
import sys
import threading
import time
import webbrowser
from functools import partial
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hud.cognitive_cockpit.daemon_bridge import CognitiveDaemonBridge
from hud.cognitive_cockpit.telemetry import get_cognitive_telemetry

# Paths
COCKPIT_DIR = Path(__file__).parent
STATE_DIR = Path.home() / ".ara" / "hud"
STATE_FILE = STATE_DIR / "cognitive_state.json"


class CockpitHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for serving cockpit files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(COCKPIT_DIR), **kwargs)

    def log_message(self, format, *args):
        pass  # Suppress logs

    def do_GET(self):
        if self.path.startswith('/cognitive_state.json'):
            self.serve_state()
        else:
            super().do_GET()

    def serve_state(self):
        try:
            if STATE_FILE.exists():
                content = STATE_FILE.read_bytes()
            else:
                content = b'{}'

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("""
╔═══════════════════════════════════════════════════════════════╗
║                  ARA COGNITIVE COCKPIT                        ║
║                                                               ║
║   Real-time visualization of cognitive state:                 ║
║   • ρ (rho) - Criticality / Edge-of-chaos                     ║
║   • D - Delusion Index (prior vs sensory balance)             ║
║   • Π - Precision ratio                                       ║
║   • Avalanche dynamics (power-law cascades)                   ║
║   • Body heat / thermal zones                                 ║
║   • Mental modes (Worker / Scientist / Chill)                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # Ensure state directory exists
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Start daemon bridge
    print("[1/3] Starting cognitive daemon bridge...")
    bridge = CognitiveDaemonBridge()
    bridge.start()
    print(f"      Telemetry output: {STATE_FILE}")

    # Start HTTP server
    print("[2/3] Starting HTTP server...")
    port = 8765
    with socketserver.TCPServer(("127.0.0.1", port), CockpitHandler) as httpd:
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        print(f"      Server running at http://127.0.0.1:{port}")

        # Open browser
        print("[3/3] Opening cockpit in browser...")
        url = f"http://127.0.0.1:{port}/cockpit.html"
        webbrowser.open(url)
        print(f"      Cockpit URL: {url}")

        print("\n" + "="*60)
        print("Cockpit is running! Press Ctrl+C to stop.")
        print("="*60 + "\n")

        # Handle shutdown
        def shutdown(sig, frame):
            print("\nShutting down...")
            bridge.stop()
            httpd.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Keep alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            shutdown(None, None)


if __name__ == "__main__":
    main()
