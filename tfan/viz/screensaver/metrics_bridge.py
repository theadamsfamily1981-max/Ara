#!/usr/bin/env python3
"""
T-FAN Metrics Bridge

Serves metrics from ~/.cache/tfan/metrics.json as JSON HTTP endpoint
for the topology screensaver.

Usage:
    python metrics_bridge.py --port 9101
    # Screensaver connects to: http://localhost:9101/metrics
"""

import argparse
import json
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

METRICS_FILE = Path.home() / ".cache" / "tfan" / "metrics.json"

class MetricsHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler that serves metrics as JSON."""

    def do_GET(self):
        if self.path in ['/metrics', '/']:
            try:
                if METRICS_FILE.exists():
                    with open(METRICS_FILE, 'r') as f:
                        data = json.load(f)
                else:
                    data = {
                        "training_active": False,
                        "step": 0,
                        "accuracy": 0.0,
                        "latency_ms": 0.0,
                        "hypervolume": 0.0,
                        "epr_cv": 0.0,
                        "topo_gap": 0.0,
                        "energy": 0.0
                    }

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            except Exception as e:
                self.send_error(500, f"Error reading metrics: {e}")
        else:
            self.send_error(404, "Not found")

    def log_message(self, format, *args):
        """Suppress request logging unless verbose."""
        pass

def run_server(port=9101, verbose=False):
    """Run metrics bridge server."""
    if verbose:
        # Re-enable logging
        MetricsHandler.log_message = BaseHTTPRequestHandler.log_message

    server = HTTPServer(('0.0.0.0', port), MetricsHandler)
    print(f"✓ Metrics bridge running on http://localhost:{port}/metrics")
    print(f"✓ Reading from: {METRICS_FILE}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Shutting down metrics bridge")
        server.shutdown()

def main():
    parser = argparse.ArgumentParser(description='T-FAN Metrics Bridge Server')
    parser.add_argument('--port', type=int, default=9101,
                       help='HTTP server port (default: 9101)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable request logging')
    args = parser.parse_args()

    # Ensure metrics directory exists
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Create default metrics file if missing
    if not METRICS_FILE.exists():
        with open(METRICS_FILE, 'w') as f:
            json.dump({
                "training_active": False,
                "step": 0,
                "accuracy": 0.0,
                "latency_ms": 0.0,
                "hypervolume": 0.0,
                "epr_cv": 0.0,
                "topo_gap": 0.0,
                "energy": 0.0,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, f, indent=2)
        print(f"✓ Created default metrics file: {METRICS_FILE}")

    run_server(port=args.port, verbose=args.verbose)

if __name__ == "__main__":
    main()
