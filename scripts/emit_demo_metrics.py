#!/usr/bin/env python
"""
Demo metrics emitter for Topo-Attention Glass.

Runs a WebSocket server streaming synthetic metrics for testing the dashboard.

Usage:
    python scripts/emit_demo_metrics.py
    python scripts/emit_demo_metrics.py --port 8765 --interval 0.5
"""

import argparse
import logging

from tfan.viz import VizStream


def main():
    parser = argparse.ArgumentParser(description="Emit demo metrics for visualization")
    parser.add_argument('--host', type=str, default='0.0.0.0', help="Host to bind to")
    parser.add_argument('--port', type=int, default=8765, help="Port to bind to")
    parser.add_argument('--interval', type=float, default=0.5, help="Update interval (seconds)")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║          Topo-Attention Glass Demo Emitter               ║
╚═══════════════════════════════════════════════════════════╝

WebSocket server: ws://{args.host}:{args.port}
Update interval: {args.interval}s
Demo mode: ON

Connect your dashboard to view live metrics!
Press Ctrl+C to stop.
""")

    # Create and run stream
    stream = VizStream(
        host=args.host,
        port=args.port,
        update_interval=args.interval,
        demo_mode=True
    )

    try:
        stream.run()
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")


if __name__ == '__main__':
    main()
