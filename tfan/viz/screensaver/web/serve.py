#!/usr/bin/env python3
"""
Simple HTTP server for T-FAN WebGL Screensaver

Serves static files with proper CORS headers for local development.
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = 8080

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS headers."""

    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Custom log format
        print(f"[{self.log_date_time_string()}] {format % args}")

def main():
    # Change to web directory
    web_dir = Path(__file__).parent
    os.chdir(web_dir)

    # Start server
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print("=" * 60)
        print("🌌 T-FAN Topology Screensaver - WebGL Server")
        print("=" * 60)
        print(f"\n✓ Server running at: http://localhost:{PORT}")
        print(f"✓ Serving from: {web_dir}")
        print("\n📖 Open in browser:")
        print(f"   http://localhost:{PORT}/index.html")
        print("\n⚡ Features:")
        print("   - Live WebSocket metrics (if API running)")
        print("   - HTTP polling fallback")
        print("   - 4 visualization modes")
        print("   - Fullscreen support")
        print("\n🔧 Controls:")
        print("   M - Cycle modes")
        print("   P - Pause/unpause")
        print("   H - Help overlay")
        print("   F - Fullscreen")
        print("\nPress Ctrl+C to stop\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n✓ Server stopped")
            sys.exit(0)

if __name__ == "__main__":
    main()
