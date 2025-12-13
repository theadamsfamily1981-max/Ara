#!/usr/bin/env python3
"""
Cognitive Cockpit GTK4 Wrapper

Embeds the HTML/JS cockpit in a GTK4 window using WebKitGTK.
Serves the static files via a local HTTP server and handles
D-Bus integration for mode switching.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# GTK4/Adwaita imports
try:
    import gi
    gi.require_version('Gtk', '4.0')
    gi.require_version('Adw', '1')
    gi.require_version('WebKit', '6.0')  # WebKitGTK 6.0 for GTK4
    from gi.repository import Gtk, Adw, GLib, Gio, WebKit
    GTK_AVAILABLE = True
except (ImportError, ValueError) as e:
    GTK_AVAILABLE = False
    print(f"GTK4/WebKit not available: {e}")
    print("Install with: sudo apt install gir1.2-webkit-6.0 libadwaita-1-dev")

logger = logging.getLogger("ara.hud.cognitive_gtk")

# Paths
COCKPIT_DIR = Path(__file__).parent
STATIC_DIR = COCKPIT_DIR / "static"
HTML_FILE = COCKPIT_DIR / "cockpit.html"
STATE_DIR = Path.home() / ".ara" / "hud"
STATE_FILE = STATE_DIR / "cognitive_state.json"


class CockpitHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler for serving cockpit files and state JSON."""

    def __init__(self, *args, state_file: Path = STATE_FILE, **kwargs):
        self.state_file = state_file
        self.directory = str(COCKPIT_DIR)
        super().__init__(*args, directory=self.directory, **kwargs)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        if self.path.startswith('/cognitive_state.json'):
            self.serve_state()
        else:
            super().do_GET()

    def serve_state(self):
        """Serve the current cognitive state JSON."""
        try:
            if self.state_file.exists():
                content = self.state_file.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', len(content))
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(content)
            else:
                # Return default state
                default = json.dumps({
                    "state_label": "INITIALIZING",
                    "criticality": {"rho": 0.85, "tau": 1.5, "state": "edge"},
                    "delusion": {"D": 1.0, "log10_D": 0, "state": "balanced"},
                    "precision": {"pi_y": 1.0, "pi_mu": 1.0, "ratio": 1.0},
                    "avalanches": {"scatter": [], "fitted_tau": 1.5, "cascade_state": "stable"},
                    "thermal": {
                        "zones": [
                            {"zone_id": 0, "name": "CPU Package", "temperature_c": 50, "status": "nominal"},
                        ],
                        "overall_status": "nominal",
                        "reflex_state": "nominal",
                        "fan_mode": "balanced",
                    },
                    "mental_mode": {"mode": "worker", "extrinsic_weight": 0.7, "intrinsic_weight": 0.3, "energy_budget": 0.5},
                    "ticker": {"message": "Waiting for telemetry...", "severity": "info"},
                    "sanity_timeline": [],
                }).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', len(default))
                self.end_headers()
                self.wfile.write(default)
        except Exception as e:
            logger.error(f"Error serving state: {e}")
            self.send_error(500)


class CockpitServer:
    """Local HTTP server for serving the cockpit."""

    def __init__(self, port: int = 0):
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> int:
        """Start the server and return the actual port."""
        handler = partial(CockpitHTTPHandler, state_file=STATE_FILE)
        self.server = HTTPServer(('127.0.0.1', self.port), handler)
        self.port = self.server.server_port

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        logger.info(f"Cockpit server started on port {self.port}")
        return self.port

    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.shutdown()
            self.thread.join(timeout=2.0)
            logger.info("Cockpit server stopped")


class CognitiveHUDWindow(Adw.ApplicationWindow if GTK_AVAILABLE else object):
    """Main GTK4 window containing the WebKit cockpit view."""

    def __init__(self, app: Adw.Application, server_port: int):
        if not GTK_AVAILABLE:
            raise RuntimeError("GTK4/WebKit not available")

        super().__init__(application=app)

        self.server_port = server_port
        self.webview: Optional[WebKit.WebView] = None

        self.setup_window()
        self.setup_webview()

    def setup_window(self):
        """Configure the window."""
        self.set_title("ARA Cognitive Cockpit")
        self.set_default_size(1200, 800)

        # Dark theme
        style_manager = Adw.StyleManager.get_default()
        style_manager.set_color_scheme(Adw.ColorScheme.FORCE_DARK)

        # Header bar
        header = Adw.HeaderBar()
        header.set_show_title(True)

        # Refresh button
        refresh_btn = Gtk.Button(icon_name="view-refresh-symbolic")
        refresh_btn.set_tooltip_text("Refresh cockpit")
        refresh_btn.connect("clicked", self.on_refresh)
        header.pack_start(refresh_btn)

        # Fullscreen button
        fullscreen_btn = Gtk.Button(icon_name="view-fullscreen-symbolic")
        fullscreen_btn.set_tooltip_text("Toggle fullscreen")
        fullscreen_btn.connect("clicked", self.on_fullscreen)
        header.pack_end(fullscreen_btn)

        # Dev tools button
        devtools_btn = Gtk.Button(icon_name="utilities-terminal-symbolic")
        devtools_btn.set_tooltip_text("Open developer tools")
        devtools_btn.connect("clicked", self.on_devtools)
        header.pack_end(devtools_btn)

        # Main box
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.append(header)

        # Content area (will hold webview)
        self.content = Gtk.Box()
        self.content.set_vexpand(True)
        self.content.set_hexpand(True)
        box.append(self.content)

        self.set_content(box)

    def setup_webview(self):
        """Set up the WebKit view."""
        # WebKit settings
        settings = WebKit.Settings()
        settings.set_enable_javascript(True)
        settings.set_enable_developer_extras(True)
        settings.set_allow_file_access_from_file_urls(True)
        settings.set_allow_universal_access_from_file_urls(True)

        # Create webview
        self.webview = WebKit.WebView()
        self.webview.set_settings(settings)
        self.webview.set_vexpand(True)
        self.webview.set_hexpand(True)

        # Connect signals
        self.webview.connect("load-changed", self.on_load_changed)
        self.webview.connect("decide-policy", self.on_decide_policy)

        # Add to content
        self.content.append(self.webview)

        # Load the cockpit
        url = f"http://127.0.0.1:{self.server_port}/cockpit.html"
        self.webview.load_uri(url)

        logger.info(f"Loading cockpit from {url}")

    def on_load_changed(self, webview, event):
        """Handle load events."""
        if event == WebKit.LoadEvent.FINISHED:
            logger.info("Cockpit loaded successfully")

    def on_decide_policy(self, webview, decision, decision_type):
        """Handle navigation decisions."""
        if decision_type == WebKit.PolicyDecisionType.NAVIGATION_ACTION:
            nav_action = decision.get_navigation_action()
            request = nav_action.get_request()
            uri = request.get_uri()

            # Allow local requests only
            if uri.startswith(f"http://127.0.0.1:{self.server_port}"):
                decision.use()
            else:
                logger.warning(f"Blocked navigation to: {uri}")
                decision.ignore()
                return True

        return False

    def on_refresh(self, button):
        """Refresh the cockpit."""
        if self.webview:
            self.webview.reload()

    def on_fullscreen(self, button):
        """Toggle fullscreen."""
        if self.is_fullscreen():
            self.unfullscreen()
        else:
            self.fullscreen()

    def on_devtools(self, button):
        """Open developer tools."""
        if self.webview:
            inspector = self.webview.get_inspector()
            inspector.show()


class CognitiveHUDApp(Adw.Application if GTK_AVAILABLE else object):
    """Main GTK4 application."""

    def __init__(self):
        if not GTK_AVAILABLE:
            raise RuntimeError("GTK4/WebKit not available")

        super().__init__(
            application_id="com.ara.cognitive.hud",
            flags=Gio.ApplicationFlags.FLAGS_NONE,
        )

        self.server = CockpitServer()
        self.window: Optional[CognitiveHUDWindow] = None

        self.connect("activate", self.on_activate)
        self.connect("shutdown", self.on_shutdown)

    def on_activate(self, app):
        """Handle application activation."""
        if self.window is None:
            # Start HTTP server
            port = self.server.start()

            # Create window
            self.window = CognitiveHUDWindow(app=self, server_port=port)

        self.window.present()

    def on_shutdown(self, app):
        """Handle application shutdown."""
        self.server.stop()


def run_cockpit():
    """Run the cognitive cockpit GTK application."""
    if not GTK_AVAILABLE:
        print("Error: GTK4/WebKit not available.")
        print("Install dependencies with:")
        print("  sudo apt install gir1.2-webkit-6.0 libadwaita-1-dev python3-gi")
        sys.exit(1)

    # Ensure state directory exists
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    app = CognitiveHUDApp()

    # Handle SIGINT gracefully
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, lambda: app.quit())

    return app.run(sys.argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sys.exit(run_cockpit())
