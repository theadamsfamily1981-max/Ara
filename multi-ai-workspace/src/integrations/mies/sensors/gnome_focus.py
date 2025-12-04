"""MIES GNOME Focus Sensor - Scavenge foreground window info.

Two backends:
1. DBus org.gnome.Shell Eval - Powerful but "unsafe" (runs GJS in shell)
2. Safe/noop backend - Returns UNKNOWN for non-GNOME or restricted systems

The sensor provides:
- Focused window WM_CLASS (e.g., "code", "firefox", "zoom")
- Window title
- Window geometry (for diegetic placement)
- Fullscreen state

Heuristics map WM_CLASS + title to ForegroundAppType.
"""

import asyncio
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any
from enum import Enum, auto
import time
import threading

from ..context import ForegroundAppType, ForegroundInfo

logger = logging.getLogger(__name__)


class SensorBackend(Enum):
    """Available sensor backends."""
    GNOME_DBUS = auto()
    SAFE_NOOP = auto()
    MOCK = auto()


# WM_CLASS patterns to app types
WM_CLASS_PATTERNS: Dict[str, ForegroundAppType] = {
    # Terminals
    "gnome-terminal": ForegroundAppType.TERMINAL,
    "kitty": ForegroundAppType.TERMINAL,
    "alacritty": ForegroundAppType.TERMINAL,
    "konsole": ForegroundAppType.TERMINAL,
    "xterm": ForegroundAppType.TERMINAL,
    "wezterm": ForegroundAppType.TERMINAL,
    "tilix": ForegroundAppType.TERMINAL,

    # IDEs
    "code": ForegroundAppType.IDE,
    "code-oss": ForegroundAppType.IDE,
    "vscodium": ForegroundAppType.IDE,
    "jetbrains-idea": ForegroundAppType.IDE,
    "jetbrains-pycharm": ForegroundAppType.IDE,
    "jetbrains-webstorm": ForegroundAppType.IDE,
    "jetbrains-clion": ForegroundAppType.IDE,
    "jetbrains-goland": ForegroundAppType.IDE,
    "jetbrains-rider": ForegroundAppType.IDE,
    "sublime_text": ForegroundAppType.IDE,
    "atom": ForegroundAppType.IDE,
    "zed": ForegroundAppType.IDE,

    # Text editors
    "gedit": ForegroundAppType.TEXT_EDITOR,
    "vim": ForegroundAppType.TEXT_EDITOR,
    "gvim": ForegroundAppType.TEXT_EDITOR,
    "neovim": ForegroundAppType.TEXT_EDITOR,
    "emacs": ForegroundAppType.TEXT_EDITOR,

    # Browsers
    "firefox": ForegroundAppType.BROWSER,
    "firefox-esr": ForegroundAppType.BROWSER,
    "chromium": ForegroundAppType.BROWSER,
    "google-chrome": ForegroundAppType.BROWSER,
    "brave-browser": ForegroundAppType.BROWSER,
    "opera": ForegroundAppType.BROWSER,
    "epiphany": ForegroundAppType.BROWSER,

    # Video calls
    "zoom": ForegroundAppType.VIDEO_CALL,
    "Zoom": ForegroundAppType.VIDEO_CALL,
    "teams": ForegroundAppType.VIDEO_CALL,
    "microsoft teams": ForegroundAppType.VIDEO_CALL,
    "slack call": ForegroundAppType.VIDEO_CALL,
    "discord": ForegroundAppType.CHAT_APP,  # Can be video too
    "skype": ForegroundAppType.VIDEO_CALL,
    "webex": ForegroundAppType.VIDEO_CALL,
    "google-meet": ForegroundAppType.VIDEO_CALL,

    # Chat apps
    "slack": ForegroundAppType.CHAT_APP,
    "element": ForegroundAppType.CHAT_APP,
    "signal": ForegroundAppType.CHAT_APP,
    "telegram": ForegroundAppType.CHAT_APP,
    "telegram-desktop": ForegroundAppType.CHAT_APP,

    # Email
    "thunderbird": ForegroundAppType.EMAIL,
    "evolution": ForegroundAppType.EMAIL,
    "geary": ForegroundAppType.EMAIL,

    # Media players
    "vlc": ForegroundAppType.MEDIA_PLAYER,
    "mpv": ForegroundAppType.MEDIA_PLAYER,
    "totem": ForegroundAppType.MEDIA_PLAYER,
    "spotify": ForegroundAppType.MEDIA_PLAYER,
    "rhythmbox": ForegroundAppType.MEDIA_PLAYER,
    "lollypop": ForegroundAppType.MEDIA_PLAYER,

    # Image editors
    "gimp": ForegroundAppType.IMAGE_EDITOR,
    "inkscape": ForegroundAppType.IMAGE_EDITOR,
    "krita": ForegroundAppType.IMAGE_EDITOR,

    # Video editors
    "kdenlive": ForegroundAppType.VIDEO_EDITOR,
    "pitivi": ForegroundAppType.VIDEO_EDITOR,
    "openshot": ForegroundAppType.VIDEO_EDITOR,
    "davinci-resolve": ForegroundAppType.VIDEO_EDITOR,

    # Games
    "steam": ForegroundAppType.FULLSCREEN_GAME,
    "lutris": ForegroundAppType.FULLSCREEN_GAME,

    # Office
    "libreoffice": ForegroundAppType.OFFICE_DOCUMENT,
    "soffice": ForegroundAppType.OFFICE_DOCUMENT,
    "abiword": ForegroundAppType.OFFICE_DOCUMENT,
    "gnumeric": ForegroundAppType.OFFICE_DOCUMENT,

    # PDF
    "evince": ForegroundAppType.PDF_READER,
    "okular": ForegroundAppType.PDF_READER,
    "zathura": ForegroundAppType.PDF_READER,

    # File manager
    "nautilus": ForegroundAppType.FILE_MANAGER,
    "dolphin": ForegroundAppType.FILE_MANAGER,
    "thunar": ForegroundAppType.FILE_MANAGER,
    "nemo": ForegroundAppType.FILE_MANAGER,

    # Settings
    "gnome-control-center": ForegroundAppType.SETTINGS,
    "systemsettings": ForegroundAppType.SETTINGS,

    # System monitor
    "gnome-system-monitor": ForegroundAppType.SYSTEM_MONITOR,
    "htop": ForegroundAppType.SYSTEM_MONITOR,
}

# Title patterns for more specific detection
TITLE_PATTERNS = [
    # Video call detection from browser
    (r"meet\.google\.com", ForegroundAppType.VIDEO_CALL),
    (r"zoom\.us", ForegroundAppType.VIDEO_CALL),
    (r"teams\.microsoft\.com", ForegroundAppType.VIDEO_CALL),
    (r"discord\.com.*/channels.*/", ForegroundAppType.VIDEO_CALL),  # In voice channel

    # Email in browser
    (r"gmail\.com", ForegroundAppType.EMAIL),
    (r"outlook\.(com|live)", ForegroundAppType.EMAIL),

    # Docs in browser
    (r"docs\.google\.com", ForegroundAppType.OFFICE_DOCUMENT),
    (r"sheets\.google\.com", ForegroundAppType.OFFICE_DOCUMENT),
]


def classify_window(wm_class: str, title: str) -> ForegroundAppType:
    """Classify window to app type using WM_CLASS and title."""
    wm_class_lower = wm_class.lower()

    # Direct WM_CLASS match
    for pattern, app_type in WM_CLASS_PATTERNS.items():
        if pattern in wm_class_lower:
            # Check title for more specific classification
            for title_pattern, title_type in TITLE_PATTERNS:
                if re.search(title_pattern, title.lower()):
                    return title_type
            return app_type

    # Title-based classification for browsers
    if any(b in wm_class_lower for b in ["firefox", "chromium", "chrome", "brave"]):
        for title_pattern, title_type in TITLE_PATTERNS:
            if re.search(title_pattern, title.lower()):
                return title_type
        return ForegroundAppType.BROWSER

    return ForegroundAppType.UNKNOWN


class GnomeFocusSensor:
    """
    Sensor for GNOME desktop focus information.

    Provides information about the currently focused window
    using DBus communication with GNOME Shell.
    """

    def __init__(
        self,
        backend: SensorBackend = SensorBackend.GNOME_DBUS,
        allow_unsafe_eval: bool = True,
        poll_interval: float = 1.0,
    ):
        self.backend = backend
        self.allow_unsafe_eval = allow_unsafe_eval
        self.poll_interval = poll_interval

        # Cached state
        self._current_info: Optional[ForegroundInfo] = None
        self._last_update: float = 0

        # Background polling
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None

        # Check backend availability
        self._backend_available = self._check_backend()

    def _check_backend(self) -> bool:
        """Check if selected backend is available."""
        if self.backend == SensorBackend.MOCK:
            return True

        if self.backend == SensorBackend.SAFE_NOOP:
            return True

        if self.backend == SensorBackend.GNOME_DBUS:
            # Try to call gdbus
            try:
                result = subprocess.run(
                    ["gdbus", "call", "--session",
                     "--dest", "org.gnome.Shell",
                     "--object-path", "/org/gnome/Shell",
                     "--method", "org.gnome.Shell.Eval", "1"],
                    capture_output=True,
                    timeout=2,
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("GNOME Shell DBus not available")
                return False

        return False

    def start(self):
        """Start background polling."""
        if self._running:
            return

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="mies-gnome-focus",
        )
        self._poll_thread.start()
        logger.info(f"GNOME focus sensor started (backend={self.backend.name})")

    def stop(self):
        """Stop background polling."""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None

    def _poll_loop(self):
        """Background polling loop."""
        while self._running:
            try:
                self._update_state()
            except Exception as e:
                logger.error(f"Focus sensor error: {e}")
            time.sleep(self.poll_interval)

    def _update_state(self):
        """Update current focus state."""
        if not self._backend_available:
            self._current_info = ForegroundInfo(
                app_type=ForegroundAppType.UNKNOWN,
                wm_class="",
                title="",
            )
            return

        if self.backend == SensorBackend.GNOME_DBUS and self.allow_unsafe_eval:
            self._current_info = self._query_gnome_dbus()
        elif self.backend == SensorBackend.MOCK:
            self._current_info = self._mock_data()
        else:
            self._current_info = ForegroundInfo(
                app_type=ForegroundAppType.UNKNOWN,
                wm_class="",
                title="",
            )

        self._last_update = time.time()

    def _query_gnome_dbus(self) -> ForegroundInfo:
        """Query GNOME Shell via DBus Eval (unsafe but powerful)."""
        # GJS snippet to get focused window info
        gjs_code = """
        (function() {
            let focusedWindow = global.display.focus_window;
            if (!focusedWindow) return JSON.stringify({});

            let rect = focusedWindow.get_frame_rect();
            return JSON.stringify({
                wm_class: focusedWindow.get_wm_class() || '',
                title: focusedWindow.get_title() || '',
                pid: focusedWindow.get_pid(),
                is_fullscreen: focusedWindow.is_fullscreen(),
                rect: [rect.x, rect.y, rect.width, rect.height]
            });
        })()
        """

        try:
            result = subprocess.run(
                ["gdbus", "call", "--session",
                 "--dest", "org.gnome.Shell",
                 "--object-path", "/org/gnome/Shell",
                 "--method", "org.gnome.Shell.Eval", gjs_code],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode != 0:
                return self._fallback_info()

            # Parse output: (true, '{"wm_class": "...", ...}')
            output = result.stdout.strip()
            # Extract JSON from DBus response
            import json
            # Find the JSON part
            match = re.search(r"'(\{.*\})'", output)
            if not match:
                return self._fallback_info()

            data = json.loads(match.group(1))

            if not data:
                return self._fallback_info()

            wm_class = data.get("wm_class", "")
            title = data.get("title", "")
            app_type = classify_window(wm_class, title)

            rect = data.get("rect")
            rect_tuple = tuple(rect) if rect else None

            return ForegroundInfo(
                app_type=app_type,
                wm_class=wm_class,
                title=title,
                rect=rect_tuple,
                is_fullscreen=data.get("is_fullscreen", False),
                pid=data.get("pid"),
            )

        except Exception as e:
            logger.debug(f"GNOME DBus query failed: {e}")
            return self._fallback_info()

    def _fallback_info(self) -> ForegroundInfo:
        """Return fallback/unknown info."""
        return ForegroundInfo(
            app_type=ForegroundAppType.UNKNOWN,
            wm_class="",
            title="",
        )

    def _mock_data(self) -> ForegroundInfo:
        """Return mock data for testing."""
        import random
        apps = [
            ("code", "main.py - MyProject - Visual Studio Code", False),
            ("firefox", "GitHub - Firefox", False),
            ("gnome-terminal", "user@host: ~/projects", False),
            ("zoom", "Zoom Meeting", False),
            ("steam_app_12345", "Game Title", True),
        ]
        wm_class, title, fullscreen = random.choice(apps)
        return ForegroundInfo(
            app_type=classify_window(wm_class, title),
            wm_class=wm_class,
            title=title,
            rect=(0, 0, 1920, 1080),
            is_fullscreen=fullscreen,
        )

    def get_state(self) -> ForegroundInfo:
        """Get current foreground info (snapshot)."""
        if self._current_info is None:
            self._update_state()
        return self._current_info or self._fallback_info()

    def get_idle_seconds(self) -> float:
        """Get system idle time in seconds."""
        try:
            # Try xprintidle (X11)
            result = subprocess.run(
                ["xprintidle"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / 1000.0
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        # Fallback: no idle info
        return 0.0


# === Factory ===

def create_focus_sensor(
    allow_unsafe: bool = True,
    mock: bool = False,
) -> GnomeFocusSensor:
    """Create a focus sensor with appropriate backend."""
    if mock:
        backend = SensorBackend.MOCK
    elif allow_unsafe:
        backend = SensorBackend.GNOME_DBUS
    else:
        backend = SensorBackend.SAFE_NOOP

    return GnomeFocusSensor(
        backend=backend,
        allow_unsafe_eval=allow_unsafe,
    )


__all__ = [
    "GnomeFocusSensor",
    "SensorBackend",
    "ForegroundInfo",
    "classify_window",
    "create_focus_sensor",
]
