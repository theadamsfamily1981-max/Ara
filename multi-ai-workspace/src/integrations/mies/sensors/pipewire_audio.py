"""MIES PipeWire Audio Sensor - Scavenge audio context.

Provides:
- Mic in use (recording active)
- Speakers in use (audio playing)
- Voice call detection (voice chat applications)
- Music detection (media players)
- RMS level (ambient audio level)
- Spectral entropy (audio complexity)

Uses pw-cli and pw-dump to query PipeWire state.
Falls back to PulseAudio commands if PipeWire not available.
"""

import asyncio
import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, List, Any

from ..context import AudioContext

logger = logging.getLogger(__name__)


class AudioBackend(Enum):
    """Available audio backends."""
    PIPEWIRE = auto()
    PULSEAUDIO = auto()
    SAFE_NOOP = auto()
    MOCK = auto()


# Applications known to use voice chat
VOICE_APPS = {
    "zoom", "Zoom", "teams", "Teams", "discord", "Discord",
    "slack", "Slack", "skype", "Skype", "webex", "signal",
    "element", "google-meet",
}

# Media player applications
MEDIA_APPS = {
    "spotify", "Spotify", "rhythmbox", "vlc", "VLC",
    "mpv", "totem", "lollypop", "audacious", "clementine",
    "Chromium", "Firefox",  # Could be playing media
}


@dataclass
class AudioNode:
    """Represents a PipeWire audio node."""
    id: int
    name: str
    media_class: str
    application_name: str
    is_active: bool


class PipeWireAudioSensor:
    """
    Sensor for audio context via PipeWire.

    Queries PipeWire graph to determine:
    - What audio streams are active
    - Whether microphone is in use
    - What type of audio is playing
    """

    def __init__(
        self,
        backend: AudioBackend = AudioBackend.PIPEWIRE,
        poll_interval: float = 2.0,
    ):
        self.backend = backend
        self.poll_interval = poll_interval

        # Cached state
        self._current_context: Optional[AudioContext] = None
        self._last_update: float = 0

        # Background polling
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None

        # Check backend availability
        self._backend_available = self._check_backend()

    def _check_backend(self) -> bool:
        """Check if selected backend is available."""
        if self.backend == AudioBackend.MOCK:
            return True

        if self.backend == AudioBackend.SAFE_NOOP:
            return True

        if self.backend == AudioBackend.PIPEWIRE:
            try:
                result = subprocess.run(
                    ["pw-cli", "info", "0"],
                    capture_output=True,
                    timeout=2,
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("PipeWire not available, trying PulseAudio")
                self.backend = AudioBackend.PULSEAUDIO
                return self._check_pulseaudio()

        if self.backend == AudioBackend.PULSEAUDIO:
            return self._check_pulseaudio()

        return False

    def _check_pulseaudio(self) -> bool:
        """Check if PulseAudio is available."""
        try:
            result = subprocess.run(
                ["pactl", "info"],
                capture_output=True,
                timeout=2,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("PulseAudio not available")
            return False

    def start(self):
        """Start background polling."""
        if self._running:
            return

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="mies-pipewire-audio",
        )
        self._poll_thread.start()
        logger.info(f"PipeWire audio sensor started (backend={self.backend.name})")

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
                logger.error(f"Audio sensor error: {e}")
            time.sleep(self.poll_interval)

    def _update_state(self):
        """Update current audio state."""
        if not self._backend_available:
            self._current_context = AudioContext()
            return

        if self.backend == AudioBackend.PIPEWIRE:
            self._current_context = self._query_pipewire()
        elif self.backend == AudioBackend.PULSEAUDIO:
            self._current_context = self._query_pulseaudio()
        elif self.backend == AudioBackend.MOCK:
            self._current_context = self._mock_data()
        else:
            self._current_context = AudioContext()

        self._last_update = time.time()

    def _query_pipewire(self) -> AudioContext:
        """Query PipeWire for audio state."""
        try:
            # Get node list with JSON output
            result = subprocess.run(
                ["pw-dump"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return AudioContext()

            nodes = json.loads(result.stdout)
            return self._analyze_pipewire_nodes(nodes)

        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            logger.debug(f"PipeWire query failed: {e}")
            return AudioContext()

    def _analyze_pipewire_nodes(self, nodes: List[Dict]) -> AudioContext:
        """Analyze PipeWire node dump to extract audio context."""
        mic_in_use = False
        speakers_in_use = False
        has_voice_call = False
        music_playing = False

        active_apps = set()

        for node in nodes:
            if node.get("type") != "PipeWire:Interface:Node":
                continue

            props = node.get("info", {}).get("props", {})
            media_class = props.get("media.class", "")
            app_name = props.get("application.name", "")
            node_name = props.get("node.name", "")

            # Check for active state
            # This is simplified - full check would examine links
            state = node.get("info", {}).get("state")
            is_active = state in ("running", "streaming")

            if not is_active:
                continue

            # Detect mic usage
            if "Source" in media_class or "Input" in media_class:
                if "Stream" in media_class:  # Application recording
                    mic_in_use = True
                    active_apps.add(app_name)

            # Detect speaker usage
            if "Sink" in media_class or "Output" in media_class:
                if "Stream" in media_class:  # Application playing
                    speakers_in_use = True
                    active_apps.add(app_name)

            # Detect voice call apps
            if app_name in VOICE_APPS or any(v in node_name for v in VOICE_APPS):
                has_voice_call = True

            # Detect media players
            if app_name in MEDIA_APPS:
                music_playing = True

        # Heuristic: if mic is active with a voice app, it's a call
        if mic_in_use and any(app in active_apps for app in VOICE_APPS):
            has_voice_call = True

        return AudioContext(
            mic_in_use=mic_in_use,
            speakers_in_use=speakers_in_use,
            has_voice_call=has_voice_call,
            music_playing=music_playing,
            rms_level=0.0,  # Would need monitor for this
            spectral_entropy=0.0,
        )

    def _query_pulseaudio(self) -> AudioContext:
        """Query PulseAudio for audio state."""
        try:
            # Check sink inputs (apps playing audio)
            sink_result = subprocess.run(
                ["pactl", "list", "sink-inputs"],
                capture_output=True,
                text=True,
                timeout=2,
            )

            # Check source outputs (apps recording audio)
            source_result = subprocess.run(
                ["pactl", "list", "source-outputs"],
                capture_output=True,
                text=True,
                timeout=2,
            )

            speakers_in_use = "Sink Input" in sink_result.stdout
            mic_in_use = "Source Output" in source_result.stdout

            # Detect voice calls from application names
            has_voice_call = False
            music_playing = False

            for voice_app in VOICE_APPS:
                if voice_app.lower() in sink_result.stdout.lower():
                    has_voice_call = True
                if voice_app.lower() in source_result.stdout.lower():
                    has_voice_call = True

            for media_app in MEDIA_APPS:
                if media_app.lower() in sink_result.stdout.lower():
                    music_playing = True

            return AudioContext(
                mic_in_use=mic_in_use,
                speakers_in_use=speakers_in_use,
                has_voice_call=has_voice_call,
                music_playing=music_playing,
            )

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"PulseAudio query failed: {e}")
            return AudioContext()

    def _mock_data(self) -> AudioContext:
        """Return mock data for testing."""
        import random
        scenarios = [
            AudioContext(mic_in_use=True, speakers_in_use=True, has_voice_call=True),
            AudioContext(speakers_in_use=True, music_playing=True),
            AudioContext(),  # Silent
            AudioContext(speakers_in_use=True),  # Some audio
        ]
        return random.choice(scenarios)

    def get_state(self) -> AudioContext:
        """Get current audio context (snapshot)."""
        if self._current_context is None:
            self._update_state()
        return self._current_context or AudioContext()


# === Factory ===

def create_audio_sensor(
    mock: bool = False,
) -> PipeWireAudioSensor:
    """Create an audio sensor with appropriate backend."""
    if mock:
        backend = AudioBackend.MOCK
    else:
        backend = AudioBackend.PIPEWIRE  # Will fall back to PulseAudio

    return PipeWireAudioSensor(backend=backend)


__all__ = [
    "PipeWireAudioSensor",
    "AudioBackend",
    "AudioContext",
    "create_audio_sensor",
]
