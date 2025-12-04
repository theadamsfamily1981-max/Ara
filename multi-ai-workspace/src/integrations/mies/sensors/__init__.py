"""MIES Sensors - Scavenger sensors for OS context.

The "Goddess in a Scrapyard" philosophy: we scavenge context from:
1. GNOME/Wayland: Foreground window, fullscreen state
2. PipeWire: Audio context, mic/speaker state
3. Biometrics (optional): Blink rate, pupil dilation, gaze

All sensors are designed to be:
- Non-blocking (async or threaded with snapshot getters)
- Gracefully degrading (return Unknown if unavailable)
- Swappable (can mock for testing or non-GNOME systems)
"""

from .gnome_focus import GnomeFocusSensor, create_focus_sensor
from .pipewire_audio import PipeWireAudioSensor, create_audio_sensor
from .biometrics import BiometricsSensor, create_biometrics_sensor

__all__ = [
    "GnomeFocusSensor",
    "PipeWireAudioSensor",
    "BiometricsSensor",
    "create_focus_sensor",
    "create_audio_sensor",
    "create_biometrics_sensor",
]
