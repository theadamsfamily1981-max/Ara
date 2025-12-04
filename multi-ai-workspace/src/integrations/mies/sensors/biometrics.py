"""MIES Biometrics Sensor - AURA-V style biometric analysis (stub).

Future integration point for:
- Blink rate estimation (cognitive load indicator)
- Pupil dilation (arousal, cognitive effort)
- Gaze tracking (attention, focus)
- Facial expression (emotional state)

This is a stub that provides the interface for future implementation.
The full version would use:
- OpenCV for face detection
- MediaPipe for face mesh
- Dlib for facial landmarks
- Custom models for blink/pupil analysis
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable, Any
from enum import Enum, auto
import threading

from ..context import BiometricState

logger = logging.getLogger(__name__)


class BiometricsBackend(Enum):
    """Available biometrics backends."""
    OPENCV_MEDIAPIPE = auto()  # Full implementation (future)
    SAFE_NOOP = auto()         # No biometrics
    MOCK = auto()              # Test data


@dataclass
class BiometricReading:
    """A single biometric reading."""
    timestamp: float
    blink_detected: bool = False
    left_eye_aspect_ratio: float = 0.0
    right_eye_aspect_ratio: float = 0.0
    pupil_diameter_left: float = 0.0
    pupil_diameter_right: float = 0.0
    gaze_x: float = 0.0  # Normalized -1 to 1
    gaze_y: float = 0.0


class BiometricsSensor:
    """
    Sensor for biometric context (stub implementation).

    In full implementation, this would:
    1. Capture webcam frames
    2. Detect face and facial landmarks
    3. Track eye state and blink rate
    4. Estimate pupil dilation
    5. Track gaze direction

    For now, provides interface stubs.
    """

    def __init__(
        self,
        backend: BiometricsBackend = BiometricsBackend.SAFE_NOOP,
        camera_index: int = 0,
        frame_rate: float = 10.0,  # Lower than video for efficiency
    ):
        self.backend = backend
        self.camera_index = camera_index
        self.frame_rate = frame_rate

        # State
        self._current_state: Optional[BiometricState] = None
        self._reading_history: list = []
        self._blink_times: list = []

        # Background processing
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_blink: Optional[Callable] = None
        self._on_fatigue_warning: Optional[Callable] = None

    def start(self):
        """Start biometric monitoring."""
        if self.backend == BiometricsBackend.SAFE_NOOP:
            logger.info("Biometrics sensor in NOOP mode (no camera access)")
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="mies-biometrics",
        )
        self._thread.start()
        logger.info(f"Biometrics sensor started (backend={self.backend.name})")

    def stop(self):
        """Stop biometric monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _processing_loop(self):
        """Main processing loop (stub)."""
        interval = 1.0 / self.frame_rate

        while self._running:
            try:
                if self.backend == BiometricsBackend.MOCK:
                    self._process_mock_frame()
                elif self.backend == BiometricsBackend.OPENCV_MEDIAPIPE:
                    self._process_real_frame()
            except Exception as e:
                logger.error(f"Biometrics processing error: {e}")

            time.sleep(interval)

    def _process_mock_frame(self):
        """Generate mock biometric data."""
        import random

        # Simulate occasional blinks
        blink = random.random() < 0.05  # ~3 blinks per second at 10fps

        if blink:
            self._blink_times.append(time.time())
            # Keep last 60 seconds of blinks
            cutoff = time.time() - 60
            self._blink_times = [t for t in self._blink_times if t > cutoff]

            if self._on_blink:
                self._on_blink()

        # Compute blink rate
        if len(self._blink_times) > 1:
            duration = self._blink_times[-1] - self._blink_times[0]
            if duration > 0:
                blink_rate = len(self._blink_times) / (duration / 60.0)
            else:
                blink_rate = 0.0
        else:
            blink_rate = 15.0  # Normal baseline

        # Mock fatigue based on blink rate
        # Normal: 15-20, Low (focus): <10, High (fatigue): >25
        if blink_rate < 10:
            estimated_fatigue = 0.2  # Low - focused
            estimated_load = 0.7  # High load
        elif blink_rate > 25:
            estimated_fatigue = 0.8  # High - tired
            estimated_load = 0.3
        else:
            estimated_fatigue = 0.4  # Normal
            estimated_load = 0.5

        self._current_state = BiometricState(
            blink_rate=blink_rate,
            pupil_dilation=0.5 + random.uniform(-0.1, 0.1),
            gaze_stability=0.8 + random.uniform(-0.1, 0.1),
            estimated_fatigue=estimated_fatigue,
            estimated_load=estimated_load,
        )

        # Check fatigue warning
        if estimated_fatigue > 0.7 and self._on_fatigue_warning:
            self._on_fatigue_warning(estimated_fatigue)

    def _process_real_frame(self):
        """Process real webcam frame (stub - not implemented)."""
        # This would:
        # 1. Capture frame from camera
        # 2. Run MediaPipe face mesh
        # 3. Extract eye landmarks
        # 4. Compute EAR (eye aspect ratio) for blink detection
        # 5. Estimate pupil size
        # 6. Track gaze

        # For now, fall back to noop
        self._current_state = BiometricState()
        logger.debug("Real biometrics processing not implemented")

    def get_state(self) -> BiometricState:
        """Get current biometric state (snapshot)."""
        if self._current_state is None:
            return BiometricState()
        return self._current_state

    def set_blink_callback(self, callback: Callable):
        """Set callback for blink events."""
        self._on_blink = callback

    def set_fatigue_callback(self, callback: Callable[[float], Any]):
        """Set callback for fatigue warnings."""
        self._on_fatigue_warning = callback

    def is_available(self) -> bool:
        """Check if biometrics are available."""
        return self.backend != BiometricsBackend.SAFE_NOOP


# === Factory ===

def create_biometrics_sensor(
    enable: bool = False,
    mock: bool = False,
) -> BiometricsSensor:
    """Create a biometrics sensor."""
    if mock:
        backend = BiometricsBackend.MOCK
    elif enable:
        # Check if we have the required libraries
        try:
            import cv2
            import mediapipe
            backend = BiometricsBackend.OPENCV_MEDIAPIPE
        except ImportError:
            logger.warning("OpenCV/MediaPipe not available, biometrics disabled")
            backend = BiometricsBackend.SAFE_NOOP
    else:
        backend = BiometricsBackend.SAFE_NOOP

    return BiometricsSensor(backend=backend)


__all__ = [
    "BiometricsSensor",
    "BiometricsBackend",
    "BiometricState",
    "BiometricReading",
    "create_biometrics_sensor",
]
