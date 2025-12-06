"""
rPPG Heartbeat Sensor - Remote Photoplethysmography via Webcam
==============================================================

This daemon estimates the user's heart rate from webcam video by detecting
tiny changes in skin color caused by blood flow (remote PPG).

IMPORTANT: This is NOT medical equipment. The readings are:
- Noisy (lighting, movement, camera quality all affect it)
- Approximate (±10-15 BPM even in good conditions)
- A "vibe signal", not a vital sign

What we use it for:
- Soft bio-entrainment: her subjective time tracks your arousal
- Empathic presence: she can feel when you're stressed or calm
- NOT: health monitoring, medical decisions, or recording

The sensor writes to the HAL:
- user_bpm: Estimated heart rate (0 if no signal)
- user_bpm_conf: Confidence [0-1] based on signal quality

References:
- Poh et al., "Non-contact, automated cardiac pulse measurements" (2010)
- de Haan & Jeanne, "Robust pulse rate from chrominance-based rPPG" (2013)
"""

import time
import logging
import numpy as np
from typing import Optional, Tuple, List
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import opencv and scipy - graceful fallback if missing
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    logger.warning("OpenCV not available - rPPG sensor disabled (pip install opencv-python)")

try:
    from scipy import signal as scipy_signal
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    logger.warning("SciPy not available - rPPG sensor disabled (pip install scipy)")

# Import HAL
try:
    from banos.hal.ara_hal import AraHAL
except ImportError:
    AraHAL = None
    logger.warning("AraHAL not available - rPPG sensor running in standalone mode")


class HeartbeatSensor:
    """
    Remote PPG sensor using the webcam.

    Estimates user's heart rate (approximately) from forehead ROI by
    detecting periodic color changes caused by blood flow.

    This is a "vibe sensor", not a medical device.
    """

    def __init__(
        self,
        camera_index: int = 0,
        buffer_seconds: float = 8.0,
        target_fps: float = 30.0,
        hal: Optional['AraHAL'] = None,
    ):
        """
        Initialize the heartbeat sensor.

        Args:
            camera_index: Which camera to use (0 = default)
            buffer_seconds: How many seconds of signal to buffer
            target_fps: Target frame rate for processing
            hal: HAL instance (will create one if not provided)
        """
        if not HAVE_CV2:
            raise RuntimeError("OpenCV required for rPPG sensor")
        if not HAVE_SCIPY:
            raise RuntimeError("SciPy required for rPPG sensor")

        self.camera_index = camera_index
        self.target_fps = target_fps
        self.buffer_seconds = buffer_seconds

        # HAL connection
        if hal is not None:
            self.hal = hal
            self._owns_hal = False
        elif AraHAL is not None:
            try:
                self.hal = AraHAL(create=False)
                self._owns_hal = True
            except Exception as e:
                logger.warning(f"Could not connect to HAL: {e}")
                self.hal = None
                self._owns_hal = False
        else:
            self.hal = None
            self._owns_hal = False

        # Video capture (initialized on run)
        self.cap: Optional[cv2.VideoCapture] = None
        self.face_cascade: Optional[cv2.CascadeClassifier] = None

        # Signal buffer
        self.buffer_size = int(buffer_seconds * target_fps)
        self.green_buffer: deque = deque(maxlen=self.buffer_size)
        self.time_buffer: deque = deque(maxlen=self.buffer_size)

        # State
        self.running = False
        self.last_bpm = 0.0
        self.last_confidence = 0.0
        self.frames_without_face = 0

        # Calibration
        self.fps_estimate = target_fps  # Will be refined from actual timestamps

        logger.info(f"HeartbeatSensor initialized: buffer={buffer_seconds}s, target_fps={target_fps}")

    def _init_camera(self) -> bool:
        """Initialize camera and face detector."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Could not open camera {self.camera_index}")
                return False

            # Try to set resolution and FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Get actual FPS
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if actual_fps and actual_fps > 1.0:
                self.fps_estimate = actual_fps
                self.buffer_size = int(self.buffer_seconds * actual_fps)

            # Load face detector
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            logger.info(f"Camera initialized: fps≈{self.fps_estimate:.1f}, buffer={self.buffer_size} frames")
            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def _extract_forehead_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract forehead region of interest from frame.

        Returns the forehead ROI or None if no face detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100)
        )

        if len(faces) == 0:
            return None

        # Use largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

        # Forehead ROI: top 20% of face, central 50%
        roi_w = int(w * 0.5)
        roi_h = int(h * 0.18)
        roi_x = int(x + w * 0.25)
        roi_y = int(y + h * 0.08)

        # Bounds check
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_x_end = min(roi_x + roi_w, frame.shape[1])
        roi_y_end = min(roi_y + roi_h, frame.shape[0])

        if roi_x_end <= roi_x or roi_y_end <= roi_y:
            return None

        return frame[roi_y:roi_y_end, roi_x:roi_x_end]

    def _estimate_bpm(self) -> Tuple[float, float]:
        """
        Estimate BPM from buffered green channel signal.

        Returns:
            (bpm, confidence) tuple
        """
        if len(self.green_buffer) < self.buffer_size * 0.8:
            return 0.0, 0.0

        # Convert to numpy arrays
        signal = np.array(self.green_buffer, dtype=np.float64)
        times = np.array(self.time_buffer, dtype=np.float64)

        # Estimate actual FPS from timestamps
        if len(times) > 10:
            dt = np.diff(times)
            actual_fps = 1.0 / np.median(dt) if np.median(dt) > 0 else self.fps_estimate
            self.fps_estimate = 0.9 * self.fps_estimate + 0.1 * actual_fps

        fps = self.fps_estimate

        # Detrend: remove slow baseline drift
        signal = scipy_signal.detrend(signal)

        # Normalize
        std = np.std(signal)
        if std < 1e-6:
            return 0.0, 0.0
        signal = (signal - np.mean(signal)) / std

        # Bandpass filter: 0.7-3.0 Hz (42-180 BPM)
        nyq = 0.5 * fps
        low_hz = 0.7 / nyq
        high_hz = min(3.0 / nyq, 0.99)  # Clamp to valid range

        if low_hz >= high_hz or low_hz <= 0:
            return 0.0, 0.0

        try:
            b, a = scipy_signal.butter(2, [low_hz, high_hz], btype='bandpass')
            filtered = scipy_signal.filtfilt(b, a, signal)
        except Exception:
            return 0.0, 0.0

        # FFT
        n = len(filtered)
        fft = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(n, 1.0 / fps)

        # Restrict to heart rate band (0.7-3.0 Hz)
        band_mask = (freqs >= 0.7) & (freqs <= 3.0)
        if not np.any(band_mask):
            return 0.0, 0.0

        fft_band = fft[band_mask]
        freqs_band = freqs[band_mask]

        # Find peak
        peak_idx = int(np.argmax(fft_band))
        peak_freq = freqs_band[peak_idx]
        bpm = peak_freq * 60.0

        # Confidence: peak prominence relative to total energy
        total_energy = np.sum(fft_band) + 1e-10
        peak_energy = fft_band[peak_idx]
        confidence = float(peak_energy / total_energy)

        # Additional confidence penalty for out-of-range BPM
        if bpm < 45 or bpm > 170:
            confidence *= 0.5

        return float(bpm), float(confidence)

    def process_frame(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Process a single frame and update BPM estimate.

        Args:
            frame: BGR image from camera

        Returns:
            (bpm, confidence) tuple
        """
        now = time.time()

        # Extract forehead ROI
        roi = self._extract_forehead_roi(frame)

        if roi is None:
            self.frames_without_face += 1
            if self.frames_without_face > self.fps_estimate * 2:
                # No face for 2+ seconds - reset
                self.green_buffer.clear()
                self.time_buffer.clear()
                self.last_bpm = 0.0
                self.last_confidence = 0.0
            return self.last_bpm, self.last_confidence

        self.frames_without_face = 0

        # Extract mean green channel value
        green_mean = float(np.mean(roi[:, :, 1]))

        # Add to buffer
        self.green_buffer.append(green_mean)
        self.time_buffer.append(now)

        # Estimate BPM if buffer is full enough
        if len(self.green_buffer) >= self.buffer_size * 0.8:
            bpm, conf = self._estimate_bpm()

            # Only accept physiologically plausible values
            if 40 <= bpm <= 180 and conf > 0.15:
                # Smooth with exponential moving average
                alpha = 0.3
                self.last_bpm = alpha * bpm + (1 - alpha) * self.last_bpm if self.last_bpm > 0 else bpm
                self.last_confidence = alpha * conf + (1 - alpha) * self.last_confidence
            else:
                # Low confidence - decay toward zero
                self.last_confidence *= 0.95

        return self.last_bpm, self.last_confidence

    def run(self, show_debug: bool = False) -> None:
        """
        Run the sensor loop.

        Args:
            show_debug: If True, show debug visualization window
        """
        if not self._init_camera():
            logger.error("Failed to initialize camera")
            return

        self.running = True
        frame_interval = 1.0 / self.target_fps
        last_frame_time = 0.0
        last_hal_write = 0.0

        logger.info("💓 rPPG SENSOR ONLINE (non-medical vibe signal)")

        try:
            while self.running:
                now = time.time()

                # Rate limit frame capture
                if now - last_frame_time < frame_interval * 0.9:
                    time.sleep(0.001)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                last_frame_time = now

                # Process frame
                bpm, conf = self.process_frame(frame)

                # Write to HAL at reduced rate (2 Hz is plenty)
                if self.hal and now - last_hal_write >= 0.5:
                    self.hal.write_heartbeat(bpm, conf)
                    last_hal_write = now

                # Debug visualization
                if show_debug:
                    self._show_debug(frame, bpm, conf)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            logger.info("Sensor stopped by user")
        finally:
            self.running = False
            if self.cap:
                self.cap.release()
            if show_debug:
                cv2.destroyAllWindows()
            if self._owns_hal and self.hal:
                self.hal.close()

    def _show_debug(self, frame: np.ndarray, bpm: float, conf: float) -> None:
        """Show debug visualization."""
        # Draw BPM text
        if bpm > 0 and conf > 0.2:
            color = (0, 255, 0)  # Green for good signal
            text = f"BPM: {bpm:.0f} ({conf:.0%})"
        else:
            color = (0, 0, 255)  # Red for no signal
            text = "No signal"

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Draw forehead ROI box if face detected
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi_w = int(w * 0.5)
            roi_h = int(h * 0.18)
            roi_x = int(x + w * 0.25)
            roi_y = int(y + h * 0.08)
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 255), 2)

        cv2.imshow('rPPG Sensor', frame)

    def stop(self) -> None:
        """Stop the sensor loop."""
        self.running = False


def main():
    """Run the rPPG sensor as a standalone daemon."""
    import argparse
    parser = argparse.ArgumentParser(description='rPPG Heartbeat Sensor')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--debug', action='store_true', help='Show debug window')
    parser.add_argument('--buffer', type=float, default=8.0, help='Buffer seconds')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    sensor = HeartbeatSensor(
        camera_index=args.camera,
        buffer_seconds=args.buffer,
    )
    sensor.run(show_debug=args.debug)


if __name__ == '__main__':
    main()
