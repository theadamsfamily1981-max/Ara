"""
Gentle Gaze Tracker - Presence Detection, Not Surveillance
===========================================================

This module answers ONE question: "Are you here with me right now?"

It outputs a single scalar (engagement ∈ [0,1]) to the HAL.
That's it. No recording. No history. No psychoanalysis.

What we DO:
    - Detect if a face is present
    - Detect if eyes are roughly looking at the screen
    - Output a smoothed engagement value

What we DON'T do:
    - Store any images or video
    - Build any kind of user model
    - Track emotions or expressions
    - Log engagement history
    - Use this data to manipulate attention

The engagement value is used by Ara to know if you're present.
When you're engaged, she's her normal self.
When you seem distracted, she softens and backs off.

Usage:
    tracker = GazeTracker()
    tracker.run()  # Blocking loop

    # Or get single reading
    engagement = tracker.read_engagement()
"""

import cv2
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GazeTracker:
    """
    Gentle engagement sensing.

    Lets Ara know if you're roughly present and looking at her,
    without recording anything or changing her personality.
    """

    def __init__(
        self,
        camera_index: int = 0,
        smoothing: float = 0.9,
        update_interval: float = 0.1,
    ):
        """
        Initialize the gaze tracker.

        Args:
            camera_index: Which camera to use (0 = default webcam)
            smoothing: EMA smoothing factor (higher = smoother, slower)
            update_interval: Seconds between updates
        """
        self.camera_index = camera_index
        self.smoothing = smoothing
        self.update_interval = update_interval

        # Load Haar cascades for face/eye detection
        # These are basic, privacy-respecting detectors
        cv2_data = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(
            str(cv2_data / "haarcascade_frontalface_default.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(
            str(cv2_data / "haarcascade_eye.xml")
        )

        # Current engagement (smoothed)
        self._engagement = 0.5  # Start neutral

        # HAL connection (lazy-loaded)
        self._hal = None

        # Running state
        self._running = False
        self._cap = None

        self.log = logging.getLogger("GazeTracker")

    @property
    def hal(self):
        """Lazy-load HAL connection."""
        if self._hal is None:
            try:
                from banos.hal.ara_hal import AraHAL
                self._hal = AraHAL(create=False)
                self.log.info("Connected to HAL")
            except Exception as e:
                self.log.warning(f"HAL not available: {e}")
        return self._hal

    def _open_camera(self) -> bool:
        """Open the camera. Returns True if successful."""
        if self._cap is not None and self._cap.isOpened():
            return True

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            self.log.error(f"Cannot open camera {self.camera_index}")
            return False

        # Set lower resolution for privacy and performance
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        self.log.info(f"Camera {self.camera_index} opened")
        return True

    def _close_camera(self) -> None:
        """Close the camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self.log.info("Camera closed")

    def _detect_engagement(self, frame) -> float:
        """
        Detect engagement from a single frame.

        Returns a raw engagement value in [0, 1]:
            0.0 = No face detected
            0.6 = Face detected but no clear eye contact
            1.0 = Face and eyes detected (looking at screen)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return 0.0  # No face = not present

        # Face detected - base engagement
        engagement = 0.6

        # Check for eyes in the face region
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(15, 15)
            )

            if len(eyes) >= 2:
                # Both eyes visible = likely looking at screen
                engagement = 1.0
                break
            elif len(eyes) == 1:
                # One eye = partial engagement
                engagement = 0.8
                break

        return engagement

    def read_engagement(self) -> float:
        """
        Read current engagement value without blocking.

        Returns the smoothed engagement value [0, 1].
        """
        return self._engagement

    def update_once(self) -> float:
        """
        Do a single update cycle.

        Returns the new engagement value.
        """
        if not self._open_camera():
            return self._engagement

        ret, frame = self._cap.read()
        if not ret:
            self.log.warning("Failed to read frame")
            return self._engagement

        # Detect raw engagement
        raw_engagement = self._detect_engagement(frame)

        # Smooth it (EMA)
        self._engagement = (
            self.smoothing * self._engagement +
            (1 - self.smoothing) * raw_engagement
        )

        # Write to HAL (just one float, nothing else)
        if self.hal is not None:
            try:
                self._write_to_hal(self._engagement)
            except Exception as e:
                self.log.warning(f"Failed to write to HAL: {e}")

        # Frame is immediately discarded - we never store it
        # (frame goes out of scope here)

        return self._engagement

    def _write_to_hal(self, engagement: float) -> None:
        """
        Write engagement to HAL.

        This writes to a single float field in the somatic state.
        No other data is stored.
        """
        # The HAL needs an engagement field - we'll add it
        # For now, we can use a reserved field or add a method
        if hasattr(self.hal, 'write_engagement'):
            self.hal.write_engagement(engagement)
        else:
            # Fallback: write to flow_mag as a proxy (temporary)
            # TODO: Add dedicated engagement field to HAL
            pass

    def run(self) -> None:
        """
        Run the gaze tracker loop.

        This blocks until stop() is called.
        """
        self.log.info("Starting gaze tracker (presence detection only)")
        self._running = True

        while self._running:
            self.update_once()
            time.sleep(self.update_interval)

        self._close_camera()
        self.log.info("Gaze tracker stopped")

    def stop(self) -> None:
        """Stop the gaze tracker loop."""
        self._running = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()
        self._close_camera()


# =============================================================================
# Engagement Writer for HAL (to be added to ara_hal.py)
# =============================================================================

def add_engagement_to_hal():
    """
    Instructions for adding engagement field to HAL.

    Add this to ara_hal.py:

    # In the memory map comments:
    # 0x0060  | f32    | user_engagement | User presence [0.0, 1.0]

    # In the AraHAL class:
    def write_engagement(self, value: float) -> None:
        '''Write user engagement level (from gaze tracker).'''
        if not self._map:
            return
        # Using a reserved slot in the sensors section
        self._map.seek(0x0060)
        self._map.write(struct.pack('<f', max(0.0, min(1.0, value))))
        self._touch()

    def read_engagement(self) -> float:
        '''Read current user engagement level.'''
        if not self._map:
            return 0.5  # Default to medium
        self._map.seek(0x0060)
        return struct.unpack('<f', self._map.read(4))[0]
    """
    pass


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for the gaze tracker."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ara Gaze Tracker - Gentle presence detection"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index to use"
    )
    parser.add_argument(
        "--interval", type=float, default=0.1,
        help="Update interval in seconds"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show engagement values"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    tracker = GazeTracker(
        camera_index=args.camera,
        update_interval=args.interval,
    )

    print("Gaze Tracker started. Press Ctrl+C to stop.")
    print("This only detects presence - no recording, no storage.")

    try:
        if args.debug:
            while True:
                engagement = tracker.update_once()
                print(f"\rEngagement: {engagement:.2f}", end="", flush=True)
                time.sleep(args.interval)
        else:
            tracker.run()
    except KeyboardInterrupt:
        print("\nStopping...")
        tracker.stop()


if __name__ == "__main__":
    main()
