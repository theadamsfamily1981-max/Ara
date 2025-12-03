#!/usr/bin/env python3
"""
Ara - Multimodal Voice & Vision Assistant

Features:
- Neural TTS using Coqui XTTS-v2 (falls back to pyttsx3 if unavailable)
- Speech recognition via Google Speech API
- Webcam vision using Ollama's llava model
- Voice cloning support with reference audio file

Requirements:
    Python 3.10 (required for Coqui TTS)
    pip install -r requirements-coqui.txt

    OR use the setup script:
    ./setup_conda.sh
"""
import sys
import threading
import time
import base64
import io
import os
import tempfile

# Suppress ALSA warnings on Linux
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
try:
    from ctypes import *
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    try:
        asound = cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
    except:
        pass
except:
    pass

# Core imports
try:
    import ollama
    import speech_recognition as sr
    import cv2
    from PIL import Image
except ImportError as e:
    print(f"Missing: {e}")
    print("pip install ollama SpeechRecognition opencv-python pillow")
    sys.exit(1)

# Try to import Coqui TTS (requires Python 3.10)
COQUI_AVAILABLE = False
try:
    from TTS.api import TTS
    import torch
    import sounddevice as sd
    import numpy as np
    COQUI_AVAILABLE = True
except ImportError:
    pass

# Fallback to pyttsx3 if Coqui not available
PYTTSX3_AVAILABLE = False
if not COQUI_AVAILABLE:
    try:
        import pyttsx3
        PYTTSX3_AVAILABLE = True
    except ImportError:
        pass

if not COQUI_AVAILABLE and not PYTTSX3_AVAILABLE:
    print("ERROR: No TTS engine available!")
    print("Install Coqui TTS (recommended, requires Python 3.10):")
    print("  pip install TTS torch torchaudio sounddevice")
    print("OR install pyttsx3 (fallback):")
    print("  pip install pyttsx3")
    sys.exit(1)

SYSTEM_PROMPT = """You are Ara, a warm and friendly AI assistant with vision.
Keep responses brief (1-2 sentences) since they're spoken aloud.
Be natural, helpful, and conversational."""

# Voice reference file for cloning (create this for best quality)
VOICE_REFERENCE = os.environ.get(
    'ARA_VOICE_REFERENCE',
    'assets/voices/ara_reference.wav'
)


class CoquiTTSEngine:
    """High-quality neural TTS using Coqui XTTS-v2."""

    def __init__(self, voice_reference=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading Coqui XTTS-v2 on {self.device}...")

        # Accept TOS automatically
        os.environ['COQUI_TOS_AGREED'] = '1'

        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.voice_reference = voice_reference
        self.sample_rate = 24000  # XTTS outputs at 24kHz

        if voice_reference and os.path.exists(voice_reference):
            print(f"  Using voice reference: {voice_reference}")
        else:
            print("  No voice reference - using default voice")
            print(f"  (Add a 3-10 second WAV file at {VOICE_REFERENCE} for voice cloning)")

    def speak(self, text):
        """Synthesize and play speech."""
        print(f"\nAra: {text}\n")

        try:
            if self.voice_reference and os.path.exists(self.voice_reference):
                # Voice cloning mode
                audio = self.tts.tts(
                    text=text,
                    speaker_wav=self.voice_reference,
                    language="en"
                )
            else:
                # Default voice mode
                audio = self.tts.tts(text=text, language="en")

            # Play audio
            audio_np = np.array(audio, dtype=np.float32)
            sd.play(audio_np, self.sample_rate)
            sd.wait()

        except Exception as e:
            print(f"  [TTS Error: {e}]")


class Pyttsx3TTSEngine:
    """Fallback TTS using pyttsx3 (works with Python 3.13)."""

    def __init__(self):
        self.engine = pyttsx3.init()
        self._setup_voice()

    def _setup_voice(self):
        """Configure for best available voice."""
        self.engine.setProperty('rate', 145)
        self.engine.setProperty('volume', 0.9)

        voices = self.engine.getProperty('voices')
        preferred = ['samantha', 'zira', 'female', 'eva', 'victoria', 'karen']

        for pref in preferred:
            for voice in voices:
                if pref in voice.name.lower() or pref in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    print(f"  Using voice: {voice.name}")
                    return

        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
            print(f"  Using voice: {voices[1].name}")
        else:
            print("  Using default voice")

    def speak(self, text):
        """Speak text aloud."""
        print(f"\nAra: {text}\n")
        self.engine.say(text)
        self.engine.runAndWait()


class Ara:
    def __init__(self, use_coqui=True):
        # Initialize TTS engine
        if use_coqui and COQUI_AVAILABLE:
            print("  Initializing Coqui TTS (neural voice)...")
            self.tts_engine = CoquiTTSEngine(voice_reference=VOICE_REFERENCE)
            self.tts_type = "coqui"
        elif PYTTSX3_AVAILABLE:
            print("  Initializing pyttsx3 TTS (fallback)...")
            self.tts_engine = Pyttsx3TTSEngine()
            self.tts_type = "pyttsx3"
        else:
            raise RuntimeError("No TTS engine available!")

        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.mic = sr.Microphone()

        # Calibrate mic on startup
        print("  Calibrating microphone...")
        with self.mic as src:
            self.recognizer.adjust_for_ambient_noise(src, duration=2)
        print("  Microphone ready!")

        # Camera
        self.camera = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.cam_running = False

        # Chat history
        self.history = []

    def speak(self, text):
        """Speak text aloud using the configured TTS engine."""
        self.tts_engine.speak(text)

    def listen(self, timeout=7):
        """Listen for speech and convert to text."""
        print("  [Listening... speak now]")
        try:
            with self.mic as src:
                audio = self.recognizer.listen(
                    src,
                    timeout=timeout,
                    phrase_time_limit=12
                )
            print("  [Processing...]")
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            print("  [No speech detected]")
            return None
        except sr.UnknownValueError:
            print("  [Couldn't understand]")
            return None
        except sr.RequestError as e:
            print(f"  [Speech API error: {e}]")
            return None

    def start_camera(self, fps=8):
        """Start webcam capture."""
        if self.cam_running:
            return

        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("  [Error: Could not open camera]")
            return

        # Lower resolution for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.camera.set(cv2.CAP_PROP_FPS, fps)

        self.cam_running = True
        threading.Thread(target=self._camera_loop, args=(fps,), daemon=True).start()
        print("  [Camera ON - press 'q' in window to close]")

    def stop_camera(self):
        """Stop webcam."""
        self.cam_running = False
        time.sleep(0.3)
        if self.camera:
            self.camera.release()
            self.camera = None
        cv2.destroyAllWindows()
        print("  [Camera OFF]")

    def _camera_loop(self, fps):
        """Background camera capture."""
        delay = 1.0 / fps
        while self.cam_running and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame.copy()

                # Show smaller preview
                small = cv2.resize(frame, (320, 240))
                cv2.imshow("Ara's View (q=close)", small)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.cam_running = False
                    break

            time.sleep(delay)

        cv2.destroyAllWindows()

    def get_frame_base64(self):
        """Get current frame as base64."""
        with self.frame_lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Compress for faster sending
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=60)
        return base64.b64encode(buf.getvalue()).decode()

    def chat(self, message, use_vision=False):
        """Get response from LLM."""
        self.history.append({"role": "user", "content": message})

        try:
            if use_vision and self.frame is not None:
                # Use vision model
                img_b64 = self.get_frame_base64()
                if img_b64:
                    response = ollama.chat(
                        model="llava",
                        messages=[{
                            "role": "user",
                            "content": message,
                            "images": [img_b64]
                        }],
                        options={"num_predict": 80}  # Shorter for speech
                    )
                else:
                    response = self._text_chat(message)
            else:
                response = self._text_chat(message)

            answer = response["message"]["content"].strip()

            # Truncate very long responses for speech
            if len(answer) > 300:
                answer = answer[:297] + "..."

        except Exception as e:
            answer = f"Sorry, I had an error: {str(e)[:50]}"

        self.history.append({"role": "assistant", "content": answer})
        return answer

    def _text_chat(self, message):
        """Text-only chat."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history[-8:])  # Less history = faster

        return ollama.chat(
            model="llama3.2",
            messages=messages,
            options={"num_predict": 80}
        )

    def needs_vision(self, text):
        """Check if query needs vision."""
        keywords = ['look', 'see', 'what is', 'what\'s', 'show', 'camera',
                   'in front', 'looking at', 'describe', 'identify', 'watch']
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)


def main():
    print()
    print("=" * 60)
    print("  ARA - Voice & Vision Assistant")
    print("=" * 60)

    # Show TTS status
    if COQUI_AVAILABLE:
        print("  TTS Engine: Coqui XTTS-v2 (neural voice)")
    elif PYTTSX3_AVAILABLE:
        print("  TTS Engine: pyttsx3 (basic)")
        print("  Note: For better voice quality, use Python 3.10:")
        print("        ./setup_conda.sh")

    print()
    print("  Voice Commands:")
    print("    'camera on/off' - Toggle webcam")
    print("    'type mode'     - Switch to typing")
    print("    'voice mode'    - Switch to voice")
    print("    'quit/exit'     - Exit")
    print()
    print("  Say 'look at this' or 'what do you see' for vision")
    print("=" * 60)
    print()

    ara = Ara()
    print(f"  Using TTS: {ara.tts_type}")
    print()
    voice_mode = True

    # Greeting
    ara.speak("Hello! I'm Ara. I can hear you and see through your camera.")

    while True:
        try:
            # Get input
            if voice_mode:
                text = ara.listen()
                if text is None:
                    continue
                print(f"You: {text}")
            else:
                text = input("You: ").strip()
                if not text:
                    continue

            text_lower = text.lower()

            # Commands
            if text_lower in ['quit', 'exit', 'q', 'bye', 'goodbye', 'stop']:
                ara.speak("Goodbye!")
                break

            if 'camera on' in text_lower or 'turn on camera' in text_lower:
                ara.start_camera(fps=8)
                ara.speak("Camera is now on.")
                continue

            if 'camera off' in text_lower or 'turn off camera' in text_lower:
                ara.stop_camera()
                ara.speak("Camera is off.")
                continue

            if 'type mode' in text_lower or 'typing mode' in text_lower:
                voice_mode = False
                print("\n  [Switched to typing mode]\n")
                continue

            if 'voice mode' in text_lower or 'speaking mode' in text_lower:
                voice_mode = True
                print("\n  [Switched to voice mode]\n")
                continue

            # Check if vision needed
            use_vision = ara.needs_vision(text)

            if use_vision and not ara.cam_running:
                print("  [Starting camera for vision...]")
                ara.start_camera(fps=8)
                time.sleep(0.5)  # Let camera warm up

            # Get response
            print("  [Thinking...]")
            response = ara.chat(text, use_vision=use_vision)
            ara.speak(response)

        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"\n  [Error: {e}]\n")

    # Cleanup
    ara.stop_camera()
    print("\nGoodbye!\n")


if __name__ == "__main__":
    main()
