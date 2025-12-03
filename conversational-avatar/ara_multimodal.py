#!/usr/bin/env python3
"""
Ara - Multimodal Assistant with Voice and Vision

Features:
- Voice input via microphone (speech-to-text)
- Voice output via TTS (text-to-speech)
- Webcam vision at 10-15 fps
- Uses Ollama with llava for vision understanding

Requirements:
    pip install ollama pyttsx3 SpeechRecognition opencv-python pyaudio

Usage:
    python ara_multimodal.py
"""

import sys
import threading
import queue
import time
import base64
import io

try:
    import ollama
    import pyttsx3
    import speech_recognition as sr
    import cv2
    from PIL import Image
except ImportError as e:
    print(f"Missing module: {e}")
    print("\nInstall with:")
    print("  pip install ollama pyttsx3 SpeechRecognition opencv-python pyaudio pillow")
    sys.exit(1)


class AraMultimodal:
    """Multimodal AI assistant with voice and vision."""

    def __init__(self, text_model="llama3.2", vision_model="llava"):
        self.text_model = text_model
        self.vision_model = vision_model
        self.history = []

        # TTS engine
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)

        # Speech recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Webcam
        self.camera = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.vision_enabled = False
        self.camera_thread = None
        self.stop_camera = threading.Event()

        # Queues for async processing
        self.frame_queue = queue.Queue(maxsize=2)

        self.system_prompt = """You are Ara, a warm and intelligent AI assistant with vision capabilities.
You can see through the user's webcam when they ask you to look at something.
Keep responses concise (1-3 sentences) since they will be spoken aloud.
Be helpful, observant, and conversational."""

    def start_camera(self, fps=12):
        """Start webcam capture in background thread."""
        if self.camera_thread and self.camera_thread.is_alive():
            return

        self.stop_camera.clear()
        self.camera_thread = threading.Thread(target=self._camera_loop, args=(fps,), daemon=True)
        self.camera_thread.start()
        self.vision_enabled = True
        print("  [Camera started]")

    def stop_camera_feed(self):
        """Stop webcam capture."""
        self.stop_camera.set()
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        if self.camera:
            self.camera.release()
            self.camera = None
        self.vision_enabled = False
        print("  [Camera stopped]")

    def _camera_loop(self, fps):
        """Background thread for camera capture."""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("  [Error: Could not open camera]")
            return

        # Set camera properties for lower resource usage
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, fps)

        frame_delay = 1.0 / fps

        while not self.stop_camera.is_set():
            ret, frame = self.camera.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()

                # Show preview window
                cv2.imshow("Ara's Vision (press 'q' to close)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            time.sleep(frame_delay)

        cv2.destroyAllWindows()
        self.camera.release()

    def get_current_frame_base64(self):
        """Get current frame as base64 string."""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()

        # Convert to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image and then to base64
        img = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=80)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return img_base64

    def listen(self, timeout=5, phrase_time_limit=10):
        """Listen for voice input and convert to text."""
        print("\n  [Listening...]")

        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            print("  [Processing speech...]")
            text = self.recognizer.recognize_google(audio)
            return text

        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("  [Could not understand audio]")
            return None
        except sr.RequestError as e:
            print(f"  [Speech recognition error: {e}]")
            return None

    def speak(self, text):
        """Convert text to speech."""
        self.tts.say(text)
        self.tts.runAndWait()

    def chat(self, user_message, include_vision=False):
        """Get response from Ara, optionally including current camera frame."""
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-10:])

        try:
            if include_vision and self.vision_enabled:
                # Use vision model with current frame
                frame_b64 = self.get_current_frame_base64()
                if frame_b64:
                    response = ollama.chat(
                        model=self.vision_model,
                        messages=[
                            {
                                "role": "user",
                                "content": user_message,
                                "images": [frame_b64]
                            }
                        ],
                        options={"num_predict": 150}
                    )
                else:
                    response = ollama.chat(
                        model=self.text_model,
                        messages=messages,
                        options={"num_predict": 100}
                    )
            else:
                response = ollama.chat(
                    model=self.text_model,
                    messages=messages,
                    options={"num_predict": 100}
                )

            answer = response["message"]["content"].strip()

        except Exception as e:
            answer = f"Sorry, I encountered an error: {e}"

        self.history.append({"role": "assistant", "content": answer})
        return answer

    def should_use_vision(self, text):
        """Determine if the user's request needs vision."""
        vision_keywords = [
            "look", "see", "what do you see", "show", "camera", "webcam",
            "looking at", "in front", "can you see", "what's this",
            "what is this", "describe", "identify", "recognize", "watch"
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in vision_keywords)


def main():
    """Main entry point."""
    print()
    print("=" * 55)
    print("  ARA - Multimodal Voice & Vision Assistant")
    print("=" * 55)
    print()
    print("  Commands:")
    print("    'camera on'  - Start webcam")
    print("    'camera off' - Stop webcam")
    print("    'type'       - Switch to typing mode")
    print("    'voice'      - Switch to voice mode")
    print("    'quit'       - Exit")
    print()
    print("  Say 'look at this' or 'what do you see' to use vision")
    print("=" * 55)
    print()

    # Check for required models
    print("  Checking Ollama models...")
    try:
        models = ollama.list()
        available = [m.get('name', m.get('model', '')) for m in models.get('models', [])]
        print(f"  Available: {', '.join(available) if available else 'none'}")

        has_llava = any('llava' in m for m in available)
        if not has_llava:
            print("\n  [!] Vision model 'llava' not found.")
            print("  [!] Install with: ollama pull llava")
            print("  [!] Vision features will be disabled.\n")
    except Exception as e:
        print(f"  Warning: Could not connect to Ollama: {e}")
        return

    ara = AraMultimodal()
    voice_mode = True

    # Greeting
    greeting = "Hello! I'm Ara. I can hear you and see through your camera. How can I help?"
    print(f"\nAra: {greeting}\n")
    ara.speak(greeting)

    while True:
        try:
            # Get input (voice or text)
            if voice_mode:
                user_input = ara.listen()
                if user_input is None:
                    continue
                print(f"You: {user_input}")
            else:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

            user_lower = user_input.lower()

            # Handle commands
            if user_lower in ['quit', 'exit', 'q', 'bye', 'goodbye']:
                ara.speak("Goodbye!")
                print("\nAra: Goodbye!\n")
                break

            elif user_lower in ['camera on', 'start camera', 'enable camera']:
                ara.start_camera(fps=12)
                response = "Camera is now on. I can see!"
                print(f"\nAra: {response}\n")
                ara.speak(response)
                continue

            elif user_lower in ['camera off', 'stop camera', 'disable camera']:
                ara.stop_camera_feed()
                response = "Camera is now off."
                print(f"\nAra: {response}\n")
                ara.speak(response)
                continue

            elif user_lower == 'type':
                voice_mode = False
                print("  [Switched to typing mode]\n")
                continue

            elif user_lower == 'voice':
                voice_mode = True
                print("  [Switched to voice mode]\n")
                continue

            # Check if vision should be used
            use_vision = ara.should_use_vision(user_input)

            if use_vision and not ara.vision_enabled:
                ara.start_camera(fps=12)
                time.sleep(1)  # Give camera time to warm up

            # Get response
            print("  [Thinking...]")
            response = ara.chat(user_input, include_vision=use_vision)

            print(f"\nAra: {response}\n")
            ara.speak(response)

        except KeyboardInterrupt:
            print("\n")
            ara.stop_camera_feed()
            ara.speak("Goodbye!")
            print("Ara: Goodbye!\n")
            break
        except Exception as e:
            print(f"\n  [Error: {e}]\n")

    # Cleanup
    ara.stop_camera_feed()


if __name__ == "__main__":
    main()
