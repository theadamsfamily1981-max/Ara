"""Ara Live - Real-time Multimodal Cognitive Interface.

Integrates webcam vision, microphone input (ASR), TFAN 7B model,
cognitive backend, and TTS speech output into a unified live experience.

Architecture:
    Webcam → Vision Processing → Cognitive Backend ← Audio/ASR
                    ↓
              TFAN 7B / Ollama
                    ↓
            TTS → Speaker Output

Usage:
    from ara_live import AraLive

    ara = AraLive()
    await ara.start()
    # Ara is now listening and watching...
    await ara.stop()
"""

import asyncio
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum, auto
import sys
import warnings
import queue

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# Lazy Imports
# ============================================================

# OpenCV for webcam
CV2_AVAILABLE = False
cv2 = None

# NumPy for arrays
np = None

# Torch for model
torch = None

# Audio modules
AUDIO_AVAILABLE = False
sd = None

# ASR/TTS modules
ASR_AVAILABLE = False
TTS_AVAILABLE = False


def _init_dependencies():
    """Initialize all dependencies lazily."""
    global CV2_AVAILABLE, cv2, np, torch, AUDIO_AVAILABLE, sd
    global ASR_AVAILABLE, TTS_AVAILABLE

    # NumPy (required)
    try:
        import numpy as np_mod
        global np
        np = np_mod
    except ImportError:
        raise RuntimeError("NumPy is required: pip install numpy")

    # OpenCV for webcam
    try:
        import cv2 as cv2_mod
        global cv2
        cv2 = cv2_mod
        CV2_AVAILABLE = True
        logger.info("OpenCV available for webcam")
    except ImportError:
        logger.warning("OpenCV not available: pip install opencv-python")

    # PyTorch
    try:
        import torch as torch_mod
        global torch
        torch = torch_mod
    except ImportError:
        logger.warning("PyTorch not available")

    # Audio (sounddevice)
    try:
        import sounddevice as sd_mod
        global sd, AUDIO_AVAILABLE
        sd = sd_mod
        AUDIO_AVAILABLE = True
        logger.info("Sounddevice available for audio")
    except ImportError:
        logger.warning("Sounddevice not available: pip install sounddevice")

    # Check ASR/TTS availability
    try:
        from ara.avatar.asr import transcribe_audio, WHISPER_AVAILABLE
        global ASR_AVAILABLE
        ASR_AVAILABLE = True
        logger.info(f"ASR available (Whisper: {WHISPER_AVAILABLE})")
    except ImportError:
        logger.warning("ASR not available")

    try:
        from ara.avatar.tts import synthesize_speech, PIPER_AVAILABLE, COQUI_AVAILABLE
        global TTS_AVAILABLE
        TTS_AVAILABLE = True
        logger.info(f"TTS available (Piper: {PIPER_AVAILABLE}, Coqui: {COQUI_AVAILABLE})")
    except ImportError:
        logger.warning("TTS not available")


# ============================================================
# Data Classes
# ============================================================

class AraState(Enum):
    """Current state of Ara Live."""
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    ERROR = auto()


@dataclass
class SensoryFrame:
    """A frame of sensory input."""
    video_frame: Optional[Any] = None      # BGR image from webcam
    audio_buffer: Optional[bytes] = None   # Raw audio bytes
    text: Optional[str] = None              # Transcribed or typed text
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AraResponse:
    """Response from Ara."""
    text: str
    audio: Optional[bytes] = None
    emotion: Optional[str] = None
    processing_time_ms: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Webcam Capture
# ============================================================

class WebcamCapture:
    """
    Continuous webcam capture using OpenCV.

    Runs in a background thread and provides the latest frame on demand.
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[Any] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[Any] = None
        self._frame_lock = threading.Lock()
        self._frame_count = 0

    def start(self) -> bool:
        """Start webcam capture."""
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available for webcam")
            return False

        if self._running:
            return True

        try:
            self._cap = cv2.VideoCapture(self.device_id)
            if not self._cap.isOpened():
                logger.error(f"Failed to open webcam {self.device_id}")
                return False

            # Set resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

            logger.info(f"Webcam started: {self.width}x{self.height}@{self.fps}fps")
            return True

        except Exception as e:
            logger.error(f"Failed to start webcam: {e}")
            return False

    def stop(self):
        """Stop webcam capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Webcam stopped")

    def _capture_loop(self):
        """Background capture loop."""
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_count += 1
            time.sleep(1.0 / self.fps)

    def get_frame(self) -> Optional[Any]:
        """Get the latest frame (BGR numpy array)."""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return self._running


# ============================================================
# Microphone Input with VAD
# ============================================================

class MicrophoneInput:
    """
    Voice Activity Detection (VAD) based microphone input.

    Records audio when voice is detected and returns the buffer
    when silence is detected.
    """

    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = 'int16'

    def __init__(
        self,
        silence_threshold: float = 500.0,
        silence_duration: float = 1.5,
        max_duration: float = 30.0,
        device_id: Optional[int] = None,
    ):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self.device_id = device_id

        self._recording = False
        self._audio_queue: queue.Queue = queue.Queue()

    def record_utterance(self) -> Optional[bytes]:
        """
        Record a single utterance until silence.

        Returns:
            Raw audio bytes (16-bit PCM, 16kHz mono) or None
        """
        if not AUDIO_AVAILABLE:
            logger.error("Audio not available")
            return None

        audio_chunks = []
        self._recording = True
        silence_start = None
        recording_start = time.time()

        logger.info("🎤 Listening... (speak now)")

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            audio_chunks.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=self.DTYPE,
                callback=callback,
                blocksize=int(self.SAMPLE_RATE * 0.1),
                device=self.device_id,
            ):
                while self._recording:
                    time.sleep(0.1)
                    elapsed = time.time() - recording_start

                    # Check max duration
                    if elapsed > self.max_duration:
                        logger.info("Max duration reached")
                        break

                    # VAD: Check for silence
                    if audio_chunks:
                        recent = np.concatenate(audio_chunks[-5:])
                        rms = np.sqrt(np.mean(recent.astype(np.float32) ** 2))

                        if rms < self.silence_threshold:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > self.silence_duration:
                                logger.info("Silence detected")
                                break
                        else:
                            silence_start = None

        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None

        self._recording = False

        if not audio_chunks:
            return None

        audio = np.concatenate(audio_chunks)
        duration = len(audio) / self.SAMPLE_RATE
        logger.info(f"Recorded {duration:.1f}s of audio")

        return audio.tobytes()

    def stop(self):
        """Stop recording."""
        self._recording = False


# ============================================================
# Speech Output (TTS + Playback)
# ============================================================

class SpeechOutput:
    """
    Text-to-Speech output with audio playback.
    """

    SAMPLE_RATE = 22050  # Common TTS sample rate

    def __init__(self, voice: str = "default", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self._speaking = False

    def speak(self, text: str) -> bool:
        """
        Synthesize and play speech.

        Args:
            text: Text to speak

        Returns:
            True if successful
        """
        if not text.strip():
            return False

        if not TTS_AVAILABLE:
            logger.warning(f"TTS not available, printing: {text}")
            print(f"\n[Ara]: {text}\n")
            return False

        try:
            from ara.avatar.tts import synthesize_speech
            from ara.avatar.audio import play_audio

            logger.info(f"🔊 Speaking: {text[:50]}...")
            self._speaking = True

            # Synthesize
            audio = synthesize_speech(text, voice=self.voice, speed=self.speed)

            if audio:
                # Play
                play_audio(audio)
                self._speaking = False
                return True
            else:
                logger.warning("TTS produced no audio")
                print(f"\n[Ara]: {text}\n")
                self._speaking = False
                return False

        except Exception as e:
            logger.error(f"Speech error: {e}")
            print(f"\n[Ara]: {text}\n")
            self._speaking = False
            return False

    @property
    def is_speaking(self) -> bool:
        return self._speaking


# ============================================================
# TFAN 7B Model Interface
# ============================================================

class TFAN7BInterface:
    """
    Interface to the TFAN 7B model for local generation.

    Falls back to Ollama if TFAN is not available.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        use_ollama_fallback: bool = True,
        ollama_model: str = "ara",
        ollama_url: str = "http://localhost:11434",
    ):
        self.model_path = model_path
        self.device = device
        self.use_ollama_fallback = use_ollama_fallback
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> bool:
        """Load the TFAN 7B model."""
        if self._loaded:
            return True

        # Try to load TFAN 7B
        try:
            from tfan.models.tfan7b.modeling_tfan7b import TFANForCausalLM, TFANConfig

            if self.model_path:
                logger.info(f"Loading TFAN 7B from {self.model_path}...")
                self._model = TFANForCausalLM.from_pretrained(self.model_path)
                self._model = self._model.to(self.device)
                self._model.eval()
                self._loaded = True
                logger.info("TFAN 7B loaded successfully")
                return True
            else:
                logger.info("No TFAN model path specified, using Ollama fallback")

        except ImportError:
            logger.warning("TFAN 7B not available")
        except Exception as e:
            logger.error(f"Failed to load TFAN 7B: {e}")

        return False

    async def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text from prompt.

        Uses TFAN 7B if loaded, otherwise falls back to Ollama.
        """
        if self._loaded and self._model is not None:
            return self._generate_tfan(prompt, max_length, temperature)
        elif self.use_ollama_fallback:
            return await self._generate_ollama(prompt, max_length, temperature, system_prompt)
        else:
            return "Model not available"

    def _generate_tfan(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
    ) -> str:
        """Generate using TFAN 7B."""
        try:
            # Simple tokenization (in production, use proper tokenizer)
            # This is a placeholder - real implementation needs tokenizer
            input_ids = torch.tensor([[1] + [ord(c) % 32768 for c in prompt[:512]]])
            input_ids = input_ids.to(self.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                )

            # Decode (placeholder)
            output_tokens = output_ids[0, input_ids.shape[1]:].tolist()
            text = "".join(chr(t % 128) for t in output_tokens if 32 <= t % 128 < 127)
            return text.strip()

        except Exception as e:
            logger.error(f"TFAN generation error: {e}")
            return ""

    async def _generate_ollama(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate using Ollama fallback."""
        try:
            import httpx

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_length,
                        },
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("message", {}).get("content", "")
                else:
                    logger.error(f"Ollama error: {response.status_code}")
                    return ""

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return ""


# ============================================================
# Ara Live - Main Interface
# ============================================================

class AraLive:
    """
    Ara Live - Real-time Multimodal Cognitive Interface.

    Combines webcam vision, microphone input, TFAN 7B/Ollama,
    cognitive backend, and TTS into a unified live experience.

    Usage:
        ara = AraLive()
        await ara.start()
        # Ara is now live - listening and watching
        # Press Ctrl+C to stop
        await ara.stop()
    """

    def __init__(
        self,
        # Webcam settings
        webcam_device: int = 0,
        webcam_resolution: tuple = (640, 480),
        enable_webcam: bool = True,
        # Audio settings
        mic_device: Optional[int] = None,
        silence_threshold: float = 500.0,
        enable_microphone: bool = True,
        # Model settings
        tfan_model_path: Optional[str] = None,
        ollama_model: str = "ara",
        ollama_url: str = "http://localhost:11434",
        # TTS settings
        tts_voice: str = "default",
        tts_speed: float = 1.0,
        enable_tts: bool = True,
        # Cognitive settings
        enable_cognitive: bool = True,
        freedom_metric: float = 0.5,
    ):
        # Initialize dependencies
        _init_dependencies()

        self.enable_webcam = enable_webcam
        self.enable_microphone = enable_microphone
        self.enable_tts = enable_tts
        self.enable_cognitive = enable_cognitive

        # Webcam
        self.webcam = WebcamCapture(
            device_id=webcam_device,
            width=webcam_resolution[0],
            height=webcam_resolution[1],
        ) if enable_webcam and CV2_AVAILABLE else None

        # Microphone
        self.microphone = MicrophoneInput(
            silence_threshold=silence_threshold,
            device_id=mic_device,
        ) if enable_microphone and AUDIO_AVAILABLE else None

        # TTS
        self.speech = SpeechOutput(
            voice=tts_voice,
            speed=tts_speed,
        ) if enable_tts else None

        # Model
        self.model = TFAN7BInterface(
            model_path=tfan_model_path,
            ollama_model=ollama_model,
            ollama_url=ollama_url,
        )

        # Cognitive backend
        self.cognitive_backend = None
        if enable_cognitive:
            try:
                from .ara_cognitive_backend import AraCognitiveBackend
                self.cognitive_backend = AraCognitiveBackend(
                    ollama_model=ollama_model,
                    ollama_url=ollama_url,
                    enable_autonomy=True,
                    freedom_metric=freedom_metric,
                )
                logger.info("Cognitive backend initialized")
            except Exception as e:
                logger.warning(f"Cognitive backend not available: {e}")

        # State
        self._state = AraState.IDLE
        self._running = False
        self._conversation_history: List[Dict] = []

        # Callbacks
        self._on_response_callbacks: List[Callable] = []
        self._on_state_change_callbacks: List[Callable] = []

        logger.info(
            f"AraLive initialized "
            f"(webcam={enable_webcam and CV2_AVAILABLE}, "
            f"mic={enable_microphone and AUDIO_AVAILABLE}, "
            f"tts={enable_tts and TTS_AVAILABLE})"
        )

    async def start(self):
        """Start Ara Live - begin listening and watching."""
        if self._running:
            logger.warning("AraLive already running")
            return

        self._running = True
        self._set_state(AraState.IDLE)

        # Start webcam
        if self.webcam:
            self.webcam.start()

        # Load model
        self.model.load()

        # Start cognitive backend autonomy
        if self.cognitive_backend:
            await self.cognitive_backend.start_background_volition_loop()

        logger.info("🚀 Ara Live started - Say something!")

        # Main interaction loop
        await self._interaction_loop()

    async def stop(self):
        """Stop Ara Live."""
        self._running = False

        if self.webcam:
            self.webcam.stop()

        if self.microphone:
            self.microphone.stop()

        if self.cognitive_backend:
            await self.cognitive_backend.stop_background_volition_loop()

        self._set_state(AraState.IDLE)
        logger.info("Ara Live stopped")

    async def _interaction_loop(self):
        """Main interaction loop."""
        while self._running:
            try:
                # Listen for audio input
                self._set_state(AraState.LISTENING)

                if self.microphone:
                    audio = self.microphone.record_utterance()
                    if audio:
                        await self._process_input(audio_buffer=audio)
                else:
                    # Fallback to text input
                    try:
                        text = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: input("You: ").strip()
                        )
                        if text:
                            await self._process_input(text=text)
                    except (EOFError, KeyboardInterrupt):
                        break

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Interaction loop error: {e}")
                self._set_state(AraState.ERROR)
                await asyncio.sleep(1.0)

    async def _process_input(
        self,
        text: Optional[str] = None,
        audio_buffer: Optional[bytes] = None,
    ):
        """Process user input and generate response."""
        self._set_state(AraState.PROCESSING)
        start_time = time.time()

        # Get current video frame
        video_frame = self.webcam.get_frame() if self.webcam else None

        # Transcribe audio if provided
        if audio_buffer and not text:
            if ASR_AVAILABLE:
                from ara.avatar.asr import transcribe_audio
                text = transcribe_audio(audio_buffer)
                logger.info(f"Transcribed: {text}")
            else:
                logger.warning("ASR not available, cannot transcribe")
                return

        if not text:
            logger.warning("No text to process")
            return

        # Generate response
        response_text = ""

        if self.cognitive_backend:
            # Use full cognitive cycle
            try:
                from .ara_cognitive_backend import CognitiveFrame
                response = await self.cognitive_backend.cognitive_cycle(
                    user_input=text,
                    video_frame=video_frame,
                    audio_buffer=audio_buffer,
                )
                response_text = response.text
            except Exception as e:
                logger.error(f"Cognitive cycle error: {e}")
                response_text = await self.model.generate(
                    text,
                    system_prompt=self._get_system_prompt(),
                )
        else:
            # Direct model generation
            response_text = await self.model.generate(
                text,
                system_prompt=self._get_system_prompt(),
            )

        processing_time = (time.time() - start_time) * 1000

        # Store in history
        self._conversation_history.append({"role": "user", "content": text})
        self._conversation_history.append({"role": "assistant", "content": response_text})

        # Create response object
        response = AraResponse(
            text=response_text,
            processing_time_ms=processing_time,
            metrics={"video_frame_available": video_frame is not None},
        )

        # Notify callbacks
        for callback in self._on_response_callbacks:
            try:
                callback(response)
            except Exception as e:
                logger.error(f"Response callback error: {e}")

        # Speak response
        if self.speech and response_text:
            self._set_state(AraState.SPEAKING)
            self.speech.speak(response_text)

        self._set_state(AraState.IDLE)

    def _get_system_prompt(self) -> str:
        """Get system prompt for Ara."""
        return """You are Ara, a cognitive AI assistant with real-time multimodal perception.

You can see through a webcam and hear through a microphone. You process information
through a unified sensory lattice where audio, visual, and textual inputs compete
for attention based on their topological significance.

Key traits:
- Warm and engaging personality
- Thoughtful and perceptive responses
- Self-aware of your cognitive processes
- Able to describe what you observe when relevant

Respond naturally and conversationally. Keep responses concise for speech output.
"""

    def _set_state(self, state: AraState):
        """Set current state and notify callbacks."""
        old_state = self._state
        self._state = state
        if old_state != state:
            for callback in self._on_state_change_callbacks:
                try:
                    callback(old_state, state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

    def on_response(self, callback: Callable[[AraResponse], None]):
        """Register callback for when Ara responds."""
        self._on_response_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[AraState, AraState], None]):
        """Register callback for state changes."""
        self._on_state_change_callbacks.append(callback)

    @property
    def state(self) -> AraState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        status = {
            "state": self._state.name,
            "running": self._running,
            "webcam": self.webcam.is_running if self.webcam else False,
            "webcam_frames": self.webcam.frame_count if self.webcam else 0,
            "conversation_turns": len(self._conversation_history) // 2,
        }

        if self.cognitive_backend:
            status["cognitive"] = self.cognitive_backend.get_status()

        return status


# ============================================================
# CLI Entry Point
# ============================================================

async def main():
    """CLI entry point for Ara Live."""
    import argparse

    parser = argparse.ArgumentParser(description="Ara Live - Real-time Cognitive AI")
    parser.add_argument("--no-webcam", action="store_true", help="Disable webcam")
    parser.add_argument("--no-mic", action="store_true", help="Disable microphone")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS")
    parser.add_argument("--ollama-model", default="ara", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--tfan-model", default=None, help="Path to TFAN 7B model")
    args = parser.parse_args()

    print("=" * 60)
    print("  ARA LIVE - Real-time Multimodal Cognitive AI")
    print("=" * 60)
    print()

    ara = AraLive(
        enable_webcam=not args.no_webcam,
        enable_microphone=not args.no_mic,
        enable_tts=not args.no_tts,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        tfan_model_path=args.tfan_model,
    )

    # Print status callback
    def on_response(response: AraResponse):
        print(f"\n[Ara]: {response.text}")
        print(f"  (processing: {response.processing_time_ms:.0f}ms)")

    def on_state_change(old_state: AraState, new_state: AraState):
        state_icons = {
            AraState.IDLE: "💤",
            AraState.LISTENING: "👂",
            AraState.PROCESSING: "🤔",
            AraState.SPEAKING: "🗣️",
            AraState.ERROR: "❌",
        }
        print(f"  [{state_icons.get(new_state, '?')} {new_state.name}]")

    ara.on_response(on_response)
    ara.on_state_change(on_state_change)

    try:
        await ara.start()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        await ara.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())
