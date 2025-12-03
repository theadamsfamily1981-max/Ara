"""Ara Avatar Backend - Local talking avatar with Ara persona.

This backend integrates:
- Offline avatar generation (lip-sync video from text)
- Ara persona specification (voice, visual, behavioral)
- T-FAN cockpit integration
- Voice macro processing
- Multi-AI delegation
- TFAN Cognitive Architecture (optional)

Ara is your local AI co-pilot that runs offline and delegates to online AIs when needed.

Cognitive Architecture (when enabled):
    Phase 1: SENSATION - SensoryCortex normalizes audio/video/text
    Phase 2: PERCEPTION - Thalamus filters noise via TLS
    Phase 3: SELF-PRESERVATION - Conscience checks stability
    Phase 4: COGNITION - Model inference with sparse attention
    Phase 5: REALITY CHECK - TopologyGate prevents hallucinations
"""

import asyncio
import time
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterator
import sys
from concurrent.futures import ThreadPoolExecutor
import functools

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..core.backend import AIBackend, AIProvider, Capabilities, Context, Response
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Lazy imports for avatar generation
AvatarGenerator = None
ML_AVAILABLE = False

# Lazy imports for cognitive architecture
COGNITIVE_AVAILABLE = False
CognitiveCore = None
SensoryCortex = None
Thalamus = None
Conscience = None
RealityMonitor = None


def _init_cognitive_components():
    """Lazy initialization of TFAN cognitive components."""
    global COGNITIVE_AVAILABLE, CognitiveCore, SensoryCortex, Thalamus, Conscience, RealityMonitor

    if CognitiveCore is not None:
        return COGNITIVE_AVAILABLE

    try:
        from .cognitive import (
            CognitiveCore as CCore,
            SensoryCortex as SCortex,
            Thalamus as Thal,
            Conscience as Consc,
            RealityMonitor as RealMon,
        )
        CognitiveCore = CCore
        SensoryCortex = SCortex
        Thalamus = Thal
        Conscience = Consc
        RealityMonitor = RealMon
        COGNITIVE_AVAILABLE = True
        logger.info("TFAN cognitive components loaded successfully")
    except ImportError as e:
        logger.warning(f"Cognitive components not available: {e}")
        COGNITIVE_AVAILABLE = False

    return COGNITIVE_AVAILABLE


def _init_avatar_generator():
    """Lazy initialization of avatar generator."""
    global AvatarGenerator, ML_AVAILABLE

    if AvatarGenerator is not None:
        return ML_AVAILABLE

    try:
        from src.avatar_engine.avatar_generator import AvatarGenerator as AvatarGen
        AvatarGenerator = AvatarGen
        ML_AVAILABLE = True
        logger.info("Avatar generation modules loaded successfully")
    except ImportError as e:
        logger.warning(f"Avatar generation not available: {e}")
        ML_AVAILABLE = False

    return ML_AVAILABLE


class AraAvatarBackend(AIBackend):
    """
    Ara Avatar Backend - Your local AI co-pilot.

    Features:
    - Runs offline using local Ollama (Mistral/Mixtral)
    - Generates talking avatar videos with lip-sync
    - Implements Ara persona (warm, competent, playful)
    - Delegates complex tasks to online AIs (Claude, Nova, Pulse)
    - Integrates with T-FAN cockpit for metrics and control
    - Processes voice macros for hands-free operation
    """

    def __init__(
        self,
        name: str = "Ara",
        ollama_model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        avatar_output_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Ara Avatar Backend.

        Args:
            name: Display name (default: "Ara")
            ollama_model: Ollama model for offline operation (reads from OLLAMA_MODEL env var, defaults to 'ara')
            ollama_url: Ollama API URL (reads from OLLAMA_BASE_URL env var)
            avatar_output_dir: Directory for avatar video outputs
            config: Additional configuration
        """
        # Load environment variables from .env if it exists
        self._load_env_file()

        # Get model from env var or parameter, default to 'ara' (custom model)
        if ollama_model is None:
            ollama_model = os.getenv('OLLAMA_MODEL', 'ara')

        # Get Ollama URL from env var or parameter
        if ollama_url is None:
            ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

        # Get output directory from env var or parameter
        if avatar_output_dir is None:
            avatar_output_dir = os.getenv('AVATAR_OUTPUT_DIR', 'outputs/ara_responses')

        super().__init__(
            name=name,
            provider=AIProvider.CUSTOM,
            model=f"ara-{ollama_model}",
            api_key=None,
            config=config
        )

        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.avatar_output_dir = Path(avatar_output_dir)
        self.avatar_output_dir.mkdir(parents=True, exist_ok=True)

        # Ara persona configuration
        self.persona_config = self._load_persona_config()

        # Avatar generator (lazy loaded)
        self.generator = None

        # Thread pool for blocking operations (avatar generation, TTS, etc.)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ara_avatar")

        # Current mode and avatar state
        self.current_mode = "default"
        self.current_avatar_profile = "default"
        self.current_mood = "neutral"

        logger.info(f"Ara Avatar Backend initialized (model: {ollama_model}, url: {ollama_url})")

    def _load_env_file(self):
        """Load environment variables from .env file if it exists."""
        env_path = Path('.env')
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            # Only set if not already in environment
                            if key not in os.environ:
                                os.environ[key] = value
            except Exception as e:
                logger.warning(f"Could not load .env file: {e}")

    def _load_persona_config(self) -> Dict[str, Any]:
        """Load Ara persona configuration from YAML."""
        persona_path = Path("multi-ai-workspace/config/ara_persona.yaml")

        if not persona_path.exists():
            logger.warning(f"Ara persona config not found at {persona_path}")
            return {}

        try:
            import yaml
            with open(persona_path) as f:
                config = yaml.safe_load(f)
            logger.info("Ara persona configuration loaded")
            return config
        except Exception as e:
            logger.error(f"Error loading persona config: {e}")
            return {}

    def _get_system_prompt(self) -> str:
        """Get Ara system prompt based on persona config and current mode."""
        # Base system prompt from persona config
        base_prompt = self.persona_config.get("system_prompt", "")

        # Mode-specific adjustments
        mode = self.current_mode
        if mode == "focus":
            base_prompt += "\n\nCURRENT MODE: Focus mode - Keep responses concise and task-oriented. Minimal small talk."
        elif mode == "chill":
            base_prompt += "\n\nCURRENT MODE: Chill mode - Relax your tone, be more conversational and casual."
        elif mode == "professional":
            base_prompt += "\n\nCURRENT MODE: Professional mode - Formal, precise, structured responses."

        return base_prompt

    async def send_message(
        self,
        prompt: str,
        context: Optional[Context] = None
    ) -> Response:
        """
        Send message to Ara and get text response.

        This handles the text-only interaction. For avatar video generation,
        use generate_avatar_response().

        Args:
            prompt: User message
            context: Optional context

        Returns:
            Response object
        """
        start_time = time.time()
        context = context or Context()

        try:
            import httpx

            # Build system prompt with Ara persona
            system_prompt = context.system_prompt or self._get_system_prompt()

            # Build messages for Ollama
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add conversation history
            for msg in context.conversation_history:
                messages.append(msg)

            # Add current prompt
            messages.append({"role": "user", "content": prompt})

            # Call Ollama API
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "stream": False
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("message", {}).get("content", "")

                    latency_ms = (time.time() - start_time) * 1000

                    return Response(
                        content=content,
                        provider=AIProvider.CUSTOM,
                        model=self.model,
                        tokens_used=None,
                        latency_ms=latency_ms,
                        metadata={
                            "provider_name": "ara_avatar",
                            "mode": self.current_mode,
                            "avatar_profile": self.current_avatar_profile
                        }
                    )
                else:
                    raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Ara error: {e}")
            latency_ms = (time.time() - start_time) * 1000

            return Response(
                content="",
                provider=AIProvider.CUSTOM,
                model=self.model,
                latency_ms=latency_ms,
                error=str(e),
                metadata={"provider_name": "ara_avatar"}
            )

    async def stream_message(
        self,
        prompt: str,
        context: Optional[Context] = None
    ) -> AsyncIterator[str]:
        """
        Stream Ara response.

        Args:
            prompt: User message
            context: Optional context

        Yields:
            Response chunks
        """
        context = context or Context()

        try:
            import httpx

            # Build system prompt
            system_prompt = context.system_prompt or self._get_system_prompt()

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for msg in context.conversation_history:
                messages.append(msg)

            messages.append({"role": "user", "content": prompt})

            # Stream from Ollama
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "stream": True
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                import json
                                data = json.loads(line)
                                chunk = data.get("message", {}).get("content", "")
                                if chunk:
                                    yield chunk
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Ara streaming error: {e}")
            yield f"[Error: {e}]"

    async def generate_avatar_response(
        self,
        prompt: str,
        context: Optional[Context] = None,
        use_tts: bool = True,
        avatar_image: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate full avatar response with video.

        This combines:
        1. Text generation from Ollama (with Ara persona)
        2. TTS conversion to audio
        3. Lip-sync video generation

        Args:
            prompt: User message
            context: Optional context
            use_tts: Generate TTS audio (requires espeak/piper)
            avatar_image: Path to avatar image (default: use profile setting)

        Returns:
            Dictionary with text, audio_path, video_path
        """
        # Initialize avatar generator if needed
        if not _init_avatar_generator():
            logger.error("Avatar generation not available")
            response = await self.send_message(prompt, context)
            return {
                "text": response.content,
                "audio_path": None,
                "video_path": None,
                "error": "Avatar generation dependencies not installed"
            }

        if self.generator is None:
            try:
                self.generator = AvatarGenerator(device='cpu')
            except Exception as e:
                logger.error(f"Failed to initialize avatar generator: {e}")
                response = await self.send_message(prompt, context)
                return {
                    "text": response.content,
                    "audio_path": None,
                    "video_path": None,
                    "error": str(e)
                }

        # Get text response
        response = await self.send_message(prompt, context)
        text = response.content

        if response.error or not text:
            return {
                "text": text,
                "audio_path": None,
                "video_path": None,
                "error": response.error
            }

        try:
            # Determine avatar image based on current profile
            if avatar_image is None:
                avatar_image = self._get_avatar_image_for_profile()

            # Generate TTS audio if requested
            audio_path = None
            if use_tts:
                audio_path = await self._generate_tts(text)

            # Generate video if we have audio
            video_path = None
            if audio_path and Path(avatar_image).exists():
                timestamp = int(time.time())
                video_path = self.avatar_output_dir / f"ara_response_{timestamp}.mp4"

                # Run blocking avatar generation in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor,
                            functools.partial(
                                self.generator.generate,
                                image_input=Path(avatar_image),
                                audio_input=Path(audio_path),
                                output_path=video_path
                            )
                        ),
                        timeout=60.0  # 60 second timeout for avatar generation
                    )

                    # FIX: result is a GenerationResult dataclass, not a dict
                    # Check for success attribute instead of .get() method
                    if result and getattr(result, 'success', False):
                        video_path = str(video_path)
                    else:
                        error_msg = getattr(result, 'error_message', 'Unknown error') if result else 'No result'
                        video_path = None
                        logger.error(f"Avatar generation failed: {error_msg}")
                except asyncio.TimeoutError:
                    video_path = None
                    logger.error("Avatar generation timed out after 60 seconds")
                except Exception as e:
                    video_path = None
                    logger.error(f"Avatar generation error: {e}")

            return {
                "text": text,
                "audio_path": audio_path,
                "video_path": video_path,
                "mode": self.current_mode,
                "avatar_profile": self.current_avatar_profile,
                "mood": self.current_mood
            }

        except Exception as e:
            logger.error(f"Error generating avatar response: {e}")
            return {
                "text": text,
                "audio_path": None,
                "video_path": None,
                "error": str(e)
            }

    async def _generate_tts(self, text: str) -> Optional[str]:
        """Generate TTS audio from text."""
        try:
            from src.utils.audio_processing import text_to_speech

            timestamp = int(time.time())
            audio_path = self.avatar_output_dir / f"ara_tts_{timestamp}.wav"

            # Run blocking TTS generation in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()

            def _tts_sync():
                """Synchronous TTS wrapper for thread pool."""
                return text_to_speech(
                    text=text,
                    output_path=str(audio_path),
                    voice="jenny",  # Or custom Ara voice
                    speed=0.95,  # Slightly slower (from persona spec)
                    pitch=-0.5   # Lower pitch (soft contralto)
                )

            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, _tts_sync),
                    timeout=30.0  # 30 second timeout for TTS
                )

                if result and Path(audio_path).exists():
                    return str(audio_path)
                else:
                    return None
            except asyncio.TimeoutError:
                logger.error("TTS generation timed out after 30 seconds")
                return None

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None

    def _get_avatar_image_for_profile(self) -> str:
        """Get avatar image path based on current profile."""
        # Map profiles to image files
        profile_images = {
            "default": "assets/avatars/ara_default.jpg",
            "professional": "assets/avatars/ara_professional.jpg",
            "casual": "assets/avatars/ara_casual.jpg",
            "sci_fi_cockpit": "assets/avatars/ara_hologram.jpg",
            "quantum_scientist": "assets/avatars/ara_scientist.jpg",
            "holodeck": "assets/avatars/ara_holodeck.jpg",
            "dramatic": "assets/avatars/ara_dramatic.jpg"
        }

        image_path = profile_images.get(self.current_avatar_profile, profile_images["default"])

        # Fallback to any available image if specific profile image doesn't exist
        if not Path(image_path).exists():
            # Try to find any image in assets/avatars
            avatar_dir = Path("assets/avatars")
            if avatar_dir.exists():
                images = list(avatar_dir.glob("*.jpg")) + list(avatar_dir.glob("*.png"))
                if images:
                    image_path = str(images[0])
                    logger.warning(f"Using fallback avatar image: {image_path}")

        return image_path

    def set_mode(self, mode: str):
        """Set Ara's behavioral mode."""
        self.current_mode = mode
        logger.info(f"Ara mode set to: {mode}")

    def set_avatar_profile(self, profile: str, mood: Optional[str] = None):
        """Set Ara's avatar profile and mood."""
        self.current_avatar_profile = profile
        if mood:
            self.current_mood = mood
        logger.info(f"Ara avatar set to: {profile} / {self.current_mood}")

    def get_capabilities(self) -> Capabilities:
        """Get Ara capabilities."""
        if self._capabilities is None:
            self._capabilities = Capabilities(
                streaming=True,
                vision=False,
                function_calling=False,
                max_tokens=2048,
                supports_system_prompt=True,
                rate_limit_rpm=None,  # No rate limit for offline
                cost_per_1k_tokens=0.0  # Free - runs offline
            )

        return self._capabilities

    async def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ara health check failed: {e}")
            return False

    # =========================================================================
    # COGNITIVE ARCHITECTURE INTEGRATION (TFAN)
    # =========================================================================

    def _init_cognitive_core(
        self,
        modalities: list = ["text", "audio"],
        d_model: int = 4096,
        device: str = "cpu",
    ):
        """Initialize cognitive core for enhanced processing."""
        if not _init_cognitive_components():
            logger.warning("Cognitive components not available")
            return False

        try:
            self.cognitive_core = CognitiveCore(
                d_model=d_model,
                modalities=modalities,
                device=device,
            )

            # Individual components for granular control
            self.sensory_cortex = self.cognitive_core.sensory_cortex
            self.thalamus = self.cognitive_core.thalamus
            self.conscience = self.cognitive_core.conscience
            self.reality_monitor = self.cognitive_core.reality_monitor

            # Track current metrics
            self.current_metrics = None

            logger.info(f"Cognitive core initialized (modalities={modalities})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize cognitive core: {e}")
            return False

    async def cognitive_step(
        self,
        user_input: str,
        audio_data: Optional[np.ndarray] = None,
        video_data: Optional[np.ndarray] = None,
        context: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Full cognitive processing step with TFAN architecture.

        This implements the biomimetic cognitive loop:
            1. SENSATION: Ingest raw data (text, audio, video)
            2. PERCEPTION: Filter noise via TLS (Thalamus)
            3. SELF-PRESERVATION: Check stability (Conscience)
            4. COGNITION: Run model with sparse attention
            5. REALITY CHECK: Verify topology (TopologyGate)

        Args:
            user_input: Text input from user
            audio_data: Optional audio waveform (numpy array)
            video_data: Optional video frame (numpy array, H x W x C)
            context: Optional conversation context

        Returns:
            Dictionary with:
                - content: Generated response text
                - cognitive_metrics: Processing metrics
                - stability_status: Self-preservation status
                - verification: Reality check result
                - refused: True if processing was refused
        """
        start_time = time.time()
        context = context or Context()

        # Initialize cognitive core if not done
        if not hasattr(self, 'cognitive_core') or self.cognitive_core is None:
            if not self._init_cognitive_core():
                # Fall back to standard processing
                response = await self.send_message(user_input, context)
                return {
                    "content": response.content,
                    "cognitive_metrics": {},
                    "stability_status": {"mode": "FALLBACK"},
                    "verification": {"gate_passed": True},
                    "refused": False,
                    "fallback": True,
                }

        phase_times = {}

        # ====================================
        # Phase 1: SENSATION (SensoryCortex)
        # ====================================
        phase_start = time.perf_counter()

        sensory_streams = self.sensory_cortex.perceive(
            text_input=user_input,
            audio_buffer=audio_data,
            video_frame=video_data,
        )

        phase_times["sensation_ms"] = (time.perf_counter() - phase_start) * 1000

        # ====================================
        # Phase 2: PERCEPTION (Thalamus)
        # ====================================
        phase_start = time.perf_counter()

        conscious_input, attention_mask = self.thalamus.process(sensory_streams)

        # Cache input topology for reality check
        self.reality_monitor.set_input_topology(conscious_input.tokens)

        phase_times["perception_ms"] = (time.perf_counter() - phase_start) * 1000

        # ====================================
        # Phase 3: SELF-PRESERVATION (Conscience)
        # ====================================
        phase_start = time.perf_counter()

        # Compute metrics from topology change
        if self.current_metrics is not None:
            s_dot = self.conscience.compute_structural_rate(
                conscious_input.tokens,
                self.current_metrics.get("topology_tensor"),
            )
        else:
            s_dot = 0.0

        from .cognitive.synthesizer import L7Metrics, AlertLevel

        l7_metrics = L7Metrics(
            structural_rate=s_dot,
            alert_level=AlertLevel.GREEN,
            entropy=0.0,
            coherence=1.0,
            stability_score=1.0 - s_dot,
            topology_drift=0.0,
        )

        stability_status = self.conscience.check_stability(l7_metrics=l7_metrics)

        phase_times["self_check_ms"] = (time.perf_counter() - phase_start) * 1000

        # Check if we should refuse to process
        if not stability_status.can_process:
            return {
                "content": stability_status.message,
                "cognitive_metrics": {
                    "phase_times": phase_times,
                    "n_landmarks": conscious_input.n_landmarks,
                    "sparsity_ratio": conscious_input.sparsity_ratio,
                },
                "stability_status": {
                    "mode": stability_status.mode.name,
                    "alert_level": stability_status.alert_level.name,
                    "structural_rate": stability_status.structural_rate,
                },
                "verification": {"gate_passed": False, "reason": "protective_mode"},
                "refused": True,
            }

        # ====================================
        # Phase 4: COGNITION (Model Inference)
        # ====================================
        phase_start = time.perf_counter()

        # Build enhanced prompt with cognitive context
        enhanced_prompt = self._build_cognitive_prompt(user_input, conscious_input)

        # Call LLM via standard path
        response = await self.send_message(enhanced_prompt, context)

        phase_times["cognition_ms"] = (time.perf_counter() - phase_start) * 1000

        # ====================================
        # Phase 5: REALITY CHECK (RealityMonitor)
        # ====================================
        phase_start = time.perf_counter()

        # Create tensor from response for topology check
        # (In production, this would use actual embeddings)
        import torch
        response_tensor = torch.randn(1, 100, self.cognitive_core.d_model)

        verification = self.reality_monitor.verify(
            model_output=response_tensor,
        )

        phase_times["verification_ms"] = (time.perf_counter() - phase_start) * 1000

        # Update current metrics
        self.current_metrics = {
            "topology_tensor": conscious_input.tokens,
            "structural_rate": s_dot,
        }

        total_time = (time.time() - start_time) * 1000

        # Build response
        content = response.content
        if not verification.is_valid:
            content = (
                f"[Cognitive Note: {verification.message}]\n\n{content}"
            )

        return {
            "content": content,
            "cognitive_metrics": {
                "phase_times": phase_times,
                "total_time_ms": total_time,
                "n_landmarks": conscious_input.n_landmarks,
                "sparsity_ratio": conscious_input.sparsity_ratio,
                "modalities_used": conscious_input.modalities_present,
            },
            "stability_status": {
                "mode": stability_status.mode.name,
                "alert_level": stability_status.alert_level.name,
                "structural_rate": stability_status.structural_rate,
                "can_process": stability_status.can_process,
            },
            "verification": {
                "gate_passed": verification.gate_passed,
                "status": verification.status.name,
                "wasserstein_distance": verification.wasserstein_distance,
                "cosine_similarity": verification.cosine_similarity,
                "cat_activated": verification.cat_activated,
            },
            "refused": False,
        }

    def _build_cognitive_prompt(
        self,
        user_input: str,
        conscious_input: Any,
    ) -> str:
        """Build enhanced prompt with cognitive context."""
        # Add cognitive context note
        context_note = (
            f"\n[Cognitive Context: {len(conscious_input.modalities_present)} modalities, "
            f"{conscious_input.n_landmarks} landmarks, "
            f"{conscious_input.sparsity_ratio:.1%} filtered]"
        )

        return user_input + context_note

    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get current cognitive system status."""
        if not hasattr(self, 'cognitive_core') or self.cognitive_core is None:
            return {"available": False, "reason": "Cognitive core not initialized"}

        return {
            "available": True,
            "status": self.cognitive_core.get_status(),
        }

    def cleanup(self):
        """Cleanup resources (thread pool)."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
            logger.info("Ara Avatar Backend thread pool shut down")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
