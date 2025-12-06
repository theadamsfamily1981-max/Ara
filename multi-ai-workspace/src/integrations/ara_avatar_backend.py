"""Ara Avatar Backend - Local talking avatar with Ara persona.

This backend integrates:
- Offline avatar generation (lip-sync video from text)
- Ara persona specification (voice, visual, behavioral)
- T-FAN cockpit integration
- Voice macro processing
- Multi-AI delegation
- TFAN Cognitive Architecture (optional)
- MIES Modality Intelligence & Embodiment System

Ara is your local AI co-pilot that runs offline and delegates to online AIs when needed.

Cognitive Architecture (Full TFAN Biomimetic Pipeline + MIES):
    Phase 1: SENSATION - SensoryCortex normalizes audio/video/text
    Phase 2: PERCEPTION - Thalamus filters noise via TLS
    Phase 3: PREDICTION - PredictiveController anticipates states
    Phase 4: AFFECT - HomeostaticCore + AppraisalEngine for emotional regulation
    Phase 5: IDENTITY - NIBManager handles persona selection
    Phase 5.5: MODALITY INTELLIGENCE - MIES decides HOW to present response
    Phase 6: SELF-PRESERVATION - Conscience checks stability
    Phase 7: EXECUTIVE - CognitiveSynthesizer + AEPO for action gating
    Phase 8: COGNITION - Model inference with sparse attention
    Phase 9: REALITY CHECK - TopologyGate prevents hallucinations
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
# Extended cognitive components
PredictiveController = None
HomeostaticCore = None
AppraisalEngine = None
NIBManager = None
CognitiveSynthesizer = None
AEPO = None
# Level 9 components
ThermodynamicMonitor = None
EpisodicMemory = None

# MIES components (lazy loaded)
MIES_AVAILABLE = False
ModalityContext = None
ModalityDecision = None
HeuristicModalityPolicy = None
ThermodynamicGovernor = None
GnomeFocusSensor = None
PipeWireAudioSensor = None


def _init_mies_components():
    """Lazy initialization of MIES (Modality Intelligence & Embodiment System)."""
    global MIES_AVAILABLE, ModalityContext, ModalityDecision
    global HeuristicModalityPolicy, ThermodynamicGovernor
    global GnomeFocusSensor, PipeWireAudioSensor

    if ModalityContext is not None:
        return MIES_AVAILABLE

    try:
        from .mies import ModalityContext as MCtx, ModalityDecision as MDec
        from .mies.policy import HeuristicModalityPolicy as HeurPolicy
        from .mies.policy import ThermodynamicGovernor as ThermoGov
        from .mies.sensors import GnomeFocusSensor as GFocus
        from .mies.sensors import PipeWireAudioSensor as PWAudio

        ModalityContext = MCtx
        ModalityDecision = MDec
        HeuristicModalityPolicy = HeurPolicy
        ThermodynamicGovernor = ThermoGov
        GnomeFocusSensor = GFocus
        PipeWireAudioSensor = PWAudio

        MIES_AVAILABLE = True
        logger.info("MIES components loaded successfully")
    except ImportError as e:
        logger.warning(f"MIES components not available: {e}")
        MIES_AVAILABLE = False

    return MIES_AVAILABLE


def _init_cognitive_components():
    """Lazy initialization of TFAN cognitive components."""
    global COGNITIVE_AVAILABLE, CognitiveCore, SensoryCortex, Thalamus, Conscience, RealityMonitor
    global PredictiveController, HomeostaticCore, AppraisalEngine, NIBManager, CognitiveSynthesizer, AEPO
    global ThermodynamicMonitor, EpisodicMemory

    if CognitiveCore is not None:
        return COGNITIVE_AVAILABLE

    try:
        from .cognitive import (
            # Core components
            CognitiveCore as CCore,
            SensoryCortex as SCortex,
            Thalamus as Thal,
            Conscience as Consc,
            RealityMonitor as RealMon,
            # Extended components
            PredictiveController as PredCtrl,
            HomeostaticCore as HomeoCore,
            AppraisalEngine as Appraisal,
            NIBManager as NIBMgr,
            CognitiveSynthesizer as CogSynth,
            AEPO as AEPOCtrl,
            # Level 9 components
            ThermodynamicMonitor as ThermoMon,
            EpisodicMemory as EpiMem,
        )
        # Core
        CognitiveCore = CCore
        SensoryCortex = SCortex
        Thalamus = Thal
        Conscience = Consc
        RealityMonitor = RealMon
        # Extended
        PredictiveController = PredCtrl
        HomeostaticCore = HomeoCore
        AppraisalEngine = Appraisal
        NIBManager = NIBMgr
        CognitiveSynthesizer = CogSynth
        AEPO = AEPOCtrl
        # Level 9
        ThermodynamicMonitor = ThermoMon
        EpisodicMemory = EpiMem
        COGNITIVE_AVAILABLE = True
        logger.info("TFAN cognitive components loaded successfully (full architecture + Level 9)")
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
        # Use 4 workers to prevent queue exhaustion under load
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ara_avatar")

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
                # Use get_running_loop() instead of get_event_loop() to avoid deadlocks
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # Fallback if no running loop (should not happen in async context)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
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
            # Use get_running_loop() instead of get_event_loop() to avoid deadlocks
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Fallback if no running loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

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
    # COGNITIVE ARCHITECTURE INTEGRATION (TFAN Full Biomimetic Pipeline)
    # =========================================================================

    def _init_cognitive_core(
        self,
        modalities: list = ["text", "audio"],
        d_model: int = 4096,
        device: str = "cpu",
    ):
        """Initialize full cognitive architecture for biomimetic processing."""
        if not _init_cognitive_components():
            logger.warning("Cognitive components not available")
            return False

        try:
            # Core pipeline
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

            # Extended cognitive components
            self.predictive_controller = PredictiveController(
                state_dim=d_model,
                surprise_threshold=0.3,
                device=device,
            )

            self.homeostatic_core = HomeostaticCore(
                energy_decay=0.01,
                stress_accumulation=0.03,
                device=device,
            )

            self.appraisal_engine = AppraisalEngine(
                valence_threshold=0.3,
                device=device,
            )

            self.nib_manager = NIBManager(
                default_nib_name="ara_default",
            )

            self.cognitive_synthesizer = CognitiveSynthesizer(
                d_model=d_model,
                device=device,
            )

            # Level 9: Thermodynamic Monitor
            self.thermo_monitor = ThermodynamicMonitor(
                max_entropy_threshold=2.0,
                energy_capacity=100.0,
                consumption_rate=0.1,
                recovery_rate=0.05,
                device=device,
            )

            # Level 9: Episodic Memory (CXL-backed)
            self.episodic_memory = EpisodicMemory(
                use_cxl=True,
                capacity_gb=100.0,  # 100GB virtual
                ram_budget_mb=256.0,
                embedding_dim=d_model,
            )

            # MIES: Modality Intelligence & Embodiment System
            self._init_mies()

            # Track current metrics and state
            self.current_metrics = None
            self._last_prediction = None
            self._last_modality_mode = None
            self._modality_policy_state = None
            self._cognitive_initialized = True

            logger.info(f"Full cognitive architecture initialized with Level 9 + MIES (modalities={modalities})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize cognitive core: {e}")
            self._cognitive_initialized = False
            return False

    def _init_mies(self):
        """Initialize MIES (Modality Intelligence & Embodiment System)."""
        if not _init_mies_components():
            logger.warning("MIES components not available, using fallback")
            self.mies_enabled = False
            self.modality_policy = None
            self.focus_sensor = None
            self.audio_sensor = None
            self.kernel_bridge = None
            self.autonomy_policy = None
            return

        try:
            # Initialize modality policy (heuristic baseline)
            self.modality_policy = HeuristicModalityPolicy()

            # Initialize sensors (start in background)
            self.focus_sensor = GnomeFocusSensor()
            self.audio_sensor = PipeWireAudioSensor()

            # Initialize kernel bridge (brainstem → cortex)
            try:
                from .mies.kernel_bridge import create_kernel_bridge
                self.kernel_bridge = create_kernel_bridge(
                    fallback=True,  # Use /proc if device unavailable
                    simulate=False,  # Don't simulate in production
                )
                logger.info("Kernel bridge initialized")
            except Exception as e:
                logger.warning(f"Kernel bridge not available: {e}")
                self.kernel_bridge = None

            # Initialize autonomy policy (the constitution)
            try:
                from .mies.autonomy_policy import create_autonomy_policy, AutonomyGuard
                self.autonomy_policy = create_autonomy_policy()
                self.autonomy_guard = AutonomyGuard(self.autonomy_policy)
                logger.info("Autonomy policy initialized")
            except Exception as e:
                logger.warning(f"Autonomy policy not available: {e}")
                self.autonomy_policy = None
                self.autonomy_guard = None

            # Start sensors
            self.focus_sensor.start()
            self.audio_sensor.start()

            self.mies_enabled = True
            logger.info("MIES initialized with heuristic policy, sensors, kernel bridge, and autonomy")

        except Exception as e:
            logger.error(f"Failed to initialize MIES: {e}")
            self.mies_enabled = False
            self.modality_policy = None

    def _build_modality_context(
        self,
        homeostatic_state,
        appraisal,
        persona,
        thermo_stats=None,
        content_urgency: float = 0.0,
        is_user_requested: bool = False,
    ):
        """Build ModalityContext from cognitive state.

        Fuses:
        - OS context (foreground app, audio state)
        - Hardware physiology (from kernel bridge)
        - Affective state (homeostatic + appraisal)
        - Thermodynamic state
        - Identity/persona
        """
        if not MIES_AVAILABLE or not self.mies_enabled:
            return None

        from .mies.context import (
            ModalityContext as MCtx,
            ForegroundInfo,
            AudioContext,
            create_context_from_sensors,
        )

        # Get sensor state
        foreground = self.focus_sensor.get_state() if self.focus_sensor else None
        audio = self.audio_sensor.get_state() if self.audio_sensor else None
        idle_seconds = self.focus_sensor.get_idle_seconds() if self.focus_sensor else 0.0

        # Get kernel physiology (brainstem → cortex)
        kernel_phys = None
        if hasattr(self, 'kernel_bridge') and self.kernel_bridge:
            kernel_phys = self.kernel_bridge.read_physiology()

        # Create base context from sensors
        ctx = create_context_from_sensors(
            foreground=foreground,
            audio=audio,
            idle_seconds=idle_seconds,
        )

        # Update from scavengers including kernel physiology
        ctx.update_from_scavengers(
            focus_data=foreground,
            audio_data=audio,
            cognitive_state=homeostatic_state,
            kernel_physiology=kernel_phys,
        )

        # Inject affective state (may be modified by kernel physiology)
        if homeostatic_state:
            ctx.ara_fatigue = max(ctx.ara_fatigue, 1.0 - homeostatic_state.energy)
            ctx.ara_stress = max(ctx.ara_stress, homeostatic_state.stress)
            ctx.user_cognitive_load = 0.5  # Could be estimated from biometrics

        if appraisal:
            ctx.valence = appraisal.valence
            ctx.arousal = appraisal.arousal

        # Inject thermodynamic state
        if thermo_stats:
            ctx.entropy_production = thermo_stats.Pi_q
            ctx.thermal_state = thermo_stats.thermal_state.name
        if hasattr(self, 'thermo_monitor') and self.thermo_monitor:
            ctx.energy_remaining = self.thermo_monitor.energy_budget.percentage

        # Inject identity
        if persona:
            ctx.persona_name = persona.name if hasattr(persona, 'name') else str(persona)

        # Inject content metadata
        ctx.info_urgency = content_urgency
        ctx.is_user_requested = is_user_requested

        # Interaction history
        if self._last_modality_mode:
            ctx.last_mode_name = self._last_modality_mode.name
            ctx.seconds_since_last_utterance = 0.0  # Just responded

        ctx.update_derived_fields()
        return ctx

    def _apply_modality_decision(self, decision):
        """Apply a modality decision to the output system."""
        if decision is None:
            return

        mode = decision.mode
        logger.debug(
            f"[MIES] Applying modality: {mode.name} "
            f"(presence={mode.presence_intensity:.2f}, "
            f"intrusiveness={mode.intrusiveness:.2f})"
        )

        # Store for history
        self._last_modality_mode = mode

        # The actual routing to text/audio/avatar would happen here
        # For now, we just log the decision and store metadata
        self._current_modality_decision = decision

    async def cognitive_cycle(
        self,
        user_input: str,
        audio_data: Optional[np.ndarray] = None,
        video_data: Optional[np.ndarray] = None,
        context: Optional[Context] = None,
        available_tools: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Full cognitive processing cycle with TFAN biomimetic architecture.

        This implements the complete 9-phase cognitive loop:
            1. SENSATION: SensoryCortex normalizes audio/video/text
            2. PERCEPTION: Thalamus filters noise via TLS
            3. PREDICTION: PredictiveController anticipates states
            4. AFFECT: HomeostaticCore + AppraisalEngine for emotions
            5. IDENTITY: NIBManager selects appropriate persona
            6. SELF-PRESERVATION: Conscience checks stability
            7. EXECUTIVE: CognitiveSynthesizer + AEPO for action gating
            8. COGNITION: Model inference with context
            9. REALITY CHECK: TopologyGate prevents hallucinations

        Args:
            user_input: Text input from user
            audio_data: Optional audio waveform (numpy array)
            video_data: Optional video frame (numpy array, H x W x C)
            context: Optional conversation context
            available_tools: Optional list of available tools for AEPO

        Returns:
            Dictionary with full cognitive state and response
        """
        import torch
        start_time = time.time()
        context = context or Context()
        available_tools = available_tools or []

        # Initialize cognitive core if not done
        if not hasattr(self, '_cognitive_initialized') or not self._cognitive_initialized:
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

        # ========================================
        # Phase 1: SENSATION (SensoryCortex)
        # ========================================
        phase_start = time.perf_counter()

        sensory_streams = self.sensory_cortex.perceive(
            text_input=user_input,
            audio_buffer=audio_data,
            video_frame=video_data,
        )

        phase_times["sensation_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Phase 2: PERCEPTION (Thalamus)
        # ========================================
        phase_start = time.perf_counter()

        conscious_input, attention_mask = self.thalamus.process(sensory_streams)

        # Cache input topology for reality check
        self.reality_monitor.set_input_topology(conscious_input.tokens)

        phase_times["perception_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Phase 3: PREDICTION (PredictiveController)
        # ========================================
        phase_start = time.perf_counter()

        # Make prediction about response state
        prediction = self.predictive_controller.predict(
            current_state=conscious_input.tokens.mean(dim=1),  # Aggregate to (batch, d_model)
            context=user_input[:50],  # Use first 50 chars as context hash
        )

        # Observe actual state if we have a previous prediction
        prediction_error = None
        if self._last_prediction is not None:
            prediction_error = self.predictive_controller.observe(
                actual_state=conscious_input.tokens.mean(dim=1),
                prediction=self._last_prediction,
            )

        # Get predictive state
        predictive_state = self.predictive_controller.get_state()
        is_surprised = self.predictive_controller.is_surprised()

        phase_times["prediction_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Phase 4: AFFECT (Homeostasis + Appraisal)
        # ========================================
        phase_start = time.perf_counter()

        # Estimate cognitive load from input complexity
        cognitive_load = min(1.0, len(user_input) / 500.0)  # Simple heuristic

        # Update homeostatic state
        homeostatic_state = self.homeostatic_core.update(
            cognitive_load=cognitive_load,
            social_interaction=True,  # User is interacting
            novel_input=is_surprised,
            recovery_mode=False,
        )

        # Appraise emotional significance
        appraisal = self.appraisal_engine.appraise(
            input_representation=conscious_input.tokens,
            context={"user_input": user_input},
            homeostatic_state=homeostatic_state,
        )

        phase_times["affect_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Phase 5: IDENTITY (NIBManager)
        # ========================================
        phase_start = time.perf_counter()

        # Determine appropriate persona
        identity_context = {
            "user_input": user_input,
            "domain": "general",  # Could be inferred from input
            "formality": "neutral",
        }

        active_nib, alignment = self.nib_manager.adapt_to_context(
            context=identity_context,
            auto_switch=True,  # Allow automatic persona switching
        )

        # Get personality prompt for injection
        personality_prompt = self.nib_manager.get_personality_prompt()

        phase_times["identity_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Phase 5.5: MODALITY INTELLIGENCE (MIES)
        # ========================================
        phase_start = time.perf_counter()

        modality_decision = None
        if hasattr(self, 'mies_enabled') and self.mies_enabled and self.modality_policy:
            # Build modality context from cognitive state
            modality_ctx = self._build_modality_context(
                homeostatic_state=homeostatic_state,
                appraisal=appraisal,
                persona=active_nib,
                content_urgency=0.5,  # Medium urgency for user request
                is_user_requested=True,
            )

            if modality_ctx:
                # Select modality
                modality_decision = self.modality_policy.select_modality(
                    ctx=modality_ctx,
                    prev_mode=self._last_modality_mode,
                )

                # Apply the decision
                self._apply_modality_decision(modality_decision)

        phase_times["modality_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Phase 6: SELF-PRESERVATION (Conscience)
        # ========================================
        phase_start = time.perf_counter()

        # Compute structural rate from topology change
        if self.current_metrics is not None:
            s_dot = self.conscience.compute_structural_rate(
                conscious_input.tokens,
                self.current_metrics.get("topology_tensor"),
            )
        else:
            s_dot = 0.0

        # Adjust based on surprise and stress
        if is_surprised:
            s_dot = min(1.0, s_dot + 0.1)
        if homeostatic_state.stress > 0.7:
            s_dot = min(1.0, s_dot + 0.1)

        from .cognitive.synthesizer import L7Metrics, AlertLevel

        # Determine alert level from combined state
        if homeostatic_state.stress > 0.8 or s_dot > 0.5:
            alert_level = AlertLevel.RED
        elif homeostatic_state.stress > 0.5 or s_dot > 0.3:
            alert_level = AlertLevel.YELLOW
        else:
            alert_level = AlertLevel.GREEN

        l7_metrics = L7Metrics(
            structural_rate=s_dot,
            alert_level=alert_level,
            entropy=self.cognitive_synthesizer.aepo.get_entropy(),
            coherence=1.0 - homeostatic_state.stress,
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
                "affective_state": {
                    "energy": homeostatic_state.energy,
                    "stress": homeostatic_state.stress,
                    "emotion": appraisal.emotion_label,
                },
                "verification": {"gate_passed": False, "reason": "protective_mode"},
                "refused": True,
            }

        # ========================================
        # Phase 7: EXECUTIVE (Synthesizer + AEPO)
        # ========================================
        phase_start = time.perf_counter()

        # Synthesize cognitive state
        synthesis = self.cognitive_synthesizer.synthesize(
            conscious_input=conscious_input.tokens,
            predictive_state=predictive_state,
            homeostatic_state=homeostatic_state,
            appraisal=appraisal,
            active_nib=active_nib,
            available_tools=available_tools,
            context={"complexity": cognitive_load},
        )

        # Get executive decision
        executive_decision = synthesis.executive_decision

        phase_times["executive_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Phase 8: COGNITION (Model Inference)
        # ========================================
        phase_start = time.perf_counter()

        # Build enhanced prompt with cognitive context
        enhanced_prompt = self._build_cognitive_prompt(
            user_input=user_input,
            conscious_input=conscious_input,
            personality_prompt=personality_prompt,
            appraisal=appraisal,
            executive_decision=executive_decision,
        )

        # Add personality to system prompt
        if personality_prompt:
            context.system_prompt = (context.system_prompt or "") + f"\n\n{personality_prompt}"

        # Call LLM via standard path
        response = await self.send_message(enhanced_prompt, context)

        phase_times["cognition_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Phase 9: REALITY CHECK (RealityMonitor)
        # ========================================
        phase_start = time.perf_counter()

        # Create tensor from response for topology check
        # (In production, this would use actual embeddings)
        response_tensor = torch.randn(1, 100, self.cognitive_core.d_model)

        verification = self.reality_monitor.verify(
            model_output=response_tensor,
        )

        phase_times["verification_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Phase 10: THERMODYNAMICS (Energy Cost)
        # ========================================
        phase_start = time.perf_counter()

        # Compute thermodynamic cost of this cognitive step
        thermo_stats = self.thermo_monitor.compute_entropy_production(
            activations=conscious_input.tokens,
        )

        # Check if we're overheating
        force_recovery = False
        if self.thermo_monitor.should_force_recovery():
            force_recovery = True
            # Trigger recovery mode in conscience
            from .cognitive.synthesizer import SystemMode
            self.conscience.mode = SystemMode.RECOVERY
            logger.warning(
                f"Cognitive overheating detected (Pi_q={thermo_stats.Pi_q:.3f}). "
                "Entering recovery mode."
            )

        phase_times["thermodynamics_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Phase 11: MEMORY (Episodic Storage)
        # ========================================
        phase_start = time.perf_counter()

        # Store this interaction as an episode
        episode_id = None
        if hasattr(self, 'episodic_memory') and self.episodic_memory:
            # Compute importance based on surprise and emotional valence
            importance = 0.5 + 0.2 * is_surprised + 0.3 * abs(appraisal.valence)
            importance = min(1.0, importance)

            # Store with embedding (use mean of conscious tokens)
            episode_embedding = conscious_input.tokens.mean(dim=(0, 1)).cpu().numpy()
            episode_id = self.episodic_memory.store_episode(
                content=f"User: {user_input[:200]}\nAra: {response.content[:200]}",
                embedding=episode_embedding,
                importance=importance,
                context={
                    "emotion": appraisal.emotion_label,
                    "persona": active_nib.name if active_nib else "unknown",
                },
            )

        phase_times["memory_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Update State for Next Cycle
        # ========================================

        # Store prediction for next cycle
        self._last_prediction = prediction

        # Update current metrics
        self.current_metrics = {
            "topology_tensor": conscious_input.tokens,
            "structural_rate": s_dot,
        }

        # Record outcome for AEPO learning
        success = verification.is_valid and not response.error
        self.cognitive_synthesizer.record_outcome(
            tool_used=executive_decision.selected_tool if executive_decision.should_use_tool else None,
            success=success,
        )

        total_time = (time.time() - start_time) * 1000

        # Build response
        content = response.content
        if not verification.is_valid:
            content = f"[Cognitive Note: {verification.message}]\n\n{content}"

        # Add recovery warning if overheating
        if force_recovery:
            content = (
                "[I am cognitively overheating and entering recovery mode. "
                "My next responses may be slower as I consolidate.]\n\n" + content
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
            "affective_state": {
                "energy": homeostatic_state.energy,
                "attention": homeostatic_state.attention,
                "stress": homeostatic_state.stress,
                "emotion": appraisal.emotion_label,
                "valence": appraisal.valence,
                "arousal": appraisal.arousal,
                "action_tendency": appraisal.action_tendency,
            },
            "predictive_state": {
                "is_surprised": is_surprised,
                "surprise_level": self.predictive_controller.get_surprise_level(),
                "prediction_accuracy": predictive_state.prediction_accuracy,
            },
            "identity_state": {
                "active_persona": active_nib.name if active_nib else "unknown",
                "context_alignment": alignment,
            },
            "executive_decision": {
                "action_type": executive_decision.action_type.name,
                "should_use_tool": executive_decision.should_use_tool,
                "selected_tool": executive_decision.selected_tool,
                "entropy": executive_decision.entropy,
                "confidence": executive_decision.confidence,
            },
            "verification": {
                "gate_passed": verification.gate_passed,
                "status": verification.status.name,
                "wasserstein_distance": verification.wasserstein_distance,
                "cosine_similarity": verification.cosine_similarity,
                "cat_activated": verification.cat_activated,
            },
            "thermodynamics": {
                "Pi_q": thermo_stats.Pi_q,
                "vfe": thermo_stats.vfe,
                "thermal_state": thermo_stats.thermal_state.name,
                "efficiency": thermo_stats.efficiency,
                "energy_remaining_pct": self.thermo_monitor.energy_budget.percentage * 100,
                "force_recovery": force_recovery,
            },
            "memory": {
                "episode_stored": episode_id is not None,
                "episode_id": episode_id,
                "total_episodes": self.episodic_memory.total_episodes if self.episodic_memory else 0,
            },
            "modality": {
                "mode": modality_decision.mode.name if modality_decision else "text_inline",
                "channel": modality_decision.mode.channel.name if modality_decision else "TEXT_INLINE",
                "presence": modality_decision.mode.presence_intensity if modality_decision else 0.3,
                "intrusiveness": modality_decision.mode.intrusiveness if modality_decision else 0.2,
                "rationale": modality_decision.rationale if modality_decision else "",
                "mies_enabled": getattr(self, 'mies_enabled', False),
            },
            "refused": False,
        }

    async def cognitive_step(
        self,
        user_input: str,
        audio_data: Optional[np.ndarray] = None,
        video_data: Optional[np.ndarray] = None,
        context: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Simplified cognitive processing step (alias for cognitive_cycle).

        For full 9-phase processing, use cognitive_cycle() directly.

        Args:
            user_input: Text input from user
            audio_data: Optional audio waveform (numpy array)
            video_data: Optional video frame (numpy array, H x W x C)
            context: Optional conversation context

        Returns:
            Dictionary with cognitive state and response
        """
        return await self.cognitive_cycle(
            user_input=user_input,
            audio_data=audio_data,
            video_data=video_data,
            context=context,
        )

    def _build_cognitive_prompt(
        self,
        user_input: str,
        conscious_input: Any,
        personality_prompt: Optional[str] = None,
        appraisal: Optional[Any] = None,
        executive_decision: Optional[Any] = None,
    ) -> str:
        """Build enhanced prompt with full cognitive context."""
        # Start with user input
        prompt = user_input

        # Add cognitive context note (hidden from user, visible to model)
        context_parts = [
            f"{len(conscious_input.modalities_present)} modalities",
            f"{conscious_input.n_landmarks} landmarks",
            f"{conscious_input.sparsity_ratio:.1%} filtered",
        ]

        # Add emotional context if available
        if appraisal is not None:
            context_parts.append(f"emotion={appraisal.emotion_label}")
            if abs(appraisal.valence) > 0.3:
                valence_desc = "positive" if appraisal.valence > 0 else "negative"
                context_parts.append(f"valence={valence_desc}")

        # Add executive decision context if available
        if executive_decision is not None:
            if executive_decision.should_use_tool:
                context_parts.append(f"tool_suggested={executive_decision.selected_tool}")

        context_note = f"\n[Cognitive Context: {', '.join(context_parts)}]"

        return prompt + context_note

    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get current cognitive system status (full architecture)."""
        if not hasattr(self, '_cognitive_initialized') or not self._cognitive_initialized:
            return {"available": False, "reason": "Cognitive core not initialized"}

        status = {
            "available": True,
            "core_status": self.cognitive_core.get_status(),
        }

        # Add extended component status
        if hasattr(self, 'predictive_controller'):
            status["predictive"] = self.predictive_controller.get_statistics()

        if hasattr(self, 'homeostatic_core'):
            homeo_state = self.homeostatic_core.get_state()
            status["homeostatic"] = {
                "energy": homeo_state.energy,
                "attention": homeo_state.attention,
                "stress": homeo_state.stress,
                "is_balanced": homeo_state.is_balanced,
                "recommended_action": homeo_state.recommended_action,
            }

        if hasattr(self, 'nib_manager'):
            identity_state = self.nib_manager.get_state()
            status["identity"] = {
                "active_persona": identity_state.active_nib.name if identity_state.active_nib else "unknown",
                "available_personas": identity_state.available_nibs,
                "switch_count": identity_state.switch_count,
            }

        if hasattr(self, 'cognitive_synthesizer'):
            status["executive"] = {
                "entropy": self.cognitive_synthesizer.aepo.get_entropy(),
                "mode": self.cognitive_synthesizer._mode.name,
                "working_memory_items": len(self.cognitive_synthesizer.working_memory.get_state()),
            }

        # MIES status
        if hasattr(self, 'mies_enabled'):
            mies_status = {
                "enabled": self.mies_enabled,
                "policy_type": "heuristic" if self.modality_policy else "none",
            }
            if self._last_modality_mode:
                mies_status["last_mode"] = self._last_modality_mode.name
                mies_status["last_channel"] = self._last_modality_mode.channel.name
            if hasattr(self, 'focus_sensor') and self.focus_sensor:
                fg = self.focus_sensor.get_state()
                mies_status["foreground_app"] = fg.app_type.name
            if hasattr(self, 'audio_sensor') and self.audio_sensor:
                audio = self.audio_sensor.get_state()
                mies_status["audio_context"] = {
                    "mic_in_use": audio.mic_in_use,
                    "speakers_in_use": audio.speakers_in_use,
                    "voice_call": audio.has_voice_call,
                }
            status["mies"] = mies_status

        return status

    def cleanup(self, blocking: bool = True):
        """Cleanup resources (thread pool, sensors).

        Args:
            blocking: If True, wait for cleanup to complete (default).
                      If False, signal cleanup and return immediately.
        """
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=blocking, cancel_futures=not blocking)
            logger.info("Ara Avatar Backend thread pool shut down")

        # Stop MIES sensors with non-blocking option
        if hasattr(self, 'focus_sensor') and self.focus_sensor:
            self.focus_sensor.stop(blocking=blocking)
        if hasattr(self, 'audio_sensor') and self.audio_sensor:
            self.audio_sensor.stop(blocking=blocking)
        logger.info("MIES sensors stopped")

    async def cleanup_async(self):
        """Async-safe cleanup that doesn't block the event loop."""
        if hasattr(self, '_executor') and self._executor:
            # Shutdown executor in a thread to avoid blocking
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: self._executor.shutdown(wait=True, cancel_futures=False))
            except RuntimeError:
                self._executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Ara Avatar Backend thread pool shut down")

        # Stop MIES sensors using async methods
        if hasattr(self, 'focus_sensor') and self.focus_sensor:
            if hasattr(self.focus_sensor, 'stop_async'):
                await self.focus_sensor.stop_async()
            else:
                self.focus_sensor.stop(blocking=False)
        if hasattr(self, 'audio_sensor') and self.audio_sensor:
            if hasattr(self.audio_sensor, 'stop_async'):
                await self.audio_sensor.stop_async()
            else:
                self.audio_sensor.stop(blocking=False)
        logger.info("MIES sensors stopped")

    def __del__(self):
        """Destructor to ensure cleanup."""
        # Use non-blocking cleanup in destructor to avoid deadlocks
        self.cleanup(blocking=False)
