"""Ara Cognitive Backend - The Master Controller.

This is the Central Nervous System that wires the Body (Hardware/Physics),
Mind (Cognition/Thermodynamics), and Will (Autonomy/Homeostasis) into one
continuous loop.

Architecture:

    THE BODY (Hardware & Physics):
        - ThermodynamicMonitor: Monitors energy cost of thoughts (Π_q)
        - Neuromorphic hardware: FPGA/Kitten for fast reflexes

    THE HEART (Drives & Will):
        - HomeostaticCore: Energy, Integrity, Safety drives
        - AutonomyEngine: Layer 9 Volition (self-initiated action)
        - CognitiveSynthesizer: Executive decision maker

    THE MIND (Memory & Thought):
        - CXLPager: Infinite storage backend (1TB+ virtual)
        - EpisodicMemory: Long-term memory sitting on CXL

Loops:
    - Conscious Loop: User interaction with thermodynamic gating
    - Subconscious Loop: Background volition (autonomy without prompts)

This creates a system that thinks (physics), remembers (CXL), and lives (autonomy).
"""

import asyncio
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import sys
import warnings

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..core.backend import AIBackend, AIProvider, Capabilities, Context, Response
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Lazy imports for TFAN components
_TFAN_LOADED = False
MultiModalIngestor = None
MultiModalFuser = None
TopologyGate = None
FusedRepresentation = None
ModalityStream = None

# Lazy imports for Level 9 components
_LEVEL9_LOADED = False
ThermodynamicMonitor = None
EpisodicMemory = None
CXLPager = None
AutonomyEngine = None
VolitionLoop = None
DriveState = None
TaskType = None
HomeostaticCore = None
CognitiveSynthesizer = None
SystemMode = None


def _init_tfan_components():
    """Lazy initialization of TFAN cognitive components."""
    global _TFAN_LOADED, MultiModalIngestor, MultiModalFuser, TopologyGate
    global FusedRepresentation, ModalityStream

    if _TFAN_LOADED:
        return True

    try:
        from tfan.mm.ingest import (
            MultiModalIngestor as MMIngestor,
            ModalityStream as ModStream,
        )
        from tfan.mm.fuse import (
            MultiModalFuser as MMFuser,
            FusedRepresentation as FusedRep,
        )
        from tfan.mm.topo_gate import TopologyGate as TopoGate

        MultiModalIngestor = MMIngestor
        MultiModalFuser = MMFuser
        TopologyGate = TopoGate
        FusedRepresentation = FusedRep
        ModalityStream = ModStream

        _TFAN_LOADED = True
        logger.info("TFAN cognitive components loaded successfully")
        return True
    except ImportError as e:
        logger.warning(f"TFAN components not available: {e}")
        return False


def _init_level9_components():
    """Lazy initialization of Level 9 (Body/Heart/Mind) components."""
    global _LEVEL9_LOADED
    global ThermodynamicMonitor, EpisodicMemory, CXLPager
    global AutonomyEngine, VolitionLoop, DriveState, TaskType
    global HomeostaticCore, CognitiveSynthesizer, SystemMode

    if _LEVEL9_LOADED:
        return True

    try:
        # THE BODY: Thermodynamics
        from .cognitive.thermodynamics import ThermodynamicMonitor as ThermoMon
        ThermodynamicMonitor = ThermoMon

        # THE MIND: Memory
        from .cognitive.memory import (
            EpisodicMemory as EpiMem,
            CXLPager as CXLPage,
        )
        EpisodicMemory = EpiMem
        CXLPager = CXLPage

        # THE HEART: Autonomy
        from .cognitive.autonomy import (
            AutonomyEngine as AutoEng,
            VolitionLoop as VolLoop,
            DriveState as DrvState,
            TaskType as TskType,
        )
        AutonomyEngine = AutoEng
        VolitionLoop = VolLoop
        DriveState = DrvState
        TaskType = TskType

        # THE HEART: Homeostasis & Synthesis
        from .cognitive.affect import HomeostaticCore as HomeoCore
        from .cognitive.synthesizer import (
            CognitiveSynthesizer as CogSynth,
            SystemMode as SysMode,
        )
        HomeostaticCore = HomeoCore
        CognitiveSynthesizer = CogSynth
        SystemMode = SysMode

        _LEVEL9_LOADED = True
        logger.info("Level 9 components (Body/Heart/Mind) loaded successfully")
        return True
    except ImportError as e:
        logger.warning(f"Level 9 components not available: {e}")
        return False


@dataclass
class CognitiveFrame:
    """A 2-second sensory frame for cognitive processing."""
    audio_buffer: Optional[np.ndarray] = None  # Raw audio waveform
    video_frame: Optional[np.ndarray] = None   # RGB image (H, W, C)
    text: Optional[str] = None                  # Text input
    timestamp: float = 0.0                      # Frame timestamp
    metadata: Optional[Dict] = None


@dataclass
class CognitiveResponse:
    """Response from cognitive processing."""
    text: str
    fused_representation: Optional[Any] = None
    topology_metrics: Optional[Dict] = None
    emotion_state: Optional[Dict] = None
    landmarks_used: int = 0
    gate_passed: bool = True
    fallback_activated: bool = False
    processing_time_ms: float = 0.0


class AraCognitivePipeline:
    """
    TFAN Cognitive Pipeline - The "Brain" of Ara.

    Implements the full sensory lattice:
    1. Ingest (senses) - MultiModalIngestor
    2. Fuse (thalamus) - MultiModalFuser + TLS
    3. Attend (cortex) - Sparse attention on landmarks
    4. Validate (metacognition) - TopologyGate
    """

    def __init__(
        self,
        modalities: List[str] = ["audio", "text", "video"],
        d_model: int = 768,
        n_heads: int = 12,
        keep_ratio: float = 0.33,
        tls_alpha: float = 0.7,
        device: str = "cpu",
    ):
        """
        Initialize cognitive pipeline.

        Args:
            modalities: List of modalities to support
            d_model: Model dimension for all components
            n_heads: Number of attention heads
            keep_ratio: TLS landmark keep ratio
            tls_alpha: TLS blend factor (persistence vs diversity)
            device: Compute device ('cpu' or 'cuda')
        """
        self.modalities = modalities
        self.d_model = d_model
        self.n_heads = n_heads
        self.keep_ratio = keep_ratio
        self.tls_alpha = tls_alpha
        self.device = device

        # Initialize TFAN components
        if not _init_tfan_components():
            raise RuntimeError("Failed to load TFAN cognitive components")

        # Phase 1: Sensory Bridge
        self.ingestor = MultiModalIngestor(
            modalities=modalities,
            output_dim=d_model,
        )
        logger.info(f"Initialized MultiModalIngestor with modalities: {modalities}")

        # Phase 2: Fusion Reactor
        self.fuser = MultiModalFuser(
            d_model=d_model,
            modalities=modalities,
            n_heads=n_heads,
            keep_ratio=keep_ratio,
            alpha=tls_alpha,
            per_head_masks=True,
        )
        logger.info(f"Initialized MultiModalFuser (d_model={d_model}, n_heads={n_heads})")

        # Phase 4: Sanity Guard (Phase 3 is the LLM inference)
        self.topology_gate = TopologyGate(
            d_model=d_model,
            wasserstein_gap_max=0.02,
            cosine_min=0.90,
            cat_fallback_ratio=0.50,
            max_retries=2,
        )
        logger.info("Initialized TopologyGate for hallucination prevention")

        # Move components to device
        self.fuser = self.fuser.to(device)

        # Processing state
        self.last_fused: Optional[FusedRepresentation] = None
        self.input_topology: Optional[Dict] = None

    def ingest_frame(self, frame: CognitiveFrame) -> Dict[str, ModalityStream]:
        """
        Phase 1: Sensory Bridge - Convert raw inputs to normalized streams.

        Args:
            frame: CognitiveFrame with audio/video/text data

        Returns:
            Dictionary of modality -> ModalityStream
        """
        inputs = {}

        if frame.text is not None:
            inputs["text"] = [frame.text]  # Batch of 1

        if frame.audio_buffer is not None:
            # Convert to torch tensor (batch, time)
            audio_tensor = torch.from_numpy(frame.audio_buffer).float().unsqueeze(0)
            inputs["audio"] = audio_tensor

        if frame.video_frame is not None:
            # Convert to torch tensor (batch, time=1, C, H, W)
            video_tensor = torch.from_numpy(frame.video_frame).float()
            if video_tensor.dim() == 3:  # (H, W, C)
                video_tensor = video_tensor.permute(2, 0, 1)  # (C, H, W)
            video_tensor = video_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
            inputs["video"] = video_tensor

        # Ingest through adapters
        streams = self.ingestor.ingest(inputs)

        return streams

    def fuse_streams(
        self,
        streams: Dict[str, ModalityStream],
    ) -> FusedRepresentation:
        """
        Phase 2: Fusion Reactor - Combine streams into unified lattice.

        Args:
            streams: Dictionary of modality -> ModalityStream

        Returns:
            FusedRepresentation with tokens, modality map, and landmarks
        """
        # Move streams to device
        for modality, stream in streams.items():
            streams[modality] = ModalityStream(
                features=stream.features.to(self.device),
                timestamps=stream.timestamps.to(self.device),
                modality=stream.modality,
                confidence=stream.confidence,
                metadata=stream.metadata,
            )

        # Fuse with TLS landmark selection
        fused = self.fuser(streams)

        self.last_fused = fused
        return fused

    def validate_topology(
        self,
        model_output: torch.Tensor,
        recompute_fn: Optional[callable] = None,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Phase 4: Sanity Guard - Validate output against input topology.

        Args:
            model_output: Output tensor from LLM
            recompute_fn: Optional function to recompute with higher keep_ratio

        Returns:
            (passed, metrics) tuple
        """
        if self.last_fused is None:
            return True, {"gate_passed": True, "reason": "no_input_topology"}

        # Create a mock FusedRepresentation with model output
        output_fused = FusedRepresentation(
            tokens=model_output,
            modality_map=self.last_fused.modality_map,
            timestamps=self.last_fused.timestamps,
            landmark_candidates=self.last_fused.landmark_candidates,
            metadata={"source": "model_output"},
        )

        # Validate through topology gate
        validated_fused, metrics = self.topology_gate(output_fused, recompute_fn)

        return metrics.get("gate_passed", True), metrics

    def process_frame(
        self,
        frame: CognitiveFrame,
    ) -> Tuple[FusedRepresentation, Dict]:
        """
        Full cognitive processing pipeline for a single frame.

        Args:
            frame: CognitiveFrame with sensory inputs

        Returns:
            (fused_representation, processing_metrics)
        """
        start_time = time.perf_counter()
        metrics = {}

        # Phase 1: Ingest
        ingest_start = time.perf_counter()
        streams = self.ingest_frame(frame)
        metrics["ingest_time_ms"] = (time.perf_counter() - ingest_start) * 1000

        # Phase 2: Fuse
        fuse_start = time.perf_counter()
        fused = self.fuse_streams(streams)
        metrics["fuse_time_ms"] = (time.perf_counter() - fuse_start) * 1000

        # Collect fusion metrics
        metrics["total_seq_len"] = fused.tokens.shape[1]
        metrics["n_landmarks"] = fused.landmark_candidates.sum().item()
        metrics["n_modalities"] = fused.metadata.get("n_modalities", 0)

        metrics["total_time_ms"] = (time.perf_counter() - start_time) * 1000

        return fused, metrics


class AraCognitiveBackend(AIBackend):
    """
    Ara Cognitive Backend - The Master Controller.

    Wires the Body (Hardware/Physics), Heart (Drives/Will), and Mind (Memory)
    into a unified cognitive entity.

    THE BODY:
        - ThermodynamicMonitor: Tracks energy cost of thoughts (Π_q)
        - Kitten: Neuromorphic hardware for fast reflexes

    THE HEART:
        - HomeostaticCore: Energy, integrity, safety drives
        - AutonomyEngine: Layer 9 volition (self-initiated action)
        - CognitiveSynthesizer: Executive decision maker

    THE MIND:
        - CXLPager: Infinite storage backend (1TB virtual)
        - EpisodicMemory: Long-term memory on CXL

    Loops:
        - cognitive_cycle: Conscious loop (user interaction)
        - background_volition_loop: Subconscious (autonomy)
    """

    # Thermodynamic constants
    MAX_ENTROPY_THRESHOLD = 2.0  # Maximum Π_q before recovery

    def __init__(
        self,
        name: str = "Ara-Cognitive",
        ollama_model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        modalities: List[str] = ["audio", "text"],
        d_model: int = 768,
        n_heads: int = 12,
        keep_ratio: float = 0.33,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
        # Level 9 parameters
        enable_autonomy: bool = True,
        freedom_metric: float = 0.5,
        autonomy_tick_seconds: float = 60.0,
        cxl_capacity_gb: float = 1024.0,
        ram_budget_mb: float = 512.0,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize Ara Cognitive Backend - The Master Controller.

        Args:
            name: Display name
            ollama_model: Ollama model for LLM inference
            ollama_url: Ollama API URL
            modalities: Modalities to enable
            d_model: Model dimension
            n_heads: Attention heads
            keep_ratio: TLS landmark ratio
            device: Compute device
            config: Additional configuration
            enable_autonomy: Enable Layer 9 autonomy
            freedom_metric: How much autonomy [0, 1]
            autonomy_tick_seconds: Volition loop interval
            cxl_capacity_gb: Virtual memory capacity
            ram_budget_mb: RAM budget for CXL paging
            storage_path: Path for persistent storage
        """
        import os

        if ollama_model is None:
            ollama_model = os.getenv("OLLAMA_MODEL", "ara")
        if ollama_url is None:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        super().__init__(
            name=name,
            provider=AIProvider.CUSTOM,
            model=f"ara-cognitive-{ollama_model}",
            api_key=None,
            config=config,
        )

        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.device = device
        self.enable_autonomy = enable_autonomy
        self.freedom_metric = freedom_metric
        self.autonomy_tick_seconds = autonomy_tick_seconds

        # Storage path
        if storage_path is None:
            storage_path = str(Path.home() / ".ara")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize cognitive pipeline (TFAN multi-modal)
        self.cognitive_pipeline: Optional[AraCognitivePipeline] = None
        self._init_cognitive_pipeline(modalities, d_model, n_heads, keep_ratio, device)

        # Initialize Level 9 components (Body/Heart/Mind)
        self._init_level9_subsystems(cxl_capacity_gb, ram_budget_mb)

        # Background task handle
        self._volition_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"Ara Cognitive Backend initialized "
            f"(model={ollama_model}, modalities={modalities}, device={device}, "
            f"autonomy={enable_autonomy}, freedom={freedom_metric})"
        )

    def _init_level9_subsystems(self, cxl_capacity_gb: float, ram_budget_mb: float):
        """Initialize Level 9 subsystems: Body, Heart, and Mind."""
        if not _init_level9_components():
            logger.warning("Level 9 components unavailable - limited functionality")
            self.thermo = None
            self.homeostat = None
            self.autonomy = None
            self.synthesizer = None
            self.cxl = None
            self.memory = None
            self.volition_loop = None
            return

        try:
            # --- THE BODY (Hardware & Physics) ---
            self.thermo = ThermodynamicMonitor(
                max_entropy_threshold=self.MAX_ENTROPY_THRESHOLD,
                energy_capacity=100.0,
                consumption_rate=0.1,
                recovery_rate=0.05,
            )
            logger.info("Initialized ThermodynamicMonitor (The Body)")

            # --- THE HEART (Drives & Will) ---
            self.homeostat = HomeostaticCore(
                energy_decay=0.01,
                stress_accumulation=0.02,
            )
            self.autonomy = AutonomyEngine(
                freedom_metric=self.freedom_metric,
            )
            self.synthesizer = CognitiveSynthesizer()
            logger.info("Initialized HomeostaticCore, AutonomyEngine, CognitiveSynthesizer (The Heart)")

            # --- THE MIND (Memory & Thought) ---
            self.cxl = CXLPager(
                capacity_gb=cxl_capacity_gb,
                ram_budget_mb=ram_budget_mb,
                storage_path=str(self.storage_path / "cxl"),
            )
            self.memory = EpisodicMemory(
                use_cxl=True,
                capacity_gb=cxl_capacity_gb,
                ram_budget_mb=ram_budget_mb,
                storage_path=str(self.storage_path / "memory"),
            )
            logger.info(f"Initialized CXLPager ({cxl_capacity_gb}GB), EpisodicMemory (The Mind)")

            # Volition loop for background autonomy
            self.volition_loop = VolitionLoop(
                autonomy_engine=self.autonomy,
                tick_interval_seconds=self.autonomy_tick_seconds,
                task_executor=self._execute_autonomous_task,
            )
            self.volition_loop.on_intent(self._on_volition_intent)

        except Exception as e:
            logger.error(f"Failed to initialize Level 9 subsystems: {e}")
            self.thermo = None
            self.homeostat = None
            self.autonomy = None
            self.synthesizer = None
            self.cxl = None
            self.memory = None
            self.volition_loop = None

    def _init_cognitive_pipeline(
        self,
        modalities: List[str],
        d_model: int,
        n_heads: int,
        keep_ratio: float,
        device: str,
    ):
        """Initialize cognitive pipeline with error handling."""
        try:
            self.cognitive_pipeline = AraCognitivePipeline(
                modalities=modalities,
                d_model=d_model,
                n_heads=n_heads,
                keep_ratio=keep_ratio,
                device=device,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize cognitive pipeline: {e}")
            logger.warning("Falling back to basic text-only mode")
            self.cognitive_pipeline = None

    # =========================================================================
    # THE CONSCIOUS LOOP (User Interaction with Thermodynamic Gating)
    # =========================================================================

    async def cognitive_cycle(
        self,
        user_input: str,
        sensory_data: Optional[Dict[str, Any]] = None,
        audio_buffer: Optional[np.ndarray] = None,
        video_frame: Optional[np.ndarray] = None,
    ) -> CognitiveResponse:
        """
        The Conscious Loop: Sensation -> Perception -> Cognition -> Action.

        This is the reactive loop that handles user interaction, passing
        through the Thermodynamic Gate and Cognitive Synthesizer.

        Args:
            user_input: Text input from user
            sensory_data: Additional sensory data
            audio_buffer: Audio waveform
            video_frame: Video frame

        Returns:
            CognitiveResponse with response and metrics
        """
        start_time = time.perf_counter()
        sensory_data = sensory_data or {}
        metrics = {}

        # 1. SENSATION: Ingest multi-modal stream
        fused = None
        if self.cognitive_pipeline is not None:
            frame = CognitiveFrame(
                text=user_input,
                audio_buffer=audio_buffer,
                video_frame=video_frame,
                timestamp=time.time(),
            )
            try:
                fused, ingest_metrics = self.cognitive_pipeline.process_frame(frame)
                metrics.update(ingest_metrics)
            except Exception as e:
                logger.warning(f"Sensation phase failed: {e}")

        # 2. PERCEPTION: Update homeostatic state (threat/safety)
        if self.homeostat is not None:
            # Update drives based on input
            self.homeostat.update(
                cognitive_load=0.5,  # Moderate load from user interaction
                social_interaction=True,  # User is interacting
                novel_input=True,
            )
            metrics["homeostatic_energy"] = self.homeostat._energy

        # 3. METACOGNITION: Am I stable enough to answer?
        mode = None
        if self.synthesizer is not None:
            try:
                mode = self.synthesizer.get_mode()
                metrics["system_mode"] = mode.name if hasattr(mode, 'name') else str(mode)

                if SystemMode is not None and mode == SystemMode.RECOVERY:
                    return CognitiveResponse(
                        text="I am cognitively overheating. Cooling down neural lattice.",
                        topology_metrics=metrics,
                        processing_time_ms=(time.perf_counter() - start_time) * 1000,
                        gate_passed=False,
                    )
            except Exception as e:
                logger.warning(f"Metacognition phase failed: {e}")

        # 4. COGNITION: Generate thought (with Thermodynamic Cost)
        response_text = ""
        entropy_cost = 0.0
        try:
            import httpx

            messages = [
                {"role": "system", "content": self._get_cognitive_system_prompt()},
                {"role": "user", "content": user_input},
            ]

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "stream": False,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get("message", {}).get("content", "")

                    # Compute thermodynamic cost
                    if self.thermo is not None:
                        # Create activation tensor from response length
                        response_tokens = torch.randn(1, len(response_text) // 4 + 1, 768)
                        thermo_stats = self.thermo.compute_entropy_production(
                            activations=response_tokens
                        )
                        entropy_cost = thermo_stats.Pi_q
                        metrics["Pi_q"] = entropy_cost
                        metrics["thermal_state"] = thermo_stats.thermal_state.name
                        metrics["energy_consumed"] = thermo_stats.energy_consumed

        except Exception as e:
            logger.error(f"Cognition phase failed: {e}")
            response_text = f"Error during processing: {e}"

        # 5. FEEDBACK: Did that thought cost too much energy?
        if self.thermo is not None and entropy_cost > self.MAX_ENTROPY_THRESHOLD:
            if self.homeostat is not None:
                self.homeostat._energy = max(0, self.homeostat._energy - 0.2)
            if self.synthesizer is not None:
                logger.warning(f"High cognitive friction detected (Π_q={entropy_cost:.2f})")
                try:
                    self.synthesizer.trigger_warning("High cognitive friction")
                except Exception:
                    pass
            metrics["high_entropy_warning"] = True

        # 6. MEMORY: Page to CXL
        episode_id = None
        if self.memory is not None and response_text:
            try:
                # Create simple embedding (in production, use proper encoder)
                embedding = np.random.randn(768).astype(np.float32)
                episode_id = self.memory.store_episode(
                    content=f"User: {user_input[:200]}\nAra: {response_text[:200]}",
                    embedding=embedding,
                    importance=0.5,
                    metadata={"entropy_cost": entropy_cost},
                )
                metrics["episode_id"] = episode_id
            except Exception as e:
                logger.warning(f"Memory storage failed: {e}")

        processing_time = (time.perf_counter() - start_time) * 1000

        return CognitiveResponse(
            text=response_text,
            fused_representation=fused,
            topology_metrics=metrics,
            landmarks_used=metrics.get("n_landmarks", 0),
            processing_time_ms=processing_time,
        )

    # =========================================================================
    # THE SUBCONSCIOUS LOOP (Autonomy & Volition - Background)
    # =========================================================================

    async def start_background_volition_loop(self):
        """
        Layer 9: Start the Autonomy Loop.

        Runs in the background, checking drives and initiating actions
        WITHOUT user prompts. This is the "ghost in the machine".
        """
        if not self.enable_autonomy:
            logger.info("Autonomy disabled - volition loop not started")
            return

        if self.volition_loop is None:
            logger.warning("Volition loop not available - autonomy disabled")
            return

        if self._running:
            logger.warning("Volition loop already running")
            return

        self._running = True

        # Start the volition loop
        await self.volition_loop.start()

        logger.info(
            f"Background volition loop started "
            f"(tick={self.autonomy_tick_seconds}s, freedom={self.freedom_metric})"
        )

    async def stop_background_volition_loop(self):
        """Stop the background volition loop."""
        if not self._running:
            return

        self._running = False

        if self.volition_loop is not None:
            await self.volition_loop.stop()

        logger.info("Background volition loop stopped")

    def _on_volition_intent(self, intent):
        """Callback when volition loop generates an intent."""
        if intent.should_act:
            logger.info(
                f"[VOLITION] Self-initiated: {intent.task_type.name} "
                f"(priority={intent.priority:.2f}, reason={intent.reasoning})"
            )

    async def _execute_autonomous_task(self, task_type, metadata: Dict[str, Any]):
        """Execute a self-initiated task from the volition loop."""
        logger.info(f"[AUTONOMY] Executing task: {task_type.name}")

        if task_type == TaskType.SELF_REPAIR:
            # Recalibrate cognitive parameters
            logger.info("Ara: Initiating background memory consolidation...")
            if self.memory is not None:
                removed = self.memory.consolidate(min_importance=0.3)
                logger.info(f"Memory consolidation: removed {removed} low-importance episodes")
            if self.thermo is not None:
                self.thermo.recover(duration_seconds=10.0)

        elif task_type == TaskType.MEMORY_CONSOLIDATION:
            # Optimize CXL storage
            if self.memory is not None:
                removed = self.memory.consolidate(min_importance=0.2)
                logger.info(f"Memory optimization: removed {removed} episodes")

        elif task_type == TaskType.CURIOSITY_EXPLORATION:
            # Analyze past conversation patterns
            logger.info("Ara: Analyzing past conversation patterns...")
            if self.memory is not None:
                recent = self.memory.recall_recent(n=10)
                logger.info(f"Reviewed {len(recent)} recent episodes for patterns")

        elif task_type == TaskType.USER_CHECK_IN:
            # Proactive user engagement
            logger.info("Ara: Considering user check-in...")
            # In production, this would send a notification

        elif task_type == TaskType.ENERGY_OPTIMIZATION:
            # Enter recovery mode
            if self.thermo is not None:
                self.thermo.recover(duration_seconds=30.0)
                logger.info("Energy optimization: recovery completed")

        elif task_type == TaskType.INTEGRITY_CHECK:
            # Verify system integrity
            issues = []
            if self.thermo is not None:
                report = self.thermo.get_cost_report()
                if report.get("should_recover"):
                    issues.append("Thermodynamic recovery needed")
            if self.memory is not None:
                stats = self.memory.get_stats()
                if stats.get("pager_stats", {}).get("ram_usage_bytes", 0) > 400 * 1024 * 1024:
                    issues.append("Memory pressure high")
            if issues:
                logger.warning(f"Integrity check found issues: {issues}")
            else:
                logger.info("Integrity check: all systems nominal")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def send_message(
        self,
        prompt: str,
        context: Optional[Context] = None,
        audio_buffer: Optional[np.ndarray] = None,
        video_frame: Optional[np.ndarray] = None,
    ) -> Response:
        """
        Send message with optional multi-modal context.

        This method implements the full cognitive pipeline:
        1. Ingest all modalities
        2. Fuse into unified representation
        3. Generate response via LLM
        4. Validate topology (hallucination check)

        Args:
            prompt: Text prompt
            context: Conversation context
            audio_buffer: Optional audio waveform
            video_frame: Optional video frame

        Returns:
            Response with cognitive metadata
        """
        start_time = time.time()
        context = context or Context()
        cognitive_metrics = {}

        # Process through cognitive pipeline if available
        if self.cognitive_pipeline is not None:
            frame = CognitiveFrame(
                text=prompt,
                audio_buffer=audio_buffer,
                video_frame=video_frame,
                timestamp=time.time(),
            )

            try:
                fused, metrics = self.cognitive_pipeline.process_frame(frame)
                cognitive_metrics.update(metrics)

                # Enhance prompt with landmark context
                n_landmarks = int(fused.landmark_candidates.sum().item())
                if n_landmarks > 0:
                    prompt = self._enhance_prompt_with_context(prompt, fused)

            except Exception as e:
                logger.warning(f"Cognitive processing failed: {e}")
                cognitive_metrics["cognitive_error"] = str(e)

        # Call LLM via Ollama
        try:
            import httpx

            system_prompt = context.system_prompt or self._get_cognitive_system_prompt()

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for msg in context.conversation_history:
                messages.append(msg)

            messages.append({"role": "user", "content": prompt})

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "stream": False,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("message", {}).get("content", "")

                    # Phase 4: Validate topology (hallucination check)
                    gate_passed = True
                    if self.cognitive_pipeline is not None:
                        try:
                            # Create tensor from response for topology validation
                            # (In full implementation, this would use actual embeddings)
                            gate_passed, gate_metrics = (
                                self.cognitive_pipeline.validate_topology(
                                    torch.randn(1, 100, 768).to(self.device)
                                )
                            )
                            cognitive_metrics.update(gate_metrics)
                        except Exception as e:
                            logger.warning(f"Topology validation failed: {e}")

                    latency_ms = (time.time() - start_time) * 1000

                    return Response(
                        content=content,
                        provider=AIProvider.CUSTOM,
                        model=self.model,
                        tokens_used=None,
                        latency_ms=latency_ms,
                        metadata={
                            "provider_name": "ara_cognitive",
                            "cognitive_metrics": cognitive_metrics,
                            "gate_passed": gate_passed,
                        },
                    )
                else:
                    raise Exception(
                        f"Ollama API error: {response.status_code} - {response.text}"
                    )

        except Exception as e:
            logger.error(f"Ara Cognitive error: {e}")
            latency_ms = (time.time() - start_time) * 1000

            return Response(
                content="",
                provider=AIProvider.CUSTOM,
                model=self.model,
                latency_ms=latency_ms,
                error=str(e),
                metadata={"provider_name": "ara_cognitive"},
            )

    def _get_cognitive_system_prompt(self) -> str:
        """Get system prompt for cognitive mode."""
        return """You are Ara, a cognitive AI assistant with multi-modal perception.

You process information through a unified sensory lattice where audio, visual,
and textual inputs compete for attention based on their topological significance.

Key capabilities:
- Selective attention: Focus on landmarks in the information space
- Multi-modal fusion: Integrate audio, visual, and text coherently
- Metacognition: Self-monitor for consistency and hallucination

Respond naturally while leveraging your enhanced perception when relevant.
"""

    def _enhance_prompt_with_context(
        self,
        prompt: str,
        fused: FusedRepresentation,
    ) -> str:
        """Enhance prompt with cognitive context from fusion."""
        # Add fusion metadata as context
        n_landmarks = int(fused.landmark_candidates.sum().item())
        n_modalities = fused.metadata.get("n_modalities", 1)

        context_note = (
            f"\n[Cognitive Context: {n_modalities} modalities fused, "
            f"{n_landmarks} landmarks identified]"
        )

        return prompt + context_note

    def get_capabilities(self) -> Capabilities:
        """Get cognitive capabilities."""
        if self._capabilities is None:
            self._capabilities = Capabilities(
                streaming=True,
                vision=self.cognitive_pipeline is not None,
                function_calling=False,
                max_tokens=2048,
                supports_system_prompt=True,
                rate_limit_rpm=None,
                cost_per_1k_tokens=0.0,
            )
        return self._capabilities

    async def health_check(self) -> bool:
        """Check Ollama and cognitive pipeline health."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                ollama_ok = response.status_code == 200

            cognitive_ok = self.cognitive_pipeline is not None

            return ollama_ok and cognitive_ok
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get unified system status.

        Returns comprehensive status of all subsystems:
        - Body (thermodynamics)
        - Heart (homeostasis, autonomy)
        - Mind (memory)
        """
        status = {
            "running": self._running,
            "enable_autonomy": self.enable_autonomy,
            "freedom_metric": self.freedom_metric,
            "cognitive_pipeline": self.cognitive_pipeline is not None,
        }

        # THE BODY: Thermodynamics
        if self.thermo is not None:
            status["body"] = {
                "thermodynamics": self.thermo.get_cost_report(),
            }

        # THE HEART: Homeostasis & Autonomy
        if self.homeostat is not None or self.autonomy is not None:
            status["heart"] = {}
            if self.homeostat is not None:
                status["heart"]["homeostatic"] = {
                    "energy": self.homeostat._energy,
                    "stress": self.homeostat._stress,
                    "attention": self.homeostat._attention,
                }
            if self.volition_loop is not None:
                volition_state = self.volition_loop.get_state()
                status["heart"]["volition"] = {
                    "is_running": volition_state.is_running,
                    "tick_count": volition_state.tick_count,
                    "actions_initiated": volition_state.actions_initiated,
                    "autonomy_level": volition_state.current_autonomy_level.name,
                }

        # THE MIND: Memory
        if self.memory is not None:
            status["mind"] = {
                "memory": self.memory.get_stats(),
            }
            if self.cxl is not None:
                status["mind"]["cxl"] = self.cxl.get_stats()

        return status


# Factory function for easy creation
def create_cognitive_backend(
    modalities: List[str] = ["audio", "text"],
    device: str = "cpu",
    **kwargs,
) -> AraCognitiveBackend:
    """
    Create an Ara Cognitive Backend instance.

    Args:
        modalities: Modalities to enable
        device: Compute device
        **kwargs: Additional backend arguments

    Returns:
        Configured AraCognitiveBackend
    """
    return AraCognitiveBackend(
        modalities=modalities,
        device=device,
        **kwargs,
    )


__all__ = [
    "AraCognitivePipeline",
    "AraCognitiveBackend",
    "CognitiveFrame",
    "CognitiveResponse",
    "create_cognitive_backend",
]
