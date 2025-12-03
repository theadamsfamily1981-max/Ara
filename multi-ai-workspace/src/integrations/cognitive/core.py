"""CognitiveCore - Unified Cognitive Processing Pipeline.

This module integrates all cognitive components into a single processing
pipeline that implements the full TFAN biomimetic architecture:

    Sensation -> Perception -> Self-Check -> Cognition -> Verification

The cognitive loop:
    1. SENSATION: Raw inputs -> SensoryCortex -> ModalityStreams
    2. PERCEPTION: ModalityStreams -> Thalamus -> ConsciousInput (filtered)
    3. SELF-PRESERVATION: ConsciousInput -> Conscience -> StabilityCheck
    4. COGNITION: ConsciousInput + AttentionMask -> Model -> RawThought
    5. REALITY CHECK: RawThought -> RealityMonitor -> VerifiedOutput

This is the "brain" that coordinates all cognitive functions.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import time
import warnings

from .senses import SensoryCortex, ModalityInput, PerceptionResult
from .thalamus import Thalamus, ConsciousInput
from .synthesizer import Conscience, SystemMode, AlertLevel, StabilityStatus, L7Metrics
from .reality_check import RealityMonitor, VerificationResult, VerificationStatus


@dataclass
class CognitiveOutput:
    """Output from full cognitive processing."""
    content: str                               # Decoded text response
    raw_output: Optional[torch.Tensor]         # Raw model output tensor
    conscious_input: Optional[ConsciousInput]  # Filtered input that was processed
    stability_status: StabilityStatus          # Self-preservation status
    verification: VerificationResult           # Reality check result

    # Metrics
    total_time_ms: float
    phase_times: Dict[str, float]
    n_landmarks: int
    sparsity_ratio: float
    modalities_used: list

    # Flags
    refused: bool = False         # True if conscience refused to process
    hallucination_blocked: bool = False  # True if reality check failed


class CognitiveCore:
    """
    The Cognitive Core - Unified Brain of Ara.

    Integrates all cognitive components into a single coherent system:
    - SensoryCortex (senses) - Multi-modal input normalization
    - Thalamus (filter) - Noise filtering via TLS
    - Conscience (self-preservation) - Stability monitoring
    - RealityMonitor (verification) - Hallucination detection

    Args:
        d_model: Model dimension (default 4096 for TFAN-7B)
        modalities: Supported modalities
        n_heads: Number of attention heads
        keep_ratio: TLS landmark keep ratio
        device: Compute device
    """

    def __init__(
        self,
        d_model: int = 4096,
        modalities: list = ["text", "audio", "video"],
        n_heads: int = 32,
        keep_ratio: float = 0.33,
        device: str = "cpu",
    ):
        self.d_model = d_model
        self.modalities = modalities
        self.n_heads = n_heads
        self.keep_ratio = keep_ratio
        self.device = device

        # Initialize cognitive components
        self.sensory_cortex = SensoryCortex(
            modalities=modalities,
            output_dim=d_model,
            device=device,
        )

        self.thalamus = Thalamus(
            d_model=d_model,
            modalities=modalities,
            n_heads=n_heads,
            keep_ratio=keep_ratio,
            device=device,
        )

        self.conscience = Conscience(
            structural_rate_threshold=0.3,
            recovery_threshold=0.1,
            device=device,
        )

        self.reality_monitor = RealityMonitor(
            d_model=d_model,
            wasserstein_gap_max=0.02,
            cosine_min=0.90,
            device=device,
        )

        # Current cognitive state
        self._current_conscious_input: Optional[ConsciousInput] = None
        self._current_metrics: Optional[L7Metrics] = None

    def cognitive_step(
        self,
        text_input: Optional[str] = None,
        audio_buffer: Optional[np.ndarray] = None,
        video_frame: Optional[np.ndarray] = None,
        model_fn: Optional[Callable[[ConsciousInput, torch.Tensor], torch.Tensor]] = None,
        decode_fn: Optional[Callable[[torch.Tensor], str]] = None,
        l7_metrics: Optional[L7Metrics] = None,
    ) -> CognitiveOutput:
        """
        Execute a full cognitive processing step.

        This is the main interface for cognitive processing. It implements
        the complete biomimetic loop:
            Sense -> Filter -> Check -> Think -> Verify

        Args:
            text_input: Text input (optional)
            audio_buffer: Audio waveform (optional)
            video_frame: Video frame (optional)
            model_fn: Function that takes (ConsciousInput, mask) and returns output tensor
            decode_fn: Function that decodes output tensor to text
            l7_metrics: Optional L7 layer metrics for stability check

        Returns:
            CognitiveOutput with all processing results
        """
        start_time = time.perf_counter()
        phase_times = {}

        # ========================================
        # Phase 1: SENSATION (SensoryCortex)
        # ========================================
        phase_start = time.perf_counter()

        sensory_streams = self.sensory_cortex.perceive(
            text_input=text_input,
            audio_buffer=audio_buffer,
            video_frame=video_frame,
        )

        phase_times["sensation_ms"] = (time.perf_counter() - phase_start) * 1000
        modalities_used = list(sensory_streams.keys())

        # ========================================
        # Phase 2: PERCEPTION (Thalamus)
        # ========================================
        phase_start = time.perf_counter()

        conscious_input, attention_mask = self.thalamus.process(sensory_streams)
        self._current_conscious_input = conscious_input

        # Cache input topology for reality check
        self.reality_monitor.set_input_topology(conscious_input.tokens)

        phase_times["perception_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Phase 3: SELF-PRESERVATION (Conscience)
        # ========================================
        phase_start = time.perf_counter()

        # Compute structural rate from conscious input if no metrics provided
        if l7_metrics is None:
            # Compute from topology change
            s_dot = self._estimate_structural_rate(conscious_input)
            l7_metrics = L7Metrics(
                structural_rate=s_dot,
                alert_level=AlertLevel.GREEN,
                entropy=0.0,
                coherence=1.0,
                stability_score=1.0 - s_dot,
                topology_drift=0.0,
            )

        stability_status = self.conscience.check_stability(l7_metrics=l7_metrics)
        self._current_metrics = l7_metrics

        phase_times["self_check_ms"] = (time.perf_counter() - phase_start) * 1000

        # Check if we should refuse to process
        if not stability_status.can_process:
            return CognitiveOutput(
                content=stability_status.message,
                raw_output=None,
                conscious_input=conscious_input,
                stability_status=stability_status,
                verification=VerificationResult(
                    status=VerificationStatus.VALIDATION_FAILED,
                    is_valid=False,
                    message="Processing refused - system in protective mode",
                    wasserstein_distance=0.0,
                    cosine_similarity=0.0,
                    gate_passed=False,
                    cat_activated=False,
                    retry_count=0,
                    metrics={},
                ),
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                phase_times=phase_times,
                n_landmarks=conscious_input.n_landmarks,
                sparsity_ratio=conscious_input.sparsity_ratio,
                modalities_used=modalities_used,
                refused=True,
                hallucination_blocked=False,
            )

        # ========================================
        # Phase 4: COGNITION (Model Inference)
        # ========================================
        phase_start = time.perf_counter()

        if model_fn is not None:
            # Use provided model function
            raw_output = model_fn(conscious_input, attention_mask)
        else:
            # Placeholder - return conscious input as "thought"
            raw_output = conscious_input.tokens

        phase_times["cognition_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Phase 5: REALITY CHECK (RealityMonitor)
        # ========================================
        phase_start = time.perf_counter()

        # Create recompute function for CAT fallback
        def recompute_with_keep_ratio(new_keep_ratio: float) -> torch.Tensor:
            # Reprocess with higher keep_ratio
            self.thalamus.keep_ratio = new_keep_ratio
            new_conscious, new_mask = self.thalamus.process(sensory_streams)
            self.thalamus.keep_ratio = self.keep_ratio  # Restore

            if model_fn is not None:
                return model_fn(new_conscious, new_mask)
            return new_conscious.tokens

        verification = self.reality_monitor.verify(
            model_output=raw_output,
            recompute_fn=recompute_with_keep_ratio if model_fn else None,
        )

        phase_times["verification_ms"] = (time.perf_counter() - phase_start) * 1000

        # ========================================
        # Decode Output
        # ========================================
        if decode_fn is not None and verification.is_valid:
            content = decode_fn(raw_output)
        elif not verification.is_valid:
            content = (
                f"I detected a potential inconsistency in my response. "
                f"{verification.message}"
            )
        else:
            content = "[Cognitive processing complete - no decoder provided]"

        total_time = (time.perf_counter() - start_time) * 1000

        return CognitiveOutput(
            content=content,
            raw_output=raw_output,
            conscious_input=conscious_input,
            stability_status=stability_status,
            verification=verification,
            total_time_ms=total_time,
            phase_times=phase_times,
            n_landmarks=conscious_input.n_landmarks,
            sparsity_ratio=conscious_input.sparsity_ratio,
            modalities_used=modalities_used,
            refused=False,
            hallucination_blocked=not verification.is_valid,
        )

    def _estimate_structural_rate(self, conscious_input: ConsciousInput) -> float:
        """Estimate structural rate from conscious input."""
        if self._current_conscious_input is None:
            return 0.0

        return self.conscience.compute_structural_rate(
            current_topology=conscious_input.tokens,
            previous_topology=self._current_conscious_input.tokens,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current cognitive system status."""
        return {
            "mode": self.conscience.mode.name,
            "alert_level": self.conscience.alert_level.name,
            "can_process": self.conscience.mode != SystemMode.PROTECTIVE,
            "reality_check_stats": self.reality_monitor.get_statistics(),
            "conscience_summary": self.conscience.get_status_summary(),
            "active_modalities": self.sensory_cortex.get_active_adapters(),
        }

    def reset(self):
        """Reset cognitive state."""
        self.conscience.reset()
        self.reality_monitor.reset_statistics()
        self._current_conscious_input = None
        self._current_metrics = None


# Convenience factory
def create_cognitive_core(
    d_model: int = 4096,
    modalities: list = ["text", "audio"],
    device: str = "cpu",
) -> CognitiveCore:
    """Create a CognitiveCore instance."""
    return CognitiveCore(
        d_model=d_model,
        modalities=modalities,
        device=device,
    )


__all__ = [
    "CognitiveCore",
    "CognitiveOutput",
    "create_cognitive_core",
]
