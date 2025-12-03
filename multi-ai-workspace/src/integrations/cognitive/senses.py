"""Phase 1: Sensory Bridge - Multi-Modal Ingestion.

The SensoryCortex normalizes all sensory inputs (text, audio, video, IMU)
into unified, time-aligned ModalityStreams for downstream processing.

This replaces the separate ASR/text processing with a unified perceptual system.

Architecture:
    Raw Input -> Adapters -> ModalityStreams

    Text:  "Hello" -> TextAdapter -> [tokens, timestamps, embeddings]
    Audio: waveform -> AudioAdapter -> [mel-specs, prosody, timestamps]
    Video: frames   -> VideoAdapter -> [ViT patches, timestamps]
    IMU:   sensors  -> IMUAdapter   -> [motion vectors, timestamps]

All streams share the same output_dim (d_model) for fusion compatibility.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import warnings
import sys
from pathlib import Path

# Add TFAN to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Lazy imports for TFAN components
_TFAN_AVAILABLE = None


def _check_tfan_available() -> bool:
    """Check if TFAN components are available."""
    global _TFAN_AVAILABLE
    if _TFAN_AVAILABLE is not None:
        return _TFAN_AVAILABLE

    try:
        from tfan.mm.ingest import MultiModalIngestor
        _TFAN_AVAILABLE = True
    except ImportError:
        _TFAN_AVAILABLE = False
        warnings.warn("TFAN mm.ingest not available. Using fallback implementation.")

    return _TFAN_AVAILABLE


@dataclass
class ModalityInput:
    """Container for raw sensory input."""
    text: Optional[str] = None
    audio_buffer: Optional[np.ndarray] = None  # (samples,) or (batch, samples)
    video_frame: Optional[np.ndarray] = None   # (H, W, C) or (batch, C, H, W)
    imu_data: Optional[np.ndarray] = None      # (samples, 6) [accel_xyz, gyro_xyz]
    timestamp: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerceptionResult:
    """Result from sensory perception."""
    streams: Dict[str, Any]  # ModalityStream objects
    active_modalities: List[str]
    total_tokens: int
    perception_time_ms: float
    confidence_scores: Dict[str, float]


class SensoryCortex:
    """
    The Sensory Bridge - Unified Multi-Modal Perception.

    Normalizes all sensory inputs into ModalityStreams that can be
    fused by the Thalamus for conscious processing.

    This is the "eyes, ears, and skin" of the cognitive system.

    Args:
        modalities: List of modalities to support
        output_dim: Unified embedding dimension (d_model)
        sample_rate: Audio sample rate (Hz)
        video_fps: Video frame rate
        deterministic: Ensure reproducible outputs
        device: Compute device
    """

    def __init__(
        self,
        modalities: List[str] = ["text", "audio", "video"],
        output_dim: int = 4096,  # Matches TFAN-7B d_model
        sample_rate: int = 16000,
        video_fps: int = 30,
        deterministic: bool = True,
        device: str = "cpu",
    ):
        self.modalities = modalities
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.deterministic = deterministic
        self.device = device

        # Initialize TFAN ingestor if available
        self.ingestor = None
        self._init_ingestor()

    def _init_ingestor(self):
        """Initialize the TFAN MultiModalIngestor."""
        if not _check_tfan_available():
            warnings.warn("Using fallback sensory processing (no TFAN)")
            return

        try:
            from tfan.mm.ingest import MultiModalIngestor

            self.ingestor = MultiModalIngestor(
                modalities=self.modalities,
                output_dim=self.output_dim,
            )

        except Exception as e:
            warnings.warn(f"Failed to initialize TFAN ingestor: {e}")
            self.ingestor = None

    def perceive(
        self,
        text_input: Optional[str] = None,
        audio_buffer: Optional[np.ndarray] = None,
        video_frame: Optional[np.ndarray] = None,
        imu_data: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Normalize raw inputs into unified ModalityStreams.

        This is the primary perception interface. Raw sensory data flows in,
        and normalized vector streams flow out.

        Args:
            text_input: Raw text string
            audio_buffer: Audio waveform (samples,) at self.sample_rate
            video_frame: Video frame (H, W, C) RGB uint8 or (C, H, W) float
            imu_data: IMU readings (samples, 6) [accel_xyz, gyro_xyz]

        Returns:
            Dict mapping modality name -> ModalityStream
            Each stream contains:
                - features: (batch, seq_len, d_model)
                - timestamps: (batch, seq_len)
                - confidence: float [0, 1]
        """
        import time
        start_time = time.perf_counter()

        # Build input dictionary
        inputs = {}
        active_modalities = []

        if text_input is not None and "text" in self.modalities:
            inputs["text"] = [text_input]  # Wrap in list for batch
            active_modalities.append("text")

        if audio_buffer is not None and "audio" in self.modalities:
            audio_tensor = self._prepare_audio(audio_buffer)
            inputs["audio"] = audio_tensor
            active_modalities.append("audio")

        if video_frame is not None and "video" in self.modalities:
            video_tensor = self._prepare_video(video_frame)
            inputs["video"] = video_tensor
            active_modalities.append("video")

        if imu_data is not None and "imu" in self.modalities:
            imu_tensor = self._prepare_imu(imu_data)
            inputs["imu"] = imu_tensor
            active_modalities.append("imu")

        # Process through TFAN ingestor or fallback
        if self.ingestor is not None:
            streams = self.ingestor.ingest(inputs)
        else:
            streams = self._fallback_ingest(inputs)

        perception_time = (time.perf_counter() - start_time) * 1000

        # Compute confidence scores per modality
        confidence_scores = {}
        for modality, stream in streams.items():
            if hasattr(stream, 'confidence'):
                confidence_scores[modality] = stream.confidence
            else:
                confidence_scores[modality] = 1.0 if modality in active_modalities else 0.0

        return streams

    def perceive_frame(self, frame: ModalityInput) -> PerceptionResult:
        """
        Perceive from a ModalityInput container.

        Args:
            frame: ModalityInput with raw sensory data

        Returns:
            PerceptionResult with streams and metadata
        """
        import time
        start_time = time.perf_counter()

        streams = self.perceive(
            text_input=frame.text,
            audio_buffer=frame.audio_buffer,
            video_frame=frame.video_frame,
            imu_data=frame.imu_data,
        )

        # Count total tokens
        total_tokens = 0
        active_modalities = []
        confidence_scores = {}

        for modality, stream in streams.items():
            if hasattr(stream, 'features') and stream.features is not None:
                total_tokens += stream.features.shape[1]
                active_modalities.append(modality)
                confidence_scores[modality] = getattr(stream, 'confidence', 1.0)

        perception_time = (time.perf_counter() - start_time) * 1000

        return PerceptionResult(
            streams=streams,
            active_modalities=active_modalities,
            total_tokens=total_tokens,
            perception_time_ms=perception_time,
            confidence_scores=confidence_scores,
        )

    def _prepare_audio(self, audio: np.ndarray) -> torch.Tensor:
        """Prepare audio buffer for processing."""
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # Add batch dimension

        return torch.from_numpy(audio.astype(np.float32)).to(self.device)

    def _prepare_video(self, video: np.ndarray) -> torch.Tensor:
        """Prepare video frame for processing."""
        # Convert HWC uint8 to BCHW float
        if video.ndim == 3:
            if video.shape[-1] == 3:  # (H, W, C)
                video = np.transpose(video, (2, 0, 1))  # (C, H, W)
            video = video[np.newaxis, np.newaxis, ...]  # (1, 1, C, H, W)
        elif video.ndim == 4:  # (B, C, H, W)
            video = video[:, np.newaxis, ...]  # (B, 1, C, H, W)

        # Normalize to [0, 1] if uint8
        if video.dtype == np.uint8:
            video = video.astype(np.float32) / 255.0

        return torch.from_numpy(video).to(self.device)

    def _prepare_imu(self, imu: np.ndarray) -> torch.Tensor:
        """Prepare IMU data for processing."""
        if imu.ndim == 2:
            imu = imu[np.newaxis, ...]  # Add batch dimension

        return torch.from_numpy(imu.astype(np.float32)).to(self.device)

    def _fallback_ingest(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback ingestion when TFAN is not available."""
        from dataclasses import dataclass

        @dataclass
        class FallbackStream:
            features: torch.Tensor
            timestamps: torch.Tensor
            modality: str
            confidence: float = 1.0
            metadata: Optional[Dict] = None

        streams = {}

        for modality, data in inputs.items():
            if data is None:
                continue

            if modality == "text":
                # Simple text embedding placeholder
                text = data[0] if isinstance(data, list) else data
                seq_len = len(text.split()) + 2  # Rough token estimate
                features = torch.randn(1, seq_len, self.output_dim, device=self.device)
                timestamps = torch.linspace(0, 1, seq_len, device=self.device).unsqueeze(0)

            elif modality == "audio":
                # Audio: estimate frames from buffer length
                if isinstance(data, torch.Tensor):
                    n_samples = data.shape[-1]
                else:
                    n_samples = len(data)
                hop_length = 160  # Standard for 16kHz
                seq_len = n_samples // hop_length
                features = torch.randn(1, seq_len, self.output_dim, device=self.device)
                timestamps = torch.linspace(0, n_samples / self.sample_rate, seq_len, device=self.device).unsqueeze(0)

            elif modality == "video":
                # Video: one feature per frame
                if isinstance(data, torch.Tensor):
                    n_frames = data.shape[1] if data.dim() == 5 else 1
                else:
                    n_frames = 1
                features = torch.randn(1, n_frames, self.output_dim, device=self.device)
                timestamps = torch.linspace(0, n_frames / self.video_fps, n_frames, device=self.device).unsqueeze(0)

            elif modality == "imu":
                # IMU: direct mapping
                if isinstance(data, torch.Tensor):
                    seq_len = data.shape[1]
                else:
                    seq_len = data.shape[0]
                features = torch.randn(1, seq_len, self.output_dim, device=self.device)
                timestamps = torch.linspace(0, 1, seq_len, device=self.device).unsqueeze(0)

            else:
                continue

            streams[modality] = FallbackStream(
                features=features,
                timestamps=timestamps,
                modality=modality,
                confidence=1.0,
                metadata={"fallback": True},
            )

        return streams

    def get_active_adapters(self) -> List[str]:
        """Return list of initialized adapter names."""
        if self.ingestor is not None:
            return list(self.ingestor.adapters.keys())
        return self.modalities


# Convenience factory
def create_sensory_cortex(
    modalities: List[str] = ["text", "audio"],
    d_model: int = 4096,
    device: str = "cpu",
) -> SensoryCortex:
    """Create a SensoryCortex instance."""
    return SensoryCortex(
        modalities=modalities,
        output_dim=d_model,
        device=device,
    )


__all__ = [
    "SensoryCortex",
    "ModalityInput",
    "PerceptionResult",
    "create_sensory_cortex",
]
