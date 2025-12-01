"""
Multi-modal input adapters.

Normalize various modalities to time-stamped feature streams:
- Text → tokens + timestamps
- Audio → log-mels + prosody + timestamps
- Video → ViT patches + timestamps
- IMU (optional) → sensor vectors + timestamps

Gates:
- Determinism on fixed seeds
- Shape/timebase invariants
- Fallback for missing streams
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import warnings

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa not available. Audio processing disabled.")

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    warnings.warn("timm not available. Video processing may be limited.")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers not available. Text processing may be limited.")


@dataclass
class ModalityStream:
    """Container for a modality stream."""
    features: torch.Tensor  # (batch, time, feat_dim)
    timestamps: torch.Tensor  # (batch, time)
    modality: str
    confidence: float = 1.0
    metadata: Optional[Dict] = None


class ModalityAdapter(nn.Module):
    """Base class for modality adapters."""

    def __init__(
        self,
        modality_name: str,
        output_dim: int,
        deterministic: bool = True,
    ):
        """
        Args:
            modality_name: Name of the modality
            output_dim: Output feature dimension
            deterministic: Ensure deterministic output for same input
        """
        super().__init__()
        self.modality_name = modality_name
        self.output_dim = output_dim
        self.deterministic = deterministic

    def forward(self, *args, **kwargs) -> ModalityStream:
        """
        Process input and return normalized stream.

        Returns:
            ModalityStream with features and timestamps
        """
        raise NotImplementedError

    def create_stub(self, batch_size: int, seq_len: int) -> ModalityStream:
        """
        Create stub stream for missing/failed modality.

        Args:
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Stub ModalityStream with [MASK_MOD] tokens
        """
        features = torch.zeros(batch_size, seq_len, self.output_dim)
        timestamps = torch.linspace(0, 1, seq_len).unsqueeze(0).expand(batch_size, -1)

        return ModalityStream(
            features=features,
            timestamps=timestamps,
            modality=self.modality_name,
            confidence=0.0,
            metadata={"is_stub": True},
        )


class TextAdapter(ModalityAdapter):
    """
    Text modality adapter.

    Tokenizes text and optionally generates timestamps.
    """

    def __init__(
        self,
        output_dim: int = 768,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        deterministic: bool = True,
    ):
        """
        Args:
            output_dim: Output embedding dimension
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            deterministic: Deterministic tokenization
        """
        super().__init__("text", output_dim, deterministic)
        self.max_length = max_length

        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                warnings.warn(f"Failed to load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None

        # Simple embedding layer (in practice, would use pre-trained model)
        self.embedding = nn.Embedding(30522, output_dim)  # BERT vocab size

    def forward(
        self,
        text: List[str],
        timestamps: Optional[torch.Tensor] = None,
    ) -> ModalityStream:
        """
        Process text input.

        Args:
            text: List of text strings (batch)
            timestamps: Optional pre-computed timestamps (batch, seq_len)

        Returns:
            ModalityStream with token embeddings
        """
        batch_size = len(text)

        if self.tokenizer is None:
            # Fallback: create stub
            return self.create_stub(batch_size, self.max_length)

        # Tokenize
        if self.deterministic:
            # Ensure deterministic behavior
            tokenized = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            tokenized = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        input_ids = tokenized["input_ids"]  # (batch, seq_len)
        seq_len = input_ids.shape[1]

        # Embed tokens
        embeddings = self.embedding(input_ids)  # (batch, seq_len, output_dim)

        # Generate timestamps if not provided
        if timestamps is None:
            # Uniform spacing
            timestamps = torch.linspace(0, 1, seq_len).unsqueeze(0).expand(batch_size, -1)

        return ModalityStream(
            features=embeddings,
            timestamps=timestamps,
            modality="text",
            confidence=1.0,
            metadata={"tokenizer": self.tokenizer.__class__.__name__},
        )


class AudioAdapter(ModalityAdapter):
    """
    Audio modality adapter.

    Converts audio to log-mel spectrograms with prosody features.
    """

    def __init__(
        self,
        output_dim: int = 768,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length: int = 160,
        deterministic: bool = True,
    ):
        """
        Args:
            output_dim: Output feature dimension
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            hop_length: Hop length for STFT
            deterministic: Deterministic processing
        """
        super().__init__("audio", output_dim, deterministic)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length

        # Projection layer from mel features to output_dim
        self.feature_proj = nn.Linear(n_mels, output_dim)

    def forward(
        self,
        audio: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> ModalityStream:
        """
        Process audio input.

        Args:
            audio: Audio waveform (batch, time) or (batch, time, 1)
            timestamps: Optional pre-computed timestamps

        Returns:
            ModalityStream with mel features
        """
        if audio.dim() == 3:
            audio = audio.squeeze(-1)  # (batch, time)

        batch_size = audio.shape[0]

        if not HAS_LIBROSA:
            # Fallback: create stub
            estimated_frames = audio.shape[1] // self.hop_length
            return self.create_stub(batch_size, estimated_frames)

        # Compute mel spectrograms
        mel_features = []
        for i in range(batch_size):
            audio_np = audio[i].cpu().numpy()

            # Compute mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio_np,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
            )

            # Convert to log scale
            log_mel = librosa.power_to_db(mel, ref=np.max)

            mel_features.append(log_mel.T)  # (time, n_mels)

        # Stack and convert to tensor
        mel_features = torch.from_numpy(np.stack(mel_features)).float()  # (batch, time, n_mels)

        # Project to output dimension
        features = self.feature_proj(mel_features)  # (batch, time, output_dim)

        # Generate timestamps if not provided
        seq_len = features.shape[1]
        if timestamps is None:
            # Time in seconds
            frame_times = np.arange(seq_len) * self.hop_length / self.sample_rate
            timestamps = torch.from_numpy(frame_times).unsqueeze(0).expand(batch_size, -1).float()

        return ModalityStream(
            features=features,
            timestamps=timestamps,
            modality="audio",
            confidence=1.0,
            metadata={
                "sample_rate": self.sample_rate,
                "n_mels": self.n_mels,
                "hop_length": self.hop_length,
            },
        )


class VideoAdapter(ModalityAdapter):
    """
    Video modality adapter.

    Extracts frame features using vision transformer (ViT).
    """

    def __init__(
        self,
        output_dim: int = 768,
        fps: int = 30,
        patch_size: int = 16,
        backbone: str = "vit_base_patch16_224",
        deterministic: bool = True,
    ):
        """
        Args:
            output_dim: Output feature dimension
            fps: Video frames per second
            patch_size: ViT patch size
            backbone: ViT backbone model name
            deterministic: Deterministic processing
        """
        super().__init__("video", output_dim, deterministic)
        self.fps = fps
        self.patch_size = patch_size
        self.backbone_name = backbone

        if HAS_TIMM:
            try:
                # Load pre-trained ViT
                self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
                self.backbone.eval()

                # Get backbone output dim
                with torch.no_grad():
                    dummy = torch.randn(1, 3, 224, 224)
                    backbone_dim = self.backbone(dummy).shape[-1]

                # Projection if needed
                if backbone_dim != output_dim:
                    self.proj = nn.Linear(backbone_dim, output_dim)
                else:
                    self.proj = nn.Identity()
            except Exception as e:
                warnings.warn(f"Failed to load ViT backbone: {e}")
                self.backbone = None
                self.proj = None
        else:
            self.backbone = None
            self.proj = None

    def forward(
        self,
        video: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> ModalityStream:
        """
        Process video input.

        Args:
            video: Video frames (batch, time, channels, height, width)
            timestamps: Optional pre-computed timestamps

        Returns:
            ModalityStream with frame features
        """
        batch_size, n_frames = video.shape[0], video.shape[1]

        if self.backbone is None:
            # Fallback: create stub
            return self.create_stub(batch_size, n_frames)

        # Extract features frame-by-frame
        features_list = []
        for i in range(batch_size):
            batch_features = []
            for t in range(n_frames):
                frame = video[i, t]  # (C, H, W)

                with torch.no_grad():
                    feat = self.backbone(frame.unsqueeze(0))  # (1, backbone_dim)
                    feat = self.proj(feat)  # (1, output_dim)

                batch_features.append(feat)

            batch_features = torch.cat(batch_features, dim=0)  # (n_frames, output_dim)
            features_list.append(batch_features)

        features = torch.stack(features_list)  # (batch, n_frames, output_dim)

        # Generate timestamps if not provided
        if timestamps is None:
            # Time in seconds
            frame_times = np.arange(n_frames) / self.fps
            timestamps = torch.from_numpy(frame_times).unsqueeze(0).expand(batch_size, -1).float()

        return ModalityStream(
            features=features,
            timestamps=timestamps,
            modality="video",
            confidence=1.0,
            metadata={
                "fps": self.fps,
                "backbone": self.backbone_name,
            },
        )


class IMUAdapter(ModalityAdapter):
    """
    IMU (Inertial Measurement Unit) adapter.

    Processes accelerometer/gyroscope data.
    """

    def __init__(
        self,
        output_dim: int = 768,
        sample_rate: int = 100,
        deterministic: bool = True,
    ):
        """
        Args:
            output_dim: Output feature dimension
            sample_rate: IMU sample rate (Hz)
            deterministic: Deterministic processing
        """
        super().__init__("imu", output_dim, deterministic)
        self.sample_rate = sample_rate

        # Simple MLP for IMU features
        # Typical IMU: 3 accel + 3 gyro = 6 channels
        self.feature_proj = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(
        self,
        imu_data: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> ModalityStream:
        """
        Process IMU input.

        Args:
            imu_data: IMU measurements (batch, time, 6)
                      [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            timestamps: Optional pre-computed timestamps

        Returns:
            ModalityStream with IMU features
        """
        batch_size, seq_len, channels = imu_data.shape

        if channels != 6:
            warnings.warn(f"Expected 6 IMU channels, got {channels}. Zero-padding.")
            # Pad or truncate to 6 channels
            if channels < 6:
                padding = torch.zeros(batch_size, seq_len, 6 - channels, device=imu_data.device)
                imu_data = torch.cat([imu_data, padding], dim=-1)
            else:
                imu_data = imu_data[:, :, :6]

        # Project to output dimension
        features = self.feature_proj(imu_data)  # (batch, seq_len, output_dim)

        # Generate timestamps if not provided
        if timestamps is None:
            time_points = np.arange(seq_len) / self.sample_rate
            timestamps = torch.from_numpy(time_points).unsqueeze(0).expand(batch_size, -1).float()

        return ModalityStream(
            features=features,
            timestamps=timestamps,
            modality="imu",
            confidence=1.0,
            metadata={"sample_rate": self.sample_rate},
        )


class MultiModalIngestor:
    """
    Orchestrator for multi-modal ingestion.

    Manages multiple adapters and handles missing modalities.
    """

    def __init__(
        self,
        modalities: List[str],
        output_dim: int = 768,
        **adapter_kwargs,
    ):
        """
        Args:
            modalities: List of modality names to support
            output_dim: Common output dimension for all modalities
            adapter_kwargs: Additional kwargs for specific adapters
        """
        self.modalities = modalities
        self.output_dim = output_dim

        self.adapters: Dict[str, ModalityAdapter] = {}

        for modality in modalities:
            if modality == "text":
                self.adapters["text"] = TextAdapter(output_dim=output_dim)
            elif modality == "audio":
                self.adapters["audio"] = AudioAdapter(output_dim=output_dim)
            elif modality == "video":
                self.adapters["video"] = VideoAdapter(output_dim=output_dim)
            elif modality == "imu":
                self.adapters["imu"] = IMUAdapter(output_dim=output_dim)
            else:
                warnings.warn(f"Unknown modality: {modality}")

    def ingest(
        self,
        inputs: Dict[str, any],
    ) -> Dict[str, ModalityStream]:
        """
        Ingest multi-modal inputs.

        Args:
            inputs: Dictionary mapping modality -> raw input

        Returns:
            Dictionary mapping modality -> ModalityStream
        """
        streams = {}

        for modality in self.modalities:
            if modality not in inputs or inputs[modality] is None:
                # Missing modality: create stub
                adapter = self.adapters.get(modality)
                if adapter is not None:
                    streams[modality] = adapter.create_stub(batch_size=1, seq_len=100)
                    warnings.warn(f"Missing modality: {modality}. Using stub.")
            else:
                # Process modality
                adapter = self.adapters[modality]
                try:
                    stream = adapter(inputs[modality])
                    streams[modality] = stream
                except Exception as e:
                    warnings.warn(f"Failed to process {modality}: {e}. Using stub.")
                    streams[modality] = adapter.create_stub(batch_size=1, seq_len=100)

        return streams
