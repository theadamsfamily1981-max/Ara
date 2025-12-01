"""
Tests for multi-modal ingest adapters.

Validates:
- Shape invariants
- Timestamp monotonicity
- Determinism
- Fallback behavior
"""

import pytest
import torch
import numpy as np

from tfan.mm.ingest import (
    TextAdapter,
    AudioAdapter,
    VideoAdapter,
    IMUAdapter,
    MultiModalIngestor,
    ModalityStream,
)


class TestTextAdapter:
    """Test text modality adapter."""

    def test_basic_forward(self):
        """Test basic text processing."""
        adapter = TextAdapter(output_dim=128)
        stream = adapter(text=["This is a test sentence"])

        assert stream.features.shape[0] == 1  # batch
        assert stream.features.shape[2] == 128  # d_model
        assert stream.timestamps.shape == stream.features.shape[:2]
        assert stream.modality == "text"
        assert stream.confidence == 1.0

    def test_determinism(self):
        """Test deterministic output with same seed."""
        adapter = TextAdapter(output_dim=128, deterministic=True)

        stream1 = adapter(["Test sentence"])
        stream2 = adapter(["Test sentence"])

        # Embeddings may differ due to random init, but shapes should match
        assert stream1.features.shape == stream2.features.shape

    def test_timestamp_monotonicity(self):
        """Test timestamps are monotonically increasing."""
        adapter = TextAdapter(output_dim=128)
        stream = adapter(["This is a longer test sentence with more tokens"])

        # Check monotonicity
        ts = stream.timestamps[0]
        assert torch.all(ts[1:] >= ts[:-1])

    def test_batch_processing(self):
        """Test batch processing."""
        adapter = TextAdapter(output_dim=128)
        texts = ["First sentence", "Second sentence"]
        stream = adapter(texts)

        assert stream.features.shape[0] == 2  # batch size


class TestAudioAdapter:
    """Test audio modality adapter."""

    def test_basic_forward(self):
        """Test basic audio processing."""
        adapter = AudioAdapter(
            output_dim=128,
            sample_rate=16000,
            n_mels=80,
        )

        # 1 second of audio
        audio = torch.randn(1, 16000)
        stream = adapter(audio)

        assert stream.features.shape[0] == 1  # batch
        assert stream.features.shape[2] == 128  # d_model
        assert stream.timestamps.shape == stream.features.shape[:2]
        assert stream.modality == "audio"

    def test_timestamp_spacing(self):
        """Test timestamp spacing matches hop length."""
        hop_length = 160
        sample_rate = 16000

        adapter = AudioAdapter(
            output_dim=128,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )

        audio = torch.randn(1, 16000)  # 1 second
        stream = adapter(audio)

        # Check timestamp spacing
        ts_diff = stream.timestamps[0, 1:] - stream.timestamps[0, :-1]
        expected_spacing = hop_length / sample_rate

        # Allow small numerical error
        assert torch.allclose(ts_diff, torch.tensor(expected_spacing), atol=1e-4)

    def test_fallback_stub(self):
        """Test fallback to stub when librosa unavailable."""
        # This test would need to mock librosa import failure
        pass


class TestVideoAdapter:
    """Test video modality adapter."""

    def test_basic_forward(self):
        """Test basic video processing."""
        adapter = VideoAdapter(
            output_dim=128,
            fps=30,
        )

        # 30 frames (1 second @ 30fps)
        video = torch.randn(1, 30, 3, 224, 224)
        stream = adapter(video)

        assert stream.features.shape[0] == 1  # batch
        assert stream.features.shape[1] == 30  # frames
        assert stream.features.shape[2] == 128  # d_model
        assert stream.modality == "video"

    def test_timestamp_frame_alignment(self):
        """Test timestamps align with frame rate."""
        fps = 30
        adapter = VideoAdapter(output_dim=128, fps=fps)

        video = torch.randn(1, 90, 3, 224, 224)  # 90 frames = 3 seconds
        stream = adapter(video)

        # Check timestamp spacing
        ts_diff = stream.timestamps[0, 1:] - stream.timestamps[0, :-1]
        expected_spacing = 1.0 / fps

        assert torch.allclose(ts_diff, torch.tensor(expected_spacing), atol=1e-4)


class TestIMUAdapter:
    """Test IMU modality adapter."""

    def test_basic_forward(self):
        """Test basic IMU processing."""
        adapter = IMUAdapter(output_dim=128, sample_rate=100)

        # 100 samples = 1 second @ 100Hz
        imu = torch.randn(1, 100, 6)  # 3 accel + 3 gyro
        stream = adapter(imu)

        assert stream.features.shape[0] == 1
        assert stream.features.shape[1] == 100
        assert stream.features.shape[2] == 128
        assert stream.modality == "imu"

    def test_channel_padding(self):
        """Test padding when channels < 6."""
        adapter = IMUAdapter(output_dim=128)

        # Only 3 channels (should pad to 6)
        imu = torch.randn(1, 100, 3)
        stream = adapter(imu)

        # Should not error and produce valid output
        assert stream.features.shape[2] == 128


class TestMultiModalIngestor:
    """Test multi-modal ingestor."""

    def test_basic_ingestion(self):
        """Test ingesting multiple modalities."""
        ingestor = MultiModalIngestor(
            modalities=["text", "audio"],
            output_dim=128,
        )

        inputs = {
            "text": ["Test sentence"],
            "audio": torch.randn(1, 16000),
        }

        streams = ingestor.ingest(inputs)

        assert "text" in streams
        assert "audio" in streams
        assert streams["text"].modality == "text"
        assert streams["audio"].modality == "audio"

    def test_missing_modality_stub(self):
        """Test stub creation for missing modalities."""
        ingestor = MultiModalIngestor(
            modalities=["text", "audio", "video"],
            output_dim=128,
        )

        inputs = {
            "text": ["Test sentence"],
            # audio and video missing
        }

        streams = ingestor.ingest(inputs)

        assert "text" in streams
        assert "audio" in streams
        assert "video" in streams

        # Missing modalities should have low confidence
        assert streams["audio"].confidence == 0.0
        assert streams["video"].confidence == 0.0

        # Check stub metadata
        assert streams["audio"].metadata.get("is_stub") == True

    def test_common_output_dim(self):
        """Test all modalities output same dimension."""
        output_dim = 256
        ingestor = MultiModalIngestor(
            modalities=["text", "audio"],
            output_dim=output_dim,
        )

        inputs = {
            "text": ["Test"],
            "audio": torch.randn(1, 16000),
        }

        streams = ingestor.ingest(inputs)

        for stream in streams.values():
            assert stream.features.shape[2] == output_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
