"""Phase 2: Thalamus - Fusion & Filtering (The Subconscious Gate).

The Thalamus fuses multi-modal sensory streams and filters noise using
Topological Landmark Selection (TLS). This is the "subconscious" that
decides what reaches conscious awareness.

Architecture:
    ModalityStreams -> Fusion -> TLS Filter -> ConsciousInput

    Key Operations:
    1. Interleave streams with sentinel tokens [AUDIO], [VIDEO], [FUSE]
    2. Add modality-specific embeddings
    3. TLS selects topologically significant landmarks (keep_ratio=0.33)
    4. Output unified token lattice with attention mask

The TLS score combines:
    - Persistence lifetime (how long features persist in topology)
    - Max-min distance (diversity/novelty)

    TLS_score = α · persistence + (1 - α) · diversity
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import warnings
import sys
from pathlib import Path

# Add TFAN to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Lazy imports
_TFAN_FUSER_AVAILABLE = None


def _check_fuser_available() -> bool:
    """Check if TFAN fuser is available."""
    global _TFAN_FUSER_AVAILABLE
    if _TFAN_FUSER_AVAILABLE is not None:
        return _TFAN_FUSER_AVAILABLE

    try:
        from tfan.mm.fuse import MultiModalFuser
        _TFAN_FUSER_AVAILABLE = True
    except ImportError:
        _TFAN_FUSER_AVAILABLE = False
        warnings.warn("TFAN mm.fuse not available. Using fallback implementation.")

    return _TFAN_FUSER_AVAILABLE


@dataclass
class ConsciousInput:
    """The filtered, fused input ready for cognitive processing."""
    tokens: torch.Tensor           # (batch, seq_len, d_model)
    modality_map: torch.Tensor     # (batch, seq_len) - modality IDs
    timestamps: torch.Tensor       # (batch, seq_len) - time alignment
    landmark_mask: torch.Tensor    # (batch, n_heads, seq_len) - attention mask
    attention_mask: torch.Tensor   # (batch, seq_len, seq_len) - full sparse mask

    # Metadata
    n_landmarks: int
    n_discarded: int
    sparsity_ratio: float
    modalities_present: List[str]
    fusion_time_ms: float


class ModalityTokens:
    """Special token IDs for modality markers."""
    PAD = 0
    TEXT = 1
    AUDIO = 2
    VIDEO = 3
    IMU = 4
    FUSE = 5       # Fusion sentinel
    MASK_MOD = 6   # Masked/missing modality


class Thalamus(nn.Module):
    """
    The Thalamic Gate - Fusion & Subconscious Filtering.

    Fuses multi-modal streams into a unified token lattice and applies
    TLS (Topological Landmark Selection) to filter noise before it
    reaches conscious processing.

    This physically removes ~67% of "boring" sensory data, focusing
    attention on topologically significant features.

    Args:
        d_model: Model dimension (must match SensoryCortex output_dim)
        modalities: List of modality names
        n_heads: Number of attention heads
        keep_ratio: Fraction of tokens to keep as landmarks (default 0.33)
        tls_alpha: TLS blend factor (persistence vs diversity)
        local_window: Local attention window size
        device: Compute device
    """

    def __init__(
        self,
        d_model: int = 4096,
        modalities: List[str] = ["text", "audio", "video"],
        n_heads: int = 32,
        keep_ratio: float = 0.33,  # Discard 67% of "boring" data
        tls_alpha: float = 0.7,
        local_window: int = 128,
        device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.modalities = modalities
        self.n_heads = n_heads
        self.keep_ratio = keep_ratio
        self.tls_alpha = tls_alpha
        self.local_window = local_window
        self.device = device

        # Initialize TFAN fuser if available
        self.fuser = None
        self._init_fuser()

        # Fallback components if TFAN not available
        if self.fuser is None:
            self._init_fallback_components()

        # Move to device
        self.to(device)

    def _init_fuser(self):
        """Initialize TFAN MultiModalFuser."""
        if not _check_fuser_available():
            return

        try:
            from tfan.mm.fuse import MultiModalFuser

            self.fuser = MultiModalFuser(
                d_model=self.d_model,
                modalities=self.modalities,
                n_heads=self.n_heads,
                keep_ratio=self.keep_ratio,
                alpha=self.tls_alpha,
                per_head_masks=True,
            )

        except Exception as e:
            warnings.warn(f"Failed to initialize TFAN fuser: {e}")
            self.fuser = None

    def _init_fallback_components(self):
        """Initialize fallback fusion components."""
        # Modality embeddings
        self.modality_embeddings = nn.Embedding(10, self.d_model)

        # Fusion token
        self.fusion_token = nn.Parameter(torch.randn(self.d_model))

        # Modality name to ID mapping
        self.modality_to_id = {
            "text": ModalityTokens.TEXT,
            "audio": ModalityTokens.AUDIO,
            "video": ModalityTokens.VIDEO,
            "imu": ModalityTokens.IMU,
        }

    def process(
        self,
        sensory_streams: Dict[str, Any],
    ) -> Tuple[ConsciousInput, torch.Tensor]:
        """
        Fuse sensory streams and filter via TLS.

        This is the main thalamic processing step:
        1. Interleave modality streams with sentinel tokens
        2. Add modality-specific embeddings
        3. Apply TLS to select landmarks
        4. Build sparse attention mask

        Args:
            sensory_streams: Dict of modality -> ModalityStream from SensoryCortex

        Returns:
            (ConsciousInput, attention_mask)
            - ConsciousInput: Fused, filtered representation
            - attention_mask: Sparse mask for SSA attention
        """
        import time
        start_time = time.perf_counter()

        if self.fuser is not None:
            # Use TFAN fuser
            conscious_input = self._process_with_tfan(sensory_streams)
        else:
            # Use fallback implementation
            conscious_input = self._process_fallback(sensory_streams)

        fusion_time = (time.perf_counter() - start_time) * 1000
        conscious_input.fusion_time_ms = fusion_time

        return conscious_input, conscious_input.attention_mask

    def _process_with_tfan(self, sensory_streams: Dict[str, Any]) -> ConsciousInput:
        """Process using TFAN MultiModalFuser."""
        # Convert to TFAN format if needed
        from tfan.mm.ingest import ModalityStream

        tfan_streams = {}
        for modality, stream in sensory_streams.items():
            if isinstance(stream, ModalityStream):
                tfan_streams[modality] = stream
            else:
                # Wrap in ModalityStream
                tfan_streams[modality] = ModalityStream(
                    features=stream.features.to(self.device),
                    timestamps=stream.timestamps.to(self.device),
                    modality=modality,
                    confidence=getattr(stream, 'confidence', 1.0),
                )

        # Fuse with TFAN
        fused = self.fuser(tfan_streams)

        # Build full attention mask from landmarks
        attention_mask = self._build_attention_mask(
            fused.landmark_candidates,
            fused.tokens.shape[1],
        )

        # Compute statistics
        n_landmarks = int(fused.landmark_candidates.sum().item())
        total_tokens = fused.tokens.shape[1]
        n_discarded = total_tokens - n_landmarks
        sparsity = n_discarded / max(total_tokens, 1)

        return ConsciousInput(
            tokens=fused.tokens,
            modality_map=fused.modality_map,
            timestamps=fused.timestamps,
            landmark_mask=fused.landmark_candidates,
            attention_mask=attention_mask,
            n_landmarks=n_landmarks,
            n_discarded=n_discarded,
            sparsity_ratio=sparsity,
            modalities_present=list(tfan_streams.keys()),
            fusion_time_ms=0.0,  # Will be set by caller
        )

    def _process_fallback(self, sensory_streams: Dict[str, Any]) -> ConsciousInput:
        """Fallback fusion without TFAN."""
        batch_size = 1
        packed_tokens = []
        packed_modality_ids = []
        packed_timestamps = []
        modalities_present = []

        for modality in self.modalities:
            if modality not in sensory_streams:
                continue

            stream = sensory_streams[modality]
            features = stream.features.to(self.device)
            timestamps = stream.timestamps.to(self.device)

            # Ensure batch dimension
            if features.dim() == 2:
                features = features.unsqueeze(0)
            if timestamps.dim() == 1:
                timestamps = timestamps.unsqueeze(0)

            batch_size = features.shape[0]
            seq_len = features.shape[1]

            # Add modality embedding
            mod_id = self.modality_to_id.get(modality, ModalityTokens.MASK_MOD)
            mod_embedding = self.modality_embeddings(
                torch.tensor([mod_id], device=self.device)
            ).squeeze(0)
            features = features + mod_embedding.unsqueeze(0).unsqueeze(0)

            packed_tokens.append(features)
            packed_modality_ids.append(
                torch.full((batch_size, seq_len), mod_id, dtype=torch.long, device=self.device)
            )
            packed_timestamps.append(timestamps)
            modalities_present.append(modality)

        # Add fusion token
        fusion_tokens = self.fusion_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        packed_tokens.append(fusion_tokens)
        packed_modality_ids.append(
            torch.full((batch_size, 1), ModalityTokens.FUSE, dtype=torch.long, device=self.device)
        )
        max_ts = max(ts.max() for ts in packed_timestamps) if packed_timestamps else 0
        packed_timestamps.append(
            torch.full((batch_size, 1), max_ts + 1.0, device=self.device)
        )

        # Concatenate
        fused_tokens = torch.cat(packed_tokens, dim=1)
        fused_modality_ids = torch.cat(packed_modality_ids, dim=1)
        fused_timestamps = torch.cat(packed_timestamps, dim=1)

        total_seq_len = fused_tokens.shape[1]

        # TLS landmark selection (fallback implementation)
        landmark_mask = self._compute_tls_landmarks(fused_tokens)

        # Build attention mask
        attention_mask = self._build_attention_mask(landmark_mask, total_seq_len)

        # Statistics
        n_landmarks = int(landmark_mask.sum().item())
        n_discarded = total_seq_len - n_landmarks
        sparsity = n_discarded / max(total_seq_len, 1)

        return ConsciousInput(
            tokens=fused_tokens,
            modality_map=fused_modality_ids,
            timestamps=fused_timestamps,
            landmark_mask=landmark_mask,
            attention_mask=attention_mask,
            n_landmarks=n_landmarks,
            n_discarded=n_discarded,
            sparsity_ratio=sparsity,
            modalities_present=modalities_present,
            fusion_time_ms=0.0,
        )

    def _compute_tls_landmarks(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute TLS landmarks (fallback implementation).

        TLS_score = α · persistence + (1 - α) · diversity
        """
        batch, seq_len, d_model = tokens.shape
        n_keep = max(int(seq_len * self.keep_ratio), 2)

        # Compute persistence (distance from centroid)
        centroid = tokens.mean(dim=1, keepdim=True)
        persistence = torch.norm(tokens - centroid, dim=2)
        persistence = persistence / (persistence.max(dim=1, keepdim=True)[0] + 1e-8)

        # Compute diversity (distance to nearest neighbors)
        distances = torch.cdist(tokens, tokens, p=2)
        k = min(10, seq_len)
        topk_distances, _ = torch.topk(distances, k=k, dim=2, largest=False)
        diversity = topk_distances[:, :, -1]
        diversity = diversity / (diversity.max(dim=1, keepdim=True)[0] + 1e-8)

        # TLS score
        tls_score = self.tls_alpha * persistence + (1 - self.tls_alpha) * diversity

        # Select top-k
        _, top_indices = torch.topk(tls_score, k=n_keep, dim=1)

        # Create mask
        landmarks = torch.zeros(batch, seq_len, dtype=torch.bool, device=self.device)
        landmarks.scatter_(1, top_indices, True)

        # Expand for heads
        landmarks = landmarks.unsqueeze(1).expand(-1, self.n_heads, -1)

        return landmarks

    def _build_attention_mask(
        self,
        landmark_mask: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Build sparse attention mask from landmarks."""
        batch = landmark_mask.shape[0]

        # Local window mask
        local_mask = self._local_window_mask(seq_len)

        # Each token attends to landmarks + local window
        if landmark_mask.dim() == 3:
            # Per-head landmarks: (batch, n_heads, seq_len)
            landmark_attn = landmark_mask.unsqueeze(2).expand(-1, -1, seq_len, -1)
        else:
            # Shared landmarks: (batch, seq_len)
            landmark_attn = landmark_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.n_heads, seq_len, -1)

        # Combine local + landmarks
        combined_mask = local_mask.unsqueeze(0).unsqueeze(0) | landmark_attn

        return combined_mask

    def _local_window_mask(self, seq_len: int) -> torch.Tensor:
        """Create local sliding window mask."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=self.device)

        for i in range(seq_len):
            start = max(0, i - self.local_window // 2)
            end = min(seq_len, i + self.local_window // 2 + 1)
            mask[i, start:end] = True

        return mask

    def get_sparsity_stats(self, conscious_input: ConsciousInput) -> Dict[str, float]:
        """Get statistics about the filtering."""
        return {
            "n_landmarks": conscious_input.n_landmarks,
            "n_discarded": conscious_input.n_discarded,
            "sparsity_ratio": conscious_input.sparsity_ratio,
            "keep_ratio": 1.0 - conscious_input.sparsity_ratio,
            "fusion_time_ms": conscious_input.fusion_time_ms,
        }


# Convenience factory
def create_thalamus(
    d_model: int = 4096,
    keep_ratio: float = 0.33,
    n_heads: int = 32,
    device: str = "cpu",
) -> Thalamus:
    """Create a Thalamus instance."""
    return Thalamus(
        d_model=d_model,
        keep_ratio=keep_ratio,
        n_heads=n_heads,
        device=device,
    )


__all__ = [
    "Thalamus",
    "ConsciousInput",
    "ModalityTokens",
    "create_thalamus",
]
