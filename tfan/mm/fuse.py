"""
Multi-modal fusion and packing.

Builds unified token lattice with modality sentinels [MOD] and [FUSE].
Integrates with TLS for per-head landmark selection.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .ingest import ModalityStream
from ..attention import TLSLandmarkSelector


@dataclass
class FusedRepresentation:
    """Container for fused multi-modal representation."""
    tokens: torch.Tensor  # (batch, total_seq_len, d_model)
    modality_map: torch.Tensor  # (batch, total_seq_len) - integer modality IDs
    timestamps: torch.Tensor  # (batch, total_seq_len)
    landmark_candidates: torch.Tensor  # (batch, n_heads, total_seq_len) - boolean mask
    metadata: Dict


class ModalityTokens:
    """Special tokens for modality markers."""
    PAD = 0
    TEXT = 1
    AUDIO = 2
    VIDEO = 3
    IMU = 4
    FUSE = 5  # Fusion token
    MASK_MOD = 6  # Masked modality (stub)


class MultiModalFuser(nn.Module):
    """
    Multi-modal fusion module.

    Packs multiple modality streams into a unified sequence with:
    - Modality-specific embeddings
    - Temporal ordering
    - TLS landmark candidate selection
    """

    def __init__(
        self,
        d_model: int,
        modalities: List[str],
        packing_order: Optional[List[str]] = None,
        late_stream_degradation: float = 0.5,
        n_heads: int = 12,
        keep_ratio: float = 0.33,
        alpha: float = 0.7,
        per_head_masks: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            modalities: List of supported modalities
            packing_order: Order for packing (default: modalities order)
            late_stream_degradation: Weight reduction for late streams
            n_heads: Number of attention heads
            keep_ratio: TLS keep ratio
            alpha: TLS alpha parameter
            per_head_masks: Use per-head landmark masks
        """
        super().__init__()
        self.d_model = d_model
        self.modalities = modalities
        self.packing_order = packing_order or modalities
        self.late_stream_degradation = late_stream_degradation
        self.n_heads = n_heads

        # Modality embeddings
        self.modality_embeddings = nn.Embedding(10, d_model)  # Support up to 10 modalities

        # Fusion token
        self.fusion_token = nn.Parameter(torch.randn(d_model))

        # TLS landmark selector
        self.tls = TLSLandmarkSelector(
            keep_ratio=keep_ratio,
            alpha=alpha,
            per_head=per_head_masks,
        )

        # Modality name to ID mapping
        self.modality_to_id = {
            "text": ModalityTokens.TEXT,
            "audio": ModalityTokens.AUDIO,
            "video": ModalityTokens.VIDEO,
            "imu": ModalityTokens.IMU,
        }

    def forward(
        self,
        streams: Dict[str, ModalityStream],
    ) -> FusedRepresentation:
        """
        Fuse multi-modal streams into unified representation.

        Args:
            streams: Dictionary of modality -> ModalityStream

        Returns:
            FusedRepresentation with packed tokens and landmarks
        """
        batch_size = list(streams.values())[0].features.shape[0]

        # Pack streams according to order
        packed_tokens = []
        packed_modality_ids = []
        packed_timestamps = []

        for modality in self.packing_order:
            if modality not in streams:
                continue

            stream = streams[modality]

            # Get modality-specific features
            features = stream.features  # (batch, seq_len, d_model)
            timestamps = stream.timestamps  # (batch, seq_len)

            # Add modality embedding
            mod_id = self.modality_to_id.get(modality, ModalityTokens.MASK_MOD)
            mod_embedding = self.modality_embeddings(
                torch.tensor([mod_id], device=features.device)
            ).squeeze(0)  # (d_model,)

            # Broadcast and add
            features = features + mod_embedding.unsqueeze(0).unsqueeze(0)

            # Apply late stream degradation if confidence is low
            if stream.confidence < 1.0:
                features = features * stream.confidence

            packed_tokens.append(features)
            packed_modality_ids.append(
                torch.full((batch_size, features.shape[1]), mod_id, dtype=torch.long)
            )
            packed_timestamps.append(timestamps)

        # Add fusion token at the end
        fusion_tokens = self.fusion_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        packed_tokens.append(fusion_tokens)
        packed_modality_ids.append(
            torch.full((batch_size, 1), ModalityTokens.FUSE, dtype=torch.long)
        )
        # Fusion token timestamp = max timestamp + 1
        max_ts = max(ts.max() for ts in packed_timestamps)
        fusion_ts = torch.full((batch_size, 1), max_ts + 1.0)
        packed_timestamps.append(fusion_ts)

        # Concatenate all modalities
        fused_tokens = torch.cat(packed_tokens, dim=1)  # (batch, total_seq_len, d_model)
        fused_modality_ids = torch.cat(packed_modality_ids, dim=1).to(fused_tokens.device)  # (batch, total_seq_len)
        fused_timestamps = torch.cat(packed_timestamps, dim=1).to(fused_tokens.device)  # (batch, total_seq_len)

        # Select landmarks using TLS
        landmark_mask = self.tls.select_landmarks(
            fused_tokens,
            n_heads=self.n_heads,
        )  # (batch, n_heads, total_seq_len) or (batch, 1, total_seq_len)

        # Package results
        fused = FusedRepresentation(
            tokens=fused_tokens,
            modality_map=fused_modality_ids,
            timestamps=fused_timestamps,
            landmark_candidates=landmark_mask,
            metadata={
                "packing_order": self.packing_order,
                "n_modalities": len(packed_tokens) - 1,  # Exclude fusion token
                "total_seq_len": fused_tokens.shape[1],
            },
        )

        return fused


def pack_and_mask(
    tokens_by_mod: Dict[str, torch.Tensor],
    timestamps_by_mod: Dict[str, torch.Tensor],
    keep_ratio: float = 0.33,
    alpha: float = 0.7,
    per_head: bool = True,
    d_model: int = 768,
    n_heads: int = 12,
) -> FusedRepresentation:
    """
    Convenience function for packing and TLS masking.

    Args:
        tokens_by_mod: Dict of modality -> token embeddings (batch, seq_len, d_model)
        timestamps_by_mod: Dict of modality -> timestamps (batch, seq_len)
        keep_ratio: TLS keep ratio
        alpha: TLS alpha
        per_head: Per-head landmark selection
        d_model: Model dimension
        n_heads: Number of heads

    Returns:
        FusedRepresentation
    """
    # Convert to ModalityStreams
    streams = {}
    for modality, tokens in tokens_by_mod.items():
        streams[modality] = ModalityStream(
            features=tokens,
            timestamps=timestamps_by_mod[modality],
            modality=modality,
            confidence=1.0,
        )

    # Create fuser
    fuser = MultiModalFuser(
        d_model=d_model,
        modalities=list(tokens_by_mod.keys()),
        n_heads=n_heads,
        keep_ratio=keep_ratio,
        alpha=alpha,
        per_head_masks=per_head,
    )

    # Fuse
    fused = fuser(streams)

    return fused
