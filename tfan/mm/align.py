"""
Multi-modal alignment wrapper using TTW-Sentry.

Exposes precursor triggers and meets p95 < 5 ms latency requirement.
"""

import torch
from typing import Dict, Tuple, Optional
from ..ttw import TTWSentry, AlignmentMetrics
from .ingest import ModalityStream


def align_streams(
    streams: Dict[str, ModalityStream],
    trigger: Optional[Dict[str, bool]] = None,
    max_iter: int = 50,
    p95_latency_ms: float = 5.0,
    coverage_target: float = 0.90,
) -> Tuple[Dict[str, ModalityStream], AlignmentMetrics]:
    """
    Align multi-modal streams to common timebase using TTW-Sentry.

    Args:
        streams: Dictionary of modality -> ModalityStream
        trigger: Optional trigger override dict (e.g., {"vfe_spike": True})
        max_iter: Maximum alignment iterations
        p95_latency_ms: Target p95 latency
        coverage_target: Minimum trigger coverage

    Returns:
        (aligned_streams, alignment_metrics)
    """
    # Create TTW-Sentry instance
    sentry = TTWSentry(
        max_iter=max_iter,
        p95_latency_ms=p95_latency_ms,
        coverage_target=coverage_target,
    )

    # Convert ModalityStreams to TTW format
    ttw_input = {}
    for modality, stream in streams.items():
        ttw_input[modality] = (stream.features, stream.timestamps)

    # Determine if we should force alignment
    force_align = False
    if trigger is not None and any(trigger.values()):
        force_align = True

    # Perform alignment
    aligned_ttw = sentry.align(ttw_input, force_align=force_align)

    # Convert back to ModalityStreams
    aligned_streams = {}
    for modality, (features, timestamps) in aligned_ttw.items():
        original_stream = streams[modality]
        aligned_streams[modality] = ModalityStream(
            features=features,
            timestamps=timestamps,
            modality=modality,
            confidence=original_stream.confidence,
            metadata=original_stream.metadata,
        )

    # Get metrics
    metrics = sentry.get_metrics()

    return aligned_streams, metrics


def validate_alignment(
    aligned_streams: Dict[str, ModalityStream],
) -> Dict[str, bool]:
    """
    Validate that all streams have compatible timebases.

    Args:
        aligned_streams: Dictionary of aligned streams

    Returns:
        Dictionary of validation checks
    """
    checks = {}

    # Check 1: All streams have same sequence length
    seq_lengths = [stream.features.shape[1] for stream in aligned_streams.values()]
    checks["uniform_length"] = len(set(seq_lengths)) == 1

    # Check 2: Timestamps are monotonic
    all_monotonic = True
    for stream in aligned_streams.values():
        ts = stream.timestamps
        monotonic = torch.all(ts[:, 1:] >= ts[:, :-1])
        if not monotonic:
            all_monotonic = False
            break
    checks["monotonic_timestamps"] = all_monotonic

    # Check 3: No NaN or Inf in features
    all_finite = True
    for stream in aligned_streams.values():
        if not torch.all(torch.isfinite(stream.features)):
            all_finite = False
            break
    checks["finite_features"] = all_finite

    return checks
