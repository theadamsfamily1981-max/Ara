"""
Retrieval Telemetry - Metrics Collection and Reporting
======================================================
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging


logger = logging.getLogger(__name__)


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class RetrievalMetrics:
    """Aggregated retrieval metrics."""
    timestamp: float = field(default_factory=time.time)

    # Counts
    total_retrievals: int = 0
    successful_retrievals: int = 0
    empty_results: int = 0

    # Latency
    avg_total_latency_us: float = 0.0
    avg_stage1_latency_us: float = 0.0
    avg_stage2_latency_us: float = 0.0
    max_total_latency_us: float = 0.0
    p99_total_latency_us: float = 0.0

    # Recall (estimated)
    avg_recall: float = 0.0

    # Parameters
    avg_oversample_factor: float = 0.0
    avg_teleo_weight: float = 0.0

    # Candidates
    avg_coarse_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# Telemetry Collector
# =============================================================================

class TelemetryCollector:
    """Collects retrieval telemetry."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size

        # Raw data
        self._total_latencies: deque = deque(maxlen=window_size)
        self._stage1_latencies: deque = deque(maxlen=window_size)
        self._stage2_latencies: deque = deque(maxlen=window_size)
        self._recalls: deque = deque(maxlen=window_size)
        self._factors: deque = deque(maxlen=window_size)
        self._teleo_weights: deque = deque(maxlen=window_size)
        self._coarse_counts: deque = deque(maxlen=window_size)

        # Counters
        self._total_retrievals = 0
        self._successful_retrievals = 0
        self._empty_results = 0

    def record(
        self,
        total_latency_us: float,
        stage1_latency_us: float,
        stage2_latency_us: float,
        coarse_count: int,
        oversample_factor: float,
        teleo_weight: float,
        recall_estimate: float = 1.0,
        success: bool = True,
    ) -> None:
        """Record a retrieval result."""
        self._total_retrievals += 1

        if success and coarse_count > 0:
            self._successful_retrievals += 1
        else:
            self._empty_results += 1

        self._total_latencies.append(total_latency_us)
        self._stage1_latencies.append(stage1_latency_us)
        self._stage2_latencies.append(stage2_latency_us)
        self._recalls.append(recall_estimate)
        self._factors.append(oversample_factor)
        self._teleo_weights.append(teleo_weight)
        self._coarse_counts.append(coarse_count)

    def get_metrics(self) -> RetrievalMetrics:
        """Compute current metrics."""
        import numpy as np

        metrics = RetrievalMetrics(
            timestamp=time.time(),
            total_retrievals=self._total_retrievals,
            successful_retrievals=self._successful_retrievals,
            empty_results=self._empty_results,
        )

        if self._total_latencies:
            latencies = list(self._total_latencies)
            metrics.avg_total_latency_us = float(np.mean(latencies))
            metrics.max_total_latency_us = float(np.max(latencies))
            metrics.p99_total_latency_us = float(np.percentile(latencies, 99))

        if self._stage1_latencies:
            metrics.avg_stage1_latency_us = float(np.mean(list(self._stage1_latencies)))

        if self._stage2_latencies:
            metrics.avg_stage2_latency_us = float(np.mean(list(self._stage2_latencies)))

        if self._recalls:
            metrics.avg_recall = float(np.mean(list(self._recalls)))

        if self._factors:
            metrics.avg_oversample_factor = float(np.mean(list(self._factors)))

        if self._teleo_weights:
            metrics.avg_teleo_weight = float(np.mean(list(self._teleo_weights)))

        if self._coarse_counts:
            metrics.avg_coarse_count = int(np.mean(list(self._coarse_counts)))

        return metrics

    def reset(self) -> None:
        """Reset all metrics."""
        self._total_latencies.clear()
        self._stage1_latencies.clear()
        self._stage2_latencies.clear()
        self._recalls.clear()
        self._factors.clear()
        self._teleo_weights.clear()
        self._coarse_counts.clear()
        self._total_retrievals = 0
        self._successful_retrievals = 0
        self._empty_results = 0


# =============================================================================
# Global Instance
# =============================================================================

_collector: Optional[TelemetryCollector] = None


def _get_collector() -> TelemetryCollector:
    global _collector
    if _collector is None:
        _collector = TelemetryCollector()
    return _collector


def record_retrieval(**kwargs) -> None:
    """Record a retrieval result to global telemetry."""
    _get_collector().record(**kwargs)


def get_metrics() -> RetrievalMetrics:
    """Get current retrieval metrics."""
    return _get_collector().get_metrics()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'RetrievalMetrics',
    'TelemetryCollector',
    'record_retrieval',
    'get_metrics',
]
