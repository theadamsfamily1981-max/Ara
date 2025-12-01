#!/usr/bin/env python
"""
Prometheus exporter for TF-A-N metrics.

Exports all hard gate metrics and training state.

Usage:
    python monitoring/prometheus_exporter.py --port 9090
"""

import argparse
import time
from typing import Optional

from prometheus_client import start_http_server, Gauge, Counter, Histogram

# Define metrics
METRICS = {
    # TTW-Sentry
    "ttw_p50_latency_ms": Gauge("tfan_ttw_p50_latency_ms", "TTW p50 latency in milliseconds"),
    "ttw_p95_latency_ms": Gauge("tfan_ttw_p95_latency_ms", "TTW p95 latency in milliseconds"),
    "ttw_p99_latency_ms": Gauge("tfan_ttw_p99_latency_ms", "TTW p99 latency in milliseconds"),
    "ttw_coverage": Gauge("tfan_ttw_coverage", "TTW alignment coverage"),

    # Sparse Attention
    "ssa_speedup": Gauge("tfan_ssa_speedup", "Sparse attention speedup factor"),
    "ssa_sparsity": Gauge("tfan_ssa_sparsity", "Sparse attention sparsity ratio"),
    "ssa_keep_ratio": Gauge("tfan_ssa_keep_ratio", "TLS landmark keep ratio"),

    # FDT Homeostat
    "epr": Gauge("tfan_epr", "Energy-Perturbation Ratio"),
    "epr_cv": Gauge("tfan_epr_cv", "EPR Coefficient of Variation"),
    "lr": Gauge("tfan_lr", "Current learning rate"),
    "temperature": Gauge("tfan_temperature", "Current temperature"),

    # PGU-MAK
    "pgu_p50_latency_ms": Gauge("tfan_pgu_p50_latency_ms", "PGU p50 latency in milliseconds"),
    "pgu_p95_latency_ms": Gauge("tfan_pgu_p95_latency_ms", "PGU p95 latency in milliseconds"),
    "pgu_p99_latency_ms": Gauge("tfan_pgu_p99_latency_ms", "PGU p99 latency in milliseconds"),
    "pgu_cache_hit_rate": Gauge("tfan_pgu_cache_hit_rate", "PGU cache hit rate"),
    "pgu_proof_success_rate": Gauge("tfan_pgu_proof_success_rate", "PGU proof success rate"),
    "pgu_timeout_rate": Gauge("tfan_pgu_timeout_rate", "PGU timeout rate"),

    # Topology
    "topology_wasserstein_gap": Gauge("tfan_topology_wasserstein_gap", "Topology Wasserstein gap"),
    "topology_cosine_similarity": Gauge("tfan_topology_cosine_similarity", "Topology cosine similarity"),

    # Emotion
    "emotion_valence": Gauge("tfan_emotion_valence", "Emotion valence (mean)"),
    "emotion_arousal": Gauge("tfan_emotion_arousal", "Emotion arousal (mean)"),
    "emotion_confidence": Gauge("tfan_emotion_confidence", "Emotion confidence (mean)"),

    # Training
    "loss": Gauge("tfan_loss", "Training loss"),
    "grad_variance": Gauge("tfan_grad_variance", "Gradient variance"),
    "step": Counter("tfan_step", "Training step counter"),

    # Hardware
    "gpu_memory_allocated_gb": Gauge("tfan_gpu_memory_allocated_gb", "GPU memory allocated (GB)"),
    "gpu_memory_reserved_gb": Gauge("tfan_gpu_memory_reserved_gb", "GPU memory reserved (GB)"),
}


class PrometheusExporter:
    """Prometheus exporter for TF-A-N."""

    def __init__(self, port: int = 9090):
        """
        Args:
            port: HTTP server port
        """
        self.port = port
        self.trainer = None
        self.running = False

    def register_trainer(self, trainer):
        """Register trainer to export metrics from."""
        self.trainer = trainer

    def start(self):
        """Start HTTP server."""
        start_http_server(self.port)
        self.running = True
        print(f"Prometheus exporter started on port {self.port}")
        print(f"Metrics available at: http://localhost:{self.port}/metrics")

    def update_metrics(self):
        """Update all metrics from trainer."""
        if self.trainer is None:
            return

        # Get latest metrics
        if not self.trainer.metrics_history:
            return

        latest = self.trainer.metrics_history[-1]

        # TTW metrics (if available)
        if hasattr(self.trainer, "ttw_sentry") and self.trainer.ttw_sentry is not None:
            ttw_metrics = self.trainer.ttw_sentry.get_metrics()
            METRICS["ttw_p50_latency_ms"].set(ttw_metrics.p50_latency_ms)
            METRICS["ttw_p95_latency_ms"].set(ttw_metrics.p95_latency_ms)
            METRICS["ttw_p99_latency_ms"].set(ttw_metrics.p99_latency_ms)
            METRICS["ttw_coverage"].set(ttw_metrics.coverage)

        # FDT metrics
        METRICS["epr"].set(latest.get("epr", 0))
        METRICS["epr_cv"].set(latest.get("epr_cv", 0))
        METRICS["lr"].set(latest.get("lr", 0))
        METRICS["temperature"].set(latest.get("temperature", 1.0))

        # PGU metrics
        pgu_metrics = self.trainer.pgu.get_metrics()
        METRICS["pgu_p50_latency_ms"].set(pgu_metrics["p50_latency_ms"])
        METRICS["pgu_p95_latency_ms"].set(pgu_metrics["p95_latency_ms"])
        METRICS["pgu_p99_latency_ms"].set(pgu_metrics["p99_latency_ms"])
        METRICS["pgu_cache_hit_rate"].set(pgu_metrics["cache_hit_rate"])
        METRICS["pgu_proof_success_rate"].set(pgu_metrics["proof_success_rate"])
        METRICS["pgu_timeout_rate"].set(pgu_metrics["timeout_rate"])

        # Training metrics
        METRICS["loss"].set(latest.get("loss", 0))
        METRICS["grad_variance"].set(latest.get("grad_variance", 0))

        # GPU metrics (if CUDA)
        try:
            import torch
            if torch.cuda.is_available():
                METRICS["gpu_memory_allocated_gb"].set(torch.cuda.memory_allocated() / 1e9)
                METRICS["gpu_memory_reserved_gb"].set(torch.cuda.memory_reserved() / 1e9)
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Prometheus exporter for TF-A-N")
    parser.add_argument("--port", type=int, default=9090, help="HTTP server port")
    args = parser.parse_args()

    exporter = PrometheusExporter(port=args.port)
    exporter.start()

    # Keep running
    try:
        while True:
            time.sleep(1)
            exporter.update_metrics()
    except KeyboardInterrupt:
        print("\nExporter stopped")


if __name__ == "__main__":
    main()
