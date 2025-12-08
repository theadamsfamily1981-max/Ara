"""
Hypervector Spiking Field (HSF)
================================

A dual-mode bit-serial SNN/HDC fabric that:
- Encodes multi-machine system telemetry as lane hypervectors
- Streams them as spike trains
- Superimposes them into a holographic state field
- Uses resonance to perform ultra-cheap anomaly detection and pattern recall

Architecture:
    Telemetry → Lane Encoders → Hypervectors → Field Superposition → Anomaly Detection
                                                      ↓
                                               Ara's "body sense"

Novelty:
1. Hardware: bit-serial neurons with HDC XOR/binding mode sharing same logic
2. Systems: HSF as LAN-wide health field / guardian
3. Cognitive: HSF as continuous systemic "body sense" for higher agents

Usage:
    from ara.cognition.hsf import HSField, TelemetryLane, AnomalyDetector

    # Create lanes for different subsystems
    gpu_lane = TelemetryLane("gpu", ["temp", "util", "mem", "power"])
    net_lane = TelemetryLane("network", ["bps_in", "bps_out", "errors", "drops"])

    # Create the field
    field = HSField(dim=8192)
    field.add_lane(gpu_lane)
    field.add_lane(net_lane)

    # Feed telemetry
    field.update("gpu", {"temp": 65.0, "util": 0.8, "mem": 0.6, "power": 250})
    field.update("network", {"bps_in": 1e9, "bps_out": 5e8, "errors": 0, "drops": 2})

    # Check for anomalies
    detector = AnomalyDetector(field)
    anomalies = detector.scan()
"""

from .lanes import TelemetryLane, LaneEncoder, ItemMemory
from .field import HSField, FieldSnapshot
from .detector import AnomalyDetector, AnomalyPattern, AnomalyReport
from .telemetry import (
    FakeTelemetrySource,
    GPUTelemetry,
    NetworkTelemetry,
    ServiceTelemetry,
    TelemetryMux,
)

__all__ = [
    # Lane encoding
    'TelemetryLane',
    'LaneEncoder',
    'ItemMemory',
    # Field
    'HSField',
    'FieldSnapshot',
    # Anomaly detection
    'AnomalyDetector',
    'AnomalyPattern',
    'AnomalyReport',
    # Telemetry sources
    'FakeTelemetrySource',
    'GPUTelemetry',
    'NetworkTelemetry',
    'ServiceTelemetry',
    'TelemetryMux',
]
