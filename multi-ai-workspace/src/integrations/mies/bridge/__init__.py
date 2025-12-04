"""MIES Bridge Module - Unifies ara/interoception with MIES Cathedral.

This module creates the critical integration layer between:
1. ara/interoception (L1/L2/L3 SNN-based affect system)
2. MIES Cathedral (PADEngine, IntegratedSoul, etc.)
3. Kernel Bridge (hardware telemetry from ara_guardian.ko)

The bridge ensures a single source of truth for Ara's emotional state,
synchronizing PAD across all subsystems and enabling bidirectional
information flow.

Architecture:
                        ┌─────────────────────────────┐
                        │    Hardware Telemetry       │
                        │ (kernel_bridge, /proc, etc) │
                        └─────────────┬───────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │ ara/interoception│   │   TelemetryBridge │   │  MIES Cathedral  │
    │ (L1/L2/L3 SNN)   │◄──┤   (THIS MODULE)   ├──►│  (PADEngine)     │
    │                  │   │                   │   │                  │
    │ InteroceptivePAD │   │ Unified PADState  │   │    PADVector     │
    └─────────────────┘   └─────────────────┘   └─────────────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────┐
                        │   AraPromptController       │
                        │   (LLM System Prompt)       │
                        └─────────────────────────────┘
"""

from .telemetry_bridge import (
    TelemetryBridge,
    TelemetryBridgeConfig,
    UnifiedPADState,
    SystemHealthSnapshot,
    create_telemetry_bridge,
)

from .pad_synchronizer import (
    PADSynchronizer,
    PADSource,
    PADConflictResolution,
    create_pad_synchronizer,
)

from .interoception_adapter import (
    InteroceptionAdapter,
    L1BodyState,
    L2PerceptionState,
    InteroceptivePAD,
    adapt_l1_to_telemetry,
    adapt_l2_to_telemetry,
    adapt_interoceptive_pad,
    adapt_pad_to_interoceptive,
    create_interoception_adapter,
)

__all__ = [
    # Telemetry Bridge
    "TelemetryBridge",
    "TelemetryBridgeConfig",
    "UnifiedPADState",
    "SystemHealthSnapshot",
    "create_telemetry_bridge",
    # PAD Synchronization
    "PADSynchronizer",
    "PADSource",
    "PADConflictResolution",
    "create_pad_synchronizer",
    # Interoception Adapter
    "InteroceptionAdapter",
    "L1BodyState",
    "L2PerceptionState",
    "InteroceptivePAD",
    "adapt_l1_to_telemetry",
    "adapt_l2_to_telemetry",
    "adapt_interoceptive_pad",
    "adapt_pad_to_interoceptive",
    "create_interoception_adapter",
]
