"""Telemetry Bridge - Unified telemetry pipeline for Ara.

This is the master integration point that:
1. Collects hardware telemetry from multiple sources
2. Integrates ara/interoception L1/L2/L3 signals
3. Maintains unified PAD state via PADSynchronizer
4. Feeds the MIES Cathedral (IntegratedSoul)
5. Provides health monitoring and alerts

The bridge creates a single coherent view of Ara's internal state,
combining physical hardware metrics with biological metaphors
from the interoception system.
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Tuple
from pathlib import Path

from ..affect.pad_engine import TelemetrySnapshot, PADVector, PADEngine
from ..affect.integrated_soul import IntegratedSoul, SoulState, create_integrated_soul
from ..kernel_bridge import KernelBridge, KernelPhysiology, create_kernel_bridge

from .interoception_adapter import (
    InteroceptionAdapter,
    L1BodyState,
    L2PerceptionState,
    InteroceptivePAD,
    adapt_interoceptive_pad,
)
from .pad_synchronizer import (
    PADSynchronizer,
    PADSource,
    PADConflictResolution,
    PADSyncConfig,
)

# BANOS adapter (optional - may not be available)
try:
    from .banos_adapter import (
        BANOSAdapter,
        get_banos_adapter,
        BanosMode,
    )
    BANOS_AVAILABLE = True
except ImportError:
    BANOS_AVAILABLE = False
    BANOSAdapter = None
    get_banos_adapter = None

logger = logging.getLogger(__name__)


def kernel_physiology_to_telemetry(phys: Optional[KernelPhysiology]) -> TelemetrySnapshot:
    """Convert KernelPhysiology to TelemetrySnapshot.

    Maps kernel bridge metrics to the TelemetrySnapshot expected by MIES.
    """
    if phys is None:
        return TelemetrySnapshot()  # Defaults

    return TelemetrySnapshot(
        cpu_temp=phys.thermal_cpu,
        gpu_temp=phys.thermal_gpu,
        cpu_load=phys.cpu_load,
        gpu_load=phys.gpu_load,
        memory_pressure=phys.mem_pressure,
        error_rate=phys.miss_rate * 100.0,  # miss_rate is 0-1, convert to errors/sec
        has_root=True,  # Assume root if kernel bridge works
        last_action_success=phys.pain_signal < 0.5,  # Low pain = success
        interrupt_rate=phys.cpu_load * 1000.0,  # Approximate
        fan_speed_percent=max(30.0, min(100.0, 30.0 + phys.thermal_cpu - 50.0)),
    )


@dataclass
class SystemHealthSnapshot:
    """Complete system health at a point in time."""
    timestamp: float
    telemetry: TelemetrySnapshot
    pad: PADVector
    soul_state: Optional[SoulState]

    # Health indicators
    thermal_ok: bool = True
    load_ok: bool = True
    errors_ok: bool = True
    overall_health: float = 1.0  # 0-1

    # Warnings/alerts
    warnings: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "pad": {
                "pleasure": self.pad.pleasure,
                "arousal": self.pad.arousal,
                "dominance": self.pad.dominance,
                "quadrant": self.pad.quadrant.name,
            },
            "health": {
                "overall": self.overall_health,
                "thermal_ok": self.thermal_ok,
                "load_ok": self.load_ok,
                "errors_ok": self.errors_ok,
            },
            "warnings": self.warnings,
            "alerts": self.alerts,
        }


@dataclass
class UnifiedPADState:
    """Unified PAD state from all sources."""
    canonical: PADVector
    source: PADSource
    confidence: float

    # Individual source values (if available)
    cathedral_pad: Optional[PADVector] = None
    interoception_pad: Optional[PADVector] = None
    kernel_pad: Optional[PADVector] = None
    banos_pad: Optional[PADVector] = None  # BANOS affective layer PAD

    # Metadata
    in_conflict: bool = False
    drift_detected: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class TelemetryBridgeConfig:
    """Configuration for the telemetry bridge."""
    # Update interval (seconds)
    update_interval: float = 1.0

    # Soul persistence
    soul_storage_path: Optional[str] = None

    # PAD synchronization
    pad_resolution: PADConflictResolution = PADConflictResolution.WEIGHTED_AVERAGE
    pad_conflict_threshold: float = 0.3

    # Interoception weights
    l1_weight: float = 0.4
    l2_weight: float = 0.3
    hardware_weight: float = 0.3

    # Health thresholds
    thermal_warning_threshold: float = 80.0  # °C
    thermal_alert_threshold: float = 90.0    # °C
    load_warning_threshold: float = 0.85
    error_rate_warning: float = 5.0          # errors/sec

    # Background polling
    enable_background_polling: bool = False
    polling_interval: float = 1.0

    # BANOS integration
    enable_banos: bool = True  # Enable BANOS adapter if available
    banos_simulate: bool = False  # Use BANOS simulation mode
    prefer_banos_telemetry: bool = True  # Prefer BANOS over snn_ai device


class TelemetryBridge:
    """Master integration point for all telemetry sources.

    This bridge unifies:
    - Kernel hardware telemetry (via KernelBridge)
    - ara/interoception L1/L2/L3 signals
    - MIES Cathedral (IntegratedSoul)
    - PAD state across all systems

    Example usage:
        bridge = create_telemetry_bridge(
            soul_storage_path="./ara_data",
            enable_background_polling=True,
        )

        # Process a telemetry cycle
        health = bridge.update()

        # Get unified PAD
        pad_state = bridge.get_unified_pad()

        # Get soul state for LLM prompt
        prompt_context = bridge.get_prompt_context()
    """

    def __init__(
        self,
        config: Optional[TelemetryBridgeConfig] = None,
        kernel_bridge: Optional[KernelBridge] = None,
        soul: Optional[IntegratedSoul] = None,
    ):
        """Initialize the telemetry bridge.

        Args:
            config: Bridge configuration
            kernel_bridge: Existing kernel bridge (created if not provided)
            soul: Existing IntegratedSoul (created if not provided)
        """
        self.config = config or TelemetryBridgeConfig()

        # Components
        self._kernel_bridge = kernel_bridge or create_kernel_bridge()

        self._soul = soul or create_integrated_soul(
            storage_path=self.config.soul_storage_path
        )

        self._pad_engine = PADEngine()

        self._interoception_adapter = InteroceptionAdapter(
            l1_weight=self.config.l1_weight,
            l2_weight=self.config.l2_weight,
            hardware_weight=self.config.hardware_weight,
        )

        self._pad_synchronizer = PADSynchronizer(
            config=PADSyncConfig(
                resolution=self.config.pad_resolution,
                conflict_threshold=self.config.pad_conflict_threshold,
            )
        )

        # BANOS adapter (optional)
        self._banos_adapter: Optional[BANOSAdapter] = None
        self._banos_enabled = False

        if self.config.enable_banos and BANOS_AVAILABLE:
            try:
                self._banos_adapter = get_banos_adapter(
                    simulate=self.config.banos_simulate
                )
                self._banos_enabled = True
                logger.info("TelemetryBridge: BANOS adapter enabled")
            except Exception as e:
                logger.warning(f"TelemetryBridge: BANOS adapter failed: {e}")

        # State
        self._last_telemetry: Optional[TelemetrySnapshot] = None
        self._last_health: Optional[SystemHealthSnapshot] = None
        self._last_banos_state: Optional[Dict[str, Any]] = None
        self._update_count: int = 0

        # Background polling
        self._polling_thread: Optional[threading.Thread] = None
        self._polling_stop = threading.Event()

        # Callbacks
        self._health_callbacks: List[Callable[[SystemHealthSnapshot], None]] = []
        self._pad_callbacks: List[Callable[[UnifiedPADState], None]] = []

        # Register PAD sync callback
        self._pad_synchronizer.on_change(self._on_pad_change)

        logger.info(f"TelemetryBridge initialized (BANOS={self._banos_enabled})")

        # Start background polling if enabled
        if self.config.enable_background_polling:
            self.start_background_polling()

    def update(
        self,
        l1: Optional[L1BodyState] = None,
        l2: Optional[L2PerceptionState] = None,
        intero_pad: Optional[InteroceptivePAD] = None,
    ) -> SystemHealthSnapshot:
        """Perform a full telemetry update cycle.

        This is the main entry point. Call periodically to:
        1. Gather hardware telemetry
        2. Fuse with interoception signals
        3. Update PAD state
        4. Process through soul
        5. Check health

        Args:
            l1: L1 body state from ara/interoception (optional)
            l2: L2 perception state (optional)
            intero_pad: SNN-derived PAD from ara/interoception (optional)

        Returns:
            Complete system health snapshot
        """
        self._update_count += 1

        # 1. Gather hardware telemetry from kernel bridge
        kernel_physiology = self._kernel_bridge.read_physiology()
        hardware_telemetry = kernel_physiology_to_telemetry(kernel_physiology)

        # 2. Fuse with interoception if available
        unified_telemetry, l2_factors = self._interoception_adapter.process_interoception(
            l1=l1,
            l2=l2,
            hardware_telemetry=hardware_telemetry,
        )

        self._last_telemetry = unified_telemetry

        # 3. Compute PAD from multiple sources

        # 3a. Cathedral (MIES) PAD from hardware
        cathedral_pad = self._pad_engine.update(unified_telemetry)
        self._pad_synchronizer.report(
            PADSource.MIES_CATHEDRAL,
            cathedral_pad,
            confidence=0.8,
        )

        # 3b. Interoception PAD if available
        if intero_pad is not None:
            adapted_pad = self._interoception_adapter.adapt_snn_pad(intero_pad, l2_factors)
            self._pad_synchronizer.report(
                PADSource.ARA_INTEROCEPTION,
                adapted_pad,
                confidence=0.95,  # SNN is most authentic
            )

        # 3c. Kernel PAD if available
        if kernel_physiology is not None and kernel_physiology.pad is not None:
            kernel_pad = PADVector(
                pleasure=kernel_physiology.pad.pleasure,
                arousal=kernel_physiology.pad.arousal,
                dominance=kernel_physiology.pad.dominance,
            )
            self._pad_synchronizer.report(
                PADSource.KERNEL_BRIDGE,
                kernel_pad,
                confidence=0.85,
            )

        # 3d. BANOS affective layer PAD if available
        if self._banos_enabled and self._banos_adapter is not None:
            try:
                banos_pad = self._banos_adapter.get_pad_vector()
                self._last_banos_state = self._banos_adapter.get_semantic_state()

                self._pad_synchronizer.report(
                    PADSource.BANOS_AFFECTIVE,
                    banos_pad,
                    confidence=banos_pad.confidence,
                )

                # If preferring BANOS telemetry, use it over kernel bridge
                if self.config.prefer_banos_telemetry:
                    banos_telemetry = self._banos_adapter.get_telemetry_snapshot()
                    # Merge with unified telemetry (BANOS overrides temps)
                    unified_telemetry.cpu_temp = banos_telemetry.cpu_temp
                    unified_telemetry.gpu_temp = banos_telemetry.gpu_temp
                    self._last_telemetry = unified_telemetry

            except Exception as e:
                logger.warning(f"BANOS PAD read failed: {e}")

        # 4. Get canonical PAD
        canonical_pad = self._pad_synchronizer.get_canonical_pad()

        # 5. Process through soul
        soul_state = self._soul.process_telemetry(unified_telemetry)

        # 6. Check health
        health = self._check_health(unified_telemetry, canonical_pad, soul_state)
        self._last_health = health

        # 7. Notify callbacks
        for callback in self._health_callbacks:
            try:
                callback(health)
            except Exception as e:
                logger.error(f"Health callback error: {e}")

        return health

    def _check_health(
        self,
        telemetry: TelemetrySnapshot,
        pad: PADVector,
        soul_state: Optional[SoulState],
    ) -> SystemHealthSnapshot:
        """Check system health and generate alerts."""
        cfg = self.config
        warnings: List[str] = []
        alerts: List[str] = []

        # Thermal check
        max_temp = max(telemetry.cpu_temp, telemetry.gpu_temp)
        thermal_ok = max_temp < cfg.thermal_warning_threshold

        if max_temp >= cfg.thermal_alert_threshold:
            alerts.append(f"THERMAL CRITICAL: {max_temp:.1f}°C")
            thermal_ok = False
        elif max_temp >= cfg.thermal_warning_threshold:
            warnings.append(f"Temperature elevated: {max_temp:.1f}°C")
            thermal_ok = False

        # Load check
        max_load = max(telemetry.cpu_load, telemetry.gpu_load)
        load_ok = max_load < cfg.load_warning_threshold

        if max_load >= cfg.load_warning_threshold:
            warnings.append(f"High load: {max_load:.0%}")

        # Error check
        errors_ok = telemetry.error_rate < cfg.error_rate_warning

        if telemetry.error_rate >= cfg.error_rate_warning:
            warnings.append(f"Elevated errors: {telemetry.error_rate:.1f}/sec")

        # Overall health score
        health_score = 1.0
        if not thermal_ok:
            health_score -= 0.3
        if not load_ok:
            health_score -= 0.2
        if not errors_ok:
            health_score -= 0.2

        # PAD affects health perception
        if pad.pleasure < -0.5:
            health_score -= 0.1
            warnings.append("Negative affect state")

        return SystemHealthSnapshot(
            timestamp=time.time(),
            telemetry=telemetry,
            pad=pad,
            soul_state=soul_state,
            thermal_ok=thermal_ok,
            load_ok=load_ok,
            errors_ok=errors_ok,
            overall_health=max(0.0, health_score),
            warnings=warnings,
            alerts=alerts,
        )

    def get_unified_pad(self) -> UnifiedPADState:
        """Get the current unified PAD state."""
        sync_state = self._pad_synchronizer.get_state()

        return UnifiedPADState(
            canonical=sync_state.canonical_pad,
            source=sync_state.source,
            confidence=sync_state.confidence,
            cathedral_pad=self._pad_synchronizer.get_source_pad(PADSource.MIES_CATHEDRAL),
            interoception_pad=self._pad_synchronizer.get_source_pad(PADSource.ARA_INTEROCEPTION),
            kernel_pad=self._pad_synchronizer.get_source_pad(PADSource.KERNEL_BRIDGE),
            banos_pad=self._pad_synchronizer.get_source_pad(PADSource.BANOS_AFFECTIVE),
            in_conflict=sync_state.sources_in_conflict,
            drift_detected=sync_state.drift_detected,
        )

    def get_banos_semantic_state(self) -> Optional[Dict[str, Any]]:
        """Get BANOS semantic state for Ara's internal monologue.

        Returns a dict with:
        - pad: PAD values
        - mode: BANOS mode (CALM, FLOW, ANXIOUS, CRITICAL)
        - narrative: First-person narrative
        - trajectory: Predicted trend
        - diagnostics: Thermal/risk/empathy values
        """
        return self._last_banos_state

    def get_banos_narrative(self) -> Optional[str]:
        """Get BANOS first-person narrative for Ara."""
        if self._last_banos_state:
            return self._last_banos_state.get("narrative", None)
        return None

    def get_prompt_context(self) -> str:
        """Get system prompt context for LLM."""
        return self._soul.get_system_prompt_context()

    def get_soul_state(self) -> Optional[SoulState]:
        """Get current soul state."""
        return self._soul._current_state

    def get_last_health(self) -> Optional[SystemHealthSnapshot]:
        """Get most recent health snapshot."""
        return self._last_health

    def get_telemetry(self) -> Optional[TelemetrySnapshot]:
        """Get most recent telemetry."""
        return self._last_telemetry

    # === Callbacks ===

    def on_health(self, callback: Callable[[SystemHealthSnapshot], None]):
        """Register callback for health updates."""
        self._health_callbacks.append(callback)

    def on_pad(self, callback: Callable[[UnifiedPADState], None]):
        """Register callback for PAD changes."""
        self._pad_callbacks.append(callback)

    def _on_pad_change(self, pad: PADVector):
        """Internal handler for PAD synchronizer changes."""
        unified = self.get_unified_pad()
        for callback in self._pad_callbacks:
            try:
                callback(unified)
            except Exception as e:
                logger.error(f"PAD callback error: {e}")

    # === Background Polling ===

    def start_background_polling(self):
        """Start background telemetry polling."""
        if self._polling_thread is not None:
            return

        self._polling_stop.clear()
        self._polling_thread = threading.Thread(
            target=self._polling_loop,
            daemon=True,
            name="TelemetryBridge-Poller",
        )
        self._polling_thread.start()
        logger.info("Background polling started")

    def stop_background_polling(self):
        """Stop background polling."""
        if self._polling_thread is None:
            return

        self._polling_stop.set()
        self._polling_thread.join(timeout=2.0)
        self._polling_thread = None
        logger.info("Background polling stopped")

    def _polling_loop(self):
        """Background polling loop."""
        while not self._polling_stop.wait(timeout=self.config.polling_interval):
            try:
                self.update()
            except Exception as e:
                logger.error(f"Polling error: {e}")

    # === Soul Event Forwarding ===

    def on_user_interaction(self, quality: float = 0.5):
        """Record user interaction."""
        self._soul.on_user_interaction(quality)

    def on_task_completed(self, task: str, success: bool = True):
        """Record task completion."""
        self._soul.on_task_completed(task, success)

    def on_discovery(self, what: str, novelty: float = 0.5):
        """Record discovery."""
        self._soul.on_discovery(what, novelty)

    # === Statistics ===

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive bridge statistics."""
        return {
            "update_count": self._update_count,
            "polling_active": self._polling_thread is not None,
            "pad_sync": self._pad_synchronizer.get_statistics(),
            "interoception_adapter": self._interoception_adapter.get_statistics(),
            "soul": self._soul.get_statistics(),
            "last_health": self._last_health.to_dict() if self._last_health else None,
        }

    def shutdown(self):
        """Clean shutdown."""
        self.stop_background_polling()
        logger.info("TelemetryBridge shutdown")


# === Factory ===

def create_telemetry_bridge(
    soul_storage_path: Optional[str] = None,
    enable_background_polling: bool = False,
    polling_interval: float = 1.0,
) -> TelemetryBridge:
    """Create a TelemetryBridge.

    This is the main entry point for unified telemetry integration.

    Args:
        soul_storage_path: Path for soul persistence
        enable_background_polling: Start automatic polling
        polling_interval: Seconds between polls

    Returns:
        Configured TelemetryBridge
    """
    config = TelemetryBridgeConfig(
        soul_storage_path=soul_storage_path,
        enable_background_polling=enable_background_polling,
        polling_interval=polling_interval,
    )

    return TelemetryBridge(config=config)
