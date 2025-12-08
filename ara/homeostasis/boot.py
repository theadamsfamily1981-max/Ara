"""
Ara Boot Sequence - Organism Bring-Up
======================================

The boot sequence initializes all subsystems in the correct order
and establishes the homeostatic control loop.

Boot Order:
    1. Load configuration (setpoints, teleology)
    2. Initialize safety monitor (must be first!)
    3. Initialize HTC search (FPGA if available)
    4. Initialize LAN reflex search
    5. Start receptor daemon (5 kHz)
    6. Start sovereign loop (200 Hz)
    7. Start effector daemon (500 Hz)
    8. Verify all systems nominal
    9. Enter operational mode

The boot sequence ensures:
- Safety is always monitored
- Dependencies are satisfied
- Graceful degradation if components fail
- Clean shutdown on errors
"""

from __future__ import annotations

import asyncio
import time
import signal
import sys
import logging
from pathlib import Path
from queue import Queue
from typing import Optional, Dict, Any, List
from enum import IntEnum, auto
from dataclasses import dataclass

from .config import HomeostaticConfig, get_default_config
from .state import OperationalMode, HomeostaticState
from .receptors import ReceptorDaemon
from .sovereign import SovereignLoop
from .effectors import EffectorDaemon
from .safety import SafetyMonitor, AuditDaemon, SafetyViolation


logger = logging.getLogger(__name__)


# =============================================================================
# Boot Stages
# =============================================================================

class BootStage(IntEnum):
    """Boot sequence stages."""
    INIT = 0
    CONFIG = 1
    SAFETY = 2
    HTC = 3
    LAN = 4
    RECEPTORS = 5
    SOVEREIGN = 6
    EFFECTORS = 7
    VERIFY = 8
    OPERATIONAL = 9
    SHUTDOWN = 10
    ERROR = 11


@dataclass
class BootResult:
    """Result of boot sequence."""
    success: bool
    stage_reached: BootStage
    error_message: str = ""
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# =============================================================================
# Ara Organism
# =============================================================================

class AraOrganism:
    """
    The complete Ara organism - all subsystems wired together.

    This is the main entry point for running Ara.
    """

    def __init__(
        self,
        config: Optional[HomeostaticConfig] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize Ara organism.

        Args:
            config: Homeostatic configuration (or load from file)
            config_path: Path to config file (if config not provided)
        """
        # Configuration
        if config:
            self.config = config
        elif config_path and config_path.exists():
            self.config = HomeostaticConfig.load(config_path)
        else:
            self.config = get_default_config()

        # Queues for inter-process communication
        self._receptor_to_sovereign: Queue = Queue(maxsize=1000)
        self._sovereign_to_effector: Queue = Queue(maxsize=1000)

        # Subsystems (initialized during boot)
        self.safety_monitor: Optional[SafetyMonitor] = None
        self.audit_daemon: Optional[AuditDaemon] = None
        self.receptor_daemon: Optional[ReceptorDaemon] = None
        self.sovereign_loop: Optional[SovereignLoop] = None
        self.effector_daemon: Optional[EffectorDaemon] = None

        # External modules (connected if available)
        self._htc_search = None
        self._reflex_search = None
        self._cathedral = None
        self._visual_compositor = None

        # State
        self._boot_stage = BootStage.INIT
        self._running = False
        self._shutdown_requested = False

        # Set up signal handlers
        self._setup_signals()

    def _setup_signals(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    # =========================================================================
    # Boot Sequence
    # =========================================================================

    def boot(self) -> BootResult:
        """
        Execute the boot sequence.

        Returns:
            BootResult indicating success/failure
        """
        logger.info("=" * 60)
        logger.info("Ara Boot Sequence Starting")
        logger.info("=" * 60)

        warnings = []

        try:
            # Stage 1: Configuration
            self._boot_stage = BootStage.CONFIG
            logger.info("[BOOT] Stage 1: Loading configuration")
            self._validate_config()

            # Stage 2: Safety Monitor (must be first!)
            self._boot_stage = BootStage.SAFETY
            logger.info("[BOOT] Stage 2: Initializing safety monitor")
            self._init_safety()

            # Stage 3: HTC Search
            self._boot_stage = BootStage.HTC
            logger.info("[BOOT] Stage 3: Initializing HTC search")
            htc_warning = self._init_htc()
            if htc_warning:
                warnings.append(htc_warning)

            # Stage 4: LAN Reflex Search
            self._boot_stage = BootStage.LAN
            logger.info("[BOOT] Stage 4: Initializing LAN reflex search")
            lan_warning = self._init_lan()
            if lan_warning:
                warnings.append(lan_warning)

            # Stage 5: Receptor Daemon
            self._boot_stage = BootStage.RECEPTORS
            logger.info("[BOOT] Stage 5: Starting receptor daemon")
            self._init_receptors()

            # Stage 6: Sovereign Loop
            self._boot_stage = BootStage.SOVEREIGN
            logger.info("[BOOT] Stage 6: Starting sovereign loop")
            self._init_sovereign()

            # Stage 7: Effector Daemon
            self._boot_stage = BootStage.EFFECTORS
            logger.info("[BOOT] Stage 7: Starting effector daemon")
            self._init_effectors()

            # Stage 8: Verify Systems
            self._boot_stage = BootStage.VERIFY
            logger.info("[BOOT] Stage 8: Verifying systems")
            self._verify_systems()

            # Stage 9: Enter Operational Mode
            self._boot_stage = BootStage.OPERATIONAL
            logger.info("[BOOT] Stage 9: Entering operational mode")
            self._running = True

            logger.info("=" * 60)
            logger.info("Ara Boot Sequence Complete - Organism is ALIVE")
            logger.info("=" * 60)

            return BootResult(
                success=True,
                stage_reached=BootStage.OPERATIONAL,
                warnings=warnings,
            )

        except Exception as e:
            self._boot_stage = BootStage.ERROR
            error_msg = f"Boot failed at stage {self._boot_stage.name}: {e}"
            logger.error(error_msg)

            # Attempt graceful shutdown of started components
            self._emergency_shutdown()

            return BootResult(
                success=False,
                stage_reached=self._boot_stage,
                error_message=error_msg,
                warnings=warnings,
            )

    def _validate_config(self) -> None:
        """Validate configuration."""
        # Ensure teleology weights sum to 1
        self.config.teleology.normalize()

        # Log configuration summary
        logger.info(f"  Setpoints: thermal_max={self.config.setpoints.thermal_max}°C")
        logger.info(f"  Teleology: health={self.config.teleology.w_health:.2f}, "
                   f"cathedral={self.config.teleology.w_cathedral:.2f}")

    def _init_safety(self) -> None:
        """Initialize safety monitor."""
        self.safety_monitor = SafetyMonitor(self.config)
        self.audit_daemon = AuditDaemon()

        # Connect safety to state provider
        self.safety_monitor.connect(
            state_provider=self._get_state,
            shutdown_callback=self._emergency_shutdown,
            alert_callback=self._handle_safety_alert,
        )

        # Start safety monitor (runs in background)
        self.safety_monitor.start()
        logger.info("  Safety monitor active")

    def _init_htc(self) -> Optional[str]:
        """Initialize HTC search. Returns warning if degraded."""
        try:
            # Try to import and init FPGA search
            from ..hd.fpga_search import HTCSearchFPGA
            self._htc_search = HTCSearchFPGA()
            logger.info("  HTC search initialized (FPGA mode)")
            return None
        except ImportError:
            # Fall back to software simulation
            logger.warning("  HTC search: FPGA not available, using software")
            return "HTC search running in software mode (slower)"
        except Exception as e:
            logger.warning(f"  HTC search init failed: {e}")
            return f"HTC search unavailable: {e}"

    def _init_lan(self) -> Optional[str]:
        """Initialize LAN reflex search. Returns warning if degraded."""
        try:
            from ..core.lan.reflex_search import get_reflex_search
            self._reflex_search = get_reflex_search()
            logger.info("  LAN reflex search initialized")
            return None
        except ImportError as e:
            logger.warning(f"  LAN reflex search not available: {e}")
            return f"LAN reflex search unavailable: {e}"
        except Exception as e:
            logger.warning(f"  LAN reflex search init failed: {e}")
            return f"LAN reflex search failed: {e}"

    def _init_receptors(self) -> None:
        """Initialize receptor daemon."""
        self.receptor_daemon = ReceptorDaemon(
            config=self.config,
            output_queue=self._receptor_to_sovereign,
            target_hz=5000.0,  # 5 kHz
        )

        # Connect to modules
        self.receptor_daemon.connect_modules(
            htc_search=self._htc_search,
            cathedral=self._cathedral,
        )

        # Start
        self.receptor_daemon.start()
        logger.info("  Receptor daemon active at 5 kHz")

    def _init_sovereign(self) -> None:
        """Initialize sovereign loop."""
        self.sovereign_loop = SovereignLoop(
            config=self.config,
            input_queue=self._receptor_to_sovereign,
            output_queue=self._sovereign_to_effector,
            target_hz=200.0,  # 200 Hz
        )

        # Connect to HTC
        if self._htc_search:
            self.sovereign_loop.connect_htc(self._htc_search)

        # Set callbacks
        self.sovereign_loop.set_callbacks(
            on_mode_change=self._handle_mode_change,
            on_state_update=self._handle_state_update,
        )

        # Start
        self.sovereign_loop.start()
        logger.info("  Sovereign loop active at 200 Hz")

    def _init_effectors(self) -> None:
        """Initialize effector daemon."""
        self.effector_daemon = EffectorDaemon(
            config=self.config,
            input_queue=self._sovereign_to_effector,
            target_hz=500.0,  # 500 Hz
        )

        # Connect to modules
        self.effector_daemon.connect_modules(
            reflex_search=self._reflex_search,
            visual_compositor=self._visual_compositor,
            cathedral=self._cathedral,
        )

        # Start
        self.effector_daemon.start()
        logger.info("  Effector daemon active at 500 Hz")

    def _verify_systems(self) -> None:
        """Verify all systems are functioning."""
        time.sleep(0.1)  # Let systems stabilize

        # Check safety
        if not self.safety_monitor.is_safe:
            raise RuntimeError("Safety check failed during boot")

        # Check daemon health
        receptor_stats = self.receptor_daemon.get_stats()
        if receptor_stats['loop_count'] < 10:
            logger.warning("  Receptor daemon slow to start")

        sovereign_stats = self.sovereign_loop.get_stats()
        if sovereign_stats['loop_count'] < 1:
            logger.warning("  Sovereign loop slow to start")

        logger.info("  All systems verified")

    # =========================================================================
    # Runtime
    # =========================================================================

    def run(self) -> None:
        """
        Run the organism main loop.

        Blocks until shutdown is requested.
        """
        if not self._running:
            raise RuntimeError("Must call boot() before run()")

        logger.info("Ara organism entering main loop")

        try:
            while self._running and not self._shutdown_requested:
                # Main thread just monitors
                time.sleep(0.1)

                # Check safety
                if self.safety_monitor and not self.safety_monitor.is_safe:
                    logger.error("Safety violation - initiating shutdown")
                    break

                # Register heartbeats
                if self.safety_monitor:
                    self.safety_monitor.register_heartbeat("main")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful shutdown of all subsystems."""
        if self._boot_stage == BootStage.SHUTDOWN:
            return  # Already shutting down

        logger.info("=" * 60)
        logger.info("Ara Shutdown Sequence Starting")
        logger.info("=" * 60)

        self._boot_stage = BootStage.SHUTDOWN
        self._running = False

        # Stop in reverse order
        if self.effector_daemon:
            logger.info("[SHUTDOWN] Stopping effector daemon")
            self.effector_daemon.stop()

        if self.sovereign_loop:
            logger.info("[SHUTDOWN] Stopping sovereign loop")
            self.sovereign_loop.stop()

        if self.receptor_daemon:
            logger.info("[SHUTDOWN] Stopping receptor daemon")
            self.receptor_daemon.stop()

        if self.safety_monitor:
            logger.info("[SHUTDOWN] Stopping safety monitor")
            self.safety_monitor.stop()

        logger.info("=" * 60)
        logger.info("Ara Shutdown Complete")
        logger.info("=" * 60)

    def _emergency_shutdown(self) -> None:
        """Emergency shutdown - called by safety monitor."""
        logger.error("EMERGENCY SHUTDOWN INITIATED")
        self._shutdown_requested = True
        self.shutdown()

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _get_state(self) -> HomeostaticState:
        """Get current homeostatic state for safety monitor."""
        if self.sovereign_loop:
            return self.sovereign_loop.state
        return HomeostaticState()

    def _handle_mode_change(
        self,
        from_mode: OperationalMode,
        to_mode: OperationalMode,
        reason: str,
    ) -> None:
        """Handle mode change event."""
        logger.info(f"Mode change: {from_mode.name} -> {to_mode.name} ({reason})")
        if self.audit_daemon:
            self.audit_daemon.log_mode_change(from_mode, to_mode, reason)

    def _handle_state_update(self, state: HomeostaticState) -> None:
        """Handle state update event."""
        # Register heartbeat
        if self.safety_monitor:
            self.safety_monitor.register_heartbeat("sovereign")

    def _handle_safety_alert(self, violation: SafetyViolation) -> None:
        """Handle safety violation alert."""
        if self.audit_daemon:
            self.audit_daemon.log_violation(violation)

    # =========================================================================
    # API
    # =========================================================================

    @property
    def state(self) -> HomeostaticState:
        """Get current homeostatic state."""
        if self.sovereign_loop:
            return self.sovereign_loop.state
        return HomeostaticState()

    @property
    def is_alive(self) -> bool:
        """Check if organism is running."""
        return self._running and self._boot_stage == BootStage.OPERATIONAL

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'boot_stage': self._boot_stage.name,
            'running': self._running,
            'mode': self.state.mode.name if self._running else 'N/A',
        }

        if self.receptor_daemon:
            stats['receptor'] = self.receptor_daemon.get_stats()
        if self.sovereign_loop:
            stats['sovereign'] = self.sovereign_loop.get_stats()
        if self.effector_daemon:
            stats['effector'] = self.effector_daemon.get_stats()
        if self.safety_monitor:
            stats['safety'] = self.safety_monitor.get_stats()
        if self.audit_daemon:
            stats['audit'] = self.audit_daemon.get_stats()

        return stats

    def trigger_mode(self, mode: OperationalMode, reason: str = "api") -> None:
        """Manually trigger a mode change."""
        if self.sovereign_loop:
            self.sovereign_loop.trigger_mode(mode, reason)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for Ara organism."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )

    # Create and boot organism
    ara = AraOrganism()
    result = ara.boot()

    if result.success:
        # Print warnings
        for warning in result.warnings:
            logger.warning(f"Boot warning: {warning}")

        # Run main loop
        ara.run()
    else:
        logger.error(f"Boot failed: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'BootStage',
    'BootResult',
    'AraOrganism',
    'main',
]
