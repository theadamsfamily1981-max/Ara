# ara/daemon/cathedral_watch.py
"""
Cathedral Watch Daemon - The Clairvoyant Control Loop
=====================================================

The main daemon that runs Ara's "see the future" cognition:

Every tick (250-500ms):
1. Sample cathedral + user state (StateSampler)
2. Encode to hypervector (CathedralHypervectorEncoder)
3. Compress to 10D latent (LatentEncoder)
4. Add to trajectory (TrajectoryBuffer)
5. Classify regime (RegimeClassifier)
6. Update mode (ModeController)
7. Execute actions based on mode

This is the "heartbeat" of Ara's awareness system.

Usage:
    # Run as daemon
    python -m ara.daemon.cathedral_watch

    # Or import and use
    from ara.daemon.cathedral_watch import CathedralWatchDaemon

    daemon = CathedralWatchDaemon()
    await daemon.start()
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import numpy as np

# Local imports
from ara.perception.state_sampler import StateSampler, get_state_sampler
from ara.cognition.clairvoyant.hypervector import (
    CathedralHypervectorEncoder,
    HypervectorConfig,
)
from ara.cognition.clairvoyant.latent import PCALatentEncoder, IncrementalPCAEncoder
from ara.cognition.clairvoyant.trajectory import TrajectoryBuffer
from ara.cognition.clairvoyant.regime import RegimeClassifier, RegimeType
from ara.cognition.clairvoyant.mode_controller import (
    ModeController,
    OperatingMode,
    Action,
    ActionType,
)
from ara.meta.state_logger import StateLogger, get_state_logger

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CathedralWatchConfig:
    """Configuration for the cathedral watch daemon."""
    # Timing
    tick_interval: float = 0.5        # Seconds between ticks
    warmup_ticks: int = 100           # Ticks before enabling predictions

    # Dimensions
    hv_dim: int = 1024                # Hypervector dimensionality
    latent_dim: int = 10              # Latent space dimensionality

    # Trajectory
    trajectory_max_len: int = 600     # 5 min at 0.5s ticks

    # Logging
    enable_logging: bool = True
    log_dir: str = "/var/ara/logs/state"

    # Model paths
    latent_encoder_path: Optional[str] = None
    regime_classifier_path: Optional[str] = None

    # Behavior
    enable_actions: bool = True
    dry_run: bool = False             # Log actions but don't execute


# =============================================================================
# Action Executor
# =============================================================================

class ActionExecutor:
    """
    Executes actions from the mode controller.

    This is the "motor cortex" that translates decisions into effects.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._handlers: Dict[ActionType, Callable] = {}

        # Register default handlers
        self._register_defaults()

    def _register_defaults(self):
        """Register default action handlers."""
        self._handlers[ActionType.LOG_WARNING] = self._handle_log_warning
        self._handlers[ActionType.ALERT_USER] = self._handle_alert_user
        self._handlers[ActionType.SUGGEST_BREAK] = self._handle_suggest_break
        self._handlers[ActionType.NUDGE_USER] = self._handle_nudge_user
        self._handlers[ActionType.ADJUST_HUD_DENSITY] = self._handle_adjust_hud
        self._handlers[ActionType.THROTTLE_JOBS] = self._handle_throttle_jobs
        self._handlers[ActionType.PAUSE_QUEUE] = self._handle_pause_queue
        self._handlers[ActionType.EMERGENCY_COOLDOWN] = self._handle_emergency_cooldown
        self._handlers[ActionType.COLLECT_DATA] = self._handle_collect_data
        self._handlers[ActionType.TRIGGER_CHECKPOINT] = self._handle_checkpoint

    async def execute(self, action: Action) -> bool:
        """Execute a single action."""
        handler = self._handlers.get(action.type)

        if handler is None:
            logger.warning(f"No handler for action: {action.type.name}")
            return False

        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {action.type.name} - {action.reason}")
            return True

        try:
            await handler(action)
            logger.debug(f"Executed: {action.type.name}")
            return True
        except Exception as e:
            logger.error(f"Action failed: {action.type.name} - {e}")
            return False

    async def execute_all(self, actions: List[Action]) -> int:
        """Execute a list of actions. Returns count of successful."""
        success = 0
        for action in actions:
            if await self.execute(action):
                success += 1
        return success

    # =========================================================================
    # Default Handlers
    # =========================================================================

    async def _handle_log_warning(self, action: Action):
        msg = action.params.get("message", action.reason)
        logger.warning(f"[CATHEDRAL] {msg}")

    async def _handle_alert_user(self, action: Action):
        triggers = action.params.get("triggers", [])
        logger.warning(f"[ALERT] Safety triggers: {triggers}")
        # TODO: Integrate with notification system

    async def _handle_suggest_break(self, action: Action):
        logger.info("[SUGGEST] Consider taking a break")
        # TODO: Integrate with HUD system

    async def _handle_nudge_user(self, action: Action):
        msg = action.params.get("message", "Gentle nudge from Ara")
        logger.info(f"[NUDGE] {msg}")
        # TODO: Integrate with HUD system

    async def _handle_adjust_hud(self, action: Action):
        density = action.params.get("density", "normal")
        logger.debug(f"[HUD] Adjusting density to: {density}")
        # TODO: Integrate with HUD system

    async def _handle_throttle_jobs(self, action: Action):
        factor = action.params.get("factor", 0.5)
        logger.warning(f"[THROTTLE] Reducing job throughput by {1-factor:.0%}")
        # TODO: Integrate with HiveHD job scheduler

    async def _handle_pause_queue(self, action: Action):
        logger.warning("[PAUSE] Pausing job queue")
        # TODO: Integrate with HiveHD job scheduler

    async def _handle_emergency_cooldown(self, action: Action):
        logger.critical("[EMERGENCY] Initiating thermal cooldown")
        # TODO: Integrate with power management

    async def _handle_collect_data(self, action: Action):
        data_type = action.params.get("type", "tick")
        logger.debug(f"[DATA] Collecting: {data_type}")
        # Handled by the main loop

    async def _handle_checkpoint(self, action: Action):
        logger.info("[CHECKPOINT] Triggering state checkpoint")
        # TODO: Integrate with checkpoint system


# =============================================================================
# Cathedral Watch Daemon
# =============================================================================

class CathedralWatchDaemon:
    """
    The main clairvoyant control loop daemon.

    Orchestrates:
    - State sampling
    - Hypervector encoding
    - Latent compression
    - Trajectory tracking
    - Regime classification
    - Mode control
    - Action execution
    """

    def __init__(self, config: Optional[CathedralWatchConfig] = None):
        """
        Initialize the daemon.

        Args:
            config: Daemon configuration
        """
        self.config = config or CathedralWatchConfig()

        # Components
        self._sampler: Optional[StateSampler] = None
        self._hv_encoder: Optional[CathedralHypervectorEncoder] = None
        self._latent_encoder: Optional[PCALatentEncoder] = None
        self._trajectory: Optional[TrajectoryBuffer] = None
        self._regime_classifier: Optional[RegimeClassifier] = None
        self._mode_controller: Optional[ModeController] = None
        self._action_executor: Optional[ActionExecutor] = None
        self._state_logger: Optional[StateLogger] = None

        # State
        self._running: bool = False
        self._tick_count: int = 0
        self._warmup_complete: bool = False
        self._start_time: Optional[datetime] = None

        # Current state cache
        self._current_z: Optional[np.ndarray] = None
        self._current_hv: Optional[np.ndarray] = None
        self._current_regime: Optional[str] = None
        self._current_mode: Optional[OperatingMode] = None

        logger.info("CathedralWatchDaemon initialized")

    def _initialize_components(self):
        """Initialize all components."""
        logger.info("Initializing cathedral watch components...")

        # State sampler
        self._sampler = get_state_sampler(tick_interval=self.config.tick_interval)

        # Hypervector encoder
        hv_config = HypervectorConfig(dim=self.config.hv_dim)
        self._hv_encoder = CathedralHypervectorEncoder(hv_config)

        # Latent encoder (use incremental PCA for online learning)
        self._latent_encoder = IncrementalPCAEncoder(
            latent_dim=self.config.latent_dim,
            batch_size=50,
        )

        # Load pre-trained encoder if available
        if self.config.latent_encoder_path:
            path = Path(self.config.latent_encoder_path)
            if path.exists():
                self._latent_encoder.load(path)
                logger.info(f"Loaded latent encoder from {path}")

        # Trajectory buffer
        self._trajectory = TrajectoryBuffer(
            max_len=self.config.trajectory_max_len,
            latent_dim=self.config.latent_dim,
        )

        # Regime classifier
        self._regime_classifier = RegimeClassifier(
            n_clusters=12,
            latent_dim=self.config.latent_dim,
        )

        # Load pre-trained classifier if available
        if self.config.regime_classifier_path:
            path = Path(self.config.regime_classifier_path)
            if path.exists():
                self._regime_classifier.load(path)
                logger.info(f"Loaded regime classifier from {path}")

        # Mode controller
        self._mode_controller = ModeController(
            regime_classifier=self._regime_classifier,
        )

        # Action executor
        self._action_executor = ActionExecutor(dry_run=self.config.dry_run)

        # State logger
        if self.config.enable_logging:
            self._state_logger = get_state_logger(
                base_dir=self.config.log_dir,
                hv_dim=self.config.hv_dim,
                latent_dim=self.config.latent_dim,
            )

        logger.info("Components initialized")

    async def _tick(self) -> Dict[str, Any]:
        """
        Execute one tick of the control loop.

        Returns:
            Tick result dict
        """
        t0 = time.time()

        # 1. Sample state
        features = await self._sampler.sample(force=True)

        # 2. Encode to hypervector
        hv = self._hv_encoder.encode_state(features)
        self._current_hv = hv

        # 3. Compress to latent space
        # During warmup, train the encoder incrementally
        if not self._warmup_complete:
            self._latent_encoder.partial_fit(hv.reshape(1, -1))

            if self._tick_count >= self.config.warmup_ticks:
                self._warmup_complete = True
                logger.info(f"Warmup complete after {self._tick_count} ticks")

        # Encode to latent (even during warmup, for trajectory tracking)
        try:
            z = self._latent_encoder.encode(hv)
        except Exception:
            # Encoder not ready yet
            z = np.zeros(self.config.latent_dim)

        self._current_z = z

        # 4. Add to trajectory
        regime_name = self._current_regime
        self._trajectory.add(z, features=features, regime=regime_name)

        # 5. Classify regime (if classifier is trained)
        if self._regime_classifier.is_trained:
            regime = self._regime_classifier.classify(z)
            self._current_regime = regime.type.name
        else:
            self._current_regime = None

        # 6. Update mode
        mode = self._mode_controller.update_mode(z, features, self._trajectory)
        self._current_mode = mode

        # 7. Get and execute actions
        actions = []
        if self.config.enable_actions:
            actions = self._mode_controller.get_actions(z, self._trajectory, features)
            if actions:
                await self._action_executor.execute_all(actions)

        # 8. Log state
        if self._state_logger:
            self._state_logger.log_tick(
                features=features,
                hv=hv,
                z=z,
                regime=self._current_regime,
                mode=mode.name,
                actions=[a.type.name for a in actions],
            )

        # Update tick count
        self._tick_count += 1
        tick_time = time.time() - t0

        return {
            "tick": self._tick_count,
            "z": z.tolist(),
            "regime": self._current_regime,
            "mode": mode.name,
            "actions": [a.type.name for a in actions],
            "tick_time_ms": tick_time * 1000,
        }

    async def start(self):
        """Start the daemon main loop."""
        if self._running:
            logger.warning("Daemon already running")
            return

        self._running = True
        self._start_time = datetime.utcnow()
        self._tick_count = 0

        # Initialize components
        self._initialize_components()

        # Start state logger session
        if self._state_logger:
            self._state_logger.start_session()

        logger.info(f"Cathedral watch daemon started (tick={self.config.tick_interval}s)")

        try:
            while self._running:
                tick_start = time.time()

                try:
                    result = await self._tick()

                    # Log status periodically
                    if self._tick_count % 100 == 0:
                        logger.info(
                            f"Tick {self._tick_count}: mode={result['mode']}, "
                            f"regime={result['regime']}, tick_time={result['tick_time_ms']:.1f}ms"
                        )

                except Exception as e:
                    logger.error(f"Tick error: {e}", exc_info=True)

                # Sleep for remaining tick interval
                elapsed = time.time() - tick_start
                sleep_time = max(0, self.config.tick_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Daemon cancelled")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the daemon."""
        if not self._running:
            return

        self._running = False

        # End state logger session
        if self._state_logger:
            summary = self._state_logger.end_session()
            logger.info(f"Session summary: {summary}")

        logger.info("Cathedral watch daemon stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current daemon status."""
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "warmup_complete": self._warmup_complete,
            "current_z": self._current_z.tolist() if self._current_z is not None else None,
            "current_regime": self._current_regime,
            "current_mode": self._current_mode.name if self._current_mode else None,
            "trajectory_length": len(self._trajectory) if self._trajectory else 0,
            "controller_status": self._mode_controller.get_status() if self._mode_controller else None,
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0,
        }


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Main entry point for the daemon."""
    import argparse

    parser = argparse.ArgumentParser(description="Cathedral Watch Daemon")
    parser.add_argument("--tick", type=float, default=0.5, help="Tick interval in seconds")
    parser.add_argument("--hv-dim", type=int, default=1024, help="Hypervector dimensionality")
    parser.add_argument("--latent-dim", type=int, default=10, help="Latent dimensionality")
    parser.add_argument("--log-dir", type=str, default="/var/ara/logs/state", help="Log directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute actions")
    parser.add_argument("--no-log", action="store_true", help="Disable state logging")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Create config
    config = CathedralWatchConfig(
        tick_interval=args.tick,
        hv_dim=args.hv_dim,
        latent_dim=args.latent_dim,
        log_dir=args.log_dir,
        dry_run=args.dry_run,
        enable_logging=not args.no_log,
    )

    # Create and start daemon
    daemon = CathedralWatchDaemon(config)

    # Handle signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(daemon.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Run
    await daemon.start()


if __name__ == "__main__":
    asyncio.run(main())
