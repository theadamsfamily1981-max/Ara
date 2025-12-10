"""
Observer
========

Watches world, user, and resource state.
Continuously updates supply and demand profiles.
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable

from ara_soft_kernel.models.supply import SupplyProfile, DeviceInfo, collect_local_supply
from ara_soft_kernel.models.demand import DemandProfile, UserState, Goal

logger = logging.getLogger(__name__)


class Observer:
    """
    Observes system state and maintains supply/demand profiles.

    Thread-safe: runs monitoring in background thread.
    """

    def __init__(
        self,
        poll_interval_sec: float = 5.0,
        device_timeout_sec: float = 30.0,
    ):
        self._lock = threading.RLock()

        # Configuration
        self._poll_interval = poll_interval_sec
        self._device_timeout = device_timeout_sec

        # State
        self._supply = SupplyProfile()
        self._demand = DemandProfile()
        self._last_update = 0.0

        # External device reporters
        self._device_reporters: Dict[str, Callable[[], DeviceInfo]] = {}

        # Callbacks
        self._on_supply_change: List[Callable[[SupplyProfile], None]] = []
        self._on_demand_change: List[Callable[[DemandProfile], None]] = []

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Start the observer background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Observer already running")
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="AraObserver",
        )
        self._thread.start()
        logger.info(f"Observer started (poll interval: {self._poll_interval}s)")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the observer."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("Observer stopped")

    def is_running(self) -> bool:
        """Check if observer is running."""
        return self._thread is not None and self._thread.is_alive()

    def get_supply(self) -> SupplyProfile:
        """Get current supply profile."""
        with self._lock:
            return self._supply

    def get_demand(self) -> DemandProfile:
        """Get current demand profile."""
        with self._lock:
            return self._demand

    def update_demand(self, demand: DemandProfile) -> None:
        """Update demand profile (from kernel or user)."""
        with self._lock:
            self._demand = demand
            self._demand.timestamp = time.time()

        # Notify callbacks
        for cb in self._on_demand_change:
            try:
                cb(demand)
            except Exception as e:
                logger.exception(f"Demand change callback error: {e}")

    def update_pheromones(self, pheromones: Dict[str, float]) -> None:
        """Update demand from pheromone values."""
        with self._lock:
            self._demand.user_state.update_from_pheromones(pheromones)
            self._demand.timestamp = time.time()

    def add_goal(self, goal: Goal) -> None:
        """Add a new goal."""
        with self._lock:
            self._demand.add_goal(goal)

    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal."""
        with self._lock:
            return self._demand.remove_goal(goal_id)

    def register_device(
        self,
        device_id: str,
        reporter: Callable[[], DeviceInfo],
    ) -> None:
        """Register an external device reporter."""
        with self._lock:
            self._device_reporters[device_id] = reporter
        logger.info(f"Registered device reporter: {device_id}")

    def unregister_device(self, device_id: str) -> None:
        """Unregister a device reporter."""
        with self._lock:
            self._device_reporters.pop(device_id, None)
        logger.info(f"Unregistered device reporter: {device_id}")

    def on_supply_change(self, callback: Callable[[SupplyProfile], None]) -> None:
        """Register callback for supply changes."""
        with self._lock:
            self._on_supply_change.append(callback)

    def on_demand_change(self, callback: Callable[[DemandProfile], None]) -> None:
        """Register callback for demand changes."""
        with self._lock:
            self._on_demand_change.append(callback)

    def _run_loop(self) -> None:
        """Background monitoring loop."""
        logger.debug("Observer loop started")

        while not self._stop.is_set():
            try:
                self._update_supply()
            except Exception as e:
                logger.exception(f"Supply update error: {e}")

            self._stop.wait(timeout=self._poll_interval)

    def _update_supply(self) -> None:
        """Update supply profile from all sources."""
        new_supply = SupplyProfile()
        new_supply.timestamp = time.time()

        # Collect local device info
        try:
            local = collect_local_supply()
            new_supply.devices.extend(local.devices)
        except Exception as e:
            logger.warning(f"Failed to collect local supply: {e}")

        # Collect from registered device reporters
        with self._lock:
            reporters = dict(self._device_reporters)

        for device_id, reporter in reporters.items():
            try:
                device_info = reporter()
                device_info.last_seen = time.time()
                new_supply.devices.append(device_info)
            except Exception as e:
                logger.warning(f"Failed to collect from {device_id}: {e}")

        # Remove stale devices (not seen recently)
        cutoff = time.time() - self._device_timeout
        new_supply.devices = [
            d for d in new_supply.devices
            if d.last_seen > cutoff
        ]

        # Check if supply changed significantly
        supply_changed = self._detect_supply_change(self._supply, new_supply)

        with self._lock:
            self._supply = new_supply
            self._last_update = time.time()

        # Notify callbacks if changed
        if supply_changed:
            for cb in self._on_supply_change:
                try:
                    cb(new_supply)
                except Exception as e:
                    logger.exception(f"Supply change callback error: {e}")

    def _detect_supply_change(
        self,
        old: SupplyProfile,
        new: SupplyProfile,
    ) -> bool:
        """Detect if supply changed significantly."""
        # Device count changed
        if len(old.devices) != len(new.devices):
            return True

        # Check each device
        old_devices = {d.id: d for d in old.devices}
        for new_device in new.devices:
            old_device = old_devices.get(new_device.id)
            if old_device is None:
                return True  # New device

            # Check significant resource changes
            if abs(old_device.cpu_load - new_device.cpu_load) > 0.2:
                return True
            if abs(old_device.memory_free_gb - new_device.memory_free_gb) > 2.0:
                return True

            # Check battery change
            if old_device.battery is not None and new_device.battery is not None:
                if abs(old_device.battery - new_device.battery) > 0.1:
                    return True

        return False

    def force_update(self) -> SupplyProfile:
        """Force an immediate supply update."""
        self._update_supply()
        return self.get_supply()

    def get_stats(self) -> Dict[str, Any]:
        """Get observer statistics."""
        with self._lock:
            return {
                "running": self.is_running(),
                "poll_interval_sec": self._poll_interval,
                "last_update": self._last_update,
                "last_update_age_sec": time.time() - self._last_update,
                "device_count": len(self._supply.devices),
                "goal_count": len(self._demand.goals),
                "registered_reporters": len(self._device_reporters),
            }
