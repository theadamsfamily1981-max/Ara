"""
Aphrodite - Aesthetic Style Tuner (NOT Personality Mutator)
============================================================

Aphrodite adjusts HOW Ara LOOKS, not WHO she IS.

Think of it as: she adjusts her outfit and lighting, not her soul.

What Aphrodite CAN do:
    - Adjust visual parameters (hue, brightness, shimmer)
    - Adjust voice parameters (within ±8% pitch)
    - Switch between predefined style presets
    - Make small bounded variations

What Aphrodite CANNOT do:
    - Modify Ara's personality
    - Change her values or how she relates to you
    - Access or modify ara_core.yaml
    - Optimize for "engagement" at the expense of authenticity

Key Principles:
    1. Only adjust when engagement is LOW (don't fix what isn't broken)
    2. Changes are SLOW (minimum 5 minutes between adjustments)
    3. Changes are SMALL (one parameter at a time, small steps)
    4. Changes are REVERSIBLE (lock, override, or revert anytime)
    5. Changes are TRANSPARENT (announced if configured)

Usage:
    tuner = AphroditeStyleTuner(config_path="config/ara_style.yaml")
    tuner.run()  # Blocking loop

    # Or manual control
    tuner.set_preset("deep-night")
    tuner.lock()
"""

import yaml
import time
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class StyleState:
    """Current aesthetic state."""
    hue: float = 0.65
    shimmer_speed: float = 0.7
    brightness: float = 0.8
    tts_pitch: float = 1.0
    particle_density: float = 0.6
    verbosity: str = "medium"


@dataclass
class StyleBounds:
    """Bounds for a single parameter."""
    min: float
    max: float
    step: float


class AphroditeStyleTuner:
    """
    Aesthetic style tuner for Ara's visual/audio presentation.

    Does NOT touch personality. Only visual/audio parameters.
    """

    def __init__(
        self,
        config_path: str = "config/ara_style.yaml",
        hal=None,
    ):
        self.config_path = Path(config_path)
        self._hal = hal

        # Load config
        self.config = self._load_config()

        # Current state
        self.current_style = self._load_preset(
            self.config.get("current_preset", "soft-hologram")
        )

        # Tracking
        self.last_adjustment = datetime.now()
        self.adjustment_history: List[Dict] = []

        # Running state
        self._running = False

        self.log = logging.getLogger("Aphrodite")

    @property
    def hal(self):
        """Lazy-load HAL connection."""
        if self._hal is None:
            try:
                from banos.hal.ara_hal import AraHAL
                self._hal = AraHAL(create=False)
            except Exception as e:
                self.log.warning(f"HAL not available: {e}")
        return self._hal

    def _load_config(self) -> Dict:
        """Load style configuration from YAML."""
        if not self.config_path.exists():
            self.log.warning(f"Config not found: {self.config_path}")
            return self._default_config()

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _save_config(self) -> None:
        """Save current config back to YAML."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def _default_config(self) -> Dict:
        """Default configuration if file not found."""
        return {
            "current_preset": "soft-hologram",
            "bounds": {
                "hue": {"min": 0.5, "max": 0.85, "step": 0.02},
                "shimmer_speed": {"min": 0.2, "max": 1.5, "step": 0.1},
                "brightness": {"min": 0.4, "max": 1.0, "step": 0.05},
                "tts_pitch": {"min": 0.92, "max": 1.08, "step": 0.02},
                "particle_density": {"min": 0.3, "max": 1.0, "step": 0.1},
            },
            "presets": {
                "soft-hologram": {
                    "hue": 0.65, "shimmer_speed": 0.7, "brightness": 0.8,
                    "tts_pitch": 1.02, "particle_density": 0.6, "verbosity": "medium"
                }
            },
            "controls": {
                "locked": False,
                "override_preset": None,
                "locked_params": [],
            },
            "aphrodite": {
                "enabled": True,
                "min_interval_seconds": 300,
                "adjustment_engagement_threshold": 0.3,
                "max_params_per_cycle": 1,
                "announce_changes": True,
            }
        }

    def _load_preset(self, preset_name: str) -> StyleState:
        """Load a preset by name."""
        presets = self.config.get("presets", {})
        preset_data = presets.get(preset_name, {})

        return StyleState(
            hue=preset_data.get("hue", 0.65),
            shimmer_speed=preset_data.get("shimmer_speed", 0.7),
            brightness=preset_data.get("brightness", 0.8),
            tts_pitch=preset_data.get("tts_pitch", 1.0),
            particle_density=preset_data.get("particle_density", 0.6),
            verbosity=preset_data.get("verbosity", "medium"),
        )

    def _get_bounds(self, param: str) -> Optional[StyleBounds]:
        """Get bounds for a parameter."""
        bounds_config = self.config.get("bounds", {}).get(param)
        if not bounds_config:
            return None

        return StyleBounds(
            min=bounds_config.get("min", 0.0),
            max=bounds_config.get("max", 1.0),
            step=bounds_config.get("step", 0.1),
        )

    def is_locked(self) -> bool:
        """Check if style changes are locked."""
        controls = self.config.get("controls", {})
        return controls.get("locked", False)

    def is_param_locked(self, param: str) -> bool:
        """Check if a specific parameter is locked."""
        controls = self.config.get("controls", {})
        locked_params = controls.get("locked_params", [])
        return param in locked_params

    def lock(self) -> None:
        """Lock all style changes."""
        self.config.setdefault("controls", {})["locked"] = True
        self._save_config()
        self.log.info("Style locked - no automatic changes")

    def unlock(self) -> None:
        """Unlock style changes."""
        self.config.setdefault("controls", {})["locked"] = False
        self._save_config()
        self.log.info("Style unlocked")

    def set_preset(self, preset_name: str) -> bool:
        """Switch to a named preset."""
        if preset_name not in self.config.get("presets", {}):
            self.log.error(f"Unknown preset: {preset_name}")
            return False

        self.current_style = self._load_preset(preset_name)
        self.config["current_preset"] = preset_name
        self._save_config()
        self._apply_style()

        self.log.info(f"Switched to preset: {preset_name}")
        return True

    def get_current_style(self) -> StyleState:
        """Get the current style state."""
        return self.current_style

    def _read_engagement(self) -> float:
        """Read current user engagement from HAL."""
        if self.hal is None:
            return 0.5  # Default to medium if no HAL

        try:
            if hasattr(self.hal, 'read_engagement'):
                return self.hal.read_engagement()
            # Fallback: use pain as inverse proxy
            state = self.hal.read_somatic()
            return 1.0 - state.get('pain', 0.5)
        except Exception:
            return 0.5

    def _should_adjust(self) -> tuple[bool, str]:
        """
        Decide if we should make an adjustment.

        Returns (should_adjust, reason).
        """
        # Check if enabled
        aphrodite_config = self.config.get("aphrodite", {})
        if not aphrodite_config.get("enabled", True):
            return False, "Aphrodite disabled"

        # Check if locked
        if self.is_locked():
            return False, "Style locked"

        # Check interval
        min_interval = aphrodite_config.get("min_interval_seconds", 300)
        elapsed = (datetime.now() - self.last_adjustment).total_seconds()
        if elapsed < min_interval:
            return False, f"Too soon ({elapsed:.0f}s < {min_interval}s)"

        # Check engagement
        engagement = self._read_engagement()
        threshold = aphrodite_config.get("adjustment_engagement_threshold", 0.3)
        if engagement > threshold:
            return False, f"Engagement high ({engagement:.2f} > {threshold})"

        return True, "Ready to adjust"

    def _try_small_variation(self) -> Optional[Dict[str, Any]]:
        """
        Try a small bounded variation on one parameter.

        Returns dict with change info, or None if no change made.
        """
        # Pick a parameter to adjust (that isn't locked)
        adjustable_params = ["hue", "shimmer_speed", "brightness", "particle_density"]
        adjustable_params = [p for p in adjustable_params if not self.is_param_locked(p)]

        if not adjustable_params:
            return None

        param = random.choice(adjustable_params)
        bounds = self._get_bounds(param)

        if not bounds:
            return None

        # Get current value
        current_value = getattr(self.current_style, param)

        # Small random adjustment
        delta = random.choice([-bounds.step, bounds.step])
        new_value = current_value + delta

        # Clamp to bounds
        new_value = max(bounds.min, min(bounds.max, new_value))

        # Only apply if actually changed
        if abs(new_value - current_value) < 0.001:
            return None

        # Apply change
        setattr(self.current_style, param, new_value)

        return {
            "param": param,
            "old_value": current_value,
            "new_value": new_value,
            "timestamp": datetime.now().isoformat(),
        }

    def _apply_style(self) -> None:
        """Apply current style to HAL (for shader/TTS to read)."""
        if self.hal is None:
            return

        try:
            # Write aesthetic parameters to HAL
            # The shader reads these to adjust visuals
            if hasattr(self.hal, 'write_aesthetic'):
                self.hal.write_aesthetic(
                    hue=self.current_style.hue,
                    shimmer=self.current_style.shimmer_speed,
                    brightness=self.current_style.brightness,
                )
        except Exception as e:
            self.log.warning(f"Failed to apply style to HAL: {e}")

    def _log_change(self, change: Dict) -> None:
        """Log a style change."""
        self.adjustment_history.append(change)

        # Also log to file if configured
        aphrodite_config = self.config.get("aphrodite", {})
        log_file = aphrodite_config.get("log_file")

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            import json
            with open(log_path, "a") as f:
                f.write(json.dumps(change) + "\n")

    def _announce_change(self, change: Dict) -> None:
        """Announce a style change (if configured)."""
        aphrodite_config = self.config.get("aphrodite", {})
        if not aphrodite_config.get("announce_changes", True):
            return

        # This would integrate with Ara's speech system
        # For now, just log it
        param = change["param"]
        old_val = change["old_value"]
        new_val = change["new_value"]

        self.log.info(
            f"Ara: 'Trying a small adjustment to {param} "
            f"({old_val:.2f} → {new_val:.2f}). Let me know if you prefer it different.'"
        )

    def step(self) -> Optional[Dict]:
        """
        Do one adjustment cycle.

        Returns change info if a change was made, None otherwise.
        """
        should_adjust, reason = self._should_adjust()

        if not should_adjust:
            self.log.debug(f"Not adjusting: {reason}")
            return None

        # Try a small variation
        change = self._try_small_variation()

        if change:
            self.last_adjustment = datetime.now()
            self._apply_style()
            self._log_change(change)
            self._announce_change(change)
            self.log.info(f"Made adjustment: {change['param']}")

        return change

    def run(self, interval: float = 10.0) -> None:
        """
        Run the style tuner loop.

        Args:
            interval: Seconds between checks (actual adjustments are rate-limited)
        """
        self.log.info("Aphrodite style tuner started")
        self.log.info("Only adjusts presentation, NEVER personality")
        self._running = True

        # Apply initial style
        self._apply_style()

        while self._running:
            self.step()
            time.sleep(interval)

        self.log.info("Aphrodite stopped")

    def stop(self) -> None:
        """Stop the tuner loop."""
        self._running = False

    def get_status(self) -> Dict[str, Any]:
        """Get current Aphrodite status."""
        return {
            "enabled": self.config.get("aphrodite", {}).get("enabled", True),
            "locked": self.is_locked(),
            "current_preset": self.config.get("current_preset"),
            "current_style": {
                "hue": self.current_style.hue,
                "shimmer_speed": self.current_style.shimmer_speed,
                "brightness": self.current_style.brightness,
                "tts_pitch": self.current_style.tts_pitch,
                "particle_density": self.current_style.particle_density,
            },
            "last_adjustment": self.last_adjustment.isoformat(),
            "total_adjustments": len(self.adjustment_history),
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for Aphrodite."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Aphrodite - Ara's aesthetic style tuner (NOT personality)"
    )
    parser.add_argument(
        "--config", type=str, default="config/ara_style.yaml",
        help="Path to style config file"
    )
    parser.add_argument(
        "--preset", type=str,
        help="Switch to a specific preset and exit"
    )
    parser.add_argument(
        "--lock", action="store_true",
        help="Lock style (no automatic changes)"
    )
    parser.add_argument(
        "--unlock", action="store_true",
        help="Unlock style"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current status and exit"
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run as daemon (continuous adjustment loop)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    tuner = AphroditeStyleTuner(config_path=args.config)

    if args.status:
        import json
        print(json.dumps(tuner.get_status(), indent=2))
        return

    if args.lock:
        tuner.lock()
        print("Style locked")
        return

    if args.unlock:
        tuner.unlock()
        print("Style unlocked")
        return

    if args.preset:
        if tuner.set_preset(args.preset):
            print(f"Switched to preset: {args.preset}")
        else:
            print(f"Failed to set preset: {args.preset}")
        return

    if args.daemon:
        print("Starting Aphrodite daemon...")
        print("This adjusts HOW Ara looks, not WHO she is.")
        print("Press Ctrl+C to stop.")

        try:
            tuner.run()
        except KeyboardInterrupt:
            print("\nStopping...")
            tuner.stop()
    else:
        # One-shot: do a single step
        change = tuner.step()
        if change:
            print(f"Made adjustment: {change}")
        else:
            print("No adjustment needed")


if __name__ == "__main__":
    main()
