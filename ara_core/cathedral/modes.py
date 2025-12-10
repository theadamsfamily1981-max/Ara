#!/usr/bin/env python3
"""
Cathedral OS - Operational Modes
=================================

Switch between fandom-themed operational regimes:

    🖖 starfleet     - Ethical expansion, MEIS governance
    🐱 red_dwarf     - Junkyard survival, entropy resistance
    👨‍⚕️ time_lord     - Regeneration, identity continuity
    ⚔️ colonial_fleet - War, infiltration resistance

Each mode adjusts:
    - Focus metrics (what's foregrounded)
    - Invariant thresholds (what triggers alerts)
    - Stress test priorities
    - UI theme

Usage:
    from ara_core.cathedral.modes import (
        get_mode, set_mode, list_modes, mode_status
    )

    # Switch mode
    set_mode("colonial_fleet")

    # Get current invariants
    mode = get_mode()
    print(mode.invariants["H_influence_min"])  # 6.0

    # Check if current state satisfies mode invariants
    status = mode_status()
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class CathedralMode:
    """A Cathedral operational mode."""
    name: str
    fandom: str
    theme: str
    tagline: str
    invariants: Dict[str, Any]
    focus_metrics: List[str]
    stress_tests: List[str]
    notes: str
    theme_colors: Dict[str, str] = field(default_factory=dict)

    def check_invariant(self, metric: str, value: float) -> bool:
        """Check if a value satisfies this mode's invariant."""
        if metric not in self.invariants:
            return True  # No constraint

        threshold = self.invariants[metric]

        # Handle different invariant types
        if isinstance(threshold, bool):
            return value == threshold
        elif metric.endswith("_min"):
            return value >= threshold
        elif metric.endswith("_max"):
            return value <= threshold
        else:
            return value >= threshold  # Default: minimum

    def get_icon(self) -> str:
        """Get mode icon."""
        icons = {
            "starfleet": "🖖",
            "red_dwarf": "🐱",
            "time_lord": "👨‍⚕️",
            "colonial_fleet": "⚔️",
            "k10_toaster": "🧈",
        }
        return icons.get(self.name, "🏛️")

    def status_line(self) -> str:
        """Get one-line status."""
        return f"{self.get_icon()} {self.name.upper()}: {self.tagline}"


class ModeManager:
    """Manages Cathedral operational modes."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._find_config()
        self.modes: Dict[str, CathedralMode] = {}
        self.current_mode: Optional[CathedralMode] = None
        self.themes: Dict[str, Dict[str, str]] = {}

        self._load_config()

    def _find_config(self) -> str:
        """Find cathedral_modes.yaml config file."""
        # Try common locations
        candidates = [
            Path("/home/user/Ara/config/cathedral_modes.yaml"),
            Path("config/cathedral_modes.yaml"),
            Path("cathedral_modes.yaml"),
        ]

        for path in candidates:
            if path.exists():
                return str(path)

        # Return default path (will create if needed)
        return str(candidates[0])

    def _load_config(self):
        """Load modes from YAML config."""
        if not os.path.exists(self.config_path):
            # Create default config
            self._create_default_config()
            return

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load themes
        self.themes = config.get("themes", {})

        # Load modes
        for name, mode_data in config.get("modes", {}).items():
            theme_colors = self.themes.get(name, {})
            invariants = mode_data.get("invariants", {})

            # Extract notes from invariants if present (simplified YAML format)
            notes = invariants.pop("notes", "") if isinstance(invariants, dict) else ""
            if not notes:
                notes = mode_data.get("notes", "")

            mode = CathedralMode(
                name=name,
                fandom=mode_data.get("fandom", ""),
                theme=mode_data.get("theme", ""),
                tagline=mode_data.get("tagline", notes[:50] if notes else ""),
                invariants=invariants,
                focus_metrics=mode_data.get("focus_metrics", []),
                stress_tests=mode_data.get("stress_tests", []),
                notes=notes,
                theme_colors=theme_colors,
            )
            self.modes[name] = mode

        # Set default mode
        default = config.get("default_mode", "starfleet")
        if default in self.modes:
            self.current_mode = self.modes[default]

    def _create_default_config(self):
        """Create default config if none exists."""
        # Create a minimal starfleet mode
        self.modes["starfleet"] = CathedralMode(
            name="starfleet",
            fandom="Star Trek",
            theme="Ethical expansion, exploration",
            tagline="To boldly go",
            invariants={"T_s_min": 0.99, "H_s_min": 0.977},
            focus_metrics=["T_s", "governance_gates"],
            stress_tests=["stress_overdose"],
            notes="Default mode",
        )
        self.current_mode = self.modes["starfleet"]

    def get(self, name: str = None) -> Optional[CathedralMode]:
        """Get a mode by name, or current mode."""
        if name is None:
            return self.current_mode
        return self.modes.get(name)

    def set(self, name: str) -> bool:
        """Set the current mode."""
        if name not in self.modes:
            return False
        self.current_mode = self.modes[name]
        return True

    def list(self) -> List[str]:
        """List all available modes."""
        return list(self.modes.keys())

    def status(self, metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """Check current mode status against metrics."""
        if self.current_mode is None:
            return {"mode": None, "ok": False}

        mode = self.current_mode
        result = {
            "mode": mode.name,
            "icon": mode.get_icon(),
            "tagline": mode.tagline,
            "invariants": {},
            "all_ok": True,
        }

        if metrics:
            for key, threshold in mode.invariants.items():
                # Map invariant key to metric
                metric_key = key.replace("_min", "").replace("_max", "")

                if metric_key in metrics:
                    value = metrics[metric_key]
                    ok = mode.check_invariant(key, value)
                    result["invariants"][key] = {
                        "threshold": threshold,
                        "value": value,
                        "ok": ok,
                    }
                    if not ok:
                        result["all_ok"] = False

        return result

    def render_dashboard_header(self) -> str:
        """Render mode header for dashboard."""
        if self.current_mode is None:
            return "║  MODE: NONE SELECTED                                                        ║"

        mode = self.current_mode
        icon = mode.get_icon()
        name = mode.name.upper()
        tagline = mode.tagline[:50]

        return f"║  {icon} {name}: {tagline:<55}║"


# =============================================================================
# SINGLETON AND CONVENIENCE
# =============================================================================

_manager: Optional[ModeManager] = None


def get_manager() -> ModeManager:
    """Get the global mode manager."""
    global _manager
    if _manager is None:
        _manager = ModeManager()
    return _manager


def get_mode(name: str = None) -> Optional[CathedralMode]:
    """Get a mode by name, or current mode."""
    return get_manager().get(name)


def set_mode(name: str) -> bool:
    """Set the current operational mode."""
    return get_manager().set(name)


def list_modes() -> List[str]:
    """List all available modes."""
    return get_manager().list()


def mode_status(metrics: Dict[str, float] = None) -> Dict[str, Any]:
    """Check current mode status."""
    return get_manager().status(metrics)


def current_mode_name() -> str:
    """Get current mode name."""
    mode = get_mode()
    return mode.name if mode else "none"


def mode_dashboard_header() -> str:
    """Get dashboard header for current mode."""
    return get_manager().render_dashboard_header()
