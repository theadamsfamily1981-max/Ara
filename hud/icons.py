#!/usr/bin/env python3
"""
hud/icons.py - Icon loading utilities for T-FAN HUD

Provides SVG icon loading and caching for the GNOME cockpit interface.
Uses the icons from gnome-tfan/icons/scalable/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GdkPixbuf, Gtk

# Icon directory
ICON_DIR = Path(__file__).parent.parent / "gnome-tfan" / "icons" / "scalable"

# Icon name mapping
ICONS = {
    # Core icons
    "tfan": "tfan-icon.svg",
    "ara": "ara-avatar-icon.svg",
    "metrics": "metrics-icon.svg",
    "topology": "topology-icon.svg",
    "training": "training-icon.svg",
    "pareto": "pareto-icon.svg",
    "work-mode": "work-mode-icon.svg",
    "relax-mode": "relax-mode-icon.svg",

    # Sanity & Criticality
    "criticality": "criticality-gauge-icon.svg",
    "sanity": "sanity-monitor-icon.svg",
    "antifragility": "antifragility-icon.svg",
    "hgf": "hgf-loop-icon.svg",
    "dau": "dau-active-icon.svg",

    # Drive states
    "drive-low": "drive-low-icon.svg",
    "drive-med": "drive-med-icon.svg",
    "drive-high": "drive-high-icon.svg",

    # Valence
    "valence-positive": "valence-positive-icon.svg",
    "valence-negative": "valence-negative-icon.svg",
}


class IconLoader:
    """Loads and caches SVG icons for the HUD."""

    def __init__(self, icon_dir: Optional[Path] = None):
        self.icon_dir = icon_dir or ICON_DIR
        self._cache: dict[str, Gtk.Picture] = {}
        self._pixbuf_cache: dict[tuple[str, int], GdkPixbuf.Pixbuf] = {}

    def get_icon_path(self, name: str) -> Optional[Path]:
        """Get the full path to an icon by name."""
        filename = ICONS.get(name)
        if filename is None:
            return None
        path = self.icon_dir / filename
        return path if path.exists() else None

    def load_picture(self, name: str, size: int = 48) -> Optional[Gtk.Picture]:
        """Load an icon as a Gtk.Picture widget."""
        path = self.get_icon_path(name)
        if path is None:
            return None

        cache_key = (name, size)
        if cache_key in self._cache:
            # Return a new Picture with the same file
            return Gtk.Picture.new_for_filename(str(path))

        try:
            picture = Gtk.Picture.new_for_filename(str(path))
            picture.set_size_request(size, size)
            picture.set_can_shrink(True)
            picture.set_keep_aspect_ratio(True)
            self._cache[cache_key] = picture
            return picture
        except Exception as e:
            print(f"[IconLoader] Failed to load {name}: {e}")
            return None

    def load_pixbuf(self, name: str, size: int = 48) -> Optional[GdkPixbuf.Pixbuf]:
        """Load an icon as a GdkPixbuf (for custom rendering)."""
        path = self.get_icon_path(name)
        if path is None:
            return None

        cache_key = (name, size)
        if cache_key in self._pixbuf_cache:
            return self._pixbuf_cache[cache_key]

        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                str(path),
                width=size,
                height=size,
                preserve_aspect_ratio=True
            )
            self._pixbuf_cache[cache_key] = pixbuf
            return pixbuf
        except Exception as e:
            print(f"[IconLoader] Failed to load pixbuf {name}: {e}")
            return None

    def get_drive_icon(self, drive_value: float) -> str:
        """Get the appropriate drive icon name based on drive value."""
        if drive_value < 0.3:
            return "drive-low"
        elif drive_value < 0.6:
            return "drive-med"
        else:
            return "drive-high"

    def get_valence_icon(self, valence: float) -> str:
        """Get the appropriate valence icon name."""
        if valence > 0.001:
            return "valence-positive"
        elif valence < -0.001:
            return "valence-negative"
        else:
            return "valence-positive"  # Default to positive for neutral


# Global singleton
_loader: Optional[IconLoader] = None


def get_icon_loader() -> IconLoader:
    """Get the global icon loader instance."""
    global _loader
    if _loader is None:
        _loader = IconLoader()
    return _loader


def list_available_icons() -> list[str]:
    """List all available icon names."""
    return list(ICONS.keys())


if __name__ == "__main__":
    # Test icon loading
    loader = get_icon_loader()
    print(f"Icon directory: {loader.icon_dir}")
    print(f"Available icons: {list_available_icons()}")

    for name in ICONS:
        path = loader.get_icon_path(name)
        status = "✓" if path and path.exists() else "✗"
        print(f"  {status} {name}: {path}")
