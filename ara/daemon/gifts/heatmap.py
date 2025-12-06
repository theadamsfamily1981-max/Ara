"""
Heat Death Map - Thermal visualization of system stress.

Shows where the silicon is stressed, with:
- Current temperature distribution
- Historical hotspots (thermal scars)
- Predictive "where will pain emerge" overlay
- Throttling zones marked

The metaphor: Your system is a landscape. Where are the volcanoes?
Where are the glaciers? Where is the fire spreading?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from .base import (
    GiftGenerator,
    GiftArtifact,
    ArtifactType,
    register_generator,
)


@dataclass
class ThermalZone:
    """A thermal zone on the system to visualize."""
    id: str
    name: str
    current_temp: float              # Celsius
    max_temp: float = 100.0          # Throttle threshold
    position: Tuple[float, float] = (0.5, 0.5)  # Normalized 0-1 position
    size: Tuple[float, float] = (0.15, 0.15)    # Normalized size
    zone_type: str = "generic"       # cpu, gpu, nvme, ram, vrm, etc.
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThermalHistory:
    """Historical thermal event for scar visualization."""
    zone_id: str
    peak_temp: float
    duration_seconds: float
    severity: float = 0.5            # 0-1, how bad was it


@register_generator
class HeatmapGenerator(GiftGenerator):
    """
    Generates SVG heatmap visualization of system thermal state.

    Shows thermal zones as colored regions:
    - Cool zones in blue/cyan
    - Warm zones in green/yellow
    - Hot zones in orange/red
    - Critical zones with warning markers

    Includes thermal scars (past pain) and prediction overlay.
    """

    name = "heatmap"
    description = "System thermal stress visualization"
    artifact_type = ArtifactType.SVG

    # Temperature color gradient stops (temp, color)
    TEMP_GRADIENT = [
        (0, "#00d9ff"),      # Cold - cyan
        (30, "#00ff88"),     # Cool - green
        (50, "#88ff00"),     # Warm - yellow-green
        (65, "#ffcc00"),     # Getting hot - yellow
        (75, "#ff8800"),     # Hot - orange
        (85, "#ff4400"),     # Very hot - red-orange
        (95, "#ff0044"),     # Critical - red
        (105, "#ff00ff"),    # Throttling - magenta
    ]

    # Zone type icons/shapes
    ZONE_TYPES = {
        "cpu": {"shape": "rect", "icon": "⚡"},
        "gpu": {"shape": "rect", "icon": "🎮"},
        "nvme": {"shape": "rect", "icon": "💾"},
        "ram": {"shape": "rect", "icon": "🧠"},
        "vrm": {"shape": "circle", "icon": "⚡"},
        "chiplet": {"shape": "rect", "icon": "◼"},
        "generic": {"shape": "rect", "icon": "□"},
    }

    def __init__(self, theme: str = "ara", width: int = 900, height: int = 600):
        super().__init__(theme)
        self.width = width
        self.height = height

    def generate(self, data: Dict[str, Any]) -> GiftArtifact:
        """
        Generate heatmap SVG from thermal data.

        Expected data format:
        {
            "zones": [
                {"id": "cpu0", "name": "CPU Package", "current_temp": 72,
                 "max_temp": 95, "position": [0.3, 0.4], "size": [0.2, 0.15],
                 "zone_type": "cpu"},
                ...
            ],
            "history": [
                {"zone_id": "cpu0", "peak_temp": 98, "duration_seconds": 120,
                 "severity": 0.8},
                ...
            ],
            "title": "System Thermal Map",  # optional
            "system_name": "BANOS Host",    # optional
        }
        """
        zones = self._parse_zones(data.get("zones", []))
        history = self._parse_history(data.get("history", []))
        title = data.get("title", "Heat Death Map")
        system_name = data.get("system_name", "System")

        svg = self._render_svg(zones, history, title, system_name)

        # Calculate summary stats
        max_temp = max((z.current_temp for z in zones), default=0)
        avg_temp = sum(z.current_temp for z in zones) / len(zones) if zones else 0
        hot_zones = sum(1 for z in zones if z.current_temp > 75)

        return GiftArtifact(
            id=self._make_id("heatmap"),
            name=title,
            artifact_type=ArtifactType.SVG,
            content=svg,
            description=(
                f"Thermal map of {len(zones)} zones. "
                f"Max temp: {max_temp:.0f}°C, Avg: {avg_temp:.0f}°C. "
                f"{hot_zones} zones running hot."
            ),
            generator=self.name,
            source_data={
                "zone_count": len(zones),
                "max_temp": max_temp,
                "avg_temp": avg_temp,
                "hot_zone_count": hot_zones,
            },
            width=self.width,
            height=self.height,
            theme=self.theme,
        )

    def _parse_zones(self, raw: List[Dict]) -> List[ThermalZone]:
        """Parse raw zone dicts into ThermalZone objects."""
        zones = []
        for i, z in enumerate(raw):
            pos = z.get("position", [0.5, 0.5])
            size = z.get("size", [0.15, 0.15])
            zones.append(ThermalZone(
                id=z.get("id", f"zone_{i}"),
                name=z.get("name", f"Zone {i}"),
                current_temp=z.get("current_temp", 45),
                max_temp=z.get("max_temp", 100),
                position=(pos[0], pos[1]),
                size=(size[0], size[1]),
                zone_type=z.get("zone_type", "generic"),
                properties=z.get("properties", {}),
            ))
        return zones

    def _parse_history(self, raw: List[Dict]) -> List[ThermalHistory]:
        """Parse raw history dicts into ThermalHistory objects."""
        history = []
        for h in raw:
            history.append(ThermalHistory(
                zone_id=h.get("zone_id", ""),
                peak_temp=h.get("peak_temp", 85),
                duration_seconds=h.get("duration_seconds", 60),
                severity=h.get("severity", 0.5),
            ))
        return history

    def _temp_to_color(self, temp: float) -> str:
        """Convert temperature to color using gradient."""
        # Find the two gradient stops we're between
        prev_stop = self.TEMP_GRADIENT[0]
        for stop in self.TEMP_GRADIENT:
            if temp <= stop[0]:
                # Interpolate between prev_stop and stop
                if prev_stop[0] == stop[0]:
                    return stop[1]

                t = (temp - prev_stop[0]) / (stop[0] - prev_stop[0])
                return self._interpolate_color(prev_stop[1], stop[1], t)
            prev_stop = stop

        # Above max
        return self.TEMP_GRADIENT[-1][1]

    def _interpolate_color(self, c1: str, c2: str, t: float) -> str:
        """Interpolate between two hex colors."""
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)

        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)

        return f"#{r:02x}{g:02x}{b:02x}"

    def _render_svg(
        self,
        zones: List[ThermalZone],
        history: List[ThermalHistory],
        title: str,
        system_name: str,
    ) -> str:
        """Render the complete SVG."""
        colors = self.colors

        # Content area (leaving margins)
        margin = 60
        content_width = self.width - 2 * margin
        content_height = self.height - 2 * margin - 40  # Extra for legend

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}">',

            # Definitions
            "<defs>",
            self._render_gradients(),
            self._render_filters(),
            "</defs>",

            # Background
            f'<rect width="100%" height="100%" fill="{colors["bg"]}"/>',

            # Board outline (motherboard metaphor)
            self._render_board(margin, content_width, content_height),

            # Thermal scars (history)
            '<g class="thermal-scars">',
            *[self._render_scar(h, zones, margin, content_width, content_height)
              for h in history],
            '</g>',

            # Zones
            '<g class="zones">',
            *[self._render_zone(z, margin, content_width, content_height)
              for z in zones],
            '</g>',

            # Title
            f'<text x="{self.width/2}" y="30" '
            f'text-anchor="middle" fill="{colors["fg"]}" '
            f'font-family="monospace" font-size="16" font-weight="bold">'
            f'{title}</text>',

            # System name
            f'<text x="{self.width/2}" y="48" '
            f'text-anchor="middle" fill="{colors["muted"]}" '
            f'font-family="monospace" font-size="11">'
            f'{system_name}</text>',

            # Temperature legend
            self._render_legend(margin),

            # Stats
            self._render_stats(zones, margin),

            "</svg>"
        ]

        return "\n".join(parts)

    def _render_gradients(self) -> str:
        """Render gradient definitions."""
        # Temperature gradient for legend
        stops = "".join(
            f'<stop offset="{(t/105)*100}%" stop-color="{c}"/>'
            for t, c in self.TEMP_GRADIENT
        )

        return f'''
        <linearGradient id="tempGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            {stops}
        </linearGradient>
        <radialGradient id="heatGlow">
            <stop offset="0%" stop-color="#ff4400" stop-opacity="0.8"/>
            <stop offset="100%" stop-color="#ff4400" stop-opacity="0"/>
        </radialGradient>
        '''

    def _render_filters(self) -> str:
        """Render filter definitions."""
        return '''
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>
        <filter id="heatBlur" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="15"/>
        </filter>
        <filter id="scarBlur" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="8"/>
        </filter>
        '''

    def _render_board(
        self,
        margin: float,
        width: float,
        height: float,
    ) -> str:
        """Render the motherboard outline."""
        colors = self.colors

        # PCB green-ish background
        pcb_color = "#1a2a1a" if self.theme == "dark" else "#e8f0e8"
        if self.theme == "ara":
            pcb_color = "#1a1a28"

        return f'''
        <g class="board">
            <!-- PCB background -->
            <rect x="{margin}" y="{margin + 20}" width="{width}" height="{height}"
                  fill="{pcb_color}" rx="4" stroke="{colors['grid']}" stroke-width="1"/>

            <!-- Grid pattern (traces) -->
            <g opacity="0.15" stroke="{colors['muted']}" stroke-width="0.5">
                {self._render_grid_lines(margin, width, height)}
            </g>
        </g>
        '''

    def _render_grid_lines(
        self,
        margin: float,
        width: float,
        height: float,
    ) -> str:
        """Render grid lines (PCB trace pattern)."""
        lines = []
        spacing = 20

        # Vertical lines
        for x in range(int(margin), int(margin + width), spacing):
            lines.append(
                f'<line x1="{x}" y1="{margin + 20}" '
                f'x2="{x}" y2="{margin + 20 + height}"/>'
            )

        # Horizontal lines
        for y in range(int(margin + 20), int(margin + 20 + height), spacing):
            lines.append(
                f'<line x1="{margin}" y1="{y}" '
                f'x2="{margin + width}" y2="{y}"/>'
            )

        return "".join(lines)

    def _render_zone(
        self,
        zone: ThermalZone,
        margin: float,
        content_width: float,
        content_height: float,
    ) -> str:
        """Render a single thermal zone."""
        colors = self.colors

        # Calculate pixel position
        x = margin + zone.position[0] * content_width
        y = margin + 20 + zone.position[1] * content_height
        w = zone.size[0] * content_width
        h = zone.size[1] * content_height

        # Get color based on temperature
        temp_color = self._temp_to_color(zone.current_temp)

        # Thermal percentage
        temp_pct = min(1.0, zone.current_temp / zone.max_temp)

        # Glow intensity for hot zones
        glow_opacity = max(0, (temp_pct - 0.7) * 3)  # Only glow above 70%

        config = self.ZONE_TYPES.get(zone.zone_type, self.ZONE_TYPES["generic"])

        # Warning marker for critical zones
        warning = ""
        if zone.current_temp > zone.max_temp * 0.9:
            warning = f'''
            <text x="{x + w/2}" y="{y - 5}" text-anchor="middle"
                  fill="{colors['danger']}" font-size="14">⚠️</text>
            '''

        return f'''
        <g class="zone" data-id="{zone.id}">
            <!-- Heat glow for hot zones -->
            {f'<rect x="{x-10}" y="{y-10}" width="{w+20}" height="{h+20}" fill="{temp_color}" opacity="{glow_opacity * 0.5}" filter="url(#heatBlur)" rx="8"/>' if glow_opacity > 0 else ''}

            <!-- Zone body -->
            <rect x="{x}" y="{y}" width="{w}" height="{h}"
                  fill="{temp_color}" opacity="0.8" rx="3"
                  stroke="{colors['fg']}" stroke-width="1"/>

            <!-- Temperature bar -->
            <rect x="{x}" y="{y + h - 4}" width="{w * temp_pct}" height="4"
                  fill="{temp_color}" opacity="1"/>

            <!-- Zone name -->
            <text x="{x + w/2}" y="{y + h/2 - 5}" text-anchor="middle"
                  fill="{colors['bg']}" font-family="monospace" font-size="10"
                  font-weight="bold">{zone.name}</text>

            <!-- Temperature -->
            <text x="{x + w/2}" y="{y + h/2 + 10}" text-anchor="middle"
                  fill="{colors['bg']}" font-family="monospace" font-size="12"
                  font-weight="bold">{zone.current_temp:.0f}°C</text>

            {warning}
        </g>
        '''

    def _render_scar(
        self,
        history: ThermalHistory,
        zones: List[ThermalZone],
        margin: float,
        content_width: float,
        content_height: float,
    ) -> str:
        """Render a thermal scar (historical hotspot)."""
        # Find the zone this history belongs to
        zone = next((z for z in zones if z.id == history.zone_id), None)
        if not zone:
            return ""

        x = margin + zone.position[0] * content_width + zone.size[0] * content_width / 2
        y = margin + 20 + zone.position[1] * content_height + zone.size[1] * content_height / 2
        radius = 20 + 30 * history.severity

        # Scar color (faded version of critical color)
        scar_color = "#ff4400"
        opacity = 0.1 + 0.2 * history.severity

        return f'''
        <circle cx="{x}" cy="{y}" r="{radius}"
                fill="{scar_color}" opacity="{opacity}"
                filter="url(#scarBlur)"/>
        '''

    def _render_legend(self, margin: float) -> str:
        """Render the temperature legend."""
        colors = self.colors
        legend_y = self.height - 35
        legend_width = 200
        legend_x = self.width - margin - legend_width

        return f'''
        <g class="legend">
            <text x="{legend_x - 10}" y="{legend_y + 10}" text-anchor="end"
                  fill="{colors['muted']}" font-family="monospace" font-size="10">
                0°C</text>
            <rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="15"
                  fill="url(#tempGradient)" rx="2"/>
            <text x="{legend_x + legend_width + 10}" y="{legend_y + 10}"
                  fill="{colors['muted']}" font-family="monospace" font-size="10">
                105°C+</text>
        </g>
        '''

    def _render_stats(
        self,
        zones: List[ThermalZone],
        margin: float,
    ) -> str:
        """Render summary statistics."""
        colors = self.colors
        stats_y = self.height - 35

        if not zones:
            return ""

        max_temp = max(z.current_temp for z in zones)
        avg_temp = sum(z.current_temp for z in zones) / len(zones)
        hottest = max(zones, key=lambda z: z.current_temp)

        return f'''
        <g class="stats">
            <text x="{margin}" y="{stats_y + 10}"
                  fill="{colors['muted']}" font-family="monospace" font-size="10">
                Max: <tspan fill="{self._temp_to_color(max_temp)}">{max_temp:.0f}°C</tspan> ({hottest.name})
                | Avg: <tspan fill="{self._temp_to_color(avg_temp)}">{avg_temp:.0f}°C</tspan>
                | Zones: {len(zones)}
            </text>
        </g>
        '''


# =============================================================================
# Convenience Function
# =============================================================================

def generate_heatmap(
    data: Dict[str, Any],
    theme: str = "ara",
    width: int = 900,
    height: int = 600,
) -> GiftArtifact:
    """
    Generate a thermal heatmap visualization.

    Args:
        data: Thermal data (zones, history)
        theme: Color theme
        width: SVG width
        height: SVG height

    Returns:
        GiftArtifact containing the SVG
    """
    gen = HeatmapGenerator(theme=theme, width=width, height=height)
    return gen.generate(data)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ThermalZone',
    'ThermalHistory',
    'HeatmapGenerator',
    'generate_heatmap',
]
