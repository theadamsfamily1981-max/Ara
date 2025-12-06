"""
Memory Layout Constellation - Stellar visualization of memory topology.

Shows memory regions as stars in a constellation, with:
- Size proportional to region size
- Brightness proportional to access frequency
- Connections showing data flow between regions
- Orbital rings for hierarchy (stack, heap, mapped, etc.)

The metaphor: Your memory is a galaxy. Where are the hot stars?
Where are the dark matter regions you forgot about?
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
class MemoryRegion:
    """A region of memory to visualize."""
    id: str
    name: str
    size_bytes: int
    access_frequency: float = 0.5    # 0-1, how often accessed
    category: str = "heap"           # stack, heap, mapped, code, etc.
    address_start: int = 0
    parent_id: Optional[str] = None  # For hierarchical regions
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryFlow:
    """Data flow between memory regions."""
    source_id: str
    target_id: str
    bandwidth: float = 0.5           # 0-1, relative bandwidth
    latency: float = 0.5             # 0-1, relative latency (0=fast)


@register_generator
class ConstellationGenerator(GiftGenerator):
    """
    Generates SVG constellation visualizations of memory topology.

    Memory regions become stars:
    - Size = memory size (log scale)
    - Brightness = access frequency
    - Color = category (stack=blue, heap=green, mapped=purple)
    - Position = arranged in orbital rings by category

    Connections become stellar winds/plasma streams.
    """

    name = "constellation"
    description = "Memory layout as stellar constellation"
    artifact_type = ArtifactType.SVG

    # Category colors and orbital radii
    CATEGORY_CONFIG = {
        "stack": {"color_key": "accent", "orbit": 0.2, "label": "Stack"},
        "heap": {"color_key": "accent3", "orbit": 0.5, "label": "Heap"},
        "mapped": {"color_key": "accent2", "orbit": 0.7, "label": "Mapped"},
        "code": {"color_key": "warning", "orbit": 0.35, "label": "Code"},
        "shared": {"color_key": "muted", "orbit": 0.85, "label": "Shared"},
        "device": {"color_key": "danger", "orbit": 0.95, "label": "Device"},
    }

    def __init__(self, theme: str = "ara", width: int = 800, height: int = 800):
        super().__init__(theme)
        self.width = width
        self.height = height
        self.center_x = width / 2
        self.center_y = height / 2
        self.max_radius = min(width, height) * 0.45

    def generate(self, data: Dict[str, Any]) -> GiftArtifact:
        """
        Generate constellation SVG from memory data.

        Expected data format:
        {
            "regions": [
                {"id": "r1", "name": "main_heap", "size_bytes": 1048576,
                 "access_frequency": 0.8, "category": "heap"},
                ...
            ],
            "flows": [
                {"source_id": "r1", "target_id": "r2", "bandwidth": 0.6},
                ...
            ],
            "title": "Process Memory Map",  # optional
        }
        """
        regions = self._parse_regions(data.get("regions", []))
        flows = self._parse_flows(data.get("flows", []))
        title = data.get("title", "Memory Constellation")

        # Calculate positions for each region
        positions = self._layout_regions(regions)

        # Generate SVG
        svg = self._render_svg(regions, flows, positions, title)

        return GiftArtifact(
            id=self._make_id("constellation"),
            name=title,
            artifact_type=ArtifactType.SVG,
            content=svg,
            description=(
                f"Stellar map of {len(regions)} memory regions. "
                f"Brighter stars = higher access frequency. "
                f"Size = memory footprint."
            ),
            generator=self.name,
            source_data={"region_count": len(regions), "flow_count": len(flows)},
            width=self.width,
            height=self.height,
            theme=self.theme,
        )

    def _parse_regions(self, raw: List[Dict]) -> List[MemoryRegion]:
        """Parse raw region dicts into MemoryRegion objects."""
        regions = []
        for r in raw:
            regions.append(MemoryRegion(
                id=r.get("id", f"r{len(regions)}"),
                name=r.get("name", "unnamed"),
                size_bytes=r.get("size_bytes", 1024),
                access_frequency=r.get("access_frequency", 0.5),
                category=r.get("category", "heap"),
                address_start=r.get("address_start", 0),
                parent_id=r.get("parent_id"),
                properties=r.get("properties", {}),
            ))
        return regions

    def _parse_flows(self, raw: List[Dict]) -> List[MemoryFlow]:
        """Parse raw flow dicts into MemoryFlow objects."""
        flows = []
        for f in raw:
            flows.append(MemoryFlow(
                source_id=f.get("source_id", ""),
                target_id=f.get("target_id", ""),
                bandwidth=f.get("bandwidth", 0.5),
                latency=f.get("latency", 0.5),
            ))
        return flows

    def _layout_regions(
        self,
        regions: List[MemoryRegion],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate (x, y) positions for each region.

        Uses orbital rings based on category, with regions spread
        around each ring.
        """
        positions = {}

        # Group by category
        by_category: Dict[str, List[MemoryRegion]] = {}
        for r in regions:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)

        # Layout each category on its orbital ring
        for category, cat_regions in by_category.items():
            config = self.CATEGORY_CONFIG.get(
                category,
                {"orbit": 0.6}
            )
            orbit_radius = config["orbit"] * self.max_radius

            # Spread regions around the ring
            n = len(cat_regions)
            for i, region in enumerate(cat_regions):
                # Angle with some jitter for visual interest
                base_angle = (2 * math.pi * i / n) if n > 0 else 0
                jitter = (hash(region.id) % 100 - 50) / 500.0
                angle = base_angle + jitter

                # Slight radius variation
                r_jitter = (hash(region.id + "r") % 100 - 50) / 500.0
                r = orbit_radius * (1 + r_jitter)

                x = self.center_x + r * math.cos(angle)
                y = self.center_y + r * math.sin(angle)
                positions[region.id] = (x, y)

        return positions

    def _render_svg(
        self,
        regions: List[MemoryRegion],
        flows: List[MemoryFlow],
        positions: Dict[str, Tuple[float, float]],
        title: str,
    ) -> str:
        """Render the complete SVG."""
        colors = self.colors

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}">',

            # Styles
            "<defs>",
            self._render_gradients(),
            self._render_filters(),
            "</defs>",

            # Background
            f'<rect width="100%" height="100%" fill="{colors["bg"]}"/>',

            # Stars background (decorative)
            self._render_background_stars(),

            # Orbital rings (visual guide)
            self._render_orbital_rings(),

            # Flows (connections between regions)
            '<g class="flows">',
            *[self._render_flow(f, positions) for f in flows],
            '</g>',

            # Regions (stars)
            '<g class="regions">',
            *[self._render_region(r, positions) for r in regions],
            '</g>',

            # Title
            f'<text x="{self.width/2}" y="30" '
            f'text-anchor="middle" fill="{colors["fg"]}" '
            f'font-family="monospace" font-size="16" font-weight="bold">'
            f'{title}</text>',

            # Legend
            self._render_legend(),

            "</svg>"
        ]

        return "\n".join(parts)

    def _render_gradients(self) -> str:
        """Render gradient definitions for glowing effects."""
        colors = self.colors
        return f'''
        <radialGradient id="starGlow">
            <stop offset="0%" stop-color="{colors['accent']}" stop-opacity="1"/>
            <stop offset="50%" stop-color="{colors['accent']}" stop-opacity="0.3"/>
            <stop offset="100%" stop-color="{colors['accent']}" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="coreGlow">
            <stop offset="0%" stop-color="#fff" stop-opacity="1"/>
            <stop offset="100%" stop-color="#fff" stop-opacity="0"/>
        </radialGradient>
        '''

    def _render_filters(self) -> str:
        """Render filter definitions for glow effects."""
        return '''
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>
        <filter id="softGlow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="8" result="blur"/>
            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>
        '''

    def _render_background_stars(self) -> str:
        """Render decorative background stars."""
        import random
        random.seed(42)  # Reproducible

        stars = []
        for _ in range(100):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            r = random.uniform(0.3, 1.2)
            opacity = random.uniform(0.3, 0.8)
            stars.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
                f'fill="{self.colors["fg"]}" opacity="{opacity:.2f}"/>'
            )
        return f'<g class="background-stars" opacity="0.5">{"".join(stars)}</g>'

    def _render_orbital_rings(self) -> str:
        """Render faint orbital rings for each category."""
        rings = []
        for category, config in self.CATEGORY_CONFIG.items():
            radius = config["orbit"] * self.max_radius
            rings.append(
                f'<circle cx="{self.center_x}" cy="{self.center_y}" '
                f'r="{radius}" fill="none" stroke="{self.colors["grid"]}" '
                f'stroke-width="0.5" stroke-dasharray="4,8" opacity="0.5"/>'
            )
        return f'<g class="orbital-rings">{"".join(rings)}</g>'

    def _render_region(
        self,
        region: MemoryRegion,
        positions: Dict[str, Tuple[float, float]],
    ) -> str:
        """Render a single memory region as a star."""
        pos = positions.get(region.id)
        if not pos:
            return ""

        x, y = pos

        # Size based on memory size (log scale)
        size_log = math.log2(max(region.size_bytes, 1))
        base_radius = 3 + (size_log - 10) * 1.5  # Scale around 1KB
        radius = max(4, min(25, base_radius))

        # Color based on category
        config = self.CATEGORY_CONFIG.get(
            region.category,
            {"color_key": "accent"}
        )
        color = self.colors[config["color_key"]]

        # Brightness/opacity based on access frequency
        core_opacity = 0.3 + 0.7 * region.access_frequency
        glow_radius = radius * (1.5 + region.access_frequency)

        # Format size for label
        size_label = self._format_size(region.size_bytes)

        return f'''
        <g class="region" data-id="{region.id}">
            <!-- Outer glow -->
            <circle cx="{x}" cy="{y}" r="{glow_radius}"
                    fill="{color}" opacity="{0.15 * region.access_frequency}"
                    filter="url(#softGlow)"/>
            <!-- Star body -->
            <circle cx="{x}" cy="{y}" r="{radius}"
                    fill="{color}" opacity="{core_opacity}"
                    filter="url(#glow)"/>
            <!-- Core highlight -->
            <circle cx="{x}" cy="{y}" r="{radius * 0.3}"
                    fill="white" opacity="{0.5 * region.access_frequency}"/>
            <!-- Label (on hover via CSS or always) -->
            <text x="{x}" y="{y + radius + 12}" text-anchor="middle"
                  fill="{self.colors['fg']}" font-family="monospace"
                  font-size="8" opacity="0.7">{region.name}</text>
            <text x="{x}" y="{y + radius + 22}" text-anchor="middle"
                  fill="{self.colors['muted']}" font-family="monospace"
                  font-size="7">{size_label}</text>
        </g>
        '''

    def _render_flow(
        self,
        flow: MemoryFlow,
        positions: Dict[str, Tuple[float, float]],
    ) -> str:
        """Render a data flow as a curved connection."""
        source_pos = positions.get(flow.source_id)
        target_pos = positions.get(flow.target_id)

        if not source_pos or not target_pos:
            return ""

        x1, y1 = source_pos
        x2, y2 = target_pos

        # Curved path using quadratic bezier
        # Control point offset perpendicular to line
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Offset control point for curve
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            # Perpendicular offset
            offset = length * 0.15
            ctrl_x = mid_x - dy/length * offset
            ctrl_y = mid_y + dx/length * offset
        else:
            ctrl_x, ctrl_y = mid_x, mid_y

        # Width and opacity based on bandwidth
        stroke_width = 0.5 + 2 * flow.bandwidth
        opacity = 0.2 + 0.5 * flow.bandwidth

        # Color: blend based on latency (fast=accent, slow=warning)
        color = self.colors["accent"] if flow.latency < 0.5 else self.colors["warning"]

        return f'''
        <path d="M {x1} {y1} Q {ctrl_x} {ctrl_y} {x2} {y2}"
              fill="none" stroke="{color}" stroke-width="{stroke_width}"
              opacity="{opacity}" stroke-linecap="round"/>
        '''

    def _render_legend(self) -> str:
        """Render a legend for the categories."""
        colors = self.colors
        items = []
        x_start = 20
        y_start = self.height - 20

        for i, (category, config) in enumerate(self.CATEGORY_CONFIG.items()):
            x = x_start + i * 80
            color = colors[config["color_key"]]
            label = config["label"]

            items.append(
                f'<circle cx="{x}" cy="{y_start}" r="4" fill="{color}"/>'
                f'<text x="{x + 8}" y="{y_start + 3}" fill="{colors["muted"]}" '
                f'font-family="monospace" font-size="8">{label}</text>'
            )

        return f'<g class="legend">{"".join(items)}</g>'

    def _format_size(self, bytes: int) -> str:
        """Format bytes as human-readable size."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if abs(bytes) < 1024:
                return f"{bytes:.0f}{unit}"
            bytes /= 1024
        return f"{bytes:.0f}TB"


# =============================================================================
# Convenience Function
# =============================================================================

def generate_constellation(
    data: Dict[str, Any],
    theme: str = "ara",
    width: int = 800,
    height: int = 800,
) -> GiftArtifact:
    """
    Generate a memory constellation visualization.

    Args:
        data: Memory topology data (regions, flows)
        theme: Color theme
        width: SVG width
        height: SVG height

    Returns:
        GiftArtifact containing the SVG
    """
    gen = ConstellationGenerator(theme=theme, width=width, height=height)
    return gen.generate(data)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'MemoryRegion',
    'MemoryFlow',
    'ConstellationGenerator',
    'generate_constellation',
]
