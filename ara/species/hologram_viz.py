#!/usr/bin/env python3
# ara/species/hologram_viz.py
"""
Holographic Visualization for AraSpeciesV3 Explanation Frames

Renders the explainability output from AraSpeciesV3 as:
- Trajectory ribbon with confidence tube
- Uncertainty fog (volumetric particle cloud)
- Ghost paths (rejected/counterfactual trajectories)
- Real-time statistics overlay

Built on VisPy for GPU-accelerated rendering.

Usage:
    python hologram_viz.py  # Standalone demo with synthetic data

Or integrate:
    from ara.species.hologram_viz import HologramScene
    scene = HologramScene(ara_instance)
    scene.run()
"""

import time
import math
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Optional VisPy import
try:
    import vispy
    from vispy import app, scene
    from vispy.scene import visuals
    from vispy.color import ColorArray
    VISPY_AVAILABLE = True
except ImportError:
    VISPY_AVAILABLE = False
    logger.warning("VisPy not available. Install with: pip install vispy")


@dataclass
class HologramConfig:
    """Configuration for hologram visualization."""
    window_size: Tuple[int, int] = (1200, 800)
    background_color: str = '#0a0a12'

    # Trajectory appearance
    trajectory_color: str = '#00ff88'
    trajectory_width: float = 3.0
    tube_color: str = '#00ff8840'  # Semi-transparent
    tube_segments: int = 16

    # Fog appearance
    fog_color: str = '#ff660030'  # Orange, very transparent
    fog_size: float = 8.0
    fog_max_particles: int = 5000

    # Ghost appearance
    ghost_colors: Dict[str, str] = None  # Initialized in __post_init__
    ghost_width: float = 1.5
    ghost_opacity: float = 0.3

    # Grid overlay
    show_grid: bool = True
    grid_color: str = '#ffffff15'
    grid_spacing: float = 20.0

    # Update rate
    update_hz: float = 30.0

    def __post_init__(self):
        if self.ghost_colors is None:
            self.ghost_colors = {
                'collision': '#ff3333',
                'uncertainty': '#ffaa00',
                'bounds': '#aa33ff',
                'ok': '#33ff33',
            }


class HologramScene:
    """
    VisPy-based holographic visualization for AraSpeciesV3.

    Consumes explanation frames and renders:
    - Main trajectory as glowing ribbon
    - Confidence tube as translucent envelope
    - Uncertainty fog as particle cloud
    - Ghost trajectories as faded alternatives
    """

    def __init__(
        self,
        ara_instance: Optional[Any] = None,
        config: Optional[HologramConfig] = None,
        grid_shape: Tuple[int, int] = (500, 500),
    ):
        if not VISPY_AVAILABLE:
            raise ImportError("VisPy is required for hologram visualization")

        self.ara = ara_instance
        self.config = config or HologramConfig()
        self.grid_shape = grid_shape

        # Create canvas and view
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=self.config.window_size,
            bgcolor=self.config.background_color,
            title='Ara Species V3 - Holographic Decision Space',
        )

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range(
            x=(0, grid_shape[1]),
            y=(0, grid_shape[0]),
        )

        # Initialize visual elements
        self._init_visuals()

        # Timer for updates
        self._timer = app.Timer(
            interval=1.0 / self.config.update_hz,
            connect=self._on_timer,
            start=False,
        )

        # Frame counter
        self._frame_count = 0
        self._last_explanation: Optional[Dict[str, Any]] = None

    def _init_visuals(self):
        """Initialize all visual elements."""
        # Background grid
        if self.config.show_grid:
            self._init_grid()

        # Trajectory line
        self.trajectory_line = visuals.Line(
            pos=np.zeros((2, 2)),
            color=self.config.trajectory_color,
            width=self.config.trajectory_width,
            parent=self.view.scene,
        )

        # Confidence tube (as mesh or multiple lines)
        self.tube_upper = visuals.Line(
            pos=np.zeros((2, 2)),
            color=self.config.tube_color,
            width=1.0,
            parent=self.view.scene,
        )
        self.tube_lower = visuals.Line(
            pos=np.zeros((2, 2)),
            color=self.config.tube_color,
            width=1.0,
            parent=self.view.scene,
        )

        # Uncertainty fog (markers)
        self.fog_scatter = visuals.Markers(parent=self.view.scene)

        # Ghost trajectories (multiple lines)
        self.ghost_lines: List[visuals.Line] = []
        for i in range(20):  # Pre-allocate
            line = visuals.Line(
                pos=np.zeros((2, 2)),
                color='#ffffff00',  # Start invisible
                width=self.config.ghost_width,
                parent=self.view.scene,
            )
            self.ghost_lines.append(line)

        # Start/goal markers
        self.start_marker = visuals.Markers(parent=self.view.scene)
        self.goal_marker = visuals.Markers(parent=self.view.scene)

        # Text overlay
        self.text_overlay = visuals.Text(
            text='',
            color='white',
            font_size=12,
            anchor_x='left',
            anchor_y='top',
            parent=self.canvas.scene,
        )
        self.text_overlay.pos = (10, 20)

    def _init_grid(self):
        """Initialize background grid lines."""
        H, W = self.grid_shape
        spacing = self.config.grid_spacing

        # Vertical lines
        x_positions = np.arange(0, W + spacing, spacing)
        for x in x_positions:
            line = visuals.Line(
                pos=np.array([[x, 0], [x, H]]),
                color=self.config.grid_color,
                width=0.5,
                parent=self.view.scene,
            )

        # Horizontal lines
        y_positions = np.arange(0, H + spacing, spacing)
        for y in y_positions:
            line = visuals.Line(
                pos=np.array([[0, y], [W, y]]),
                color=self.config.grid_color,
                width=0.5,
                parent=self.view.scene,
            )

    def update_from_explanation(self, explanation: Dict[str, Any]):
        """Update visuals from an explanation frame."""
        self._last_explanation = explanation
        visuals_data = explanation.get('visuals', {})

        # Update trajectory
        trajectory = visuals_data.get('trajectory', [])
        self._update_trajectory(trajectory)

        # Update confidence tube
        confidence_tube = visuals_data.get('confidence_tube', [])
        self._update_tube(trajectory, confidence_tube)

        # Update fog
        fog_nodes = visuals_data.get('fog_nodes', [])
        self._update_fog(fog_nodes)

        # Update ghosts
        ghosts = visuals_data.get('ghosts', [])
        self._update_ghosts(ghosts)

        # Update text overlay
        self._update_text(explanation)

        self.canvas.update()

    def _update_trajectory(self, trajectory: List[List[float]]):
        """Update main trajectory line."""
        if len(trajectory) < 2:
            self.trajectory_line.visible = False
            return

        # Convert [y, x] to [x, y] for VisPy
        points = np.array([[p[1], p[0]] for p in trajectory], dtype=np.float32)
        self.trajectory_line.set_data(pos=points)
        self.trajectory_line.visible = True

        # Update start marker
        if len(points) > 0:
            self.start_marker.set_data(
                pos=points[:1],
                face_color='#00ff00',
                edge_color='white',
                size=15,
            )

    def _update_tube(self, trajectory: List[List[float]], radii: List[float]):
        """Update confidence tube visualization."""
        if len(trajectory) < 2 or len(radii) < 2:
            self.tube_upper.visible = False
            self.tube_lower.visible = False
            return

        # Compute tube boundaries perpendicular to trajectory
        points = np.array([[p[1], p[0]] for p in trajectory], dtype=np.float32)
        radii = np.array(radii[:len(points)], dtype=np.float32)

        # Simple perpendicular offset (could be improved with proper normals)
        upper_points = []
        lower_points = []

        for i in range(len(points)):
            if i == 0:
                direction = points[1] - points[0]
            elif i == len(points) - 1:
                direction = points[-1] - points[-2]
            else:
                direction = points[i + 1] - points[i - 1]

            # Normalize and get perpendicular
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length
            perp = np.array([-direction[1], direction[0]])

            r = radii[i] if i < len(radii) else radii[-1]
            upper_points.append(points[i] + perp * r)
            lower_points.append(points[i] - perp * r)

        upper_points = np.array(upper_points, dtype=np.float32)
        lower_points = np.array(lower_points, dtype=np.float32)

        self.tube_upper.set_data(pos=upper_points)
        self.tube_lower.set_data(pos=lower_points)
        self.tube_upper.visible = True
        self.tube_lower.visible = True

    def _update_fog(self, fog_nodes: List[List[float]]):
        """Update uncertainty fog particles."""
        if len(fog_nodes) == 0:
            self.fog_scatter.visible = False
            return

        # Limit particles
        if len(fog_nodes) > self.config.fog_max_particles:
            indices = np.random.choice(
                len(fog_nodes),
                self.config.fog_max_particles,
                replace=False
            )
            fog_nodes = [fog_nodes[i] for i in indices]

        # Convert [y, x, density] to positions and colors
        positions = np.array([[f[1], f[0]] for f in fog_nodes], dtype=np.float32)
        densities = np.array([f[2] if len(f) > 2 else 0.5 for f in fog_nodes], dtype=np.float32)

        # Color based on density
        colors = np.zeros((len(fog_nodes), 4), dtype=np.float32)
        colors[:, 0] = 1.0  # Red
        colors[:, 1] = 0.4 * (1 - densities)  # Less green at high density
        colors[:, 2] = 0.0  # Blue
        colors[:, 3] = 0.3 * densities  # Alpha based on density

        self.fog_scatter.set_data(
            pos=positions,
            face_color=colors,
            edge_color=None,
            size=self.config.fog_size * densities + 2,
        )
        self.fog_scatter.visible = True

    def _update_ghosts(self, ghosts: List[Dict[str, Any]]):
        """Update ghost trajectory visualizations."""
        # Hide all ghost lines first
        for line in self.ghost_lines:
            line.visible = False

        for i, ghost in enumerate(ghosts[:len(self.ghost_lines)]):
            states = ghost.get('states', [])
            if len(states) < 2:
                continue

            reason = ghost.get('reason', 'ok')
            color = self.config.ghost_colors.get(reason, '#ffffff')

            # Add alpha for fading
            if len(color) == 7:  # #RRGGBB
                color = color + hex(int(self.config.ghost_opacity * 255))[2:].zfill(2)

            points = np.array([[s[1], s[0]] for s in states], dtype=np.float32)
            self.ghost_lines[i].set_data(pos=points, color=color)
            self.ghost_lines[i].visible = True

    def _update_text(self, explanation: Dict[str, Any]):
        """Update text overlay."""
        meta = explanation.get('meta', {})
        stats = explanation.get('statistics', {})
        summary = explanation.get('text_summary', '')

        planner_stats = stats.get('planner', {})
        wm_stats = stats.get('world_model', {})

        lines = [
            f"Frame: {self._frame_count}",
            f"Mode: {meta.get('mode', 'Unknown')}",
            f"Iterations: {planner_stats.get('iterations', 0)}",
            f"Best Cost: {planner_stats.get('best_cost', 0):.1f}",
            f"WM Updates: {wm_stats.get('updates', 0)}",
            "",
            summary[:80] + ("..." if len(summary) > 80 else ""),
        ]

        self.text_overlay.text = "\n".join(lines)

    def _on_timer(self, event):
        """Timer callback for continuous updates."""
        self._frame_count += 1

        if self.ara is not None:
            # Get fresh explanation from Ara
            explanation = self.ara.explain_decision()
            self.update_from_explanation(explanation)
        elif self._last_explanation is not None:
            # Use last explanation (for demo mode)
            pass

    def run(self):
        """Start the visualization."""
        self._timer.start()
        self.canvas.show()
        app.run()

    def stop(self):
        """Stop the visualization."""
        self._timer.stop()


# ---------------------------------------------------------------------------
# Standalone Demo
# ---------------------------------------------------------------------------

class DemoExplanationGenerator:
    """Generate synthetic explanation frames for demo."""

    def __init__(self, grid_shape: Tuple[int, int] = (200, 200)):
        self.grid_shape = grid_shape
        self.t = 0

    def generate(self) -> Dict[str, Any]:
        """Generate a synthetic explanation frame."""
        H, W = self.grid_shape
        self.t += 1

        # Synthetic trajectory (curved path)
        num_points = 30
        t_vals = np.linspace(0, 1, num_points)
        base_y = H // 2 + 30 * np.sin(2 * np.pi * t_vals + self.t * 0.05)
        base_x = 20 + (W - 40) * t_vals
        trajectory = [[float(base_y[i]), float(base_x[i])] for i in range(num_points)]

        # Confidence tube (varies along path)
        confidence_tube = [2.0 + 3.0 * np.sin(i * 0.3 + self.t * 0.1) ** 2
                          for i in range(num_points)]

        # Uncertainty fog
        num_fog = 500
        fog_y = np.random.randint(0, H, num_fog)
        fog_x = np.random.randint(0, W, num_fog)
        fog_density = np.random.random(num_fog) * 0.5 + 0.5
        fog_nodes = [[float(fog_y[i]), float(fog_x[i]), float(fog_density[i])]
                     for i in range(num_fog)]

        # Ghost trajectories
        ghosts = []
        for _ in range(5):
            offset = np.random.randn() * 20
            ghost_y = base_y + offset + np.random.randn(num_points) * 5
            ghost_x = base_x + np.random.randn(num_points) * 3
            ghosts.append({
                'reason': np.random.choice(['collision', 'uncertainty', 'bounds']),
                'reason_code': np.random.randint(1, 4),
                'cost': np.random.random() * 500,
                'states': [[float(ghost_y[i]), float(ghost_x[i])]
                          for i in range(num_points)],
            })

        return {
            'meta': {
                'timestamp': time.time(),
                'mode': 'Demo_Mode',
                'iteration': self.t,
            },
            'visuals': {
                'trajectory': trajectory,
                'confidence_tube': confidence_tube,
                'fog_nodes': fog_nodes,
                'ghosts': ghosts,
            },
            'statistics': {
                'world_model': {'updates': self.t * 20},
                'planner': {
                    'iterations': self.t,
                    'best_cost': 50 + 20 * np.sin(self.t * 0.1),
                },
            },
            'text_summary': f"Demo frame {self.t}: navigating synthetic uncertainty field",
        }


def demo():
    """Run standalone visualization demo."""
    if not VISPY_AVAILABLE:
        print("VisPy is required. Install with: pip install vispy")
        return

    print("=" * 70)
    print("Hologram Visualization Demo")
    print("=" * 70)
    print("Controls:")
    print("  - Mouse drag: Pan")
    print("  - Scroll: Zoom")
    print("  - Close window to exit")
    print("=" * 70)

    # Create demo generator
    demo_gen = DemoExplanationGenerator(grid_shape=(200, 200))

    # Create scene without Ara instance
    scene = HologramScene(
        ara_instance=None,
        grid_shape=(200, 200),
    )

    # Custom timer that uses demo generator
    def on_demo_timer(event):
        scene._frame_count += 1
        explanation = demo_gen.generate()
        scene.update_from_explanation(explanation)

    scene._timer.disconnect()
    scene._timer.connect(on_demo_timer)

    # Run
    scene.run()


if __name__ == "__main__":
    demo()
