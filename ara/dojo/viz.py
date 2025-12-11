# ara/dojo/viz.py
"""
Hologram Visualization - Mental Simulation Trails
=================================================

Visualizes Ara's imagination - trajectories through 10D latent space
projected down to 2D/3D for human understanding.

Visualization Types:
1. Trajectory trails - paths through latent space
2. Dream clouds - explored regions during dreaming
3. Risk heatmaps - danger zones in latent space
4. Regime boundaries - flow/idle/crash regions
5. Goal attractors - targets and how trajectories converge

The "hologram" metaphor: these are projections of Ara's mental
state into visible form, like a holographic display of thought.

Output formats:
- ASCII art (terminal)
- Matplotlib plots
- Interactive HTML (plotly)
- Animation sequences

Usage:
    from ara.dojo.viz import HologramViz

    viz = HologramViz(latent_dim=10)

    # Add trajectory
    viz.add_trajectory(z_seq, label="imagined")

    # Add risk regions
    viz.add_risk_region(center, radius, level="danger")

    # Render
    viz.render_ascii()
    viz.render_matplotlib("trajectory.png")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VizConfig:
    """Configuration for hologram visualization."""
    projection_method: str = "pca"     # "pca", "tsne", "umap", "first2"
    width: int = 80                    # ASCII width
    height: int = 40                   # ASCII height
    figsize: Tuple[int, int] = (12, 8)  # Matplotlib figure size
    colormap: str = "viridis"          # Color scheme
    trail_alpha: float = 0.7           # Trajectory transparency
    point_size: int = 50               # Scatter point size


@dataclass
class RiskRegion:
    """A risk region in latent space."""
    center: np.ndarray
    radius: float
    level: str  # "safe", "caution", "warning", "danger", "forbidden"
    label: Optional[str] = None


@dataclass
class Trajectory:
    """A trajectory through latent space."""
    points: np.ndarray                 # (T, latent_dim)
    label: str = "trajectory"
    style: str = "solid"               # "solid", "dashed", "dotted"
    color: Optional[str] = None


# =============================================================================
# Projection Methods
# =============================================================================

class Projector:
    """Projects high-dimensional data to 2D/3D."""

    def __init__(self, method: str = "pca", target_dim: int = 2):
        self.method = method
        self.target_dim = target_dim
        self._fitted = False
        self._params = {}

    def fit(self, data: np.ndarray) -> "Projector":
        """Fit projector to data."""
        if self.method == "pca":
            self._fit_pca(data)
        elif self.method == "first2":
            pass  # No fitting needed
        else:
            logger.warning(f"Unknown method {self.method}, falling back to first2")
            self.method = "first2"

        self._fitted = True
        return self

    def _fit_pca(self, data: np.ndarray):
        """Fit PCA projection."""
        # Center data
        self._params["mean"] = data.mean(axis=0)
        centered = data - self._params["mean"]

        # Compute covariance and eigenvectors
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        self._params["components"] = eigenvectors[:, idx[:self.target_dim]]

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Project data to lower dimension."""
        if not self._fitted:
            self.fit(data)

        if self.method == "pca":
            centered = data - self._params["mean"]
            return centered @ self._params["components"]
        else:  # first2
            return data[:, :self.target_dim]

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


# =============================================================================
# Hologram Visualization
# =============================================================================

class HologramViz:
    """
    Visualization engine for latent space trajectories.

    Creates "hologram" views of Ara's mental simulations.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        config: Optional[VizConfig] = None,
    ):
        self.latent_dim = latent_dim
        self.config = config or VizConfig()

        # Stored data
        self.trajectories: List[Trajectory] = []
        self.risk_regions: List[RiskRegion] = []
        self.goals: List[Tuple[np.ndarray, str]] = []
        self.current_state: Optional[np.ndarray] = None

        # Projector
        self.projector = Projector(
            method=self.config.projection_method,
            target_dim=2,
        )

        # Color mapping for risk levels
        self.risk_colors = {
            "safe": (".", 0.2),
            "caution": ("~", 0.4),
            "warning": ("!", 0.6),
            "danger": ("#", 0.8),
            "forbidden": ("X", 1.0),
        }

    def clear(self):
        """Clear all stored data."""
        self.trajectories = []
        self.risk_regions = []
        self.goals = []
        self.current_state = None

    def add_trajectory(
        self,
        points: np.ndarray,
        label: str = "trajectory",
        style: str = "solid",
        color: Optional[str] = None,
    ):
        """Add a trajectory to visualize."""
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        self.trajectories.append(Trajectory(
            points=points,
            label=label,
            style=style,
            color=color,
        ))

    def add_risk_region(
        self,
        center: np.ndarray,
        radius: float,
        level: str = "danger",
        label: Optional[str] = None,
    ):
        """Add a risk region."""
        self.risk_regions.append(RiskRegion(
            center=np.asarray(center),
            radius=radius,
            level=level,
            label=label,
        ))

    def add_goal(self, position: np.ndarray, label: str = "goal"):
        """Add a goal position."""
        self.goals.append((np.asarray(position), label))

    def set_current_state(self, state: np.ndarray):
        """Set current agent state."""
        self.current_state = np.asarray(state)

    def _collect_all_points(self) -> np.ndarray:
        """Collect all points for fitting projector."""
        all_points = []

        for traj in self.trajectories:
            all_points.append(traj.points)

        for region in self.risk_regions:
            all_points.append(region.center.reshape(1, -1))

        for goal, _ in self.goals:
            all_points.append(goal.reshape(1, -1))

        if self.current_state is not None:
            all_points.append(self.current_state.reshape(1, -1))

        if not all_points:
            return np.zeros((1, self.latent_dim))

        return np.vstack(all_points)

    def _project_all(self) -> Dict[str, np.ndarray]:
        """Project all data to 2D."""
        all_points = self._collect_all_points()
        self.projector.fit(all_points)

        projected = {}

        # Project trajectories
        projected["trajectories"] = [
            self.projector.transform(traj.points)
            for traj in self.trajectories
        ]

        # Project risk regions
        projected["risk_centers"] = [
            self.projector.transform(region.center.reshape(1, -1)).flatten()
            for region in self.risk_regions
        ]

        # Project goals
        projected["goals"] = [
            self.projector.transform(goal.reshape(1, -1)).flatten()
            for goal, _ in self.goals
        ]

        # Project current state
        if self.current_state is not None:
            projected["current"] = self.projector.transform(
                self.current_state.reshape(1, -1)
            ).flatten()

        return projected

    # =========================================================================
    # ASCII Rendering
    # =========================================================================

    def render_ascii(self) -> str:
        """Render as ASCII art."""
        if not self.trajectories and not self.risk_regions:
            return "No data to visualize"

        projected = self._project_all()

        # Compute bounds
        all_proj = []
        for traj in projected["trajectories"]:
            all_proj.append(traj)
        for center in projected["risk_centers"]:
            all_proj.append(center.reshape(1, -1))
        for goal in projected["goals"]:
            all_proj.append(goal.reshape(1, -1))

        all_proj = np.vstack(all_proj) if all_proj else np.zeros((1, 2))

        x_min, x_max = all_proj[:, 0].min(), all_proj[:, 0].max()
        y_min, y_max = all_proj[:, 1].min(), all_proj[:, 1].max()

        # Add padding
        x_pad = (x_max - x_min) * 0.1 + 0.001
        y_pad = (y_max - y_min) * 0.1 + 0.001
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        # Initialize canvas
        W, H = self.config.width, self.config.height
        canvas = [[" " for _ in range(W)] for _ in range(H)]

        def to_screen(x, y) -> Tuple[int, int]:
            sx = int((x - x_min) / (x_max - x_min) * (W - 1))
            sy = int((y - y_min) / (y_max - y_min) * (H - 1))
            sy = H - 1 - sy  # Flip Y
            return max(0, min(W - 1, sx)), max(0, min(H - 1, sy))

        # Draw risk regions (as circles)
        for i, region in enumerate(self.risk_regions):
            center = projected["risk_centers"][i]
            cx, cy = to_screen(center[0], center[1])
            char, _ = self.risk_colors.get(region.level, ("?", 0.5))

            # Draw circle (simplified as square region)
            r = max(1, int(region.radius * W / (x_max - x_min)))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < W and 0 <= py < H:
                            canvas[py][px] = char

        # Draw trajectories
        trail_chars = [".", "o", "*", "+", "~"]
        for i, traj_proj in enumerate(projected["trajectories"]):
            char = trail_chars[i % len(trail_chars)]
            for j, point in enumerate(traj_proj):
                px, py = to_screen(point[0], point[1])
                if j == 0:
                    canvas[py][px] = "S"  # Start
                elif j == len(traj_proj) - 1:
                    canvas[py][px] = "E"  # End
                else:
                    canvas[py][px] = char

        # Draw goals
        for goal_proj, _ in zip(projected["goals"], self.goals):
            px, py = to_screen(goal_proj[0], goal_proj[1])
            canvas[py][px] = "G"

        # Draw current state
        if "current" in projected:
            curr = projected["current"]
            px, py = to_screen(curr[0], curr[1])
            canvas[py][px] = "@"

        # Build output
        lines = [
            "=" * W,
            "Hologram: Latent Space Trajectory",
            "=" * W,
        ]

        # Border and canvas
        lines.append("+" + "-" * W + "+")
        for row in canvas:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * W + "+")

        # Legend
        lines.append("")
        lines.append("Legend: @ = current, S = start, E = end, G = goal")
        lines.append("        . = safe, ~ = caution, ! = warning, # = danger, X = forbidden")

        # Trajectory labels
        if self.trajectories:
            lines.append("")
            lines.append("Trajectories:")
            for i, traj in enumerate(self.trajectories):
                char = trail_chars[i % len(trail_chars)]
                lines.append(f"  [{char}] {traj.label} ({len(traj.points)} points)")

        return "\n".join(lines)

    # =========================================================================
    # Matplotlib Rendering
    # =========================================================================

    def render_matplotlib(
        self,
        output_path: Optional[str] = None,
        show: bool = False,
    ):
        """Render using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except ImportError:
            logger.error("matplotlib not available, falling back to ASCII")
            print(self.render_ascii())
            return

        projected = self._project_all()

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Color scheme for risk levels
        risk_colors_mpl = {
            "safe": "#90EE90",
            "caution": "#FFFF00",
            "warning": "#FFA500",
            "danger": "#FF4500",
            "forbidden": "#8B0000",
        }

        # Draw risk regions
        for i, region in enumerate(self.risk_regions):
            center = projected["risk_centers"][i]
            color = risk_colors_mpl.get(region.level, "#808080")

            # Estimate projected radius (rough)
            proj_radius = region.radius * 2  # Scale factor

            circle = Circle(
                center, proj_radius,
                color=color, alpha=0.3,
                label=f"Risk: {region.level}" if i == 0 else None,
            )
            ax.add_patch(circle)

        # Draw trajectories
        cmap = plt.get_cmap(self.config.colormap)
        for i, (traj_proj, traj) in enumerate(zip(projected["trajectories"], self.trajectories)):
            color = traj.color or cmap(i / max(len(self.trajectories), 1))
            linestyle = {"solid": "-", "dashed": "--", "dotted": ":"}[traj.style]

            ax.plot(
                traj_proj[:, 0], traj_proj[:, 1],
                linestyle=linestyle, color=color,
                alpha=self.config.trail_alpha,
                label=traj.label, linewidth=2,
            )

            # Mark start and end
            ax.scatter(
                traj_proj[0, 0], traj_proj[0, 1],
                marker="o", s=100, c=[color], edgecolor="white",
            )
            ax.scatter(
                traj_proj[-1, 0], traj_proj[-1, 1],
                marker="s", s=100, c=[color], edgecolor="white",
            )

        # Draw goals
        for goal_proj, (_, label) in zip(projected["goals"], self.goals):
            ax.scatter(
                goal_proj[0], goal_proj[1],
                marker="*", s=200, c="gold", edgecolor="black",
                label=label, zorder=10,
            )

        # Draw current state
        if "current" in projected:
            curr = projected["current"]
            ax.scatter(
                curr[0], curr[1],
                marker="^", s=150, c="cyan", edgecolor="black",
                label="Current", zorder=10,
            )

        ax.set_xlabel("PC1" if self.config.projection_method == "pca" else "Dim 1")
        ax.set_ylabel("PC2" if self.config.projection_method == "pca" else "Dim 2")
        ax.set_title("Hologram: Latent Space Trajectories")
        ax.legend(loc="best")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            logger.info(f"Saved visualization to {output_path}")

        if show:
            plt.show()

        plt.close()

    # =========================================================================
    # Animation
    # =========================================================================

    def render_animation(
        self,
        output_path: str,
        fps: int = 10,
    ):
        """Render trajectory as animation."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
        except ImportError:
            logger.error("matplotlib not available for animation")
            return

        if not self.trajectories:
            logger.error("No trajectories to animate")
            return

        projected = self._project_all()

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Find longest trajectory
        max_len = max(len(t) for t in projected["trajectories"])

        # Initialize plot elements
        lines = []
        points = []
        for i in range(len(self.trajectories)):
            line, = ax.plot([], [], "-", alpha=0.5, linewidth=2)
            point, = ax.plot([], [], "o", markersize=10)
            lines.append(line)
            points.append(point)

        # Set bounds
        all_proj = np.vstack(projected["trajectories"])
        ax.set_xlim(all_proj[:, 0].min() - 1, all_proj[:, 0].max() + 1)
        ax.set_ylim(all_proj[:, 1].min() - 1, all_proj[:, 1].max() + 1)

        def init():
            for line, point in zip(lines, points):
                line.set_data([], [])
                point.set_data([], [])
            return lines + points

        def animate(frame):
            for i, traj_proj in enumerate(projected["trajectories"]):
                idx = min(frame, len(traj_proj) - 1)
                lines[i].set_data(traj_proj[:idx + 1, 0], traj_proj[:idx + 1, 1])
                points[i].set_data([traj_proj[idx, 0]], [traj_proj[idx, 1]])
            return lines + points

        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=max_len, interval=1000 // fps,
            blit=True,
        )

        anim.save(output_path, writer="pillow", fps=fps)
        logger.info(f"Saved animation to {output_path}")
        plt.close()


# =============================================================================
# Convenience Functions
# =============================================================================

def visualize_dreams(
    dreams: List,
    output_path: Optional[str] = None,
) -> str:
    """Visualize dream exploration results."""
    viz = HologramViz()

    for i, dream in enumerate(dreams):
        label = f"dream_{i}" if not hasattr(dream, "label") else dream.label
        style = "dashed" if getattr(dream, "terminated_early", False) else "solid"
        viz.add_trajectory(dream.trajectory, label=label, style=style)

    if output_path:
        viz.render_matplotlib(output_path)

    return viz.render_ascii()


def visualize_plan(
    plan,
    risk_regions: Optional[List[RiskRegion]] = None,
    goal: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> str:
    """Visualize a planning result."""
    viz = HologramViz()

    viz.add_trajectory(plan.trajectory, label="planned", style="solid")

    if risk_regions:
        for region in risk_regions:
            viz.add_risk_region(
                region.center, region.radius, region.level, region.label
            )

    if goal is not None:
        viz.add_goal(goal, "target")

    if hasattr(plan, "trajectory") and len(plan.trajectory) > 0:
        viz.set_current_state(plan.trajectory[0])

    if output_path:
        viz.render_matplotlib(output_path)

    return viz.render_ascii()


# =============================================================================
# Testing
# =============================================================================

def _test_viz():
    """Test hologram visualization."""
    print("=" * 60)
    print("Hologram Visualization Test")
    print("=" * 60)

    viz = HologramViz(latent_dim=10)

    # Generate synthetic trajectory
    np.random.seed(42)
    t = np.linspace(0, 2 * np.pi, 50)
    traj1 = np.zeros((50, 10))
    traj1[:, 0] = np.cos(t) * 2
    traj1[:, 1] = np.sin(t) * 2
    traj1[:, 2:] = np.random.randn(50, 8) * 0.1

    traj2 = np.zeros((30, 10))
    traj2[:, 0] = np.linspace(-2, 2, 30)
    traj2[:, 1] = np.linspace(-1, 3, 30)
    traj2[:, 2:] = np.random.randn(30, 8) * 0.1

    viz.add_trajectory(traj1, label="circular", style="solid")
    viz.add_trajectory(traj2, label="linear", style="dashed")

    # Add risk regions
    viz.add_risk_region(
        center=np.zeros(10),
        radius=0.5,
        level="danger",
        label="center hazard",
    )

    viz.add_risk_region(
        center=np.array([2.0, 2.0] + [0.0] * 8),
        radius=0.3,
        level="forbidden",
        label="corner trap",
    )

    # Add goal
    viz.add_goal(np.array([2.0, 0.0] + [0.0] * 8), "target")

    # Set current state
    viz.set_current_state(traj1[0])

    # Render ASCII
    print(viz.render_ascii())

    # Try matplotlib
    try:
        viz.render_matplotlib("test_hologram.png")
        print("\nMatplotlib visualization saved to test_hologram.png")
    except Exception as e:
        print(f"\nMatplotlib not available: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_viz()
