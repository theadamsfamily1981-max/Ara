#!/usr/bin/env python3
"""
phase_diagram.py - GUTC (λ, Π) Control Manifold Visualization Tool

Render the GUTC control manifold with:
  - Theoretical regions (Healthy, ASD-like, Psychotic, Anesthetic, Manic, Depressive)
  - Development / Psychosis / Recovery trajectories
  - Capacity contours C(λ, Π) peaking at λ=5 (proxy for critical λ≈1)
  - Optional subject points from a CSV file

The capacity function follows:
    C(λ, Π) = Π · exp(-(λ - 5)² / 2σ²)

This captures:
  - Peak computational capacity at criticality (λ = 1, mapped to λ = 5)
  - Precision-dependent scaling (higher Π → higher capacity)
  - Gaussian falloff away from criticality

Usage:
    # Basic manifold with capacity contours
    python phase_diagram.py -o lambda_pi_manifold.png

    # With subject overlay from CSV
    python phase_diagram.py \\
        -i subjects.csv \\
        --lambda-col lambda_hat \\
        --pi-col Pi_hat \\
        --group-col group \\
        --label-col id \\
        -o lambda_pi_with_subjects.png

    # Without capacity contours
    python phase_diagram.py --no-capacity -o manifold_simple.png

    # High-resolution PDF for publication
    python phase_diagram.py -o figure1.pdf --dpi 600

Reference:
    See GUTC_Theoretical_Connections.md, Appendix B for theoretical background.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# Capacity Function
# =============================================================================

def capacity_function(lam: np.ndarray, pi: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Compute GUTC capacity C(λ, Π).

    C(λ, Π) = Π · exp(-(λ - 5)² / 2σ²)

    Args:
        lam: Criticality values (λ), scaled so λ=5 represents critical point
        pi: Precision values (Π)
        sigma: Width of the critical ridge (default 1.5)

    Returns:
        Capacity values C(λ, Π)
    """
    return pi * np.exp(-((lam - 5.0) ** 2) / (2.0 * sigma ** 2))


def add_capacity_contours(ax, n_levels: int = 6, sigma: float = 1.5,
                          show_filled: bool = True, colormap: str = 'YlOrRd',
                          use_imshow: bool = False):
    """
    Add contours of the capacity function C(λ, Π) to the given axes.

    Args:
        ax: Matplotlib axes
        n_levels: Number of contour levels
        sigma: Width parameter for the Gaussian ridge
        show_filled: Whether to show filled contours (subtle background)
        colormap: Matplotlib colormap name ('YlOrRd', 'viridis', etc.)
        use_imshow: Use imshow for smooth gradient instead of contourf

    Returns:
        Legend handle for capacity contours
    """
    # Grid in λ (x-axis) and Π (y-axis)
    lam_vals = np.linspace(0, 10, 200)
    pi_vals = np.linspace(0, 8, 160)
    Lambda, Pi_grid = np.meshgrid(lam_vals, pi_vals)

    # Capacity landscape
    Capacity = capacity_function(Lambda, Pi_grid, sigma)

    # Normalize to [0, 1] for clean visualization
    C_min, C_max = Capacity.min(), Capacity.max()
    if C_max <= 0:
        return None

    C_norm = (Capacity - C_min) / (C_max - C_min + 1e-12)

    levels = np.linspace(0.15, 0.95, n_levels)

    # Background: either imshow (smooth) or contourf (stepped)
    if show_filled:
        if use_imshow:
            # Smooth gradient background
            ax.imshow(
                C_norm,
                origin='lower',
                extent=[lam_vals.min(), lam_vals.max(),
                        pi_vals.min(), pi_vals.max()],
                aspect='auto',
                cmap=colormap,
                alpha=0.25,
                zorder=0
            )
        else:
            # Stepped contour fill
            ax.contourf(
                Lambda, Pi_grid, C_norm,
                levels=20,
                cmap=colormap,
                alpha=0.15,
                zorder=0
            )

    # Line contours (on normalized capacity)
    cs = ax.contour(
        Lambda, Pi_grid, C_norm,
        levels=levels,
        colors='darkgoldenrod',
        linestyles='-',
        linewidths=0.7,
        alpha=0.6,
        zorder=1
    )

    # Labels showing normalized capacity
    ax.clabel(cs, inline=True, fontsize=6,
              fmt=lambda v: f'C={v:.2f}')

    # Dummy handle for legend
    cap_handle = plt.Line2D(
        [], [], color='darkgoldenrod', linewidth=0.7, alpha=0.6,
        label=r'Capacity $C(\lambda,\Pi)$'
    )
    return cap_handle


# =============================================================================
# Manifold Plotting
# =============================================================================

# Region definitions: (x, y, width, height, label, color)
REGIONS = [
    (4.5, 2.0, 1.0, 3.0, 'Healthy', 'green'),
    (2.0, 4.5, 2.5, 2.5, 'ASD-like', 'orange'),
    (5.5, 4.5, 2.5, 2.5, 'Psychotic', 'red'),
    (0.0, 0.0, 3.0, 2.0, 'Anesthetic', 'gray'),
    (7.0, 0.0, 3.0, 3.0, 'Manic', 'purple'),
    (2.0, 0.5, 2.0, 2.0, 'Depressive', 'steelblue'),
]

# Trajectory definitions: ((x1, y1), (x2, y2), label, rotation)
TRAJECTORIES = [
    ((1.5, 0.8), (4.8, 3.0), 'Development', 25),
    ((5.0, 3.5), (7.5, 6.0), 'Psychosis', 35),
    ((7.5, 6.0), (6.0, 4.0), 'Recovery', -30),
    ((6.0, 4.0), (5.0, 3.0), None, 0),  # Recovery continuation
]


def plot_lambda_pi_manifold(show_capacity: bool = True,
                            show_glow: bool = True,
                            sigma: float = 1.5,
                            colormap: str = 'YlOrRd',
                            use_imshow: bool = False):
    """
    Create the base (λ, Π) manifold figure with regions + trajectories.

    Args:
        show_capacity: Whether to show capacity contours
        show_glow: Whether to show glow effect on critical line
        sigma: Width parameter for capacity function
        colormap: Matplotlib colormap for capacity background
        use_imshow: Use smooth gradient instead of contour fill

    Returns:
        fig, ax, cap_handle: Figure, axes, and capacity legend handle
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Axis limits and labels
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_xlabel(r'$\lambda$ (Criticality)', fontsize=11)
    ax.set_ylabel(r'$\Pi$ (Precision)', fontsize=11)
    ax.set_title(r'GUTC Control Manifold: $(\lambda, \Pi)$ Phase Space',
                 fontsize=13, fontweight='bold')

    # Critical line λ = 5 (proxy for λ=1)
    if show_glow:
        for width, alpha in [(8, 0.05), (6, 0.1), (4, 0.15), (2, 0.25)]:
            ax.axvline(x=5, lw=width, color='gold', alpha=alpha, zorder=2)
    ax.axvline(x=5, linestyle='--', linewidth=2, color='black',
               label=r'$\lambda = 1$ (critical)', zorder=3)

    # Regions
    for x, y, w, h, label, color in REGIONS:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.02,rounding_size=0.2',
            facecolor=color, alpha=0.35,
            edgecolor=color, linewidth=1.5,
            zorder=4
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            zorder=5
        )

    # Capacity contours
    cap_handle = None
    if show_capacity:
        cap_handle = add_capacity_contours(ax, sigma=sigma, colormap=colormap,
                                           use_imshow=use_imshow)

    # Trajectories
    for (x1, y1), (x2, y2), label, rotation in TRAJECTORIES:
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', linewidth=2.5, color='#2980b9'),
            zorder=6
        )
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.3, label, fontsize=9, rotation=rotation,
                    ha='center', va='bottom', color='#2980b9', fontweight='bold')

    # Axis annotations
    ax.text(2.5, -0.5, 'Subcritical\n$(\\lambda < 1)$',
            ha='center', va='top', fontsize=9, color='#34495e')
    ax.text(5.0, -0.5, 'Critical\n$(\\lambda \\approx 1)$',
            ha='center', va='top', fontsize=9, color='#27ae60', fontweight='bold')
    ax.text(7.5, -0.5, 'Supercritical\n$(\\lambda > 1)$',
            ha='center', va='top', fontsize=9, color='#c0392b')

    # Π annotations
    ax.text(-0.6, 1.5, r'Low $\Pi$',
            ha='center', va='center', fontsize=9, rotation=90)
    ax.text(-0.6, 6.0, r'High $\Pi$',
            ha='center', va='center', fontsize=9, rotation=90)

    return fig, ax, cap_handle


# =============================================================================
# Subject Data Handling
# =============================================================================

def load_subjects(csv_path: str,
                  lambda_col: str = 'lambda',
                  pi_col: str = 'pi',
                  group_col: str = None,
                  label_col: str = None):
    """
    Load subject data from a CSV file.

    Args:
        csv_path: Path to CSV file
        lambda_col: Column name for λ̂ values
        pi_col: Column name for Π̂ values
        group_col: Optional column for group labels
        label_col: Optional column for point labels

    Returns:
        List of (lam, pi, group, label) tuples
    """
    subjects = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lam = float(row[lambda_col])
                pi = float(row[pi_col])
            except (KeyError, ValueError):
                continue
            group = row.get(group_col) if group_col else None
            label = row.get(label_col) if label_col else None
            subjects.append((lam, pi, group, label))
    return subjects


def plot_points_on_manifold(subjects, ax, group_col_present: bool,
                            scatter_size: float = 50.0):
    """
    Overlay subject points on an existing manifold axis.

    Args:
        subjects: List of (lam, pi, group, label) tuples
        ax: Matplotlib axes
        group_col_present: Whether groups are specified
        scatter_size: Size of scatter points

    Returns:
        List of legend handles
    """
    handles = []
    group_to_color = {}

    # Color palette for groups
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12',
              '#1abc9c', '#e67e22', '#95a5a6']

    if group_col_present:
        # Assign colors per group
        groups = sorted({s[2] for s in subjects if s[2] is not None})
        for idx, g in enumerate(groups):
            group_to_color[g] = colors[idx % len(colors)]

        for lam, pi, group, label in subjects:
            color = group_to_color.get(group, 'black')
            ax.scatter(lam, pi, s=scatter_size, color=color, alpha=0.9,
                       edgecolors='white', linewidth=1, zorder=10)
            if label:
                ax.annotate(label, (lam, pi), xytext=(5, 5),
                            textcoords='offset points', fontsize=7,
                            bbox=dict(boxstyle='round,pad=0.2',
                                      fc='white', alpha=0.7))

        # Legend handles
        for g, color in group_to_color.items():
            h = plt.Line2D([], [], marker='o', linestyle='',
                           color=color, markersize=8, label=str(g))
            handles.append(h)
    else:
        # All points same color
        for lam, pi, group, label in subjects:
            ax.scatter(lam, pi, s=scatter_size, color='black', alpha=0.9,
                       edgecolors='white', linewidth=1, zorder=10)
            if label:
                ax.annotate(label, (lam, pi), xytext=(5, 5),
                            textcoords='offset points', fontsize=7,
                            bbox=dict(boxstyle='round,pad=0.2',
                                      fc='white', alpha=0.7))
        if subjects:
            h = plt.Line2D([], [], marker='o', linestyle='',
                           color='black', markersize=8, label='Subjects')
            handles.append(h)

    return handles


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GUTC (λ, Π) Control Manifold Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s -o manifold.png
    %(prog)s -i subjects.csv --lambda-col lambda_hat --pi-col Pi_hat -o fig.pdf
    %(prog)s --no-capacity --no-glow -o simple.png
        """
    )

    # Input/output
    parser.add_argument(
        '-i', '--input', type=str, default=None,
        help='Path to CSV with subject data (optional)'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='lambda_pi_manifold.png',
        help='Output file path (extension determines format)'
    )
    parser.add_argument(
        '--dpi', type=int, default=300,
        help='DPI for saved figure (default: 300)'
    )

    # CSV columns
    parser.add_argument(
        '--lambda-col', type=str, default='lambda',
        help='Column name for λ̂ values (default: lambda)'
    )
    parser.add_argument(
        '--pi-col', type=str, default='pi',
        help='Column name for Π̂ values (default: pi)'
    )
    parser.add_argument(
        '--group-col', type=str, default=None,
        help='Column name for group labels (optional)'
    )
    parser.add_argument(
        '--label-col', type=str, default=None,
        help='Column name for point labels (optional)'
    )

    # Display options
    parser.add_argument(
        '--no-capacity', action='store_true',
        help='Disable capacity contours'
    )
    parser.add_argument(
        '--no-glow', action='store_true',
        help='Disable critical line glow effect'
    )
    parser.add_argument(
        '--sigma', type=float, default=1.5,
        help='Width of capacity ridge (default: 1.5)'
    )
    parser.add_argument(
        '--colormap', type=str, default='YlOrRd',
        choices=['YlOrRd', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'],
        help='Colormap for capacity background (default: YlOrRd)'
    )
    parser.add_argument(
        '--smooth', action='store_true',
        help='Use smooth gradient (imshow) instead of contour fill'
    )
    parser.add_argument(
        '--diagnose', action='store_true',
        help='Print GUTC diagnosis for each subject point'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Display plot interactively'
    )

    args = parser.parse_args()

    # Create manifold
    show_capacity = not args.no_capacity
    show_glow = not args.no_glow
    fig, ax, cap_handle = plot_lambda_pi_manifold(
        show_capacity=show_capacity,
        show_glow=show_glow,
        sigma=args.sigma,
        colormap=args.colormap,
        use_imshow=args.smooth
    )

    legend_handles = []
    if cap_handle is not None:
        legend_handles.append(cap_handle)

    # Load and plot subject data
    subjects = []
    if args.input is not None and Path(args.input).exists():
        subjects = load_subjects(
            args.input,
            lambda_col=args.lambda_col,
            pi_col=args.pi_col,
            group_col=args.group_col,
            label_col=args.label_col
        )
        if subjects:
            group_col_present = args.group_col is not None
            subject_handles = plot_points_on_manifold(
                subjects, ax=ax, group_col_present=group_col_present
            )
            legend_handles.extend(subject_handles)
            print(f"Loaded {len(subjects)} subjects from {args.input}")

    # Diagnostic integration
    if args.diagnose and subjects:
        try:
            from gutc_diagnostic_engine import GUTCDiagnosticEngine
            engine = GUTCDiagnosticEngine(verbose=False)
            print("\n" + "=" * 70)
            print("GUTC DIAGNOSTIC REPORTS")
            print("=" * 70)
            for lam, pi, group, label in subjects:
                diagnosis = engine.diagnose(lam, pi)
                subj_id = label or group or f"({lam:.1f}, {pi:.1f})"
                print(f"\n[Subject: {subj_id}]")
                print(f"  State: {diagnosis.state_name}")
                print(f"  Capacity: C = {diagnosis.capacity:.3f}")
                print(f"  Risk: {diagnosis.risk_level}")
                print(f"  λ-strategy: {diagnosis.repair_vector.lambda_strategy}")
                print(f"  Π-strategy: {diagnosis.repair_vector.pi_strategy}")
        except ImportError:
            print("Warning: gutc_diagnostic_engine.py not found. Skipping diagnosis.")

    # Legend
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8, loc='upper left',
                  title='Legend', framealpha=0.9)

    plt.tight_layout()

    # Save
    output_path = Path(args.output)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    # Show
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
