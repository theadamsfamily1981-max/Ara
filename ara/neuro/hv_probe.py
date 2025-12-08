"""
Hypervector Probe - Visualization and Analysis
===============================================

Visual probe for analyzing hypervectors produced by the Homeostatic LIF network.

Provides:
1. Cluster statistics (cosine similarity within/between classes)
2. 2D visualization via PCA/t-SNE
3. Status HV analysis (fabric health encoding)
4. Concept codebook for human-readable decoding

This probe lets you see exactly what CorrSpike-HDC will see at line rate.
"""

from __future__ import annotations
import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Check for PyTorch availability
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# Probe Configuration
# ============================================================================

@dataclass
class HVProbeConfig:
    """Configuration for the HV probe."""
    hv_dim: int = 1024
    status_hv_dim: int = 64
    anomaly_threshold: float = 0.3
    cluster_threshold: float = 0.7


# ============================================================================
# Cosine Statistics
# ============================================================================

def compute_cosine_stats(
    hv: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute cosine similarity statistics for normal vs anomaly classes.

    Args:
        hv: Hypervectors [N, D]
        labels: Class labels [N] (0=normal, 1=anomaly)

    Returns:
        Dictionary of similarity statistics
    """
    # Normalize
    hv_float = hv.float()
    hv_norm = hv_float / (hv_float.norm(dim=1, keepdim=True) + 1e-8)

    mask_normal = (labels == 0)
    mask_anom = (labels == 1)

    stats = {}

    # Normal intra-class similarity
    if mask_normal.sum() > 1:
        hv_normals = hv_norm[mask_normal]

        # Compute pairwise similarities
        sim_matrix = hv_normals @ hv_normals.t()
        n = sim_matrix.size(0)

        # Mask diagonal
        mask_offdiag = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
        intra_vals = sim_matrix[mask_offdiag]

        stats["normal_intra_mean"] = intra_vals.mean().item()
        stats["normal_intra_std"] = intra_vals.std().item()
        stats["normal_intra_min"] = intra_vals.min().item()
        stats["normal_intra_max"] = intra_vals.max().item()

        # Normal cluster center
        center_norm = hv_normals.mean(0, keepdim=True)
        center_norm = center_norm / (center_norm.norm() + 1e-8)

        # Similarity to center
        sim_to_center = (hv_normals @ center_norm.t()).squeeze()
        stats["normal_center_mean"] = sim_to_center.mean().item()
        stats["normal_center_std"] = sim_to_center.std().item()
    else:
        stats["normal_intra_mean"] = float("nan")
        stats["normal_intra_std"] = float("nan")
        stats["normal_center_mean"] = float("nan")
        stats["normal_center_std"] = float("nan")

    # Anomaly vs normal center
    if mask_normal.sum() > 0 and mask_anom.sum() > 0:
        hv_normals = hv_norm[mask_normal]
        hv_anoms = hv_norm[mask_anom]

        center_norm = hv_normals.mean(0, keepdim=True)
        center_norm = center_norm / (center_norm.norm() + 1e-8)

        sim_anom_to_center = (hv_anoms @ center_norm.t()).squeeze()

        if sim_anom_to_center.dim() == 0:
            sim_anom_to_center = sim_anom_to_center.unsqueeze(0)

        stats["anom_vs_normal_center_mean"] = sim_anom_to_center.mean().item()
        stats["anom_vs_normal_center_std"] = sim_anom_to_center.std().item()
        stats["anom_vs_normal_center_min"] = sim_anom_to_center.min().item()
        stats["anom_vs_normal_center_max"] = sim_anom_to_center.max().item()

        # Separation margin
        stats["separation_margin"] = (
            stats["normal_center_mean"] - stats["anom_vs_normal_center_mean"]
        )
    else:
        stats["anom_vs_normal_center_mean"] = float("nan")
        stats["anom_vs_normal_center_std"] = float("nan")
        stats["separation_margin"] = float("nan")

    return stats


# ============================================================================
# Visualization
# ============================================================================

def pca_plot(
    hv: torch.Tensor,
    labels: torch.Tensor,
    title: str = "HV Space (PCA)",
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[np.ndarray]:
    """
    Create 2D PCA visualization of hypervectors.

    Args:
        hv: Hypervectors [N, D]
        labels: Class labels [N]
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display plot

    Returns:
        2D coordinates [N, 2] if sklearn available, else None
    """
    if not HAS_SKLEARN:
        print("Warning: sklearn not available, skipping PCA")
        return None

    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot")
        return None

    hv_np = hv.cpu().numpy().astype(np.float32)
    labels_np = labels.cpu().numpy().astype(np.int32)

    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(hv_np)

    # Plot
    plt.figure(figsize=(8, 6))

    # Normal class (blue)
    mask_normal = labels_np == 0
    if mask_normal.any():
        plt.scatter(
            coords[mask_normal, 0],
            coords[mask_normal, 1],
            c='tab:blue',
            alpha=0.5,
            s=20,
            label='Normal'
        )

    # Anomaly class (red)
    mask_anom = labels_np == 1
    if mask_anom.any():
        plt.scatter(
            coords[mask_anom, 0],
            coords[mask_anom, 1],
            c='tab:red',
            alpha=0.5,
            s=20,
            label='Anomaly'
        )

    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return coords


def tsne_plot(
    hv: torch.Tensor,
    labels: torch.Tensor,
    title: str = "HV Space (t-SNE)",
    save_path: Optional[str] = None,
    show: bool = True,
    perplexity: int = 30
) -> Optional[np.ndarray]:
    """
    Create 2D t-SNE visualization of hypervectors.

    Note: t-SNE is slower but often shows better cluster separation.
    """
    if not HAS_SKLEARN or not HAS_MATPLOTLIB:
        print("Warning: sklearn/matplotlib not available")
        return None

    hv_np = hv.cpu().numpy().astype(np.float32)
    labels_np = labels.cpu().numpy().astype(np.int32)

    # Adjust perplexity if needed
    n_samples = hv_np.shape[0]
    perplexity = min(perplexity, n_samples // 4)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(hv_np)

    # Plot
    plt.figure(figsize=(8, 6))

    mask_normal = labels_np == 0
    if mask_normal.any():
        plt.scatter(
            coords[mask_normal, 0],
            coords[mask_normal, 1],
            c='tab:blue',
            alpha=0.5,
            s=20,
            label='Normal'
        )

    mask_anom = labels_np == 1
    if mask_anom.any():
        plt.scatter(
            coords[mask_anom, 0],
            coords[mask_anom, 1],
            c='tab:red',
            alpha=0.5,
            s=20,
            label='Anomaly'
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return coords


# ============================================================================
# Status HV Analysis
# ============================================================================

def analyze_status_hv(
    hv_full: torch.Tensor,
    labels: torch.Tensor,
    hv_dim: int,
    status_dim: int
) -> Dict[str, any]:
    """
    Analyze the status HV portion for fabric health encoding.

    Args:
        hv_full: Full hypervector [N, hv_dim + status_dim]
        labels: Class labels [N]
        hv_dim: Main HV dimension
        status_dim: Status HV dimension

    Returns:
        Dictionary with status analysis
    """
    status_hv = hv_full[:, hv_dim:hv_dim + status_dim]

    # Convert to {0, 1} for bit analysis
    status_bits = (status_hv > 0).float()

    results = {
        "status_dim": status_dim,
    }

    for cls in [0, 1]:
        cls_name = "normal" if cls == 0 else "anomaly"
        mask = labels == cls

        if not mask.any():
            continue

        bits_cls = status_bits[mask]

        # Mean bit activation per dimension
        mean_bits = bits_cls.mean(dim=0)

        # Which bits are consistently active/inactive?
        active_dims = (mean_bits > 0.7).sum().item()
        inactive_dims = (mean_bits < 0.3).sum().item()
        variable_dims = status_dim - active_dims - inactive_dims

        results[f"{cls_name}_active_dims"] = active_dims
        results[f"{cls_name}_inactive_dims"] = inactive_dims
        results[f"{cls_name}_variable_dims"] = variable_dims
        results[f"{cls_name}_mean_bits_sample"] = mean_bits[:10].tolist()

    # Discriminative bits (different between classes)
    if (labels == 0).any() and (labels == 1).any():
        mean_normal = status_bits[labels == 0].mean(0)
        mean_anom = status_bits[labels == 1].mean(0)
        bit_diff = (mean_normal - mean_anom).abs()

        # Bits that differ significantly
        discriminative = (bit_diff > 0.3).sum().item()
        results["discriminative_bits"] = discriminative
        results["top_discriminative_dims"] = bit_diff.argsort(descending=True)[:5].tolist()

    return results


# ============================================================================
# Concept Codebook
# ============================================================================

class ConceptCodebook:
    """
    Maps hypervector regions to human-readable concept names.

    Used to decode what CorrSpike-HDC is actually "seeing".
    """

    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        self.concepts: Dict[str, torch.Tensor] = {}
        self.rng = np.random.default_rng(seed)

    def add_concept(self, name: str, hv: Optional[torch.Tensor] = None) -> None:
        """Add a concept to the codebook."""
        if hv is None:
            # Generate random bipolar HV
            hv = torch.from_numpy(
                self.rng.choice([-1, 1], size=self.dim).astype(np.float32)
            )
        self.concepts[name] = hv

    def add_from_samples(
        self,
        name: str,
        hvs: torch.Tensor,
        binarize: bool = True
    ) -> None:
        """
        Add a concept as the centroid of sample HVs.

        Useful for learning concepts from training data.
        """
        centroid = hvs.float().mean(0)
        if binarize:
            centroid = torch.sign(centroid)
            centroid = torch.where(centroid == 0, torch.ones_like(centroid), centroid)
        self.concepts[name] = centroid

    def decode(
        self,
        hv: torch.Tensor,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Decode a hypervector to concept names with similarities.

        Args:
            hv: Input hypervector [D] or [1, D]
            top_k: Number of top matches to return

        Returns:
            List of (concept_name, similarity) tuples
        """
        if hv.dim() == 2:
            hv = hv.squeeze(0)

        hv_norm = hv.float() / (hv.float().norm() + 1e-8)

        results = []
        for name, concept_hv in self.concepts.items():
            concept_norm = concept_hv.float() / (concept_hv.float().norm() + 1e-8)
            sim = (hv_norm @ concept_norm).item()
            results.append((name, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def decode_batch(
        self,
        hvs: torch.Tensor,
        top_k: int = 1
    ) -> List[str]:
        """Decode a batch of HVs to their top concept."""
        return [self.decode(hv, top_k=top_k)[0][0] for hv in hvs]

    def save(self, path: str) -> None:
        """Save codebook to file."""
        data = {
            name: hv.cpu().numpy().tolist()
            for name, hv in self.concepts.items()
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load codebook from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        for name, vec in data.items():
            self.concepts[name] = torch.tensor(vec)


# ============================================================================
# Full Probe Class
# ============================================================================

class HVProbe:
    """
    Complete hypervector probe for analysis and visualization.

    Usage:
        probe = HVProbe(hv_dim=1024, status_dim=64)
        probe.add_samples(hvs, labels)
        probe.report()
        probe.visualize()
    """

    def __init__(
        self,
        hv_dim: int = 1024,
        status_dim: int = 64
    ):
        self.hv_dim = hv_dim
        self.status_dim = status_dim

        self.hvs: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.codebook = ConceptCodebook(hv_dim)

    def add_samples(
        self,
        hvs: torch.Tensor,
        labels: torch.Tensor
    ) -> None:
        """Add sample HVs and labels for analysis."""
        self.hvs = hvs
        self.labels = labels

    def learn_concepts_from_data(self) -> None:
        """Learn NORMAL and ANOMALY concepts from loaded data."""
        if self.hvs is None or self.labels is None:
            raise ValueError("No samples loaded")

        main_hv = self.hvs[:, :self.hv_dim]

        # Normal concept
        normal_hvs = main_hv[self.labels == 0]
        if len(normal_hvs) > 0:
            self.codebook.add_from_samples("NORMAL_OPERATING", normal_hvs)

        # Anomaly concept
        anom_hvs = main_hv[self.labels == 1]
        if len(anom_hvs) > 0:
            self.codebook.add_from_samples("ANOMALY_DETECTED", anom_hvs)

    def report(self) -> Dict[str, any]:
        """Generate comprehensive analysis report."""
        if self.hvs is None or self.labels is None:
            raise ValueError("No samples loaded")

        main_hv = self.hvs[:, :self.hv_dim]

        report = {
            "n_samples": len(self.hvs),
            "n_normal": (self.labels == 0).sum().item(),
            "n_anomaly": (self.labels == 1).sum().item(),
            "hv_dim": self.hv_dim,
            "status_dim": self.status_dim,
        }

        # Cosine stats
        cosine_stats = compute_cosine_stats(main_hv, self.labels)
        report["cosine_stats"] = cosine_stats

        # Status analysis
        status_analysis = analyze_status_hv(
            self.hvs, self.labels, self.hv_dim, self.status_dim
        )
        report["status_analysis"] = status_analysis

        return report

    def print_report(self) -> None:
        """Print human-readable report."""
        report = self.report()

        print("\n" + "=" * 60)
        print("Hypervector Probe Report")
        print("=" * 60)

        print(f"\nSamples: {report['n_samples']} total")
        print(f"  Normal: {report['n_normal']}")
        print(f"  Anomaly: {report['n_anomaly']}")
        print(f"  Main HV dim: {report['hv_dim']}")
        print(f"  Status HV dim: {report['status_dim']}")

        print("\nCosine Similarity Stats (Main HV):")
        for k, v in report["cosine_stats"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Quality assessment
        margin = report["cosine_stats"].get("separation_margin", 0)
        if margin > 0.4:
            quality = "EXCELLENT - Clear separation"
        elif margin > 0.2:
            quality = "GOOD - Reasonable separation"
        elif margin > 0.0:
            quality = "FAIR - Some overlap"
        else:
            quality = "POOR - Significant overlap"
        print(f"\nCluster Quality: {quality}")

        print("\nStatus HV Analysis:")
        sa = report["status_analysis"]
        if "discriminative_bits" in sa:
            print(f"  Discriminative bits: {sa['discriminative_bits']}/{report['status_dim']}")

        print("\n" + "=" * 60)

    def visualize(
        self,
        method: str = "pca",
        save_dir: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Visualize hypervector space.

        Args:
            method: "pca" or "tsne"
            save_dir: Optional directory to save plots
            show: Whether to display plots
        """
        if self.hvs is None or self.labels is None:
            raise ValueError("No samples loaded")

        main_hv = self.hvs[:, :self.hv_dim]
        status_hv = self.hvs[:, self.hv_dim:self.hv_dim + self.status_dim]

        save_main = None
        save_status = None
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_main = f"{save_dir}/main_hv_{method}.png"
            save_status = f"{save_dir}/status_hv_{method}.png"

        plot_fn = pca_plot if method == "pca" else tsne_plot

        # Main HV visualization
        plot_fn(
            main_hv,
            self.labels,
            title=f"Main Hypervector Space ({method.upper()})",
            save_path=save_main,
            show=show
        )

        # Status HV visualization
        if self.status_dim > 2:
            plot_fn(
                status_hv,
                self.labels,
                title=f"Status Hypervector Space ({method.upper()})",
                save_path=save_status,
                show=show
            )


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate the HV probe."""
    print("=" * 60)
    print("Hypervector Probe Demo")
    print("=" * 60)

    # Create synthetic data
    hv_dim = 512
    status_dim = 32
    n_samples = 200

    # Generate clustered HVs
    rng = np.random.default_rng(42)

    # Normal cluster center
    normal_center = rng.choice([-1, 1], size=hv_dim).astype(np.float32)

    # Generate normals (perturb center slightly)
    n_normal = 150
    normal_hvs = np.tile(normal_center, (n_normal, 1))
    flip_mask = rng.random((n_normal, hv_dim)) < 0.1  # 10% bit flip
    normal_hvs[flip_mask] *= -1

    # Generate anomalies (random)
    n_anom = 50
    anom_hvs = rng.choice([-1, 1], size=(n_anom, hv_dim)).astype(np.float32)

    # Combine
    main_hv = np.vstack([normal_hvs, anom_hvs])
    labels = np.array([0] * n_normal + [1] * n_anom)

    # Random status HV
    status_hv = rng.choice([-1, 1], size=(n_samples, status_dim)).astype(np.float32)

    # Full HV
    hv_full = np.hstack([main_hv, status_hv])

    # Convert to torch
    hv_full_t = torch.from_numpy(hv_full)
    labels_t = torch.from_numpy(labels)

    # Create probe
    probe = HVProbe(hv_dim=hv_dim, status_dim=status_dim)
    probe.add_samples(hv_full_t, labels_t)
    probe.learn_concepts_from_data()

    # Report
    probe.print_report()

    # Decode some samples
    print("\nConcept Decoding (first 5 samples):")
    for i in range(5):
        decoded = probe.codebook.decode(hv_full_t[i, :hv_dim], top_k=2)
        actual = "NORMAL" if labels[i] == 0 else "ANOMALY"
        print(f"  Sample {i} ({actual}): {decoded}")

    # Visualize (if available)
    if HAS_MATPLOTLIB and HAS_SKLEARN:
        print("\nGenerating visualizations...")
        probe.visualize(method="pca", show=False, save_dir="/tmp")
        print("Plots saved to /tmp/")


if __name__ == '__main__':
    demo()
