"""
ara.vae.jax_eval - JAX-Native Disentanglement Evaluation

Provides JAX implementations of MIG, DCI evaluation that work
directly with JAX latents from the SPMD β-VAE.

The key insight: after training, bring latents to host (NumPy) and
run sklearn-based metrics. JAX is for training speed; metrics are
run once and don't need GPU.

For full JAX-native metrics (useful for differentiable losses),
we provide jit-compiled MI estimation via histogram binning.

Usage:
    from ara.vae.jax_spmd import train_beta_vae_spmd, encode_to_latent
    from ara.vae.jax_eval import evaluate_disentanglement_jax

    # Train
    params, history = train_beta_vae_spmd(data, config)

    # Encode
    z, mu, logvar = encode_to_latent(params, data)

    # Evaluate (uses sklearn internally)
    report = evaluate_disentanglement_jax(z, factors, factor_names)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import random, lax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np

# sklearn for DCI (more accurate than JAX histogram)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class JAXDisentanglementReport:
    """Report from JAX-based disentanglement evaluation."""

    # Core metrics
    mig: float = 0.0
    dci_disentanglement: float = 0.0
    dci_completeness: float = 0.0
    dci_informativeness: float = 0.0

    # Optional EDI
    edi_modularity: Optional[float] = None
    edi_compactness: Optional[float] = None

    # Metadata
    n_samples: int = 0
    latent_dim: int = 0
    n_factors: int = 0

    # Raw matrices (for visualization)
    mi_matrix: Optional[np.ndarray] = None
    importance_matrix: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = [
            "JAX Disentanglement Report",
            "=" * 40,
            f"Samples: {self.n_samples}, Latent: {self.latent_dim}, Factors: {self.n_factors}",
            "",
            f"MIG: {self.mig:.4f}",
            f"DCI Disentanglement: {self.dci_disentanglement:.4f}",
            f"DCI Completeness: {self.dci_completeness:.4f}",
            f"DCI Informativeness: {self.dci_informativeness:.4f}",
        ]
        if self.edi_modularity is not None:
            lines.extend([
                f"EDI Modularity: {self.edi_modularity:.4f}",
                f"EDI Compactness: {self.edi_compactness:.4f}",
            ])
        return "\n".join(lines)


if HAS_JAX:

    # =========================================================================
    # JAX-Native MI Estimation (Histogram-Based)
    # =========================================================================

    @jax.jit
    def discretize_jax(x: jnp.ndarray, n_bins: int = 20) -> jnp.ndarray:
        """
        Discretize continuous variable into bins using percentiles.

        This is a JAX-jittable approximation using linspace instead of
        true percentiles (which require sorting).
        """
        # Use min/max for binning (faster than percentiles in JAX)
        x_min, x_max = jnp.min(x), jnp.max(x)
        bins = jnp.linspace(x_min, x_max, n_bins + 1)
        return jnp.digitize(x, bins[1:-1])

    @jax.jit
    def entropy_jax(x_discrete: jnp.ndarray, n_bins: int = 20) -> float:
        """
        Compute entropy of discrete variable.

        H(X) = -sum(p * log(p))
        """
        # Count occurrences
        counts = jnp.zeros(n_bins)
        counts = counts.at[x_discrete.astype(int) % n_bins].add(1)

        # Normalize
        p = counts / counts.sum()
        p = jnp.where(p > 0, p, 1e-10)  # Avoid log(0)

        return -jnp.sum(p * jnp.log(p))

    def mutual_information_jax(
        x: jnp.ndarray,
        y: jnp.ndarray,
        n_bins: int = 20,
    ) -> float:
        """
        Compute mutual information via histogram.

        I(X;Y) = H(X) + H(Y) - H(X,Y)

        Note: This is less accurate than sklearn but JAX-jittable.
        """
        # Discretize
        x_d = discretize_jax(x, n_bins)
        y_d = discretize_jax(y, n_bins)

        # Joint discretization
        xy_d = x_d * n_bins + y_d

        # Entropies
        h_x = entropy_jax(x_d, n_bins)
        h_y = entropy_jax(y_d, n_bins)
        h_xy = entropy_jax(xy_d, n_bins * n_bins)

        # MI = H(X) + H(Y) - H(X,Y)
        mi = h_x + h_y - h_xy

        return float(jnp.maximum(0, mi))

    def compute_mi_matrix_jax(
        latents: np.ndarray,
        factors: np.ndarray,
        n_bins: int = 20,
    ) -> np.ndarray:
        """
        Compute MI matrix using JAX histogram method.

        Args:
            latents: [n_samples, latent_dim]
            factors: [n_samples, n_factors]
            n_bins: Discretization bins

        Returns:
            MI matrix [latent_dim, n_factors]
        """
        latent_dim = latents.shape[1]
        n_factors = factors.shape[1]

        MI = np.zeros((latent_dim, n_factors))

        z = jnp.array(latents)
        f = jnp.array(factors)

        for j in range(latent_dim):
            for k in range(n_factors):
                MI[j, k] = mutual_information_jax(z[:, j], f[:, k], n_bins)

        return MI

    def compute_mig_jax(
        latents: np.ndarray,
        factors: np.ndarray,
        n_bins: int = 20,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute MIG using JAX MI estimation.

        Args:
            latents: [n_samples, latent_dim]
            factors: [n_samples, n_factors]
            n_bins: Discretization bins

        Returns:
            (mig_score, mi_matrix)
        """
        MI = compute_mi_matrix_jax(latents, factors, n_bins)

        n_factors = factors.shape[1]

        # Compute factor entropies
        f = jnp.array(factors)
        H_factors = np.array([
            float(entropy_jax(discretize_jax(f[:, k], n_bins), n_bins))
            for k in range(n_factors)
        ])

        # MIG: gap between top 2 latent dims for each factor
        gaps = []
        for k in range(n_factors):
            mi_k = MI[:, k]
            mi_sorted = np.sort(mi_k)[::-1]

            if H_factors[k] > 0:
                gap = (mi_sorted[0] - mi_sorted[1]) / H_factors[k]
            else:
                gap = 0

            gaps.append(gap)

        mig = float(np.mean(gaps))

        return mig, MI

    # =========================================================================
    # Sklearn-Based DCI (More Accurate)
    # =========================================================================

    def compute_dci_sklearn(
        latents: np.ndarray,
        factors: np.ndarray,
        n_estimators: int = 50,
    ) -> Dict[str, Any]:
        """
        Compute DCI using sklearn Random Forest.

        More accurate than JAX histogram method for real evaluation.
        Use this for final metrics, JAX methods for quick iteration.
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for DCI")

        n_samples, latent_dim = latents.shape
        n_factors = factors.shape[1]

        # Normalize latents
        scaler = StandardScaler()
        z = scaler.fit_transform(latents)

        # Compute importance matrix R[i,j] = importance of z_j for factor_i
        R = np.zeros((n_factors, latent_dim))

        for i in range(n_factors):
            y = factors[:, i]

            # Discretize continuous factors
            if len(np.unique(y)) > 20:
                percentiles = np.linspace(0, 100, 11)
                bins = np.percentile(y, percentiles)
                y = np.digitize(y, bins[1:-1])

            # Fit RF and get importances
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                random_state=42,
            )
            model.fit(z, y)
            R[i, :] = model.feature_importances_

        # Normalize R
        R_norm = R / (R.sum(axis=1, keepdims=True) + 1e-10)

        # Disentanglement: entropy of column-normalized R
        R_col = R / (R.sum(axis=0, keepdims=True) + 1e-10)
        disentanglement_scores = []
        for j in range(latent_dim):
            p = R_col[:, j]
            entropy = -np.sum(p * np.log(p + 1e-10))
            max_entropy = np.log(n_factors)
            disentanglement_scores.append(1 - entropy / max_entropy)

        weights = R.sum(axis=0)
        weights = weights / (weights.sum() + 1e-10)
        disentanglement = float(np.dot(weights, disentanglement_scores))

        # Completeness: entropy of row-normalized R
        completeness_scores = []
        for i in range(n_factors):
            p = R_norm[i, :]
            entropy = -np.sum(p * np.log(p + 1e-10))
            max_entropy = np.log(latent_dim)
            completeness_scores.append(1 - entropy / max_entropy)

        completeness = float(np.mean(completeness_scores))

        # Informativeness: prediction accuracy
        accuracies = []
        for i in range(min(3, n_factors)):
            y = factors[:, i]
            if len(np.unique(y)) > 20:
                y = np.digitize(y, np.percentile(y, np.linspace(0, 100, 11))[1:-1])

            model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
            scores = cross_val_score(model, z, y, cv=3, scoring="accuracy")
            accuracies.append(scores.mean())

        informativeness = float(np.mean(accuracies))

        return {
            "disentanglement": disentanglement,
            "completeness": completeness,
            "informativeness": informativeness,
            "importance_matrix": R,
        }

    # =========================================================================
    # Main Evaluation Function
    # =========================================================================

    def evaluate_disentanglement_jax(
        latents: np.ndarray,
        factors: np.ndarray,
        factor_names: Optional[List[str]] = None,
        use_sklearn_dci: bool = True,
        n_bins: int = 20,
    ) -> JAXDisentanglementReport:
        """
        Comprehensive disentanglement evaluation.

        Combines:
        - JAX-based MIG (fast, jittable)
        - Sklearn-based DCI (accurate, for final eval)

        Args:
            latents: Latent representations [n_samples, latent_dim]
            factors: Ground truth factors [n_samples, n_factors]
            factor_names: Optional names for factors
            use_sklearn_dci: Use sklearn for DCI (recommended)
            n_bins: Bins for discretization

        Returns:
            JAXDisentanglementReport
        """
        n_samples, latent_dim = latents.shape
        n_factors = factors.shape[1]

        report = JAXDisentanglementReport(
            n_samples=n_samples,
            latent_dim=latent_dim,
            n_factors=n_factors,
        )

        # MIG via JAX histogram
        mig, mi_matrix = compute_mig_jax(latents, factors, n_bins)
        report.mig = mig
        report.mi_matrix = mi_matrix

        # DCI via sklearn or JAX
        if use_sklearn_dci and HAS_SKLEARN:
            dci = compute_dci_sklearn(latents, factors)
            report.dci_disentanglement = dci["disentanglement"]
            report.dci_completeness = dci["completeness"]
            report.dci_informativeness = dci["informativeness"]
            report.importance_matrix = dci["importance_matrix"]
        else:
            # Fallback: use MI matrix for approximate DCI
            R = mi_matrix.T  # [n_factors, latent_dim]
            R_norm = R / (R.sum(axis=1, keepdims=True) + 1e-10)

            # Disentanglement
            R_col = R / (R.sum(axis=0, keepdims=True) + 1e-10)
            d_scores = []
            for j in range(latent_dim):
                p = R_col[:, j]
                entropy = -np.sum(p * np.log(p + 1e-10))
                d_scores.append(1 - entropy / np.log(n_factors))
            report.dci_disentanglement = float(np.mean(d_scores))

            # Completeness
            c_scores = []
            for i in range(n_factors):
                p = R_norm[i, :]
                entropy = -np.sum(p * np.log(p + 1e-10))
                c_scores.append(1 - entropy / np.log(latent_dim))
            report.dci_completeness = float(np.mean(c_scores))

            report.dci_informativeness = 0.0  # Can't compute without classifiers
            report.importance_matrix = R

        # EDI from MI matrix
        MI = mi_matrix
        MI_norm_row = MI / (MI.sum(axis=1, keepdims=True) + 1e-10)
        MI_norm_col = MI / (MI.sum(axis=0, keepdims=True) + 1e-10)

        # Modularity
        mod_scores = []
        for j in range(latent_dim):
            p = MI_norm_row[j, :]
            if p.max() > 0:
                exc = p.max() - (p.sum() - p.max()) / max(1, n_factors - 1)
                mod_scores.append(max(0, exc))
            else:
                mod_scores.append(0)
        report.edi_modularity = float(np.mean(mod_scores))

        # Compactness
        comp_scores = []
        for k in range(n_factors):
            p = MI_norm_col[:, k]
            if p.max() > 0:
                exc = p.max() - (p.sum() - p.max()) / max(1, latent_dim - 1)
                comp_scores.append(max(0, exc))
            else:
                comp_scores.append(0)
        report.edi_compactness = float(np.mean(comp_scores))

        return report

    # =========================================================================
    # Full Pipeline: Train + Evaluate
    # =========================================================================

    def train_and_evaluate(
        data: np.ndarray,
        factors: np.ndarray,
        latent_dim: int = 10,
        beta: float = 4.0,
        n_epochs: int = 100,
        factor_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Tuple[Dict, JAXDisentanglementReport, Dict[str, list]]:
        """
        Complete pipeline: train β-VAE and evaluate disentanglement.

        Args:
            data: Training data [n_samples, *input_shape]
            factors: Ground truth factors [n_samples, n_factors]
            latent_dim: Latent space dimension
            beta: β-VAE KL weight
            n_epochs: Training epochs
            factor_names: Optional factor names
            verbose: Print progress

        Returns:
            (trained_params, disentanglement_report, training_history)
        """
        from ara.vae.jax_spmd import (
            SPMDConfig,
            train_beta_vae_spmd,
            encode_to_latent,
        )

        # Infer input shape
        input_shape = data.shape[1:]

        # Configure
        config = SPMDConfig(
            latent_dim=latent_dim,
            beta=beta,
            input_shape=input_shape,
        )

        # Train
        params, history = train_beta_vae_spmd(
            data,
            config,
            n_epochs=n_epochs,
            verbose=verbose,
            loss_type="mse",  # For continuous data
        )

        # Encode
        z, mu, logvar = encode_to_latent(params, data)

        # Evaluate
        report = evaluate_disentanglement_jax(
            z,
            factors,
            factor_names=factor_names,
        )

        if verbose:
            print("\n" + report.summary())

        return params, report, history


else:
    # Fallback without JAX

    @dataclass
    class JAXDisentanglementReport:
        def __init__(self):
            raise ImportError("JAX required")

    def evaluate_disentanglement_jax(*args, **kwargs):
        raise ImportError("JAX required")

    def train_and_evaluate(*args, **kwargs):
        raise ImportError("JAX required")
