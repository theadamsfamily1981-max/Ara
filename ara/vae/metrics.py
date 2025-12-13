"""
ara.vae.metrics - Disentanglement Metrics

Implements standard metrics for evaluating latent space disentanglement:
- DCI (Disentanglement, Completeness, Informativeness)
- MIG (Mutual Information Gap)
- SAP (Separated Attribute Predictability)
- EDI (Exclusivity Disentanglement Index)

These metrics quantify:
- Modularity: One latent dim encodes one factor
- Compactness: One factor encoded by few dims
- Explicitness: Factors are easy to read out

For synthetic benchmarks with known ground-truth factors, we compute
full disentanglement scores. For real data (EEG, behavior), we use
DCI-style predictor analysis on available labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Optional imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class DisentanglementReport:
    """Container for disentanglement evaluation results."""

    # DCI scores
    dci_disentanglement: float = 0.0  # Modularity
    dci_completeness: float = 0.0  # Compactness
    dci_informativeness: float = 0.0  # Explicitness

    # MIG score
    mig: float = 0.0

    # SAP score
    sap: float = 0.0

    # EDI scores (if computed)
    edi_modularity: Optional[float] = None
    edi_compactness: Optional[float] = None
    edi_explicitness: Optional[float] = None

    # Per-factor breakdown
    per_factor_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Metadata
    n_samples: int = 0
    n_factors: int = 0
    latent_dim: int = 0

    def summary(self) -> str:
        """Return summary string."""
        lines = [
            "Disentanglement Report",
            "=" * 40,
            f"Samples: {self.n_samples}, Factors: {self.n_factors}, Latent: {self.latent_dim}",
            "",
            "DCI Scores:",
            f"  Disentanglement (modularity): {self.dci_disentanglement:.3f}",
            f"  Completeness (compactness):   {self.dci_completeness:.3f}",
            f"  Informativeness (explicitness): {self.dci_informativeness:.3f}",
            "",
            f"MIG: {self.mig:.3f}",
            f"SAP: {self.sap:.3f}",
        ]

        if self.edi_modularity is not None:
            lines.extend([
                "",
                "EDI Scores:",
                f"  Modularity:   {self.edi_modularity:.3f}",
                f"  Compactness:  {self.edi_compactness:.3f}",
                f"  Explicitness: {self.edi_explicitness:.3f}",
            ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dci_disentanglement": self.dci_disentanglement,
            "dci_completeness": self.dci_completeness,
            "dci_informativeness": self.dci_informativeness,
            "mig": self.mig,
            "sap": self.sap,
            "edi_modularity": self.edi_modularity,
            "edi_compactness": self.edi_compactness,
            "edi_explicitness": self.edi_explicitness,
            "n_samples": self.n_samples,
            "n_factors": self.n_factors,
            "latent_dim": self.latent_dim,
        }


def compute_dci(
    z: np.ndarray,
    factors: np.ndarray,
    predictor: str = "rf",
    n_estimators: int = 100,
) -> Dict[str, float]:
    """
    Compute DCI (Disentanglement, Completeness, Informativeness).

    Uses predictor-based importance to measure how latent dims relate to factors.

    Args:
        z: Latent representations [n_samples, latent_dim]
        factors: Ground truth factors [n_samples, n_factors]
        predictor: "rf" (Random Forest) or "lasso"
        n_estimators: Number of trees for RF

    Returns:
        Dict with disentanglement, completeness, informativeness scores
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required for DCI computation")

    n_samples, latent_dim = z.shape
    n_factors = factors.shape[1]

    # Normalize latents
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    # Compute importance matrix R[i,j] = importance of z_j for factor_i
    R = np.zeros((n_factors, latent_dim))

    for i in range(n_factors):
        y = factors[:, i]

        # Discretize if continuous
        if len(np.unique(y)) > 20:
            y = _discretize(y, n_bins=10)

        if predictor == "rf":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                random_state=42,
            )
            model.fit(z_scaled, y)
            R[i, :] = model.feature_importances_

        elif predictor == "lasso":
            # Use L1-regularized logistic regression
            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=1000,
                random_state=42,
            )
            model.fit(z_scaled, y)
            # Use absolute coefficients as importance
            R[i, :] = np.abs(model.coef_).mean(axis=0)

    # Normalize R
    R = R / (R.sum(axis=1, keepdims=True) + 1e-10)

    # Disentanglement: For each latent dim, how concentrated is its contribution?
    # High if each dim contributes to only one factor
    disentanglement = _compute_disentanglement(R)

    # Completeness: For each factor, how concentrated is it in few dims?
    # High if each factor uses only one dim
    completeness = _compute_completeness(R)

    # Informativeness: Prediction accuracy
    informativeness = _compute_informativeness(z_scaled, factors)

    return {
        "disentanglement": disentanglement,
        "completeness": completeness,
        "informativeness": informativeness,
        "importance_matrix": R,
    }


def _compute_disentanglement(R: np.ndarray) -> float:
    """
    Disentanglement score from importance matrix.

    For each latent dim j, compute 1 - H(R[:,j]) / log(K)
    where K is number of factors.
    """
    n_factors, latent_dim = R.shape

    # Normalize columns
    R_col = R / (R.sum(axis=0, keepdims=True) + 1e-10)

    scores = []
    for j in range(latent_dim):
        p = R_col[:, j]
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(n_factors)
        scores.append(1 - entropy / max_entropy)

    # Weight by column sum (importance of each dim)
    weights = R.sum(axis=0)
    weights = weights / (weights.sum() + 1e-10)

    return float(np.dot(weights, scores))


def _compute_completeness(R: np.ndarray) -> float:
    """
    Completeness score from importance matrix.

    For each factor i, compute 1 - H(R[i,:]) / log(D)
    where D is latent dim.
    """
    n_factors, latent_dim = R.shape

    scores = []
    for i in range(n_factors):
        p = R[i, :]
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(latent_dim)
        scores.append(1 - entropy / max_entropy)

    return float(np.mean(scores))


def _compute_informativeness(z: np.ndarray, factors: np.ndarray) -> float:
    """Compute average prediction accuracy across factors."""
    n_factors = factors.shape[1]
    accuracies = []

    for i in range(n_factors):
        y = factors[:, i]
        if len(np.unique(y)) > 20:
            y = _discretize(y, n_bins=10)

        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        scores = cross_val_score(model, z, y, cv=3, scoring="accuracy")
        accuracies.append(scores.mean())

    return float(np.mean(accuracies))


def compute_mig(
    z: np.ndarray,
    factors: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Compute MIG (Mutual Information Gap).

    For each factor, computes MI with all latent dims, then takes the
    gap between top two. High MIG = each factor mainly uses one dim.

    Args:
        z: Latent representations [n_samples, latent_dim]
        factors: Ground truth factors [n_samples, n_factors]
        n_bins: Bins for discretization

    Returns:
        MIG score (0-1)
    """
    n_factors = factors.shape[1]
    latent_dim = z.shape[1]

    # Discretize latents
    z_discrete = np.zeros_like(z, dtype=int)
    for j in range(latent_dim):
        z_discrete[:, j] = _discretize(z[:, j], n_bins)

    gaps = []
    for i in range(n_factors):
        y = factors[:, i]
        if len(np.unique(y)) > n_bins:
            y = _discretize(y, n_bins)

        # Compute MI with each latent dim
        mi_values = []
        for j in range(latent_dim):
            mi = _mutual_information(z_discrete[:, j], y)
            mi_values.append(mi)

        mi_values = np.array(mi_values)

        # Normalize by entropy of factor
        h_y = _entropy(y)
        if h_y > 0:
            mi_values = mi_values / h_y

        # Gap between top two
        sorted_mi = np.sort(mi_values)[::-1]
        gap = sorted_mi[0] - sorted_mi[1] if len(sorted_mi) > 1 else sorted_mi[0]
        gaps.append(gap)

    return float(np.mean(gaps))


def compute_sap(
    z: np.ndarray,
    factors: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute SAP (Separated Attribute Predictability).

    For each factor, trains linear classifiers on each latent dim separately,
    then computes gap between top two prediction scores.

    Args:
        z: Latent representations [n_samples, latent_dim]
        factors: Ground truth factors [n_samples, n_factors]
        n_bins: Bins for continuous factors

    Returns:
        SAP score (0-1)
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required for SAP computation")

    n_factors = factors.shape[1]
    latent_dim = z.shape[1]

    gaps = []
    for i in range(n_factors):
        y = factors[:, i]
        if len(np.unique(y)) > n_bins:
            y = _discretize(y, n_bins)

        # Score each latent dim separately
        scores = []
        for j in range(latent_dim):
            z_j = z[:, j:j+1]  # Single feature
            try:
                model = LogisticRegression(max_iter=500, random_state=42)
                cv_scores = cross_val_score(model, z_j, y, cv=3, scoring="accuracy")
                scores.append(cv_scores.mean())
            except Exception:
                scores.append(0.0)

        scores = np.array(scores)

        # Gap between top two
        sorted_scores = np.sort(scores)[::-1]
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        gaps.append(gap)

    return float(np.mean(gaps))


def compute_edi(
    z: np.ndarray,
    factors: np.ndarray,
    n_bins: int = 20,
) -> Dict[str, float]:
    """
    Compute EDI (Exclusivity Disentanglement Index).

    A newer metric (2024) that uses exclusive mutual information
    to better handle distributed representations.

    Args:
        z: Latent representations [n_samples, latent_dim]
        factors: Ground truth factors [n_samples, n_factors]
        n_bins: Bins for discretization

    Returns:
        Dict with modularity, compactness, explicitness scores
    """
    n_samples, latent_dim = z.shape
    n_factors = factors.shape[1]

    # Discretize latents and factors
    z_discrete = np.zeros_like(z, dtype=int)
    for j in range(latent_dim):
        z_discrete[:, j] = _discretize(z[:, j], n_bins)

    factors_discrete = np.zeros_like(factors, dtype=int)
    for i in range(n_factors):
        if len(np.unique(factors[:, i])) > n_bins:
            factors_discrete[:, i] = _discretize(factors[:, i], n_bins)
        else:
            factors_discrete[:, i] = factors[:, i].astype(int)

    # Compute MI matrix
    MI = np.zeros((latent_dim, n_factors))
    for j in range(latent_dim):
        for i in range(n_factors):
            MI[j, i] = _mutual_information(z_discrete[:, j], factors_discrete[:, i])

    # Normalize each row and column
    MI_norm_row = MI / (MI.sum(axis=1, keepdims=True) + 1e-10)
    MI_norm_col = MI / (MI.sum(axis=0, keepdims=True) + 1e-10)

    # Modularity: For each latent dim, how exclusive is its information?
    modularity_scores = []
    for j in range(latent_dim):
        # Exclusivity = max MI share minus average of others
        p = MI_norm_row[j, :]
        if p.max() > 0:
            exclusivity = p.max() - (p.sum() - p.max()) / max(1, n_factors - 1)
            modularity_scores.append(max(0, exclusivity))
        else:
            modularity_scores.append(0)
    modularity = float(np.mean(modularity_scores))

    # Compactness: For each factor, how concentrated in few dims?
    compactness_scores = []
    for i in range(n_factors):
        p = MI_norm_col[:, i]
        if p.max() > 0:
            exclusivity = p.max() - (p.sum() - p.max()) / max(1, latent_dim - 1)
            compactness_scores.append(max(0, exclusivity))
        else:
            compactness_scores.append(0)
    compactness = float(np.mean(compactness_scores))

    # Explicitness: Average normalized MI (how much info is captured)
    H_factors = np.array([_entropy(factors_discrete[:, i]) for i in range(n_factors)])
    explicitness = float(np.mean(MI.max(axis=0) / (H_factors + 1e-10)))

    return {
        "modularity": modularity,
        "compactness": compactness,
        "explicitness": explicitness,
    }


def evaluate_disentanglement(
    z: np.ndarray,
    factors: np.ndarray,
    factor_names: Optional[List[str]] = None,
    compute_all: bool = True,
) -> DisentanglementReport:
    """
    Comprehensive disentanglement evaluation.

    Computes all standard metrics and returns a structured report.

    Args:
        z: Latent representations [n_samples, latent_dim]
        factors: Ground truth factors [n_samples, n_factors]
        factor_names: Optional names for factors
        compute_all: If True, compute all metrics (may be slow)

    Returns:
        DisentanglementReport with all scores
    """
    n_samples, latent_dim = z.shape
    n_factors = factors.shape[1]

    report = DisentanglementReport(
        n_samples=n_samples,
        n_factors=n_factors,
        latent_dim=latent_dim,
    )

    # Always compute DCI (most robust)
    dci = compute_dci(z, factors)
    report.dci_disentanglement = dci["disentanglement"]
    report.dci_completeness = dci["completeness"]
    report.dci_informativeness = dci["informativeness"]

    if compute_all:
        # MIG
        report.mig = compute_mig(z, factors)

        # SAP
        report.sap = compute_sap(z, factors)

        # EDI
        edi = compute_edi(z, factors)
        report.edi_modularity = edi["modularity"]
        report.edi_compactness = edi["compactness"]
        report.edi_explicitness = edi["explicitness"]

    # Per-factor breakdown (DCI importance)
    if factor_names is None:
        factor_names = [f"factor_{i}" for i in range(n_factors)]

    R = dci["importance_matrix"]
    for i, name in enumerate(factor_names):
        report.per_factor_scores[name] = {
            "top_latent_dim": int(np.argmax(R[i, :])),
            "max_importance": float(np.max(R[i, :])),
            "entropy": float(-np.sum(R[i, :] * np.log(R[i, :] + 1e-10))),
        }

    return report


# =============================================================================
# DCI-Lite for Real Data
# =============================================================================

def compute_dci_lite(
    z: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    DCI-style analysis for real data with observable labels.

    For real EEG/behavioral data where true generative factors are unknown,
    we use available labels (task condition, clinical group, HGF parameters)
    as proxy factors.

    Args:
        z: Latent representations [n_samples, latent_dim]
        labels: Observable labels [n_samples, n_labels]
        label_names: Names for each label column

    Returns:
        Dict with importance matrix, prediction accuracies, and interpretation
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required")

    n_samples, latent_dim = z.shape
    n_labels = labels.shape[1] if labels.ndim > 1 else 1
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    if label_names is None:
        label_names = [f"label_{i}" for i in range(n_labels)]

    # Standardize latents
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    # Compute importance and accuracy for each label
    results = {
        "importance_matrix": np.zeros((n_labels, latent_dim)),
        "accuracies": {},
        "top_dims": {},
    }

    for i in range(n_labels):
        y = labels[:, i]
        n_unique = len(np.unique(y))

        if n_unique > 20:
            # Continuous: use regression
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            model.fit(z_scaled, y)
            results["importance_matrix"][i, :] = model.feature_importances_
            scores = cross_val_score(model, z_scaled, y, cv=3, scoring="r2")
            results["accuracies"][label_names[i]] = float(scores.mean())
        else:
            # Categorical: use classification
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(z_scaled, y)
            results["importance_matrix"][i, :] = model.feature_importances_
            scores = cross_val_score(model, z_scaled, y, cv=3, scoring="accuracy")
            results["accuracies"][label_names[i]] = float(scores.mean())

        # Top dims for this label
        top_k = min(3, latent_dim)
        top_indices = np.argsort(results["importance_matrix"][i, :])[-top_k:][::-1]
        results["top_dims"][label_names[i]] = top_indices.tolist()

    # Compute approximate modularity (how sparse is each column?)
    R = results["importance_matrix"]
    R_norm = R / (R.sum(axis=0, keepdims=True) + 1e-10)
    sparsity = 1 - np.mean([_entropy(R_norm[:, j]) / np.log(n_labels + 1e-10)
                           for j in range(latent_dim) if R[:, j].sum() > 0])
    results["approximate_modularity"] = float(max(0, sparsity))

    return results


# =============================================================================
# Helper Functions
# =============================================================================

def _discretize(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Discretize continuous variable into bins."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(x, percentiles)
    bins = np.unique(bins)  # Remove duplicates
    return np.digitize(x, bins[1:-1])


def _entropy(x: np.ndarray) -> float:
    """Compute entropy of discrete variable."""
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-10)))


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute mutual information between two discrete variables."""
    # Joint distribution
    xy = np.column_stack([x, y])
    unique_xy, counts_xy = np.unique(xy, axis=0, return_counts=True)
    p_xy = counts_xy / counts_xy.sum()

    # Marginals
    _, counts_x = np.unique(x, return_counts=True)
    _, counts_y = np.unique(y, return_counts=True)
    p_x = counts_x / counts_x.sum()
    p_y = counts_y / counts_y.sum()

    # MI = sum p(x,y) log(p(x,y) / p(x)p(y))
    mi = 0.0
    x_map = {v: i for i, v in enumerate(np.unique(x))}
    y_map = {v: i for i, v in enumerate(np.unique(y))}

    for (xi, yi), p_joint in zip(unique_xy, p_xy):
        px = p_x[x_map[xi]]
        py = p_y[y_map[yi]]
        if p_joint > 0 and px > 0 and py > 0:
            mi += p_joint * np.log(p_joint / (px * py) + 1e-10)

    return float(max(0, mi))
