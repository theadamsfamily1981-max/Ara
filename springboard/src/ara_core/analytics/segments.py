"""
Segmentation and clustering for small datasets.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


def segment_by_performance(
    df: pd.DataFrame,
    target: str,
    n_segments: int = 3,
) -> Dict[str, Any]:
    """
    Segment data by target performance (top/middle/bottom).

    Args:
        df: Input DataFrame
        target: Target column name
        n_segments: Number of segments (default 3: top/middle/bottom)

    Returns:
        Dictionary with segment info and characteristics
    """
    target_series = pd.to_numeric(df[target], errors="coerce")
    valid_mask = ~target_series.isna()
    valid_df = df[valid_mask].copy()
    valid_target = target_series[valid_mask]

    # Create segments based on percentiles
    percentiles = np.linspace(0, 100, n_segments + 1)
    thresholds = np.percentile(valid_target, percentiles)

    segments = []
    segment_labels = ["bottom", "middle", "top"] if n_segments == 3 else [f"segment_{i}" for i in range(n_segments)]

    for i in range(n_segments):
        lower = thresholds[i]
        upper = thresholds[i + 1]

        if i == n_segments - 1:
            mask = (valid_target >= lower) & (valid_target <= upper)
        else:
            mask = (valid_target >= lower) & (valid_target < upper)

        segment_df = valid_df[mask]

        if len(segment_df) == 0:
            continue

        # Find distinguishing characteristics
        characteristics = []
        for col in df.columns:
            if col == target:
                continue

            try:
                if df[col].dtype == 'object' or df[col].nunique() < 10:
                    # Categorical: find most common value
                    segment_mode = segment_df[col].mode()
                    overall_mode = df[col].mode()
                    if len(segment_mode) > 0 and len(overall_mode) > 0:
                        if segment_mode.iloc[0] != overall_mode.iloc[0]:
                            characteristics.append(f"{col}='{segment_mode.iloc[0]}' is more common")
                else:
                    # Numeric: compare means
                    segment_mean = pd.to_numeric(segment_df[col], errors="coerce").mean()
                    overall_mean = pd.to_numeric(df[col], errors="coerce").mean()
                    if pd.notna(segment_mean) and pd.notna(overall_mean) and overall_mean != 0:
                        diff_pct = (segment_mean - overall_mean) / abs(overall_mean) * 100
                        if abs(diff_pct) > 10:
                            direction = "higher" if diff_pct > 0 else "lower"
                            characteristics.append(f"{col} is {abs(diff_pct):.0f}% {direction}")
            except Exception:
                continue

        segments.append({
            "label": segment_labels[i] if i < len(segment_labels) else f"segment_{i}",
            "size": len(segment_df),
            "target_range": f"{lower:.2f} - {upper:.2f}",
            "target_mean": float(valid_target[mask].mean()),
            "characteristics": characteristics[:5],  # Top 5 distinguishing features
        })

    return {
        "segments": segments,
        "target": target,
        "total_rows": len(df),
    }


def find_clusters(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_clusters: int = 3,
) -> Dict[str, Any]:
    """
    Simple clustering using k-means on numeric features.

    Args:
        df: Input DataFrame
        features: Features to use (auto-detect numeric if None)
        n_clusters: Number of clusters

    Returns:
        Dictionary with cluster assignments and characteristics
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "sklearn not installed - run: pip install scikit-learn"}

    # Auto-detect numeric features
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(features) < 2:
        return {"error": "Need at least 2 numeric features for clustering"}

    # Prepare data
    X = df[features].copy()
    X = X.fillna(X.mean())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Analyze clusters
    df_with_labels = df.copy()
    df_with_labels["_cluster"] = labels

    clusters = []
    for i in range(n_clusters):
        cluster_df = df_with_labels[df_with_labels["_cluster"] == i]

        # Find distinguishing features
        characteristics = []
        for feat in features:
            cluster_mean = cluster_df[feat].mean()
            overall_mean = df[feat].mean()
            if overall_mean != 0:
                diff_pct = (cluster_mean - overall_mean) / abs(overall_mean) * 100
                if abs(diff_pct) > 15:
                    direction = "high" if diff_pct > 0 else "low"
                    characteristics.append(f"{direction} {feat}")

        clusters.append({
            "cluster_id": i,
            "size": len(cluster_df),
            "pct_of_total": round(len(cluster_df) / len(df) * 100, 1),
            "characteristics": characteristics[:5],
        })

    return {
        "clusters": clusters,
        "features_used": features,
        "n_clusters": n_clusters,
        "labels": labels.tolist(),
    }
