"""
Clustering algorithms for heat consumption analysis.
Implements K-Means++, DBSCAN, and Gaussian Mixture Models.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import structlog

logger = structlog.get_logger(__name__)


def run_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run K-Means++ clustering.

    Args:
        df: DataFrame with features
        n_clusters: Number of clusters (default: 4)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with cluster labels, centroids, inertia, silhouette score
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        logger.warning("sklearn not available, skipping K-Means")
        return {"cluster_labels": []}

    # Create feature vector for clustering
    # Using: [mean_Q, beta1, intercept, R2, cv_Q, norm_deviation]
    feature_cols = ["q", "t_out", "hdd"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].dropna()

    if len(X) < n_clusters:
        logger.warning("Not enough data for K-Means")
        return {"cluster_labels": [0] * len(df)}

    # Fit K-Means with k-means++ initialization
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    model.fit(X)

    # Get cluster assignments
    labels = model.labels_

    # Calculate silhouette score
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = 0.0

    logger.info(
        "K-Means clustering completed",
        n_clusters=n_clusters,
        inertia=model.inertia_,
        silhouette=silhouette,
    )

    # Align with original dataframe
    full_labels = [-1] * len(df)
    for i, idx in enumerate(X.index):
        full_labels[idx] = int(labels[i])

    return {
        "cluster_labels": full_labels,
        "centroids": model.cluster_centers_.tolist(),
        "inertia": float(model.inertia_),
        "silhouette_score": float(silhouette),
        "params": {
            "method": "KMeans",
            "init": "k-means++",
            "n_clusters": n_clusters,
            "n_init": 10,
            "max_iter": 300,
            "random_state": random_state,
        },
    }


def run_dbscan(
    df: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
) -> Dict[str, Any]:
    """
    Run DBSCAN clustering.

    Args:
        df: DataFrame with features
        eps: Maximum distance between samples (default: 0.5)
        min_samples: Minimum samples per cluster (default: 5)

    Returns:
        Dictionary with labels, core samples, noise mask
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        logger.warning("sklearn not available, skipping DBSCAN")
        return {"cluster_labels": []}

    # Select features
    feature_cols = ["q", "t_out", "hdd"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].dropna()

    if len(X) < min_samples:
        logger.warning("Not enough data for DBSCAN")
        return {"cluster_labels": [-1] * len(df)}

    # Fit DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    # Identify core samples and noise
    core_samples = model.core_sample_indices_
    noise_mask = labels == -1

    logger.info(
        "DBSCAN clustering completed",
        n_clusters=len(set(labels)) - (1 if -1 in labels else 0),
        noise_count=noise_mask.sum(),
    )

    # Align with original dataframe
    full_labels = [-1] * len(df)
    for i, idx in enumerate(X.index):
        full_labels[idx] = int(labels[i])

    return {
        "labels": full_labels,
        "cluster_labels": full_labels,
        "core_samples": core_samples.tolist(),
        "noise_mask": noise_mask.tolist(),
        "params": {
            "method": "DBSCAN",
            "eps": eps,
            "min_samples": min_samples,
        },
    }


def run_gmm(
    df: pd.DataFrame,
    n_components: int = 4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run Gaussian Mixture Model clustering.

    Args:
        df: DataFrame with features
        n_components: Number of mixture components (default: 4)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with soft assignments, log likelihood, BIC
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        logger.warning("sklearn not available, skipping GMM")
        return {"cluster_labels": []}

    # Select features
    feature_cols = ["q", "t_out", "hdd"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].dropna()

    if len(X) < n_components * 2:
        logger.warning("Not enough data for GMM")
        return {"cluster_labels": [0] * len(df)}

    # Fit GMM
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    model.fit(X)

    # Get predictions and probabilities
    labels = model.predict(X)
    probabilities = model.predict_proba(X)

    # Get log likelihood and BIC
    log_likelihood = model.score(X) * len(X)
    bic = model.bic(X)

    logger.info(
        "GMM clustering completed",
        n_components=n_components,
        bic=bic,
        log_likelihood=log_likelihood,
    )

    # Align with original dataframe
    full_labels = [0] * len(df)
    full_probs = [[0.0] * n_components] * len(df)

    for i, idx in enumerate(X.index):
        full_labels[idx] = int(labels[i])
        full_probs[idx] = probabilities[i].tolist()

    return {
        "cluster_labels": full_labels,
        "soft_assignments": full_probs,
        "log_likelihood": float(log_likelihood),
        "bic": float(bic),
        "params": {
            "method": "GaussianMixture",
            "n_components": n_components,
            "covariance_type": "full",
            "n_init": 10,
            "max_iter": 300,
            "random_state": random_state,
        },
    }


def calculate_efficiency_class(
    norm_deviation_pct: float,
    efficiency_thresholds: dict = None,
) -> str:
    """
    Calculate efficiency class based on norm deviation.

    Classes:
        A: >15% below norm (excellent)
        B: 5-15% below norm (good)
        C: ±5% of norm (normal)
        D: 5-15% above norm (poor)
        E: >15% above norm (critical)

    Args:
        norm_deviation_pct: Percentage deviation from norm
        efficiency_thresholds: Custom threshold dictionary

    Returns:
        Efficiency class letter (A, B, C, D, E)
    """
    if efficiency_thresholds is None:
        efficiency_thresholds = {
            "A": {"min": -100, "max": -15},
            "B": {"min": -15, "max": -5},
            "C": {"min": -5, "max": 5},
            "D": {"min": 5, "max": 15},
            "E": {"min": 15, "max": 100},
        }

    for cls, thresholds in efficiency_thresholds.items():
        if thresholds["min"] <= norm_deviation_pct < thresholds["max"]:
            return cls

    return "C"  # Default
