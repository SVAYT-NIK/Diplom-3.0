"""
Anomaly detection for heat consumption data.
Implements EWMA, Isolation Forest, and Local Outlier Factor.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import structlog

logger = structlog.get_logger(__name__)


def run_ewma_anomaly(
    df: pd.DataFrame,
    span: int = 14,
    std_multiplier: float = 3.0,
) -> Dict[str, Any]:
    """
    Run Exponentially Weighted Moving Average anomaly detection.

    Args:
        df: DataFrame with date and q columns
        span: Span for EWMA calculation (default: 14)
        std_multiplier: Number of standard deviations for threshold (default: 3.0)

    Returns:
        Dictionary with EWMA series, anomaly flags, and trigger days
    """
    q = df["q"].dropna()

    if len(q) < span:
        logger.warning("Not enough data for EWMA")
        return {"anomaly_flags": [False] * len(df)}

    # Calculate EWMA
    ewma = q.ewm(span=span, adjust=False).mean()

    # Calculate rolling standard deviation
    rolling_std = q.ewm(span=span, adjust=False).std()

    # Calculate residuals
    residuals = q - ewma

    # Detect anomalies
    threshold = std_multiplier * rolling_std
    anomaly_flags = np.abs(residuals) > threshold

    # Find trigger days
    trigger_days = q.index[anomaly_flags].tolist()

    logger.info(
        "EWMA anomaly detection completed",
        anomaly_count=anomaly_flags.sum(),
        total=len(q),
    )

    # Align with original dataframe
    full_anomaly_flags = [False] * len(df)
    for i, idx in enumerate(q.index):
        orig_idx = df.index[df["date"] == idx]
        if len(orig_idx) > 0:
            full_anomaly_flags[orig_idx[0]] = anomaly_flags.iloc[i]

    return {
        "ewma_series": ewma.tolist(),
        "anomaly_flags": full_anomaly_flags,
        "trigger_days": [str(d.date()) if hasattr(d, "date") else str(d) for d in trigger_days],
        "params": {
            "method": "EWMA",
            "span": span,
            "std_multiplier": std_multiplier,
        },
    }


def run_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run Isolation Forest anomaly detection.

    Args:
        df: DataFrame with features
        contamination: Expected proportion of outliers (default: 0.1)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with anomaly scores and labels
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        logger.warning("sklearn not available, skipping Isolation Forest")
        return {"anomaly_flags": [False] * len(df)}

    # Select numeric features
    feature_cols = ["q", "t_out", "hdd"] if "hdd" in df.columns else ["q"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].dropna()

    if len(X) < 10:
        logger.warning("Not enough data for Isolation Forest")
        return {"anomaly_flags": [False] * len(df)}

    # Fit model
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    model.fit(X)

    # Get predictions and scores
    predictions = model.predict(X)
    scores = model.decision_function(X)

    # Convert to anomaly flags (-1 = anomaly, 1 = normal)
    anomaly_flags = predictions == -1

    logger.info(
        "Isolation Forest completed",
        anomaly_count=anomaly_flags.sum(),
        total=len(X),
    )

    # Align with original dataframe
    full_anomaly_flags = [False] * len(df)
    full_scores = [0.0] * len(df)

    for i, idx in enumerate(X.index):
        full_anomaly_flags[idx] = bool(anomaly_flags.iloc[i]) if hasattr(anomaly_flags, "iloc") else bool(anomaly_flags[i])
        full_scores[idx] = float(scores[i])

    return {
        "anomaly_scores": full_scores,
        "anomaly_flags": full_anomaly_flags,
        "labels": [-1 if f else 1 for f in full_anomaly_flags],
        "contamination": contamination,
        "params": {
            "method": "IsolationForest",
            "contamination": contamination,
            "n_estimators": 100,
            "random_state": random_state,
        },
    }


def run_lof(
    df: pd.DataFrame,
    n_neighbors: int = 20,
    contamination: float = 0.1,
) -> Dict[str, Any]:
    """
    Run Local Outlier Factor anomaly detection.

    Args:
        df: DataFrame with features
        n_neighbors: Number of neighbors for LOF (default: 20)
        contamination: Expected proportion of outliers (default: 0.1)

    Returns:
        Dictionary with LOF scores and outlier mask
    """
    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        logger.warning("sklearn not available, skipping LOF")
        return {"anomaly_flags": [False] * len(df)}

    # Select numeric features
    feature_cols = ["q", "t_out", "hdd"] if "hdd" in df.columns else ["q"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].dropna()

    if len(X) < n_neighbors:
        logger.warning("Not enough data for LOF")
        return {"anomaly_flags": [False] * len(df)}

    # Fit model
    model = LocalOutlierFactor(
        n_neighbors=min(n_neighbors, len(X) - 1),
        contamination=contamination,
        novelty=False,
    )
    predictions = model.fit_predict(X)
    scores = model.negative_outlier_factor_

    # Convert to anomaly flags (-1 = anomaly, 1 = normal)
    outlier_mask = predictions == -1

    logger.info(
        "LOF completed",
        anomaly_count=outlier_mask.sum(),
        total=len(X),
    )

    # Align with original dataframe
    full_outlier_mask = [False] * len(df)
    full_scores = [0.0] * len(df)

    for i, idx in enumerate(X.index):
        full_outlier_mask[idx] = bool(outlier_mask[i])
        full_scores[idx] = float(scores[i])

    return {
        "lof_scores": full_scores,
        "outlier_mask": full_outlier_mask,
        "anomaly_flags": full_outlier_mask,
        "neighbors": n_neighbors,
        "params": {
            "method": "LocalOutlierFactor",
            "n_neighbors": min(n_neighbors, len(X) - 1),
            "contamination": contamination,
        },
    }


def consensus_anomaly_detection(
    df: pd.DataFrame,
    methods: List[str] = None,
    min_consensus: int = 2,
) -> Dict[str, Any]:
    """
    Run consensus-based anomaly detection using multiple methods.

    An anomaly is flagged only if at least min_consensus methods agree.

    Args:
        df: DataFrame with features
        methods: List of methods to use (default: ["ewma", "isolation_forest", "lof"])
        min_consensus: Minimum number of methods that must agree (default: 2)

    Returns:
        Dictionary with combined anomaly flags
    """
    if methods is None:
        methods = ["ewma", "isolation_forest", "lof"]

    all_flags = []

    # Run each method
    if "ewma" in methods:
        ewma_result = run_ewma_anomaly(df)
        all_flags.append(ewma_result["anomaly_flags"])

    if "isolation_forest" in methods:
        if_result = run_isolation_forest(df)
        all_flags.append(if_result["anomaly_flags"])

    if "lof" in methods:
        lof_result = run_lof(df)
        all_flags.append(lof_result["anomaly_flags"])

    if not all_flags:
        return {"anomaly_flags": [False] * len(df)}

    # Count agreements
    flag_matrix = np.array(all_flags).T
    consensus_count = np.sum(flag_matrix, axis=1)

    # Final anomaly flags based on consensus
    final_flags = consensus_count >= min_consensus

    logger.info(
        "Consensus anomaly detection completed",
        methods_used=len(methods),
        min_consensus=min_consensus,
        anomaly_count=final_flags.sum(),
    )

    return {
        "anomaly_flags": final_flags.tolist(),
        "consensus_counts": consensus_count.tolist(),
        "methods_used": methods,
        "min_consensus": min_consensus,
    }
