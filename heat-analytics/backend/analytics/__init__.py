"""
Analytics module for heat consumption analysis.
"""
from backend.analytics.features import prepare_features, create_feature_matrix
from backend.analytics.regression import (
    run_ols_regression,
    run_huber_regression,
    run_ridge_regression,
    run_lasso_regression,
    run_quantile_regression,
)
from backend.analytics.timeseries import (
    run_decomposition,
    run_holt_winters,
    run_prophet_forecast,
)
from backend.analytics.anomaly import (
    run_ewma_anomaly,
    run_isolation_forest,
    run_lof,
    consensus_anomaly_detection,
)
from backend.analytics.clustering import (
    run_kmeans,
    run_dbscan,
    run_gmm,
    calculate_efficiency_class,
)

__all__ = [
    # Features
    "prepare_features",
    "create_feature_matrix",
    # Regression
    "run_ols_regression",
    "run_huber_regression",
    "run_ridge_regression",
    "run_lasso_regression",
    "run_quantile_regression",
    # Time series
    "run_decomposition",
    "run_holt_winters",
    "run_prophet_forecast",
    # Anomaly detection
    "run_ewma_anomaly",
    "run_isolation_forest",
    "run_lof",
    "consensus_anomaly_detection",
    # Clustering
    "run_kmeans",
    "run_dbscan",
    "run_gmm",
    "calculate_efficiency_class",
]
