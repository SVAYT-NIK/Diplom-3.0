"""
Regression models for heat consumption analysis.
Implements OLS, Huber, Ridge, Lasso, and Quantile regression.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import structlog

from backend.analytics.features import create_feature_matrix

logger = structlog.get_logger(__name__)


def run_ols_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run Ordinary Least Squares (OLS) regression.

    Args:
        df: DataFrame with features and target

    Returns:
        Dictionary with coefficients, R2, p-values, predictions, residuals
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("statsmodels not available, skipping OLS")
        return {"predictions": [], "residuals": []}

    X, y, feature_names = create_feature_matrix(df)

    # Add constant for intercept
    X_const = sm.add_constant(X)

    # Fit OLS model
    model = sm.OLS(y, X_const)
    results = model.fit()

    # Get predictions and residuals
    predictions = results.predict(X_const)
    residuals = results.resid

    # Calculate metrics
    r_squared = results.rsquared
    adj_r_squared = results.rsquared_adj

    # Get confidence intervals
    conf_int = results.conf_int(alpha=0.05)

    logger.info(
        "OLS regression completed",
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
    )

    return {
        "predictions": predictions.tolist(),
        "residuals": residuals.tolist(),
        "beta0": results.params[0],  # Intercept
        "beta1": results.params[1] if len(results.params) > 1 else 0,  # First coefficient
        "coefficients": dict(zip(["const"] + feature_names, results.params.tolist())),
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "p_values": dict(zip(["const"] + feature_names, results.pvalues.tolist())),
        "confidence_intervals": conf_int.values.tolist(),
        "params": {
            "method": "OLS",
            "alpha": 0.05,
        },
    }


def run_huber_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run Robust Regression using Huber loss.

    Args:
        df: DataFrame with features and target

    Returns:
        Dictionary with coefficients, residuals, predictions
    """
    try:
        from sklearn.linear_model import HuberRegressor
    except ImportError:
        logger.warning("sklearn not available, skipping Huber")
        return {"predictions": [], "residuals": []}

    X, y, feature_names = create_feature_matrix(df)

    # Fit Huber model
    model = HuberRegressor(epsilon=1.35, max_iter=1000)
    model.fit(X, y)

    # Get predictions and residuals
    predictions = model.predict(X)
    residuals = y - predictions

    logger.info(
        "Huber regression completed",
        n_iterations=model.n_iter_,
    )

    return {
        "predictions": predictions.tolist(),
        "residuals": residuals.tolist(),
        "beta0": model.intercept_,
        "beta1": model.coef_[0] if len(model.coef_) > 0 else 0,
        "coefficients": dict(zip(feature_names, model.coef_.tolist())),
        "params": {
            "method": "Huber",
            "epsilon": 1.35,
            "max_iter": 1000,
        },
    }


def run_ridge_regression(df: pd.DataFrame, alpha: float = 1.0) -> Dict[str, Any]:
    """
    Run Ridge regression with L2 regularization.

    Args:
        df: DataFrame with features and target
        alpha: Regularization strength

    Returns:
        Dictionary with coefficients, predictions, residuals
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        logger.warning("sklearn not available, skipping Ridge")
        return {"predictions": [], "residuals": []}

    X, y, feature_names = create_feature_matrix(df)

    # Fit Ridge model
    model = Ridge(alpha=alpha)
    model.fit(X, y)

    # Get predictions and residuals
    predictions = model.predict(X)
    residuals = y - predictions

    # Calculate R2
    r_squared = model.score(X, y)

    logger.info("Ridge regression completed", alpha=alpha, r_squared=r_squared)

    return {
        "predictions": predictions.tolist(),
        "residuals": residuals.tolist(),
        "beta0": model.intercept_,
        "beta1": model.coef_[0] if len(model.coef_) > 0 else 0,
        "coefficients": dict(zip(feature_names, model.coef_.tolist())),
        "r_squared": r_squared,
        "params": {
            "method": "Ridge",
            "alpha": alpha,
        },
    }


def run_lasso_regression(df: pd.DataFrame, alpha: float = 0.01) -> Dict[str, Any]:
    """
    Run Lasso regression with L1 regularization.

    Args:
        df: DataFrame with features and target
        alpha: Regularization strength

    Returns:
        Dictionary with coefficients, predictions, residuals
    """
    try:
        from sklearn.linear_model import Lasso
    except ImportError:
        logger.warning("sklearn not available, skipping Lasso")
        return {"predictions": [], "residuals": []}

    X, y, feature_names = create_feature_matrix(df)

    # Fit Lasso model
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X, y)

    # Get predictions and residuals
    predictions = model.predict(X)
    residuals = y - predictions

    # Calculate R2
    r_squared = model.score(X, y)

    logger.info("Lasso regression completed", alpha=alpha, r_squared=r_squared)

    return {
        "predictions": predictions.tolist(),
        "residuals": residuals.tolist(),
        "beta0": model.intercept_,
        "beta1": model.coef_[0] if len(model.coef_) > 0 else 0,
        "coefficients": dict(zip(feature_names, model.coef_.tolist())),
        "r_squared": r_squared,
        "params": {
            "method": "Lasso",
            "alpha": alpha,
        },
    }


def run_quantile_regression(
    df: pd.DataFrame,
    quantiles: list = None,
) -> Dict[str, Any]:
    """
    Run Quantile regression for multiple quantiles.

    Args:
        df: DataFrame with features and target
        quantiles: List of quantiles to estimate (default: [0.25, 0.5, 0.75])

    Returns:
        Dictionary with quantile estimates and intervals
    """
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]

    try:
        import statsmodels.api as sm
        from statsmodels.regression.quantile_regression import QuantReg
    except ImportError:
        logger.warning("statsmodels not available, skipping Quantile")
        return {"predictions": [], "residuals": []}

    X, y, feature_names = create_feature_matrix(df)
    X_const = sm.add_constant(X)

    results_by_quantile = {}

    for q in quantiles:
        model = QuantReg(y, X_const)
        result = model.fit(q=q)
        results_by_quantile[f"q{q}"] = {
            "coefficients": dict(zip(["const"] + feature_names, result.params.tolist())),
            "predictions": result.predict(X_const).tolist(),
        }

    # Calculate interval width
    if "q0.25" in results_by_quantile and "q0.75" in results_by_quantile:
        pred_25 = np.array(results_by_quantile["q0.25"]["predictions"])
        pred_75 = np.array(results_by_quantile["q0.75"]["predictions"])
        interval_width = pred_75 - pred_25
    else:
        interval_width = None

    logger.info("Quantile regression completed", quantiles=quantiles)

    return {
        "predictions": results_by_quantile.get("q0.5", {}).get("predictions", []),
        "residuals": [],  # Not directly available for quantile regression
        "quantile_results": results_by_quantile,
        "interval_widths": interval_width.tolist() if interval_width is not None else None,
        "params": {
            "method": "Quantile",
            "quantiles": quantiles,
        },
    }
