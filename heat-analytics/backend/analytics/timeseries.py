"""
Time series analysis for heat consumption data.
Implements decomposition, Holt-Winters, and Prophet forecasting.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


def run_decomposition(df: pd.DataFrame, period: int = 7) -> Dict[str, Any]:
    """
    Run seasonal decomposition of time series.

    Args:
        df: DataFrame with date and q columns
        period: Seasonal period (default: 7 for weekly)

    Returns:
        Dictionary with trend, seasonal, residual components
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        logger.warning("statsmodels not available, skipping decomposition")
        return {}

    # Prepare time series
    ts = df.set_index("date")["q"]
    ts = ts.resample("D").mean().fillna(method="ffill")

    if len(ts) < period * 2:
        logger.warning("Not enough data for decomposition")
        return {}

    try:
        # Run decomposition
        result = seasonal_decompose(ts, model="additive", period=period)

        return {
            "trend": result.trend.dropna().tolist(),
            "seasonal": result.seasonal.dropna().tolist(),
            "residual": result.resid.dropna().tolist(),
            "observed": result.observed.dropna().tolist(),
            "params": {
                "method": "seasonal_decompose",
                "period": period,
                "model": "additive",
            },
        }
    except Exception as e:
        logger.error("Decomposition failed", error=str(e))
        return {}


def run_holt_winters(
    df: pd.DataFrame,
    forecast_periods: int = 7,
) -> Dict[str, Any]:
    """
    Run Holt-Winters exponential smoothing.

    Args:
        df: DataFrame with date and q columns
        forecast_periods: Number of periods to forecast

    Returns:
        Dictionary with forecasts, fitted values, and parameters
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        logger.warning("statsmodels not available, skipping Holt-Winters")
        return {}

    # Prepare time series
    ts = df.set_index("date")["q"]
    ts = ts.resample("D").mean().fillna(method="ffill")

    if len(ts) < 14:
        logger.warning("Not enough data for Holt-Winters")
        return {}

    try:
        # Fit model with trend and seasonality
        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
        )
        fitted = model.fit()

        # Get forecasts
        forecast = fitted.forecast(forecast_periods)

        return {
            "fitted": fitted.fittedvalues.dropna().tolist(),
            "forecast_7d": forecast.tolist(),
            "alpha": fitted.params["smoothing_level"],
            "beta": fitted.params["smoothing_trend"],
            "gamma": fitted.params["smoothing_seasonal"],
            "params": {
                "method": "Holt-Winters",
                "trend": "add",
                "seasonal": "add",
                "seasonal_periods": 7,
            },
        }
    except Exception as e:
        logger.error("Holt-Winters failed", error=str(e))
        return {}


def run_prophet_forecast(
    df: pd.DataFrame,
    forecast_periods: int = 7,
) -> Dict[str, Any]:
    """
    Run Facebook Prophet forecasting.

    Args:
        df: DataFrame with date and q columns
        forecast_periods: Number of days to forecast

    Returns:
        Dictionary with forecasts and uncertainty intervals
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not available, skipping")
        return {}

    # Prepare data for Prophet (requires ds and y columns)
    prophet_df = df[["date", "q"]].copy()
    prophet_df.columns = ["ds", "y"]
    prophet_df = prophet_df.dropna()

    if len(prophet_df) < 30:
        logger.warning("Not enough data for Prophet (need at least 30 points)")
        return {}

    try:
        # Fit model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
        )
        model.fit(prophet_df)

        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)

        # Extract components
        forecast_df = forecast.tail(forecast_periods)

        return {
            "forecast_df": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict("records"),
            "uncertainty_intervals": {
                "lower": forecast_df["yhat_lower"].tolist(),
                "upper": forecast_df["yhat_upper"].tolist(),
            },
            "trend": forecast["trend"].tolist(),
            "weekly": forecast["weekly"].tolist() if "weekly" in forecast.columns else [],
            "params": {
                "method": "Prophet",
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": False,
            },
        }
    except Exception as e:
        logger.error("Prophet failed", error=str(e))
        return {}
