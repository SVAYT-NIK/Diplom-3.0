"""
Feature engineering for heat consumption analysis.
Prepares features including T_out, HDD, and lag variables.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import structlog

logger = structlog.get_logger(__name__)


# Weather data stub for Abakan (can be replaced with real API)
# Format: {month: avg_temp}
ABAKAN_WEATHER_STUB = {
    1: -20.5,   # January
    2: -18.2,   # February
    3: -10.5,   # March
    4: -2.0,    # April
    5: 8.5,     # May
    6: 17.0,    # June
    7: 20.5,    # July
    8: 18.0,    # August
    9: 10.5,    # September
    10: 1.0,    # October
    11: -10.0,  # November
    12: -17.5,  # December
}

# Base temperature for HDD calculation (°C)
BASE_TEMP = 18.0


def prepare_features(
    df: pd.DataFrame,
    weather_data: Optional[pd.DataFrame] = None,
    add_lags: bool = True,
    lag_days: list = None,
) -> pd.DataFrame:
    """
    Prepare features for heat consumption modeling.

    Args:
        df: Input DataFrame with columns: date, q, t1, t2
        weather_data: Optional external weather DataFrame with columns: date, t_out
        add_lags: Whether to add lag features
        lag_days: List of lag days to create (default: [1, 7, 14])

    Returns:
        DataFrame with additional features: t_out, hdd, lags
    """
    if lag_days is None:
        lag_days = [1, 7, 14]

    df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Add outdoor temperature
    if weather_data is not None:
        df = df.merge(weather_data, on="date", how="left")
    else:
        # Use stub data based on month
        df["t_out"] = df["date"].dt.month.map(ABAKAN_WEATHER_STUB)

    # Calculate Heating Degree Days (HDD)
    df["hdd"] = np.maximum(BASE_TEMP - df["t_out"], 0)

    # Add lag features for Q
    if add_lags:
        for lag in lag_days:
            df[f"q_lag_{lag}"] = df["q"].shift(lag)

    # Add rolling statistics
    df["q_rolling_mean_7"] = df["q"].rolling(window=7, min_periods=1).mean()
    df["q_rolling_std_7"] = df["q"].rolling(window=7, min_periods=1).std()

    # Add day of week and month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Add heating season flag (October to April in Russia)
    df["heating_season"] = df["month"].apply(lambda m: 1 if m in [10, 11, 12, 1, 2, 3, 4] else 0)

    # Fill NaN values created by lags/rolling
    df = df.fillna(method="bfill").fillna(method="ffill").fillna(0)

    logger.info("Features prepared", rows=len(df), features=list(df.columns))

    return df


def create_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "q",
    feature_cols: list = None,
) -> tuple:
    """
    Create feature matrix (X) and target vector (y) for modeling.

    Args:
        df: DataFrame with features
        target_col: Name of target column
        feature_cols: List of feature columns to use (default: all numeric except date)

    Returns:
        Tuple of (X, y, feature_names)
    """
    if feature_cols is None:
        # Default features
        feature_cols = [
            "t_out",
            "hdd",
            "q_lag_1",
            "q_lag_7",
            "q_rolling_mean_7",
            "day_of_week",
            "month",
            "is_weekend",
            "heating_season",
        ]

    # Filter to available columns
    available_features = [f for f in feature_cols if f in df.columns]

    X = df[available_features].values
    y = df[target_col].values

    return X, y, available_features


def get_weather_stub_df(
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Generate stub weather DataFrame for a date range.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with columns: date, t_out
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    weather_df = pd.DataFrame({
        "date": dates,
        "t_out": [ABAKAN_WEATHER_STUB[d.month] + np.random.normal(0, 3) for d in dates],
    })

    return weather_df
