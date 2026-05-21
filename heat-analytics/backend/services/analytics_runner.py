"""
Analytics runner - orchestrates the analysis pipeline.
"""
import json
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog

from backend.models.reading import DailyReading
from backend.models.analysis import AnalysisResult
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
)
from backend.analytics.clustering import (
    run_kmeans,
    run_dbscan,
    run_gmm,
)
from backend.analytics.features import prepare_features
from backend.config.settings import settings

logger = structlog.get_logger(__name__)


async def run_analysis_pipeline(
    building_id: int,
    run_id: str,
    models: List[str],
):
    """
    Run complete analysis pipeline for a building.

    Args:
        building_id: ID of the building to analyze
        run_id: Unique identifier for this analysis run
        models: List of model types to run
    """
    logger.info("Starting analysis pipeline", run_id=run_id, building_id=building_id)

    # Get reading data from database
    async with AsyncSession(get_db_engine()) as session:
        result = await session.execute(
            select(DailyReading)
            .where(DailyReading.building_id == building_id)
            .order_by(DailyReading.date)
        )
        readings = result.scalars().all()

    if not readings:
        logger.error("No readings found", building_id=building_id)
        return

    # Convert to DataFrame
    import pandas as pd

    data = []
    for r in readings:
        data.append({
            "date": r.date,
            "q": r.q,
            "t1": r.t1,
            "t2": r.t2,
            "p1": r.p1,
            "p2": r.p2,
            "ns_codes": r.ns_codes,
        })

    df = pd.DataFrame(data)

    # Prepare features (add T_out, HDD, lags)
    df_features = prepare_features(df)

    # Store results
    results_to_save = []

    # Run regression models
    if "ols" in models:
        ols_results = run_ols_regression(df_features)
        results_to_save.extend(_format_results(ols_results, run_id, building_id, "ols"))

    if "huber" in models:
        huber_results = run_huber_regression(df_features)
        results_to_save.extend(_format_results(huber_results, run_id, building_id, "huber"))

    if "ridge" in models:
        ridge_results = run_ridge_regression(df_features)
        results_to_save.extend(_format_results(ridge_results, run_id, building_id, "ridge"))

    if "lasso" in models:
        lasso_results = run_lasso_regression(df_features)
        results_to_save.extend(_format_results(lasso_results, run_id, building_id, "lasso"))

    if "quantile" in models:
        quantile_results = run_quantile_regression(df_features)
        results_to_save.extend(_format_results(quantile_results, run_id, building_id, "quantile"))

    # Run time series models
    if "decomposition" in models:
        decomp_results = run_decomposition(df_features)
        results_to_save.extend(_format_results(decomp_results, run_id, building_id, "decomposition"))

    if "holt_winters" in models:
        hw_results = run_holt_winters(df_features)
        results_to_save.extend(_format_results(hw_results, run_id, building_id, "holt_winters"))

    if "prophet" in models:
        prophet_results = run_prophet_forecast(df_features)
        results_to_save.extend(_format_results(prophet_results, run_id, building_id, "prophet"))

    # Run anomaly detection
    if "ewma" in models:
        ewma_results = run_ewma_anomaly(df_features)
        results_to_save.extend(_format_results(ewma_results, run_id, building_id, "ewma"))

    if "isolation_forest" in models:
        if_results = run_isolation_forest(df_features)
        results_to_save.extend(_format_results(if_results, run_id, building_id, "isolation_forest"))

    if "lof" in models:
        lof_results = run_lof(df_features)
        results_to_save.extend(_format_results(lof_results, run_id, building_id, "lof"))

    # Run clustering
    if "kmeans" in models:
        kmeans_results = run_kmeans(df_features)
        results_to_save.extend(_format_results(kmeans_results, run_id, building_id, "kmeans"))

    if "dbscan" in models:
        dbscan_results = run_dbscan(df_features)
        results_to_save.extend(_format_results(dbscan_results, run_id, building_id, "dbscan"))

    if "gmm" in models:
        gmm_results = run_gmm(df_features)
        results_to_save.extend(_format_results(gmm_results, run_id, building_id, "gmm"))

    # Save results to database
    async with AsyncSession(get_db_engine()) as session:
        for result_data in results_to_save:
            analysis_result = AnalysisResult(**result_data)
            session.add(analysis_result)

        await session.commit()

    logger.info(
        "Analysis pipeline completed",
        run_id=run_id,
        results_count=len(results_to_save),
    )


def _format_results(
    results: Dict[str, Any],
    run_id: str,
    building_id: int,
    model_type: str,
) -> List[Dict[str, Any]]:
    """Format analysis results for database storage."""
    formatted = []

    # Results can be per-row or aggregate
    if "predictions" in results:
        # Per-row results (e.g., predictions, residuals)
        predictions = results["predictions"]
        residuals = results.get("residuals", [None] * len(predictions))
        anomaly_scores = results.get("anomaly_scores", [None] * len(predictions))
        anomaly_flags = results.get("anomaly_flags", [False] * len(predictions))
        cluster_ids = results.get("cluster_ids", [None] * len(predictions))

        for i in range(len(predictions)):
            formatted.append({
                "run_id": run_id,
                "building_id": building_id,
                "model_type": model_type,
                "predicted_q": float(predictions[i]) if predictions[i] is not None else None,
                "residual": float(residuals[i]) if residuals[i] is not None else None,
                "anomaly_score": float(anomaly_scores[i]) if anomaly_scores[i] is not None else None,
                "anomaly_flag": bool(anomaly_flags[i]) if anomaly_flags[i] is not None else False,
                "cluster_id": int(cluster_ids[i]) if cluster_ids[i] is not None else None,
                "efficiency_class": results.get("efficiency_class"),
                "norm_deviation_pct": results.get("norm_deviation_pct"),
                "params": json.dumps(results.get("params", {})),
            })
    else:
        # Aggregate results
        formatted.append({
            "run_id": run_id,
            "building_id": building_id,
            "model_type": model_type,
            "predicted_q": results.get("predicted_q"),
            "residual": results.get("residual"),
            "anomaly_score": results.get("anomaly_score"),
            "anomaly_flag": results.get("anomaly_flag", False),
            "cluster_id": results.get("cluster_id"),
            "efficiency_class": results.get("efficiency_class"),
            "norm_deviation_pct": results.get("norm_deviation_pct"),
            "params": json.dumps(results.get("params", {})),
        })

    return formatted


# Import here to avoid circular imports
from backend.services.db import get_db_engine
