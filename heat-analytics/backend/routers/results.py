"""
Results router for retrieving analysis results.
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import structlog
import json

from backend.services.db import get_db
from backend.models.analysis import (
    AnalysisResult,
    AnalysisResultResponse,
    AnalysisResultsListResponse,
    AnalysisRunSummary,
)
from backend.models.building import Building

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/results/{run_id}", response_model=AnalysisResultsListResponse)
async def get_results(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get analysis results for a specific run.

    Args:
        run_id: Analysis run ID

    Returns:
        List of analysis results with summary
    """
    # Get all results for this run
    result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.run_id == run_id)
    )
    results = result.scalars().all()

    if not results:
        raise HTTPException(
            status_code=404,
            detail="Analysis run not found or still processing",
        )

    # Get building info
    building_id = results[0].building_id
    building_result = await db.execute(
        select(Building).where(Building.id == building_id)
    )
    building = building_result.scalar_one_or_none()

    # Aggregate statistics
    total_records = len(results)
    anomaly_count = sum(1 for r in results if r.anomaly_flag)
    models_used = list(set(r.model_type for r in results))

    # Calculate average efficiency class
    efficiency_classes = [r.efficiency_class for r in results if r.efficiency_class]
    avg_efficiency = None
    if efficiency_classes:
        # Most common class
        from collections import Counter
        avg_efficiency = Counter(efficiency_classes).most_common(1)[0][0]

    # Calculate average norm deviation
    norm_deviations = [r.norm_deviation_pct for r in results if r.norm_deviation_pct is not None]
    avg_norm_deviation = sum(norm_deviations) / len(norm_deviations) if norm_deviations else None

    run_summary = AnalysisRunSummary(
        run_id=run_id,
        building_id=building_id,
        total_records=total_records,
        anomaly_count=anomaly_count,
        models_used=models_used,
        avg_efficiency_class=avg_efficiency,
        avg_norm_deviation=avg_norm_deviation,
        created_at=results[0].created_at,
    )

    return AnalysisResultsListResponse(
        results=[
            AnalysisResultResponse(
                id=r.id,
                building_id=r.building_id,
                run_id=r.run_id,
                model_type=r.model_type,
                predicted_q=r.predicted_q,
                residual=r.residual,
                anomaly_score=r.anomaly_score,
                anomaly_flag=r.anomaly_flag,
                cluster_id=r.cluster_id,
                efficiency_class=r.efficiency_class,
                norm_deviation_pct=r.norm_deviation_pct,
                params=json.loads(r.params) if r.params else None,
                created_at=r.created_at,
            )
            for r in results
        ],
        total=total_records,
        run_summary=run_summary,
    )


@router.get("/results/{run_id}/chart")
async def get_chart_data(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get chart data for visualization (Recharts format).

    Args:
        run_id: Analysis run ID

    Returns:
        JSON data formatted for Recharts LineChart
    """
    result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.run_id == run_id)
    )
    results = result.scalars().all()

    if not results:
        raise HTTPException(status_code=404, detail="Analysis run not found")

    # Group by date (need to join with readings to get dates)
    from backend.models.reading import DailyReading

    chart_data = []
    for r in results:
        reading_result = await db.execute(
            select(DailyReading).where(
                DailyReading.building_id == r.building_id,
                DailyReading.id == r.id  # This might need adjustment
            )
        )
        reading = reading_result.scalar_one_or_none()

        if reading:
            chart_data.append({
                "date": reading.date.strftime("%Y-%m-%d"),
                "actual_q": reading.q,
                "predicted_q": r.predicted_q,
                "residual": r.residual,
                "anomaly": r.anomaly_flag,
            })

    # Sort by date
    chart_data.sort(key=lambda x: x["date"])

    return {
        "run_id": run_id,
        "data": chart_data,
    }


@router.get("/results/{run_id}/anomalies")
async def get_anomalies(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get only anomalous records from analysis results.

    Args:
        run_id: Analysis run ID

    Returns:
        List of anomalous records with details
    """
    result = await db.execute(
        select(AnalysisResult)
        .where(AnalysisResult.run_id == run_id)
        .where(AnalysisResult.anomaly_flag == True)
    )
    anomalies = result.scalars().all()

    if not anomalies:
        return {"anomalies": [], "total": 0}

    # Get detailed info including NS codes
    from backend.models.reading import DailyReading

    anomaly_details = []
    for a in anomalies:
        reading_result = await db.execute(
            select(DailyReading).where(
                DailyReading.building_id == a.building_id,
                DailyReading.id == a.id
            )
        )
        reading = reading_result.scalar_one_or_none()

        if reading:
            anomaly_details.append({
                "id": a.id,
                "date": reading.date.strftime("%Y-%m-%d"),
                "q": reading.q,
                "predicted_q": a.predicted_q,
                "residual": a.residual,
                "anomaly_score": a.anomaly_score,
                "ns_codes": reading.ns_codes,
                "model_type": a.model_type,
                "efficiency_class": a.efficiency_class,
            })

    return {
        "anomalies": anomaly_details,
        "total": len(anomaly_details),
    }


@router.get("/results/{run_id}/clustering")
async def get_clustering_data(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get clustering results for visualization.

    Args:
        run_id: Analysis run ID

    Returns:
        Data formatted for scatter plot visualization
    """
    result = await db.execute(
        select(AnalysisResult)
        .where(AnalysisResult.run_id == run_id)
        .where(AnalysisResult.cluster_id.isnot(None))
    )
    results = result.scalars().all()

    if not results:
        return {"clusters": [], "total": 0}

    cluster_data = []
    for r in results:
        cluster_data.append({
            "id": r.id,
            "cluster_id": r.cluster_id,
            "efficiency_class": r.efficiency_class,
            "norm_deviation_pct": r.norm_deviation_pct,
            "anomaly_score": r.anomaly_score,
        })

    return {
        "clusters": cluster_data,
        "total": len(cluster_data),
    }
