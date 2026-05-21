"""
Analysis router for running analytics pipelines.
"""
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog

from backend.services.db import get_db
from backend.models.building import Building
from backend.models.analysis import AnalysisResult
from backend.services.analytics_runner import run_analysis_pipeline

logger = structlog.get_logger(__name__)

router = APIRouter()


class AnalysisRequest:
    """Request schema for analysis."""

    def __init__(
        self,
        building_id: int,
        models: Optional[List[str]] = None,
    ):
        self.building_id = building_id
        self.models = models or [
            "ols",
            "huber",
            "isolation_forest",
            "kmeans",
        ]


@router.post("/analyze")
async def start_analysis(
    request: dict,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Start analysis pipeline for a building.

    Args:
        building_id: ID of the building to analyze
        models: List of model types to run (ols, huber, ridge, lasso, quantile,
                isolation_forest, lof, kmeans, dbscan, gmm, prophet, holt_winters)

    Returns:
        Run ID for tracking analysis progress
    """
    building_id = request.get("building_id")
    models = request.get("models", ["ols", "huber", "isolation_forest", "kmeans"])

    if not building_id:
        raise HTTPException(status_code=400, detail="building_id is required")

    # Verify building exists
    result = await db.execute(
        select(Building).where(Building.id == building_id)
    )
    building = result.scalar_one_or_none()

    if not building:
        raise HTTPException(status_code=404, detail="Building not found")

    # Check if building has data
    from backend.models.reading import DailyReading
    readings_result = await db.execute(
        select(DailyReading).where(DailyReading.building_id == building_id)
    )
    readings = readings_result.scalars().all()

    if not readings:
        raise HTTPException(
            status_code=400,
            detail="No data available for this building",
        )

    # Generate unique run ID
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    logger.info(
        "Starting analysis",
        run_id=run_id,
        building_id=building_id,
        models=models,
    )

    # Run analysis in background
    background_tasks.add_task(
        run_analysis_pipeline,
        building_id=building_id,
        run_id=run_id,
        models=models,
    )

    return {
        "status": "started",
        "run_id": run_id,
        "building_id": building_id,
        "models": models,
        "message": "Analysis started in background",
    }


@router.get("/analysis/{run_id}/status")
async def get_analysis_status(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get status of an analysis run.

    Args:
        run_id: Analysis run ID

    Returns:
        Status and progress information
    """
    result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.run_id == run_id)
    )
    results = result.scalars().all()

    if not results:
        # Check if run_id exists at all
        raise HTTPException(
            status_code=404,
            detail="Analysis run not found or still processing",
        )

    # Aggregate status
    total_records = len(results)
    anomaly_count = sum(1 for r in results if r.anomaly_flag)
    models_used = list(set(r.model_type for r in results))

    return {
        "run_id": run_id,
        "status": "completed",
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "models_used": models_used,
        "created_at": results[0].created_at.isoformat() if results else None,
    }
