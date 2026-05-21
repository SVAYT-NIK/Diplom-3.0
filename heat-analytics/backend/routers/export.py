"""
Export router for generating PDF/CSV reports.
"""
from io import BytesIO
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog

from backend.services.db import get_db
from backend.models.analysis import AnalysisResult
from backend.models.building import Building
from backend.services.report_generator import generate_pdf_report, generate_csv_report

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/export/pdf")
async def export_pdf(
    request: dict,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate and download a PDF report.

    Args:
        run_id: Analysis run ID to include in report

    Returns:
        PDF file as attachment
    """
    run_id = request.get("run_id")

    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")

    # Get analysis results
    result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.run_id == run_id)
    )
    results = result.scalars().all()

    if not results:
        raise HTTPException(status_code=404, detail="Analysis run not found")

    # Get building info
    building_id = results[0].building_id
    building_result = await db.execute(
        select(Building).where(Building.id == building_id)
    )
    building = building_result.scalar_one_or_none()

    try:
        # Generate PDF
        pdf_buffer = generate_pdf_report(
            building=building,
            results=results,
            run_id=run_id,
        )

        return StreamingResponse(
            BytesIO(pdf_buffer),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=report_{run_id}.pdf"
            },
        )
    except Exception as e:
        logger.error("PDF generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


@router.post("/export/csv")
async def export_csv(
    request: dict,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate and download a CSV report.

    Args:
        run_id: Analysis run ID to include in report

    Returns:
        CSV file as attachment
    """
    run_id = request.get("run_id")

    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")

    # Get analysis results
    result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.run_id == run_id)
    )
    results = result.scalars().all()

    if not results:
        raise HTTPException(status_code=404, detail="Analysis run not found")

    # Get building info
    building_id = results[0].building_id
    building_result = await db.execute(
        select(Building).where(Building.id == building_id)
    )
    building = building_result.scalar_one_or_none()

    try:
        # Generate CSV
        csv_content = generate_csv_report(
            building=building,
            results=results,
            run_id=run_id,
        )

        return StreamingResponse(
            BytesIO(csv_content.encode("utf-8")),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=report_{run_id}.csv"
            },
        )
    except Exception as e:
        logger.error("CSV generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")


@router.get("/export/{run_id}/summary")
async def get_export_summary(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get summary data for export preview.

    Args:
        run_id: Analysis run ID

    Returns:
        Summary statistics for the report
    """
    # Get analysis results
    result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.run_id == run_id)
    )
    results = result.scalars().all()

    if not results:
        raise HTTPException(status_code=404, detail="Analysis run not found")

    # Aggregate statistics
    total_records = len(results)
    anomaly_count = sum(1 for r in results if r.anomaly_flag)

    # Efficiency class distribution
    efficiency_dist = {}
    for r in results:
        if r.efficiency_class:
            efficiency_dist[r.efficiency_class] = efficiency_dist.get(r.efficiency_class, 0) + 1

    # Model types used
    models_used = list(set(r.model_type for r in results))

    # Average metrics
    avg_residual = sum(abs(r.residual or 0) for r in results) / total_records
    avg_anomaly_score = sum(r.anomaly_score or 0 for r in results) / total_records

    norm_deviations = [r.norm_deviation_pct for r in results if r.norm_deviation_pct is not None]
    avg_norm_deviation = sum(norm_deviations) / len(norm_deviations) if norm_deviations else 0

    return {
        "run_id": run_id,
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": (anomaly_count / total_records * 100) if total_records > 0 else 0,
        "efficiency_distribution": efficiency_dist,
        "models_used": models_used,
        "avg_residual": round(avg_residual, 6),
        "avg_anomaly_score": round(avg_anomaly_score, 4),
        "avg_norm_deviation": round(avg_norm_deviation, 2),
    }
