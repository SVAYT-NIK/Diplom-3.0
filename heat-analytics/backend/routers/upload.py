"""
Upload router for handling Excel file uploads.
"""
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog

from backend.services.db import get_db
from backend.models.building import Building, BuildingResponse
from backend.models.reading import DailyReading, DailyReadingCreate
from backend.models.audit import AuditLog
from backend.services.parser import parse_excel_file
from backend.config.settings import settings

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/upload", response_model=BuildingResponse)
async def upload_excel(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload an Excel file with heat consumption data.

    The file will be parsed, and data will be saved to the database.
    If a building with the same address exists, data will be added to it.
    Otherwise, a new building will be created.

    Returns:
        Building information with ID
    """
    logger.info("Received file upload", filename=file.filename, size=file.size)

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.allowed_extensions}"
        )

    # Validate file size
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB"
        )

    # Save file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename

    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Parse Excel file
        df, metadata = parse_excel_file(str(temp_path))

        logger.info("File parsed successfully", rows=len(df))

        # Extract or create building
        building_info = ExcelParser_building_info(df, metadata)
        building = await get_or_create_building(db, building_info)

        # Save daily readings
        await save_readings(db, building.id, df)

        # Log audit
        audit_entry = AuditLog(
            user="system",
            action="upload",
            payload={
                "filename": file.filename,
                "building_id": building.id,
                "rows_parsed": len(df),
            }
        )
        db.add(audit_entry)
        await db.commit()

        logger.info(
            "Upload completed",
            building_id=building.id,
            rows=len(df)
        )

        return BuildingResponse(
            id=building.id,
            address=building.address,
            area_m2=building.area_m2,
            year_built=building.year_built,
            heating_type=building.heating_type,
            norm_gcal_m2=building.norm_gcal_m2,
            created_at=building.created_at
        )

    except ValueError as e:
        logger.error("Parse error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        # Clean up temp file
        if temp_path.exists():
            os.remove(temp_path)
        if temp_dir and Path(temp_dir).exists():
            os.rmdir(temp_dir)


def ExcelParser_building_info(df, metadata) -> dict:
    """Extract building information from parsed data."""
    # Try to get from metadata first
    consumer = metadata.get("consumer", "")
    address = consumer if consumer else "Unknown Building"

    # Try to extract area from metadata or use default
    area_m2 = 1000.0  # Default value, should be updated by user

    return {
        "address": address,
        "area_m2": area_m2,
        "year_built": None,
        "heating_type": None,
        "norm_gcal_m2": None,
    }


async def get_or_create_building(
    db: AsyncSession,
    building_info: dict
) -> Building:
    """Get existing building or create new one."""
    # Try to find by address
    result = await db.execute(
        select(Building).where(Building.address == building_info["address"])
    )
    building = result.scalar_one_or_none()

    if building:
        logger.info("Found existing building", address=building.address)
        return building

    # Create new building
    building = Building(**building_info)
    db.add(building)
    await db.commit()
    await db.refresh(building)

    logger.info("Created new building", address=building.address)
    return building


async def save_readings(db: AsyncSession, building_id: int, df):
    """Save daily readings from dataframe to database."""
    readings_to_save = []

    for _, row in df.iterrows():
        if row.get("Дата") is None:
            continue

        reading_data = {
            "building_id": building_id,
            "date": row["Дата"],
            "t1": row.get("T1"),
            "t2": row.get("T2"),
            "p1": row.get("P1"),
            "p2": row.get("P2"),
            "v1": row.get("V1"),
            "v2": row.get("V2"),
            "m1": row.get("M1"),
            "m2": row.get("M2"),
            "q": row.get("Q"),
            "dt": row.get("d T"),
            "dv": row.get("d V"),
            "dm": row.get("d M"),
            "imbalance": row.get("Небаланс"),
            "ns_codes": row.get("НС"),
            "status": row.get("Состояние"),
        }

        readings_to_save.append(DailyReadingCreate(**reading_data))

    # Bulk insert
    if readings_to_save:
        readings = [DailyReading(**r.model_dump()) for r in readings_to_save]
        db.add_all(readings)
        await db.commit()

        logger.info("Saved readings", count=len(readings))
