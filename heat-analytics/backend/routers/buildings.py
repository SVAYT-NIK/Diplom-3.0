"""
Buildings router for CRUD operations.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import structlog

from backend.services.db import get_db
from backend.models.building import (
    Building,
    BuildingCreate,
    BuildingResponse,
    BuildingListResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/buildings", response_model=BuildingListResponse)
async def get_buildings(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    """
    Get list of all buildings with pagination.

    Returns:
        List of buildings with metadata
    """
    # Get total count
    count_result = await db.execute(select(func.count()).select_from(Building))
    total = count_result.scalar() or 0

    # Get buildings
    result = await db.execute(select(Building).offset(skip).limit(limit))
    buildings = result.scalars().all()

    return BuildingListResponse(
        buildings=[
            BuildingResponse(
                id=b.id,
                address=b.address,
                area_m2=b.area_m2,
                year_built=b.year_built,
                heating_type=b.heating_type,
                norm_gcal_m2=b.norm_gcal_m2,
                created_at=b.created_at,
            )
            for b in buildings
        ],
        total=total,
    )


@router.get("/buildings/{building_id}", response_model=BuildingResponse)
async def get_building(
    building_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific building by ID.

    Args:
        building_id: Building ID

    Returns:
        Building information
    """
    result = await db.execute(
        select(Building).where(Building.id == building_id)
    )
    building = result.scalar_one_or_none()

    if not building:
        raise HTTPException(status_code=404, detail="Building not found")

    return BuildingResponse(
        id=building.id,
        address=building.address,
        area_m2=building.area_m2,
        year_built=building.year_built,
        heating_type=building.heating_type,
        norm_gcal_m2=building.norm_gcal_m2,
        created_at=building.created_at,
    )


@router.post("/buildings", response_model=BuildingResponse)
async def create_building(
    building_data: BuildingCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new building manually.

    Args:
        building_data: Building information

    Returns:
        Created building information
    """
    building = Building(**building_data.model_dump())
    db.add(building)
    await db.commit()
    await db.refresh(building)

    logger.info("Created building manually", building_id=building.id)

    return BuildingResponse(
        id=building.id,
        address=building.address,
        area_m2=building.area_m2,
        year_built=building.year_built,
        heating_type=building.heating_type,
        norm_gcal_m2=building.norm_gcal_m2,
        created_at=building.created_at,
    )


@router.put("/buildings/{building_id}", response_model=BuildingResponse)
async def update_building(
    building_id: int,
    building_data: BuildingCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Update an existing building.

    Args:
        building_id: Building ID
        building_data: Updated building information

    Returns:
        Updated building information
    """
    result = await db.execute(
        select(Building).where(Building.id == building_id)
    )
    building = result.scalar_one_or_none()

    if not building:
        raise HTTPException(status_code=404, detail="Building not found")

    # Update fields
    for key, value in building_data.model_dump(exclude_unset=True).items():
        setattr(building, key, value)

    await db.commit()
    await db.refresh(building)

    logger.info("Updated building", building_id=building.id)

    return BuildingResponse(
        id=building.id,
        address=building.address,
        area_m2=building.area_m2,
        year_built=building.year_built,
        heating_type=building.heating_type,
        norm_gcal_m2=building.norm_gcal_m2,
        created_at=building.created_at,
    )


@router.delete("/buildings/{building_id}")
async def delete_building(
    building_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a building and all associated data.

    Args:
        building_id: Building ID
    """
    result = await db.execute(
        select(Building).where(Building.id == building_id)
    )
    building = result.scalar_one_or_none()

    if not building:
        raise HTTPException(status_code=404, detail="Building not found")

    await db.delete(building)
    await db.commit()

    logger.info("Deleted building", building_id=building_id)

    return {"message": "Building deleted successfully"}
