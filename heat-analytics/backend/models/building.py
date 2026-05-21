"""
SQLAlchemy ORM models and Pydantic schemas for buildings.
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

from backend.services.db import Base


class Building(Base):
    """SQLAlchemy model for multi-apartment buildings."""

    __tablename__ = "buildings"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String(500), nullable=False, index=True)
    area_m2 = Column(Float, nullable=False)
    year_built = Column(Integer, nullable=True)
    heating_type = Column(String(100), nullable=True)
    norm_gcal_m2 = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    readings = relationship("DailyReading", back_populates="building", cascade="all, delete-orphan")
    analysis_results = relationship("AnalysisResult", back_populates="building", cascade="all, delete-orphan")


class BuildingBase(BaseModel):
    """Base Pydantic schema for building data."""

    address: str = Field(..., min_length=1, max_length=500)
    area_m2: float = Field(..., gt=0)
    year_built: Optional[int] = Field(None, ge=1800, le=2030)
    heating_type: Optional[str] = Field(None, max_length=100)
    norm_gcal_m2: Optional[float] = Field(None, gt=0)


class BuildingCreate(BuildingBase):
    """Schema for creating a new building."""

    pass


class BuildingResponse(BuildingBase):
    """Schema for building response with ID and timestamps."""

    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class BuildingListResponse(BaseModel):
    """Schema for list of buildings."""

    buildings: List[BuildingResponse]
    total: int
