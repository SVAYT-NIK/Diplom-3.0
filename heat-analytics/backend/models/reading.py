"""
SQLAlchemy ORM models and Pydantic schemas for daily readings.
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

from backend.services.db import Base


class DailyReading(Base):
    """SQLAlchemy model for daily heat consumption readings."""

    __tablename__ = "daily_readings"

    id = Column(Integer, primary_key=True, index=True)
    building_id = Column(Integer, ForeignKey("buildings.id"), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    
    # Temperature
    t1 = Column(Float, nullable=True)  # Supply temperature
    t2 = Column(Float, nullable=True)  # Return temperature
    
    # Pressure
    p1 = Column(Float, nullable=True)  # Supply pressure
    p2 = Column(Float, nullable=True)  # Return pressure
    
    # Volume flow
    v1 = Column(Float, nullable=True)  # Supply volume
    v2 = Column(Float, nullable=True)  # Return volume
    
    # Mass flow
    m1 = Column(Float, nullable=True)  # Supply mass
    m2 = Column(Float, nullable=True)  # Return mass
    
    # Heat energy
    q = Column(Float, nullable=True)  # Heat consumption (Gcal)
    
    # Differences
    dt = Column(Float, nullable=True)  # Temperature difference
    dv = Column(Float, nullable=True)  # Volume difference
    dm = Column(Float, nullable=True)  # Mass difference
    
    # Imbalance
    imbalance = Column(Float, nullable=True)
    
    # Non-standard situations codes (comma-separated)
    ns_codes = Column(Text, nullable=True)
    
    # Status
    status = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    building = relationship("Building", back_populates="readings")


class DailyReadingBase(BaseModel):
    """Base Pydantic schema for daily reading data."""

    date: datetime
    t1: Optional[float] = None
    t2: Optional[float] = None
    p1: Optional[float] = None
    p2: Optional[float] = None
    v1: Optional[float] = None
    v2: Optional[float] = None
    m1: Optional[float] = None
    m2: Optional[float] = None
    q: Optional[float] = None
    dt: Optional[float] = None
    dv: Optional[float] = None
    dm: Optional[float] = None
    imbalance: Optional[float] = None
    ns_codes: Optional[str] = None
    status: Optional[str] = None


class DailyReadingCreate(DailyReadingBase):
    """Schema for creating a new daily reading."""

    building_id: int


class DailyReadingResponse(DailyReadingBase):
    """Schema for daily reading response with ID."""

    id: int
    building_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class DailyReadingListResponse(BaseModel):
    """Schema for list of daily readings."""

    readings: List[DailyReadingResponse]
    total: int
