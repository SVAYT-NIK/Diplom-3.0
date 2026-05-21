"""
SQLAlchemy ORM models and Pydantic schemas for analysis results.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON

from backend.services.db import Base


class AnalysisResult(Base):
    """SQLAlchemy model for analysis results."""

    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    building_id = Column(Integer, ForeignKey("buildings.id"), nullable=False, index=True)
    run_id = Column(String(100), nullable=False, index=True)  # Unique identifier for analysis run
    
    # Model type: ols, huber, ridge, lasso, quantile, prophet, holt_winters, etc.
    model_type = Column(String(100), nullable=False)
    
    # Predictions and metrics
    predicted_q = Column(Float, nullable=True)
    residual = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    anomaly_flag = Column(Boolean, default=False)
    
    # Clustering
    cluster_id = Column(Integer, nullable=True)
    efficiency_class = Column(String(10), nullable=True)  # A, B, C, D, E
    norm_deviation_pct = Column(Float, nullable=True)
    
    # Model parameters stored as JSON
    params = Column(Text, nullable=True)  # JSON string of hyperparameters
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    building = relationship("Building", back_populates="analysis_results")


class AnalysisResultBase(BaseModel):
    """Base Pydantic schema for analysis result data."""

    run_id: str
    model_type: str
    predicted_q: Optional[float] = None
    residual: Optional[float] = None
    anomaly_score: Optional[float] = None
    anomaly_flag: bool = False
    cluster_id: Optional[int] = None
    efficiency_class: Optional[str] = None
    norm_deviation_pct: Optional[float] = None
    params: Optional[Dict[str, Any]] = None


class AnalysisResultCreate(AnalysisResultBase):
    """Schema for creating a new analysis result."""

    building_id: int


class AnalysisResultResponse(AnalysisResultBase):
    """Schema for analysis result response with ID."""

    id: int
    building_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisRunSummary(BaseModel):
    """Schema for summarizing an analysis run."""

    run_id: str
    building_id: int
    total_records: int
    anomaly_count: int
    models_used: List[str]
    avg_efficiency_class: Optional[str]
    avg_norm_deviation: Optional[float]
    created_at: datetime


class AnalysisResultsListResponse(BaseModel):
    """Schema for list of analysis results."""

    results: List[AnalysisResultResponse]
    total: int
    run_summary: Optional[AnalysisRunSummary] = None
