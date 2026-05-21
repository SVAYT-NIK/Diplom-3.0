"""
SQLAlchemy ORM models and Pydantic schemas for audit logging.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Text
from pydantic import BaseModel, Field

from backend.services.db import Base


class AuditLog(Base):
    """SQLAlchemy model for audit logging."""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user = Column(String(100), nullable=True)  # Can be None for system actions
    action = Column(String(100), nullable=False, index=True)  # upload, analyze, export, etc.
    payload = Column(Text, nullable=True)  # JSON string of action details


class AuditLogBase(BaseModel):
    """Base Pydantic schema for audit log data."""

    user: Optional[str] = None
    action: str
    payload: Optional[Dict[str, Any]] = None


class AuditLogCreate(AuditLogBase):
    """Schema for creating a new audit log entry."""

    pass


class AuditLogResponse(AuditLogBase):
    """Schema for audit log response with ID."""

    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


class AuditLogListResponse(BaseModel):
    """Schema for list of audit logs."""

    logs: List[AuditLogResponse]
    total: int
