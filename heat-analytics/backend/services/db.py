"""
SQLAlchemy database service for async SQLite operations.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from pathlib import Path

from backend.config.settings import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


# Database engine
engine = None
async_session_maker = None


def get_db_engine():
    """Get or create the database engine."""
    global engine
    
    if engine is None:
        db_path = Path(settings.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        connection_url = f"sqlite+aiosqlite:///{db_path}"
        engine = create_async_engine(
            connection_url,
            echo=False,
            future=True,
        )
    
    return engine


def get_session_maker():
    """Get or create the session maker."""
    global async_session_maker
    
    if async_session_maker is None:
        async_session_maker = async_sessionmaker(
            get_db_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    
    return async_session_maker


async def init_db():
    """Initialize database tables."""
    from backend.models.building import Building
    from backend.models.reading import DailyReading
    from backend.models.analysis import AnalysisResult
    from backend.models.audit import AuditLog
    
    engine = get_db_engine()
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Dependency for getting database session."""
    session_maker = get_session_maker()
    
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
