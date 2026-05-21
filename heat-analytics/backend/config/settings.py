"""
Application settings and configuration.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    db_path: str = "/app/data/heat_analytics.db"
    
    # Analytics parameters
    norm_hdd: int = 4500  # Normal heating degree days for Abakan
    anomaly_threshold: float = 3.0  # Standard deviations for anomaly detection
    cluster_k: int = 4  # Number of clusters for K-Means
    
    # Logging
    log_level: str = "INFO"
    
    # Upload settings
    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = [".xlsx", ".xls"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
