"""
Routers package - API endpoints.
"""
from backend.routers.upload import router as upload_router
from backend.routers.analysis import router as analysis_router
from backend.routers.results import router as results_router
from backend.routers.buildings import router as buildings_router
from backend.routers.export import router as export_router

__all__ = [
    "upload_router",
    "analysis_router",
    "results_router",
    "buildings_router",
    "export_router",
]
