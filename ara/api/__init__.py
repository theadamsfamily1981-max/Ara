"""
ARA API Package

Unified API combining:
- Avatar generation endpoints (src/api)
- TFAN training/metrics endpoints (api/)

Usage:
    from ara.api import create_app

    app = create_app()
    # Run with: uvicorn ara.api:app
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent paths for imports
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Try to import FastAPI
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def create_app(
    title: str = "ARA API",
    version: str = "0.1.0",
    enable_avatar: bool = True,
    enable_tfan: bool = True,
    enable_cors: bool = True,
) -> Optional["FastAPI"]:
    """
    Create unified ARA FastAPI application.

    Args:
        title: API title
        version: API version
        enable_avatar: Include avatar generation endpoints
        enable_tfan: Include TFAN training/metrics endpoints
        enable_cors: Enable CORS middleware

    Returns:
        FastAPI application instance
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title=title,
        version=version,
        description="Unified ARA API for avatar generation and TFAN model operations",
    )

    # Add CORS middleware
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Health check
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "version": version,
            "avatar_enabled": enable_avatar,
            "tfan_enabled": enable_tfan,
        }

    # Include avatar router
    if enable_avatar:
        try:
            from src.api.routes import router as avatar_router
            app.include_router(avatar_router, prefix="/avatar", tags=["avatar"])
        except ImportError:
            pass

    # Include TFAN router
    if enable_tfan:
        try:
            from api.routers.ara_router import router as tfan_router
            app.include_router(tfan_router, prefix="/tfan", tags=["tfan"])
        except ImportError:
            pass

    return app


# Create default app instance
app = None
if FASTAPI_AVAILABLE:
    try:
        app = create_app()
    except Exception:
        pass


__all__ = [
    "create_app",
    "app",
    "FASTAPI_AVAILABLE",
]
