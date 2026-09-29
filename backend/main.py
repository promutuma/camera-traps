"""
Wildlife Camera Trap Auto-Analyzer — FastAPI backend entry point.
Run in development:  uvicorn backend.main:app --reload --port 8000
Run in production:   uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import os
import sys

# Load .env before anything else so all modules see the right env vars.
# Falls back silently if python-dotenv is not installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import logging
from contextlib import asynccontextmanager
import asyncio

# ---------------------------------------------------------------------------
# Windows-specific env fixes (must happen before any torch/cv2 import)
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import subprocess as _sp
    _nvidia_gpu = False
    try:
        _smi = _sp.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=5, text=True,
        )
        _nvidia_gpu = _smi.returncode == 0 and bool(_smi.stdout.strip())
    except Exception:
        pass
    if not _nvidia_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import multiprocessing as _mp
    _mp.freeze_support()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
from typing import Optional

from backend.models.state import AppState, AppConfig
from backend.routers import (
    config as config_router,
    images as images_router,
    results as results_router,
    statistics as statistics_router,
    history as history_router,
    diagnostics as diagnostics_router,
    ecological as ecological_router,
    qc as qc_router,
    stations as stations_router,
    review as review_router,
    community as community_router,
    spatial as spatial_router,
    species as species_router,
    corridor as corridor_router,
    project as project_router,
    arcgis as arcgis_router,
    exports as exports_router,
    storage as storage_router,
    retrain as retrain_router,
    cameras as cameras_router,
    session as session_router,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blocking model loader — runs in a thread so the event loop stays free
# ---------------------------------------------------------------------------

def _load_all_models(state: AppState, project_root: Path) -> None:
    from backend.services.model_loader import load_all_models
    load_all_models(state, project_root)


def _load_speciesnet(state: AppState) -> None:
    from backend.services.model_loader import load_speciesnet
    load_speciesnet(state)


# ---------------------------------------------------------------------------
# Lifespan — core models at startup; SpeciesNet loads in background
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    state: AppState = app.state.app_state
    project_root = Path(__file__).parent.parent
    logger.info("Loading core AI models and services...")
    try:
        await asyncio.to_thread(_load_all_models, state, project_root)
    except Exception as exc:
        state.models_error = str(exc)
        logger.exception("Failed to load models: %s", exc)

    try:
        from backend.routers.project import _sync_speciesnet_coords
        _sync_speciesnet_coords(state)
    except Exception as exc:
        logger.warning("Could not sync SpeciesNet coords from active project: %s", exc)

    sn_task: Optional[asyncio.Task] = None
    if not state.config.enable_low_spec:
        async def _bg_speciesnet():
            await asyncio.to_thread(_load_speciesnet, state)

        sn_task = asyncio.create_task(_bg_speciesnet())
        logger.info("SpeciesNet download/load started in background.")

    yield

    if sn_task and not sn_task.done():
        sn_task.cancel()
        try:
            await sn_task
        except asyncio.CancelledError:
            pass

    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app_state = AppState()

    application = FastAPI(
        title="ViumbeLens API",
        version="2.0.0",
        lifespan=lifespan,
    )
    application.state.app_state = app_state

    # CORS — read allowed origins from env; falls back to Vite dev server
    _cors_raw = os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    )
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _session_secret = os.environ.get(
        "SESSION_SECRET", "dev-session-secret-change-in-production"
    )
    application.add_middleware(SessionMiddleware, secret_key=_session_secret)

    # API routes
    prefix = "/api"
    application.include_router(session_router.router, prefix=prefix)
    application.include_router(config_router.router, prefix=prefix)
    application.include_router(images_router.router, prefix=prefix)
    application.include_router(results_router.router, prefix=prefix)
    application.include_router(statistics_router.router, prefix=prefix)
    application.include_router(history_router.router, prefix=prefix)
    application.include_router(diagnostics_router.router, prefix=prefix)
    application.include_router(ecological_router.router, prefix=prefix)
    application.include_router(qc_router.router, prefix=prefix)
    application.include_router(stations_router.router, prefix=prefix)
    application.include_router(review_router.router, prefix=prefix)
    application.include_router(community_router.router, prefix=prefix)
    application.include_router(spatial_router.router, prefix=prefix)
    application.include_router(species_router.router, prefix=prefix)
    application.include_router(corridor_router.router, prefix=prefix)
    application.include_router(project_router.router, prefix=prefix)
    application.include_router(arcgis_router.router, prefix=prefix)
    application.include_router(exports_router.router, prefix=prefix)
    application.include_router(storage_router.router, prefix=prefix)
    application.include_router(retrain_router.router, prefix=prefix)
    application.include_router(cameras_router.router, prefix=prefix)

    # Serve built React app in production
    dist_path = Path(__file__).parent.parent / "frontend" / "dist"
    if dist_path.exists():
        application.mount("/assets", StaticFiles(directory=str(dist_path / "assets")), name="assets")

        from fastapi.responses import FileResponse as _FileResponse

        @application.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            return _FileResponse(str(dist_path / "index.html"))

    return application


app = create_app()
