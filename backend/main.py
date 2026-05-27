"""
Wildlife Camera Trap Auto-Analyzer — FastAPI backend entry point.
Run in development:  uvicorn backend.main:app --reload --port 8000
Run in production:   uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import os
import sys
import logging
from contextlib import asynccontextmanager

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
from pathlib import Path

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
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — load all models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    state: AppState = app.state.app_state
    logger.info("Loading AI models and services...")

    try:
        # Add project root to sys.path so `core/` is importable
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from core.ocr_processor import OCRProcessor
        from core.animal_detector import AnimalDetector, MegaDetectorWrapper
        from core.bioclip_classifier import BioClipClassifier
        from core.day_night_classifier import DayNightClassifier
        from core.db_manager import DatabaseManager
        from core.station_manager import StationManager
        from core.review_engine import ReviewEngine
        from core.privacy_scrubber import PrivacyScrubber
        from core.community_observer import CommunityObserver
        from core.species_library import SpeciesLibrary
        from core.spatial_exporter import SpatialExporter
        from core.corridor_analyzer import CorridorAnalyzer
        from core.project_config import ProjectConfig
        from core.arcgis_sync import ArcGISSync
        from core.independence_engine import IndependenceEngine
        from core.qc_engine import QCEngine

        cfg = state.config

        # Set PyTorch thread count
        try:
            import torch
            _threads = 1 if cfg.enable_low_spec else cfg.cpu_threads
            torch.set_num_threads(_threads)
        except Exception:
            pass

        logger.info("Loading OCR...")
        state.ocr_model = OCRProcessor(low_spec=cfg.enable_low_spec)

        logger.info("Loading MegaDetector...")
        state.md_model = MegaDetectorWrapper(
            confidence_threshold=cfg.detection_confidence,
            low_spec=cfg.enable_low_spec,
        )

        logger.info("Loading BioClip...")
        state.bio_model = BioClipClassifier(
            species_list=AnimalDetector.WILDLIFE_CLASSES,
            low_spec=cfg.enable_low_spec,
        )

        logger.info("Loading Day/Night classifier...")
        state.dn_model = DayNightClassifier()

        # Services
        state.db_manager = DatabaseManager()
        state.station_manager = StationManager()
        state.review_engine = ReviewEngine()
        state.scrubber = PrivacyScrubber(blur_strength=cfg.blur_strength)
        state.community_observer = CommunityObserver()
        state.species_library = SpeciesLibrary()
        state.spatial_exporter = SpatialExporter()
        state.corridor_analyzer = CorridorAnalyzer()
        state.project_config = ProjectConfig()
        state.arcgis_sync = ArcGISSync()
        state.independence_engine = IndependenceEngine(window_minutes=cfg.independence_window)
        state.qc_engine = QCEngine()

        state.models_loaded = True
        logger.info("All models loaded successfully.")

    except Exception as exc:
        state.models_error = str(exc)
        logger.exception("Failed to load models: %s", exc)

    yield

    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app_state = AppState()

    application = FastAPI(
        title="WildlifeID Pro API",
        version="2.0.0",
        lifespan=lifespan,
    )
    application.state.app_state = app_state

    # CORS — allow the Vite dev server
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    prefix = "/api"
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

    # Serve built React app in production
    dist_path = Path(__file__).parent.parent / "frontend" / "dist"
    if dist_path.exists():
        # Serve static assets (JS/CSS/images) directly
        application.mount("/assets", StaticFiles(directory=str(dist_path / "assets")), name="assets")

        # SPA catch-all: any non-API path returns index.html so React Router works
        from fastapi.responses import FileResponse as _FileResponse

        @application.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            index = dist_path / "index.html"
            return _FileResponse(str(index))

    return application


app = create_app()
