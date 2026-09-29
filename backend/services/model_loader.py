"""Load / reload ML models shared by startup and config updates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from backend.models.state import AppState

logger = logging.getLogger(__name__)


def load_speciesnet(state: AppState) -> None:
    """Load SpeciesNet unless Low-Spec mode is enabled."""
    cfg = state.config
    if state.speciesnet_loading:
        return
    if cfg.enable_low_spec:
        state.speciesnet_model = None
        state.speciesnet_error = "SpeciesNet disabled (Low-Spec mode)"
        return

    state.speciesnet_loading = True
    state.speciesnet_error = None
    try:
        from core.speciesnet_classifier import SpeciesNetWrapper

        logger.info("Loading SpeciesNet classifier...")
        state.speciesnet_model = SpeciesNetWrapper(
            low_spec=False,
            lat=cfg.speciesnet_lat,
            lng=cfg.speciesnet_lng,
            country=cfg.speciesnet_country,
        )
        if state.speciesnet_model.classifier is None:
            state.speciesnet_error = state.speciesnet_model.load_error or "SpeciesNet failed to load"
            logger.warning("SpeciesNet unavailable: %s", state.speciesnet_error)
        else:
            logger.info("SpeciesNet loaded.")
            _restore_retrain_layer(state)
    except Exception as exc:
        state.speciesnet_model = None
        state.speciesnet_error = str(exc)
        logger.warning("SpeciesNet failed to load (non-fatal): %s", exc)
    finally:
        state.speciesnet_loading = False


def unload_speciesnet(state: AppState) -> None:
    """Release SpeciesNet to save RAM (Low-Spec mode)."""
    state.speciesnet_model = None
    state.speciesnet_error = "SpeciesNet disabled (Low-Spec mode)"
    state.speciesnet_loading = False
    logger.info("SpeciesNet unloaded (Low-Spec mode).")


def reload_speciesnet(state: AppState) -> None:
    """Reload or unload SpeciesNet based on current config."""
    if state.config.enable_low_spec:
        unload_speciesnet(state)
    else:
        load_speciesnet(state)


def _restore_retrain_layer(state: AppState) -> None:
    if not state.db_manager or not state.retrain_engine or not state.speciesnet_model:
        return
    try:
        active = state.db_manager.get_active_retrain_run()
        if active and active.get("job_id"):
            artifact = state.retrain_engine.load_artifact(active["job_id"])
            if artifact:
                state.speciesnet_model.set_correction_layer(artifact)
                logger.info("Restored active retrain correction layer: %s", active["job_id"])
    except Exception as exc:
        logger.warning("Could not restore active retrain model: %s", exc)


def load_all_models(state: AppState, project_root: Path) -> None:
    """Blocking loader for OCR, MegaDetector, Day/Night, and services."""
    import os
    import sys
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*no accelerator.*",
        category=UserWarning,
    )

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from core.ocr_processor import OCRProcessor
    from core.animal_detector import MegaDetectorWrapper
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
    from core.retrain_engine import RetrainEngine
    from backend.services.file_manager import FileManager

    cfg = state.config

    try:
        import torch
        _threads = 1 if cfg.enable_low_spec else cfg.cpu_threads
        torch.set_num_threads(_threads)
    except Exception:
        pass

    logger.info("Loading OCR...")
    try:
        state.ocr_model = OCRProcessor(low_spec=cfg.enable_low_spec)
    except Exception as exc:
        state.ocr_model = None
        logger.warning("OCR failed to load (non-fatal): %s", exc)

    logger.info("Loading MegaDetector...")
    state.md_model = MegaDetectorWrapper(
        confidence_threshold=cfg.detection_confidence,
        low_spec=cfg.enable_low_spec,
    )

    logger.info("Loading Day/Night classifier...")
    state.dn_model = DayNightClassifier()

    db_path = os.environ.get("DB_PATH", "wildlife_data.db")
    state.db_manager = DatabaseManager(db_path)

    uploads_dir = Path(db_path).parent / "uploads"
    state.file_manager = FileManager(uploads_dir, state.db_manager)
    state.file_manager.reconcile_missing_files()

    scrubbed_dir = uploads_dir / "scrubbed"
    scrubbed_dir.mkdir(exist_ok=True)
    state.scrubber = PrivacyScrubber(
        output_dir=str(scrubbed_dir),
        blur_strength=cfg.blur_strength,
        audit_path=str(scrubbed_dir / "privacy_audit.json"),
    )
    state.station_manager = StationManager(db_path)
    state.review_engine = ReviewEngine(db_path)
    state.community_observer = CommunityObserver(db_path)
    state.species_library = SpeciesLibrary(db_path)
    state.spatial_exporter = SpatialExporter()
    state.corridor_analyzer = CorridorAnalyzer()
    state.project_config = ProjectConfig(db_path)
    state.arcgis_sync = ArcGISSync(db_path=db_path)
    state.independence_engine = IndependenceEngine(window_minutes=cfg.independence_window)
    state.qc_engine = QCEngine()
    state.retrain_engine = RetrainEngine(db_manager=state.db_manager, uploads_dir=uploads_dir)

    state.models_loaded = True
    logger.info("Core models and services loaded.")
