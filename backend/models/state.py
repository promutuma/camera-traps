"""
Shared application state — holds loaded AI models and runtime config.
Instantiated once at startup via the FastAPI lifespan and stored on app.state.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class AppConfig:
    enable_ocr: bool = True
    enable_detection: bool = True
    enable_day_night: bool = True
    enable_scrubbing: bool = True
    enable_low_spec: bool = False
    cpu_threads: int = field(default_factory=lambda: max(1, (os.cpu_count() or 4) // 4))
    detection_confidence: float = 0.35
    brightness_threshold: int = 100
    ocr_strip_height: float = 0.10
    blur_strength: int = 51
    review_confidence_threshold: float = 0.90
    reviewer_id: str = "anonymous"
    default_station_id: str = "Station-1"
    independence_window: int = 30
    trap_nights_default: int = 30
    # SpeciesNet geographic prior — defaults to Kenya (East Africa)
    speciesnet_lat: float = -1.0
    speciesnet_lng: float = 37.0
    speciesnet_country: str = "KEN"


@dataclass
class AppState:
    config: AppConfig = field(default_factory=AppConfig)

    # AI models (populated during lifespan startup)
    ocr_model: Optional[Any] = None
    md_model: Optional[Any] = None          # MegaDetector v5a
    md_v1000_model: Optional[Any] = None   # MegaDetector v1000 (redwood)
    bio_model: Optional[Any] = None         # BioClip classifier
    speciesnet_model: Optional[Any] = None  # Google SpeciesNet classifier
    dn_model: Optional[Any] = None

    # Service objects
    db_manager: Optional[Any] = None
    station_manager: Optional[Any] = None
    review_engine: Optional[Any] = None
    scrubber: Optional[Any] = None
    community_observer: Optional[Any] = None
    species_library: Optional[Any] = None
    spatial_exporter: Optional[Any] = None
    corridor_analyzer: Optional[Any] = None
    project_config: Optional[Any] = None
    arcgis_sync: Optional[Any] = None
    independence_engine: Optional[Any] = None
    qc_engine: Optional[Any] = None

    models_loaded: bool = False
    models_error: Optional[str] = None
