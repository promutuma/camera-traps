"""Pydantic request/response models for the FastAPI app."""

from __future__ import annotations
from typing import Optional, List, Any, Dict
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ConfigResponse(BaseModel):
    enable_ocr: bool
    enable_detection: bool
    enable_day_night: bool
    enable_scrubbing: bool
    enable_low_spec: bool
    cpu_threads: int
    detection_confidence: float
    brightness_threshold: int
    ocr_strip_height: float
    blur_strength: int
    review_confidence_threshold: float
    reviewer_id: str
    default_station_id: str
    independence_window: int
    trap_nights_default: int
    speciesnet_lat: float
    speciesnet_lng: float
    speciesnet_country: str


class ConfigUpdate(BaseModel):
    enable_ocr: Optional[bool] = None
    enable_detection: Optional[bool] = None
    enable_day_night: Optional[bool] = None
    enable_scrubbing: Optional[bool] = None
    enable_low_spec: Optional[bool] = None
    cpu_threads: Optional[int] = None
    detection_confidence: Optional[float] = None
    brightness_threshold: Optional[int] = None
    ocr_strip_height: Optional[float] = None
    blur_strength: Optional[int] = None
    review_confidence_threshold: Optional[float] = None
    reviewer_id: Optional[str] = None
    default_station_id: Optional[str] = None
    independence_window: Optional[int] = None
    trap_nights_default: Optional[int] = None
    speciesnet_lat: Optional[float] = None
    speciesnet_lng: Optional[float] = None
    speciesnet_country: Optional[str] = None


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class JobStatus(BaseModel):
    job_id: str
    status: str          # queued | running | done | error
    total: int
    completed: int
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class DetectionResult(BaseModel):
    image_id: Optional[int] = None
    filename: str
    filepath: Optional[str] = None
    station_id: str
    primary_label: Optional[str] = None
    species_label: Optional[str] = None
    detected_animal: Optional[str] = None
    detection_confidence: Optional[float] = None
    detection_method: Optional[str] = None
    bioclip_confidence: Optional[float] = None
    speciesnet_confidence: Optional[float] = None
    agreement: Optional[str] = None
    model_breakdown: Optional[Any] = None
    day_night: Optional[str] = None
    capture_date: Optional[str] = None
    capture_time: Optional[str] = None
    temperature: Optional[str] = None
    user_notes: Optional[str] = None
    scrubbed: Optional[bool] = False
    scrubbed_path: Optional[str] = None


class ResultUpdate(BaseModel):
    detected_animal: Optional[str] = None
    species_label: Optional[str] = None
    user_notes: Optional[str] = None
    station_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------

class StationCreate(BaseModel):
    station_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    stratum: Optional[str] = None
    habitat: Optional[str] = None
    notes: Optional[str] = None


class StationUpdate(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    stratum: Optional[str] = None
    habitat: Optional[str] = None
    notes: Optional[str] = None


class DeploymentCreate(BaseModel):
    station_id: str
    start_date: str
    end_date: str
    camera_id: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------

class ReviewAction(BaseModel):
    reviewer_id: str
    corrected_label: Optional[str] = None
    notes: Optional[str] = None
    bbox: Optional[List[float]] = None


class BulkFlagRequest(BaseModel):
    filenames: List[str]
    reviewer_id: str = "anonymous"
    notes: str = ""


# ---------------------------------------------------------------------------
# Community observations
# ---------------------------------------------------------------------------

class ObservationCreate(BaseModel):
    observer_name: str
    observation_type: str
    species: Optional[str] = None
    count: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    date: Optional[str] = None
    time: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# ArcGIS
# ---------------------------------------------------------------------------

class ArcGISPushRequest(BaseModel):
    url: str
    token: str
    layer_id: Optional[int] = 0


# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------

class ProjectUpdate(BaseModel):
    project_name: Optional[str] = None
    survey_area: Optional[str] = None
    notes: Optional[str] = None
    thresholds: Optional[Dict[str, Any]] = None
