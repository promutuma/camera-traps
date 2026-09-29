"""
Shared Darwin Core / Wildlife Insights formatting helpers.

Used by both backend/routers/exports.py (per-detection export) and
backend/routers/ecological.py (per-IDE-event export) so field definitions
and defaults — scientificName mapping, countryCode default, camera ID
resolution, etc. — can't drift out of sync between the two independent
export paths again. The two routers still fetch differently-shaped source
data (raw per-image SQL rows vs. aggregated IDE-event rows), but both build
their final records through these same functions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def darwin_core_row(
    *,
    occurrence_id: Any,
    event_date: Optional[str],
    detected_animal: Optional[str],
    confidence: Any,
    latitude: Any,
    longitude: Any,
    filename: Optional[str],
    station_id: Optional[str],
    camera_id: Optional[str] = None,
    verification_status: Any = None,
    country_code: str = "ET",
) -> Dict[str, Any]:
    """Build one Darwin Core occurrence record."""
    scientific_name = "Animalia" if detected_animal == "Animal" else (detected_animal or "Biota")
    return {
        "occurrenceID": occurrence_id,
        "basisOfRecord": "MachineObservation",
        "eventDate": event_date,
        "vernacularName": detected_animal,
        "scientificName": scientific_name,
        "scientificNameConfidence": confidence,
        "decimalLatitude": latitude,
        "decimalLongitude": longitude,
        "geodeticDatum": "WGS84",
        "associatedMedia": filename,
        "locality": station_id,
        "cameraID": camera_id,
        "identificationVerificationStatus": verification_status,
        "individualCount": 1,
        "countryCode": country_code,
        "georeferenceProtocol": "Camera Trap GPS",
    }


def darwin_core_csv(rows: List[Dict[str, Any]]) -> str:
    """Rows built by `darwin_core_row` -> CSV text."""
    return pd.DataFrame(rows).to_csv(index=False)


def wildlife_insights_image_entry(
    *,
    image_id: Any,
    filename: Optional[str],
    station_id: Optional[str],
    timestamp: Optional[str],
    camera_id: Optional[str] = None,
    species_common_name: Optional[str] = None,
    confidence: Any = None,
    bounding_box: Any = None,
    ide_id: Any = None,
) -> Dict[str, Any]:
    """Build one Wildlife Insights image entry."""
    observation = None
    if species_common_name:
        observation = {
            "species_common_name": species_common_name,
            "confidence": confidence,
            "bounding_box": bounding_box,
        }
        if ide_id is not None:
            observation["ide_id"] = ide_id
    return {
        "image_id": image_id,
        "filename": filename,
        "station_id": station_id,
        "camera_id": camera_id,
        "timestamp": timestamp,
        "observations": [observation] if observation else [],
    }


def wildlife_insights_package(
    project_details: Optional[Dict[str, Any]],
    deployments: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the full Wildlife Insights upload package."""
    return {
        "version": "1.0",
        "provider": "ViumbeLens AI",
        "project": project_details or {
            "project_name": "ViumbeLens Default Survey",
            "notes": "Generated from default local SQLite database",
        },
        "deployments": deployments,
        "images": images,
    }
