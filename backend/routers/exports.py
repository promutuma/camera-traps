"""FastAPI Router for Darwin Core and Wildlife Insights standardized exports."""

import json
import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from backend.models.state import AppState
from backend.routers.deps import get_state
from core.standard_exports import (
    darwin_core_csv,
    darwin_core_row,
    wildlife_insights_image_entry,
    wildlife_insights_package,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/exports", tags=["exports"])


def _mark_exported(state: AppState, export_type: str) -> None:
    try:
        conn = state.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id FROM detections
            WHERE is_exported = 0 AND detected_animal IS NOT NULL
        ''')
        detection_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        if detection_ids:
            state.db_manager.mark_exports_as_exported(detection_ids, export_type)
    except Exception as e:
        logger.warning(f"Could not mark detections as exported: {e}")


@router.get("/darwin-core")
def export_darwin_core(state: AppState = Depends(get_state)):
    """Export camera trap data formatted to the Darwin Core Standard (CSV)."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="Database manager not ready")

    _mark_exported(state, "darwin_core")

    conn = state.db_manager.get_connection()
    try:
        query = """
            SELECT
                d.id AS occurrenceID,
                (i.capture_date || 'T' || i.capture_time) AS eventDate,
                i.capture_date AS captureDate,
                i.station_id AS station_id,
                i.camera_id AS camera_id,
                d.detected_animal AS vernacularName,
                d.confidence AS scientificNameConfidence,
                s.gps_lat AS decimalLatitude,
                s.gps_lon AS decimalLongitude,
                i.filename AS associatedMedia,
                d.agreement AS identificationVerificationStatus
            FROM images i
            JOIN detections d ON i.id = d.image_id
            LEFT JOIN stations s ON i.station_id = s.station_id
            ORDER BY i.capture_date DESC, i.capture_time DESC
        """
        df = pd.read_sql_query(query, conn)

        # Camera explicitly selected at upload time (images.camera_id) takes
        # priority; deployment-window inference (deployments.sd_card_id)
        # fills the gaps for images uploaded without one selected.
        camera_ids = (
            state.station_manager.resolve_camera_ids(
                df, station_col="station_id", date_col="captureDate", existing_col="camera_id"
            )
            if state.station_manager is not None
            else pd.Series([None] * len(df))
        )

        rows = [
            darwin_core_row(
                occurrence_id=row["occurrenceID"],
                event_date=row["eventDate"],
                detected_animal=row["vernacularName"],
                confidence=row["scientificNameConfidence"],
                latitude=row["decimalLatitude"],
                longitude=row["decimalLongitude"],
                filename=row["associatedMedia"],
                station_id=row["station_id"],
                camera_id=camera_ids.loc[idx] if pd.notna(camera_ids.loc[idx]) else None,
                verification_status=row["identificationVerificationStatus"],
            )
            for idx, row in df.iterrows()
        ]
        csv_text = darwin_core_csv(rows)

        return StreamingResponse(
            iter([csv_text]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=viumbelens_darwin_core.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    finally:
        conn.close()


@router.get("/wildlife-insights")
def export_wildlife_insights(state: AppState = Depends(get_state)):
    """Export camera trap data matching the Wildlife Insights batch upload schema (JSON)."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="Database manager not ready")

    _mark_exported(state, "wildlife_insights")

    conn = state.db_manager.get_connection()
    try:
        # Retrieve active project details
        project_details = {}
        if state.project_config:
            active_proj = state.project_config.get_active_project()
            if active_proj:
                project_details = {
                    "project_id": active_proj.get("id"),
                    "project_name": active_proj.get("name"),
                    "survey_area": active_proj.get("survey_area"),
                    "notes": active_proj.get("notes"),
                }

        # Retrieve deployments
        deployments_df = pd.read_sql_query("SELECT * FROM deployments", conn) \
            if "deployments" in pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].values \
            else pd.DataFrame()

        # Retrieve images and detections
        images_query = """
            SELECT
                i.id AS image_id,
                i.filename,
                i.station_id,
                i.camera_id,
                i.capture_date,
                i.capture_time,
                d.detected_animal,
                d.confidence,
                d.bbox
            FROM images i
            LEFT JOIN detections d ON i.id = d.image_id
        """
        images_df = pd.read_sql_query(images_query, conn)

        # Camera explicitly selected at upload time takes priority;
        # deployment-window inference fills the gaps — resolved per-image so
        # consumers don't have to cross-reference the deployments list
        # against station_id + date themselves.
        camera_ids = (
            state.station_manager.resolve_camera_ids(
                images_df, station_col="station_id", date_col="capture_date", existing_col="camera_id"
            )
            if state.station_manager is not None
            else pd.Series([None] * len(images_df))
        )

        images = []
        for idx, row in images_df.iterrows():
            bbox_data = None
            if row["bbox"]:
                try:
                    bbox_data = json.loads(row["bbox"])
                except Exception:
                    pass

            images.append(wildlife_insights_image_entry(
                image_id=row["image_id"],
                filename=row["filename"],
                station_id=row["station_id"],
                camera_id=camera_ids.loc[idx] if pd.notna(camera_ids.loc[idx]) else None,
                timestamp=f"{row['capture_date']}T{row['capture_time']}" if row["capture_date"] else None,
                species_common_name=row["detected_animal"],
                confidence=row["confidence"],
                bounding_box=bbox_data,
            ))

        insights_package = wildlife_insights_package(
            project_details,
            deployments_df.to_dict(orient="records") if not deployments_df.empty else [],
            images,
        )

        return JSONResponse(
            content=insights_package,
            headers={"Content-Disposition": "attachment; filename=viumbelens_wildlife_insights.json"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    finally:
        conn.close()
