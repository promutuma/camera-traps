"""FastAPI Router for Darwin Core and Wildlife Insights standardized exports."""

import io
import json
import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from backend.models.state import AppState
from backend.routers.deps import get_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/exports", tags=["exports"])


@router.get("/darwin-core")
def export_darwin_core(state: AppState = Depends(get_state)):
    """Export camera trap data formatted to the Darwin Core Standard (CSV)."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="Database manager not ready")

    # Mark detections as exported
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
            state.db_manager.mark_exports_as_exported(detection_ids, 'darwin_core')
    except Exception as e:
        logger.warning(f"Could not mark detections as exported: {e}")

    conn = state.db_manager.get_connection()
    try:
        # Join images, detections, stations, and optionally review actions to see human corrections
        query = """
            SELECT 
                d.id AS occurrenceID,
                'MachineObservation' AS basisOfRecord,
                (i.capture_date || 'T' || i.capture_time) AS eventDate,
                d.detected_animal AS vernacularName,
                d.confidence AS scientificNameConfidence,
                s.latitude AS decimalLatitude,
                s.longitude AS decimalLongitude,
                'WGS84' AS geodeticDatum,
                i.filename AS associatedMedia,
                i.station_id AS locality,
                d.agreement AS identificationVerificationStatus
            FROM images i
            JOIN detections d ON i.id = d.image_id
            LEFT JOIN stations s ON i.station_id = s.station_id
            ORDER BY i.capture_date DESC, i.capture_time DESC
        """
        df = pd.read_sql_query(query, conn)
        
        # Add basic Darwin Core fields
        df["scientificName"] = df["vernacularName"].apply(
            lambda x: "Animalia" if x == "Animal" else (x or "Biota")
        )
        df["individualCount"] = 1
        df["countryCode"] = "ET"  # Default to Ethiopia (Gambella)
        df["georeferenceProtocol"] = "Camera Trap GPS"

        # Stream CSV response
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=viumbelens_darwin_core.csv"},
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    finally:
        conn.close()


@router.get("/wildlife-insights")
def export_wildlife_insights(state: AppState = Depends(get_state)):
    """Export camera trap data matching the Wildlife Insights batch upload schema (JSON)."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="Database manager not ready")

    # Mark detections as exported
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
            state.db_manager.mark_exports_as_exported(detection_ids, 'wildlife_insights')
    except Exception as e:
        logger.warning(f"Could not mark detections as exported: {e}")

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
                i.capture_date,
                i.capture_time,
                d.detected_animal,
                d.confidence,
                d.bbox
            FROM images i
            LEFT JOIN detections d ON i.id = d.image_id
        """
        images_df = pd.read_sql_query(images_query, conn)

        # Structure the Wildlife Insights upload package
        insights_package = {
            "version": "1.0",
            "provider": "ViumbeLens AI",
            "project": project_details or {
                "project_name": "ViumbeLens Default Survey",
                "notes": "Generated from default local SQLite database"
            },
            "deployments": deployments_df.to_dict(orient="records") if not deployments_df.empty else [],
            "images": []
        }

        for _, row in images_df.iterrows():
            # Parse bbox JSON if exists
            bbox_data = None
            if row["bbox"]:
                try:
                    bbox_data = json.loads(row["bbox"])
                except Exception:
                    pass

            insights_package["images"].append({
                "image_id": row["image_id"],
                "filename": row["filename"],
                "station_id": row["station_id"],
                "timestamp": f"{row['capture_date']}T{row['capture_time']}" if row["capture_date"] else None,
                "observations": [
                    {
                        "species_common_name": row["detected_animal"],
                        "confidence": row["confidence"],
                        "bounding_box": bbox_data
                    }
                ] if row["detected_animal"] else []
            })

        return JSONResponse(
            content=insights_package,
            headers={"Content-Disposition": "attachment; filename=viumbelens_wildlife_insights.json"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    finally:
        conn.close()
