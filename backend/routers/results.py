"""Tab 2 — Review Results: fetch, edit, export."""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.models.state import AppState
from backend.models.schemas import ResultUpdate
from backend.routers.deps import get_state
from core.results_exporter import (
    build_export_filename,
    build_export_info_dict,
    build_export_metadata_rows,
    build_json_export,
    compute_export_stats,
    content_disposition_attachment,
    detections_csv_bytes,
    detections_excel_bytes,
    detections_json_bytes,
    prepare_detections_detail_export,
    prepare_images_export,
)

router = APIRouter(prefix="/results", tags=["results"])


def _fetch_export_df(
    state: AppState,
    species: Optional[str] = None,
    day_night: Optional[str] = None,
    min_conf: Optional[float] = None,
    max_conf: Optional[float] = None,
    station: Optional[str] = None,
):
    return state.db_manager.get_history_df(
        station_id=station or None,
        species=species or None,
        day_night=day_night or None,
        min_conf=min_conf,
        max_conf=max_conf,
    )


def _export_filters(
    species: Optional[str],
    day_night: Optional[str],
    min_conf: Optional[float],
    max_conf: Optional[float],
    station: Optional[str],
) -> Dict[str, Any]:
    return {
        k: v
        for k, v in {
            "species": species,
            "day_night": day_night,
            "min_conf": min_conf,
            "max_conf": max_conf,
            "station": station,
        }.items()
        if v is not None
    }


def _mark_exported(state: AppState, raw_df, export_type: str) -> None:
    try:
        if raw_df is not None and not raw_df.empty and "detection_id" in raw_df.columns:
            detection_ids = [int(x) for x in raw_df["detection_id"].dropna().tolist()]
            if detection_ids:
                state.db_manager.mark_exports_as_exported(detection_ids, export_type)
    except Exception as e:
        logging.warning("Could not mark detections as exported: %s", e)


def _active_project_meta(state: AppState) -> Dict[str, Optional[str]]:
    try:
        if state.project_config:
            project = state.project_config.get_active_project()
            if project:
                return {
                    "name": str(project.get("name") or "project"),
                    "area": str(project.get("survey_area") or "").strip() or None,
                }
    except Exception:
        pass
    return {"name": "project", "area": None}


def _build_export_bundle(
    state: AppState,
    raw_df,
    export_kind: str,
    file_format: str,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    project = _active_project_meta(state)
    stats = compute_export_stats(raw_df)
    active_filters = filters or None
    filename = build_export_filename(
        export_kind=export_kind,
        file_format=file_format,
        project_name=project["name"],
        project_area=project.get("area"),
        stats=stats,
        filters=active_filters,
    )
    metadata_rows = build_export_metadata_rows(
        export_kind=export_kind,
        file_format=file_format,
        filename=filename,
        project_name=project["name"],
        project_area=project.get("area"),
        stats=stats,
        filters=active_filters,
    )
    export_info = build_export_info_dict(
        export_kind=export_kind,
        file_format=file_format,
        filename=filename,
        project_name=project["name"],
        project_area=project.get("area"),
        stats=stats,
        filters=active_filters,
    )
    return {
        "filename": filename,
        "metadata_rows": metadata_rows,
        "export_info": export_info,
    }


@router.get("")
def get_results(
    state: AppState = Depends(get_state),
    species: Optional[str] = Query(None),
    day_night: Optional[str] = Query(None),
    min_conf: Optional[float] = Query(None),
    max_conf: Optional[float] = Query(None),
    station: Optional[str] = Query(None),
    limit: int = Query(5000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    df = state.db_manager.get_history_df(
        station_id=station or None,
        species=species or None,
        day_night=day_night or None,
        min_conf=min_conf,
        max_conf=max_conf,
        limit=limit,
        offset=offset,
    )
    if df.empty:
        return {"total": 0, "limit": limit, "offset": offset, "items": []}

    total = state.db_manager.count_history_df(
        station_id=station or None,
        species=species or None,
        day_night=day_night or None,
        min_conf=min_conf,
        max_conf=max_conf,
    )
    return {"total": total, "limit": limit, "offset": offset, "items": df.fillna("").to_dict(orient="records")}


@router.patch("/{detection_id}")
def update_result(detection_id: int, update: ResultUpdate, state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    fields = update.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    state.db_manager.update_detection(detection_id, fields)
    return {"ok": True, "detection_id": detection_id}


@router.delete("/{detection_id}")
def delete_result(detection_id: int, state: AppState = Depends(get_state)):
    """Delete a detection result. If it's the last detection for an image, mark image for deletion."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    try:
        success = state.db_manager.delete_detection(detection_id)
        if not success:
            raise HTTPException(status_code=404, detail="Detection not found")
        return {"ok": True, "deleted_id": detection_id, "message": "Detection deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("")
def delete_results_batch(state: AppState = Depends(get_state), detection_ids: str = Query(...)):
    """Delete multiple detection results (CSV of IDs)."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    try:
        ids = [int(x.strip()) for x in detection_ids.split(",") if x.strip()]
        if not ids:
            raise HTTPException(status_code=400, detail="No detection IDs provided")
        count = state.db_manager.delete_detections_batch(ids)
        return {"ok": True, "deleted_count": count, "message": f"Deleted {count} detections"}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid detection IDs")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/excel")
def export_excel(
    state: AppState = Depends(get_state),
    species: Optional[str] = Query(None),
    day_night: Optional[str] = Query(None),
    min_conf: Optional[float] = Query(None),
    max_conf: Optional[float] = Query(None),
    station: Optional[str] = Query(None),
):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")

    raw_df = _fetch_export_df(state, species, day_night, min_conf, max_conf, station)
    filters = _export_filters(species, day_night, min_conf, max_conf, station)
    bundle = _build_export_bundle(state, raw_df, "results", "xlsx", filters)
    images_df = prepare_images_export(raw_df)
    detections_df = prepare_detections_detail_export(raw_df)
    _mark_exported(state, raw_df, "excel")

    return StreamingResponse(
        io.BytesIO(detections_excel_bytes(
            images_df, detections_df, raw_df, metadata_rows=bundle["metadata_rows"],
        )),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": content_disposition_attachment(bundle["filename"])},
    )


@router.get("/export/csv")
def export_csv(
    state: AppState = Depends(get_state),
    species: Optional[str] = Query(None),
    day_night: Optional[str] = Query(None),
    min_conf: Optional[float] = Query(None),
    max_conf: Optional[float] = Query(None),
    station: Optional[str] = Query(None),
):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")

    raw_df = _fetch_export_df(state, species, day_night, min_conf, max_conf, station)
    filters = _export_filters(species, day_night, min_conf, max_conf, station)
    bundle = _build_export_bundle(state, raw_df, "results", "csv", filters)
    images_df = prepare_images_export(raw_df)
    _mark_exported(state, raw_df, "csv")

    return StreamingResponse(
        io.BytesIO(detections_csv_bytes(images_df, metadata_rows=bundle["metadata_rows"])),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": content_disposition_attachment(bundle["filename"])},
    )


@router.get("/export/json")
def export_json(
    state: AppState = Depends(get_state),
    species: Optional[str] = Query(None),
    day_night: Optional[str] = Query(None),
    min_conf: Optional[float] = Query(None),
    max_conf: Optional[float] = Query(None),
    station: Optional[str] = Query(None),
):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")

    raw_df = _fetch_export_df(state, species, day_night, min_conf, max_conf, station)
    filters = _export_filters(species, day_night, min_conf, max_conf, station)
    bundle = _build_export_bundle(state, raw_df, "results", "json", filters)
    images_df = prepare_images_export(raw_df)
    payload = build_json_export(
        raw_df,
        images_df,
        filters=filters,
        export_info=bundle["export_info"],
    )
    _mark_exported(state, raw_df, "json")

    return StreamingResponse(
        io.BytesIO(detections_json_bytes(payload)),
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": content_disposition_attachment(bundle["filename"])},
    )
