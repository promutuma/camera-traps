"""Tab 4 — Analysis History."""

import io

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.models.state import AppState
from backend.routers.deps import get_state
from core.results_exporter import (
    build_export_filename,
    build_export_metadata_rows,
    compute_export_stats,
    content_disposition_attachment,
    detections_csv_bytes,
    prepare_images_export,
)

router = APIRouter(prefix="/history", tags=["history"])


def _project_meta(state: AppState):
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


@router.get("")
def get_history(
    state: AppState = Depends(get_state),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    df = state.db_manager.get_history_df(limit=limit, offset=offset)
    return df.fillna("").to_dict(orient="records")


@router.delete("")
def clear_history(state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    state.db_manager.clear_history()
    return {"ok": True}


@router.get("/export/csv")
def export_csv(state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    raw_df = state.db_manager.get_history_df()
    export_df = prepare_images_export(raw_df)
    project = _project_meta(state)
    stats = compute_export_stats(raw_df)
    filename = build_export_filename(
        export_kind="analysis-history",
        file_format="csv",
        project_name=project["name"],
        project_area=project.get("area"),
        stats=stats,
    )
    metadata_rows = build_export_metadata_rows(
        export_kind="analysis-history",
        file_format="csv",
        filename=filename,
        project_name=project["name"],
        project_area=project.get("area"),
        stats=stats,
    )
    return StreamingResponse(
        io.BytesIO(detections_csv_bytes(export_df, metadata_rows=metadata_rows)),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": content_disposition_attachment(filename)},
    )
