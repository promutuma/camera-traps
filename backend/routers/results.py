"""Tab 2 — Review Results: fetch, edit, export."""

from __future__ import annotations
import io
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.models.state import AppState
from backend.models.schemas import ResultUpdate
from backend.routers.deps import get_state

router = APIRouter(prefix="/results", tags=["results"])


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
    )
    if df.empty:
        return {"total": 0, "limit": limit, "offset": offset, "items": []}

    total = len(df)
    page = df.iloc[offset: offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "items": page.fillna("").to_dict(orient="records")}


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
def export_excel(state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")

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
            state.db_manager.mark_exports_as_exported(detection_ids, 'excel')
    except Exception as e:
        import logging
        logging.warning(f"Could not mark detections as exported: {e}")

    import pandas as pd
    df = state.db_manager.get_history_df()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=wildlife_results.xlsx"},
    )


@router.get("/export/csv")
def export_csv(state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")

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
            state.db_manager.mark_exports_as_exported(detection_ids, 'csv')
    except Exception as e:
        import logging
        logging.warning(f"Could not mark detections as exported: {e}")

    df = state.db_manager.get_history_df()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=wildlife_results.csv"},
    )


@router.get("/export/json")
def export_json(state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    df = state.db_manager.get_history_df()
    return df.fillna("").to_dict(orient="records")
