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
):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    df = state.db_manager.get_history_df()
    if df.empty:
        return []

    if species:
        df = df[df["detected_animal"].str.contains(species, case=False, na=False)]
    if day_night:
        df = df[df["day_night"] == day_night]
    if min_conf is not None:
        df = df[df["detection_confidence"] >= min_conf]
    if max_conf is not None:
        df = df[df["detection_confidence"] <= max_conf]
    if station:
        df = df[df["station_id"] == station]

    return df.fillna("").to_dict(orient="records")


@router.patch("/{detection_id}")
def update_result(detection_id: int, update: ResultUpdate, state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    fields = update.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    state.db_manager.update_detection(detection_id, fields)
    return {"ok": True, "detection_id": detection_id}


@router.get("/export/excel")
def export_excel(state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
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
