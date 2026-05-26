"""Tab 6 — Ecological Analytics: IDE, RAI, richness, accumulation, activity."""

import io
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from backend.models.state import AppState
from backend.routers.deps import get_state

router = APIRouter(prefix="/ecological", tags=["ecological"])


def _get_data(state: AppState):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    import pandas as pd
    df = state.db_manager.get_history_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No data — process images first")
    return df


def _get_engine(state: AppState):
    from core.independence_engine import IndependenceEngine
    return IndependenceEngine(window_minutes=state.config.independence_window)


@router.get("/ide")
def get_ide(state: AppState = Depends(get_state)):
    df = _get_data(state)
    engine = _get_engine(state)
    enriched = engine.compute_ides(df, default_station=state.config.default_station_id)
    summary = engine.get_ide_summary(enriched)
    return {
        "summary": summary.fillna("").to_dict(orient="records"),
        "enriched": enriched.fillna("").to_dict(orient="records"),
    }


@router.get("/rai")
def get_rai(state: AppState = Depends(get_state), trap_nights: int = Query(30)):
    df = _get_data(state)
    engine = _get_engine(state)
    enriched = engine.compute_ides(df, default_station=state.config.default_station_id)
    summary = engine.get_ide_summary(enriched)
    stations = summary["station_id"].unique().tolist() if not summary.empty else []
    trap_map = {s: trap_nights for s in stations}
    rai_df = engine.compute_rai(summary, trap_map)
    return rai_df.fillna("").to_dict(orient="records")


@router.get("/timeline")
def get_timeline(state: AppState = Depends(get_state)):
    df = _get_data(state)
    engine = _get_engine(state)
    enriched = engine.compute_ides(df, default_station=state.config.default_station_id)
    summary = engine.get_ide_summary(enriched)
    if summary.empty or "first_detection" not in summary.columns:
        return []
    import pandas as pd
    tl = summary.dropna(subset=["first_detection"]).copy()
    tl["first_detection"] = pd.to_datetime(tl["first_detection"], errors="coerce")
    tl = tl.dropna(subset=["first_detection"])
    tl["date"] = tl["first_detection"].dt.date.astype(str)
    pivot = (
        tl.groupby(["date", "species"]).size()
        .reset_index(name="count")
    )
    return pivot.to_dict(orient="records")


@router.get("/richness")
def get_richness(state: AppState = Depends(get_state)):
    df = _get_data(state)
    engine = _get_engine(state)
    enriched = engine.compute_ides(df, default_station=state.config.default_station_id)
    summary = engine.get_ide_summary(enriched)
    richness = engine.compute_species_richness(summary)
    return richness.fillna("").to_dict(orient="records")


@router.get("/accumulation")
def get_accumulation(state: AppState = Depends(get_state)):
    df = _get_data(state)
    engine = _get_engine(state)
    enriched = engine.compute_ides(df, default_station=state.config.default_station_id)
    summary = engine.get_ide_summary(enriched)
    accum = engine.compute_species_accumulation(summary)
    return accum.fillna("").to_dict(orient="records")


@router.get("/group-size")
def get_group_size(state: AppState = Depends(get_state)):
    df = _get_data(state)
    engine = _get_engine(state)
    enriched = engine.compute_ides(df, default_station=state.config.default_station_id)
    group_df = engine.compute_mean_group_size(enriched)
    return group_df.fillna("").to_dict(orient="records")


@router.get("/visitation")
def get_visitation(state: AppState = Depends(get_state), trap_nights: int = Query(30)):
    df = _get_data(state)
    engine = _get_engine(state)
    enriched = engine.compute_ides(df, default_station=state.config.default_station_id)
    summary = engine.get_ide_summary(enriched)
    stations = summary["station_id"].unique().tolist() if not summary.empty else []
    trap_map = {s: trap_nights for s in stations}
    visit_df, heatmap_df = engine.compute_visitation_rate(summary, trap_map)
    return {
        "visitation": visit_df.fillna("").to_dict(orient="records"),
        "heatmap": heatmap_df.fillna("").to_dict(orient="records"),
    }


@router.get("/export")
def export_ecological(state: AppState = Depends(get_state), trap_nights: int = Query(30)):
    df = _get_data(state)
    engine = _get_engine(state)
    enriched = engine.compute_ides(df, default_station=state.config.default_station_id)
    summary = engine.get_ide_summary(enriched)
    buf = io.StringIO()
    summary.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ecological_export.csv"},
    )
