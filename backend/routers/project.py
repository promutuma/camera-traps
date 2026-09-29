"""Tab 14 — Project Configuration."""

from __future__ import annotations

import io
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.models.state import AppState
from backend.models.schemas import ProjectUpdate
from backend.routers.deps import get_state

router = APIRouter(prefix="/project", tags=["project"])


def _sync_project_to_config(state: AppState) -> None:
    """
    Push active project settings into AppConfig and live service objects.
    Called at startup, on project switch, and on project update.
    """
    if not state.project_config:
        return
    project = state.project_config.get_active_project()
    if not project:
        return

    cfg = state.config
    thresholds = state.project_config.get_thresholds(int(project["id"]))

    if project.get("name"):
        cfg.default_station_id = cfg.default_station_id or "STATION_001"
    if thresholds.get("confidence_threshold") is not None:
        cfg.detection_confidence = float(thresholds["confidence_threshold"])
    if thresholds.get("independence_window_min") is not None:
        cfg.independence_window = int(thresholds["independence_window_min"])
    if project.get("speciesnet_lat") is not None:
        cfg.speciesnet_lat = float(project["speciesnet_lat"])
    if project.get("speciesnet_lng") is not None:
        cfg.speciesnet_lng = float(project["speciesnet_lng"])
    if project.get("speciesnet_country"):
        cfg.speciesnet_country = str(project["speciesnet_country"])

    if state.md_model:
        state.md_model.set_confidence_threshold(cfg.detection_confidence)
    if state.dn_model and thresholds.get("confidence_threshold"):
        pass


def _sync_speciesnet_coords(state: AppState) -> None:
    """Update SpeciesNet geo context from the active project."""
    if not state.project_config or not state.speciesnet_model:
        return
    project = state.project_config.get_active_project()
    if not project:
        return
    lat = project.get("speciesnet_lat")
    lng = project.get("speciesnet_lng")
    country = project.get("speciesnet_country")
    try:
        if hasattr(state.speciesnet_model, "update_geo_context"):
            state.speciesnet_model.update_geo_context(lat, lng, country)
    except Exception:
        pass


@router.get("")
def get_project(state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    project = state.project_config.get_active_project()
    if not project:
        raise HTTPException(status_code=404, detail="No active project found")
    pid = int(project["id"])
    return {
        "project_id": pid,
        "project_name": project.get("name"),
        "survey_area": project.get("survey_area"),
        "notes": project.get("notes"),
        "thresholds": state.project_config.get_thresholds(pid),
        "speciesnet_lat": project.get("speciesnet_lat"),
        "speciesnet_lng": project.get("speciesnet_lng"),
        "speciesnet_country": project.get("speciesnet_country"),
        "baseline": state.project_config.get_baseline(pid),
    }


@router.patch("")
def update_project(body: ProjectUpdate, state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    project = state.project_config.get_active_project()
    if not project:
        raise HTTPException(status_code=404, detail="No active project found")
    pid = int(project["id"])

    updates: Dict[str, Any] = {}
    if body.project_name is not None:
        updates["name"] = body.project_name
    if body.survey_area is not None:
        updates["survey_area"] = body.survey_area
    if body.notes is not None:
        updates["notes"] = body.notes
    if body.speciesnet_lat is not None:
        updates["speciesnet_lat"] = body.speciesnet_lat
    if body.speciesnet_lng is not None:
        updates["speciesnet_lng"] = body.speciesnet_lng
    if body.speciesnet_country is not None:
        updates["speciesnet_country"] = body.speciesnet_country

    if updates:
        state.project_config.update_project(pid, **updates)

    if body.thresholds:
        state.project_config.save_thresholds(pid, body.thresholds)

    _sync_project_to_config(state)
    _sync_speciesnet_coords(state)
    return get_project(state)


@router.get("/list")
def list_projects(state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    df = state.project_config.get_projects()
    return df.fillna("").to_dict(orient="records")


@router.post("/active/{project_id}")
def set_active_project(project_id: int, state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    if not state.project_config.set_active_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    _sync_project_to_config(state)
    _sync_speciesnet_coords(state)
    return {"ok": True, "project_id": project_id}


@router.post("/create")
def create_project(
    name: str = Query(...),
    survey_area: Optional[str] = Query(None),
    notes: Optional[str] = Query(None),
    state: AppState = Depends(get_state),
):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        pid = state.project_config.create_project(name, survey_area=survey_area, notes=notes)
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            raise HTTPException(status_code=409, detail="Project name already exists") from exc
        raise
    state.project_config.set_active_project(pid)
    _sync_project_to_config(state)
    _sync_speciesnet_coords(state)
    return {"ok": True, "project_id": pid}


@router.delete("/{project_id}")
def delete_project(project_id: int, state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    active = state.project_config.get_active_project()
    if active and int(active["id"]) == project_id:
        raise HTTPException(status_code=409, detail="Cannot delete active project")
    if not state.project_config.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True}


@router.post("/baseline/lock")
def lock_baseline(state: AppState = Depends(get_state)):
    if not state.project_config or not state.db_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    project = state.project_config.get_active_project()
    if not project:
        raise HTTPException(status_code=404, detail="No active project found")
    pid = int(project["id"])

    df = state.db_manager.get_history_df()
    total_images = len(df.drop_duplicates("id")) if not df.empty and "id" in df.columns else 0
    baseline_data = {
        "project_name": project.get("name"),
        "total_images": total_images,
        "records": int(len(df)),
    }
    state.project_config.lock_baseline(pid, baseline_data)
    return {"ok": True, "baseline": state.project_config.get_baseline(pid)}


@router.get("/export")
def export_project(state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    project = state.project_config.get_active_project()
    if not project:
        raise HTTPException(status_code=404, detail="No active project found")
    pid = int(project["id"])
    payload = state.project_config.export_project(pid)
    filename = f"project_config_{project.get('name', 'project')}.json"
    return StreamingResponse(
        io.BytesIO(payload.encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
