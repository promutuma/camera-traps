"""Tab 14 — Project Configuration."""

import io
import json
from typing import Optional, Dict, Any
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
    proj = state.project_config.get_active_project()
    if not proj:
        return

    # ── SpeciesNet geographic prior ──────────────────────────────────────
    lat = proj.get("speciesnet_lat")
    lng = proj.get("speciesnet_lng")
    country = proj.get("speciesnet_country")
    if lat is not None:
        state.config.speciesnet_lat = float(lat)
    if lng is not None:
        state.config.speciesnet_lng = float(lng)
    if country:
        state.config.speciesnet_country = str(country)
    if state.speciesnet_model is not None:
        if lat is not None:
            state.speciesnet_model.lat = float(lat)
        if lng is not None:
            state.speciesnet_model.lng = float(lng)
        if country:
            state.speciesnet_model.country = str(country)

    # ── Independence window ──────────────────────────────────────────────
    thresholds = state.project_config.get_thresholds(proj["id"])
    win = thresholds.get("independence_window_min")
    if win is not None:
        try:
            state.config.independence_window = int(win)
            if state.independence_engine is not None:
                state.independence_engine.window_minutes = int(win)
        except (TypeError, ValueError):
            pass


# Keep old name as alias so main.py import still works
_sync_speciesnet_coords = _sync_project_to_config


@router.get("")
def get_project(state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    cfg = state.project_config.get_active_project()
    if cfg:
        cfg["project_name"] = cfg.pop("name", "")
        cfg["thresholds"] = state.project_config.get_thresholds(cfg["id"])
    return cfg or {}


@router.patch("")
def update_project(body: ProjectUpdate, state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")

    proj = state.project_config.get_active_project()
    if not proj:
        raise HTTPException(status_code=404, detail="No active project found")

    fields = body.model_dump(exclude_none=True)
    if "project_name" in fields:
        fields["name"] = fields.pop("project_name")

    thresholds = fields.pop("thresholds", None)
    if thresholds:
        state.project_config.save_thresholds(proj["id"], thresholds)

    if fields:
        state.project_config.update_project(proj["id"], **fields)

    # Propagate coordinate changes to the live SpeciesNet config immediately
    _sync_speciesnet_coords(state)

    return {"ok": True}


@router.get("/list")
def list_projects(state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    df = state.project_config.get_projects()
    return df.to_dict(orient="records")


@router.post("/active/{project_id}")
def set_active_project(project_id: int, state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    success = state.project_config.set_active_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    _sync_speciesnet_coords(state)
    return {"ok": True}


@router.post("/create")
def create_project(name: str = Query(...), survey_area: str = "", notes: str = "", state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    project_id = state.project_config.create_project(name=name, survey_area=survey_area, notes=notes)
    if project_id is None:
        raise HTTPException(status_code=400, detail="Project name already exists")
    state.project_config.set_active_project(project_id)
    return {"ok": True, "project_id": project_id}


@router.delete("/{project_id}")
def delete_project(project_id: int, state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    active = state.project_config.get_active_project()
    if active and active["id"] == project_id:
        raise HTTPException(status_code=400, detail="Cannot delete active project")
    success = state.project_config.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True}


@router.post("/baseline/lock")
def lock_baseline(state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    proj = state.project_config.get_active_project()
    if not proj:
        raise HTTPException(status_code=404, detail="No active project found")
    
    metrics = {"total_images": 0}
    if state.db_manager:
        stats = state.db_manager.get_hash_stats()
        metrics["total_images"] = stats.get("total_images", 0)
        
    state.project_config.lock_baseline(proj["id"], metrics)
    return {"ok": True}


@router.get("/export")
def export_project(state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    proj = state.project_config.get_active_project()
    if not proj:
        raise HTTPException(status_code=404, detail="No active project found")
    payload = state.project_config.export_project(proj["id"])
    return StreamingResponse(
        io.BytesIO(payload.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=project_config_{proj['id']}.json"},
    )
