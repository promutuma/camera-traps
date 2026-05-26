"""Tab 14 — Project Configuration."""

import io
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from backend.models.state import AppState
from backend.models.schemas import ProjectUpdate
from backend.routers.deps import get_state

router = APIRouter(prefix="/project", tags=["project"])


@router.get("")
def get_project(state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    cfg = state.project_config
    return cfg.__dict__ if hasattr(cfg, "__dict__") else {}


@router.patch("")
def update_project(body: ProjectUpdate, state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    for key, value in body.model_dump(exclude_none=True).items():
        if hasattr(state.project_config, key):
            setattr(state.project_config, key, value)
    return {"ok": True}


@router.post("/baseline/lock")
def lock_baseline(state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    if hasattr(state.project_config, "lock_baseline"):
        state.project_config.lock_baseline()
    return {"ok": True}


@router.get("/export")
def export_project(state: AppState = Depends(get_state)):
    if not state.project_config:
        raise HTTPException(status_code=503, detail="Service not ready")
    data = state.project_config.__dict__ if hasattr(state.project_config, "__dict__") else {}
    payload = json.dumps(data, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(payload.encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=project_config.json"},
    )
