"""Camera registry — physical camera units, independent of which station
they're currently deployed at."""

from fastapi import APIRouter, Depends, HTTPException
from backend.models.state import AppState
from backend.models.schemas import CameraCreate, CameraUpdate
from backend.routers.deps import get_state

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("")
def list_cameras(state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    df = state.station_manager.get_cameras()
    return df.fillna("").to_dict(orient="records") if hasattr(df, "to_dict") else []


@router.post("")
def add_camera(body: CameraCreate, state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    ok = state.station_manager.add_camera(
        camera_id=body.camera_id,
        model=body.model or "",
        serial_number=body.serial_number or "",
        status=body.status or "active",
        notes=body.notes or "",
    )
    if not ok:
        raise HTTPException(status_code=409, detail="Camera ID already exists")
    return {"ok": True}


@router.patch("/{camera_id}")
def update_camera(camera_id: str, body: CameraUpdate, state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    updates = body.model_dump(exclude_none=True)
    if updates:
        state.station_manager.update_camera(camera_id, **updates)
    return {"ok": True}


@router.delete("/{camera_id}")
def delete_camera(camera_id: str, state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.station_manager.delete_camera(camera_id)
    return {"ok": True}
