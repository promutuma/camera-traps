"""Tab 8 — Stations & Deployments."""

from fastapi import APIRouter, Depends, HTTPException
from backend.models.state import AppState
from backend.models.schemas import StationCreate, StationUpdate, DeploymentCreate
from backend.routers.deps import get_state

router = APIRouter(prefix="/stations", tags=["stations"])


@router.get("")
def list_stations(state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    df = state.station_manager.get_stations()
    return df.fillna("").to_dict(orient="records") if hasattr(df, "to_dict") else []


@router.post("")
def add_station(body: StationCreate, state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.station_manager.add_station(**body.model_dump())
    return {"ok": True}


@router.patch("/{station_id}")
def update_station(station_id: str, body: StationUpdate, state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.station_manager.update_station(station_id, **body.model_dump(exclude_none=True))
    return {"ok": True}


@router.delete("/{station_id}")
def delete_station(station_id: str, state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.station_manager.delete_station(station_id)
    return {"ok": True}


@router.get("/deployments")
def list_deployments(state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    df = state.station_manager.get_deployments()
    return df.fillna("").to_dict(orient="records") if hasattr(df, "to_dict") else []


@router.post("/deployments")
def add_deployment(body: DeploymentCreate, state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.station_manager.add_deployment(**body.model_dump())
    return {"ok": True}


@router.get("/summary")
def station_summary(state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    summary = state.station_manager.get_station_summary()
    return summary if isinstance(summary, list) else (
        summary.fillna("").to_dict(orient="records") if hasattr(summary, "to_dict") else []
    )


@router.get("/map")
def station_map(state: AppState = Depends(get_state)):
    """Return GeoJSON FeatureCollection of stations with lat/lon."""
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    df = state.station_manager.get_stations()
    stations = df.fillna("").to_dict(orient="records") if hasattr(df, "to_dict") else []
    features = []
    for s in stations:
        lat = s.get("latitude") or s.get("lat")
        lon = s.get("longitude") or s.get("lon")
        if lat and lon:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": s,
            })
    return {"type": "FeatureCollection", "features": features}
