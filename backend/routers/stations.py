"""Tab 8 — Stations & Deployments."""

from fastapi import APIRouter, Depends, HTTPException, Query
from backend.models.state import AppState
from backend.models.schemas import StationCreate, StationUpdate, DeploymentCreate
from backend.routers.deps import get_state

router = APIRouter(prefix="/stations", tags=["stations"])


@router.get("")
def list_stations(state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    df = state.station_manager.get_stations()
    if not hasattr(df, "to_dict"):
        return []
    records = df.fillna("").to_dict(orient="records")
    current_cameras = state.station_manager.get_current_camera_ids()
    mapped = []
    for r in records:
        mapped.append({
            "station_id":   r.get("station_id"),
            "latitude":     r.get("gps_lat"),
            "longitude":    r.get("gps_lon"),
            "stratum":      r.get("habitat_stratum"),
            "habitat":      "",   # kept for backward compat
            "camera_model": r.get("camera_model", ""),
            "team_member":  r.get("team_member", ""),
            "notes":        r.get("notes"),
            "current_camera_id": current_cameras.get(r.get("station_id")),
        })
    return mapped


@router.post("/{station_id}/camera")
def assign_camera(station_id: str, camera_id: str = Query(...), state: AppState = Depends(get_state)):
    """
    Map a camera to a station right now — updates the deployment covering
    today if one exists, otherwise starts a new open-ended deployment.
    """
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    if not camera_id.strip():
        raise HTTPException(status_code=400, detail="camera_id is required")
    state.station_manager.set_current_camera(station_id, camera_id.strip())
    return {"ok": True}


@router.post("")
def add_station(body: StationCreate, state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    ok = state.station_manager.add_station(
        station_id=body.station_id,
        gps_lat=body.latitude,
        gps_lon=body.longitude,
        habitat_stratum=body.stratum or "",
        camera_model=body.camera_model or "",
        team_member=body.team_member or "",
        notes=body.notes or "",
    )
    if not ok:
        raise HTTPException(status_code=409, detail="Station ID already exists")
    return {"ok": True}


@router.patch("/{station_id}")
def update_station(station_id: str, body: StationUpdate, state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    body_dict = body.model_dump(exclude_none=True)
    update_data = {}
    if "latitude" in body_dict:
        update_data["gps_lat"] = body_dict["latitude"]
    if "longitude" in body_dict:
        update_data["gps_lon"] = body_dict["longitude"]
    if "stratum" in body_dict:
        update_data["habitat_stratum"] = body_dict["stratum"]
    if "camera_model" in body_dict:
        update_data["camera_model"] = body_dict["camera_model"]
    if "team_member" in body_dict:
        update_data["team_member"] = body_dict["team_member"]
    if "notes" in body_dict:
        update_data["notes"] = body_dict["notes"]

    if update_data:
        state.station_manager.update_station(station_id, **update_data)
    return {"ok": True}


@router.delete("/{station_id}")
def delete_station(station_id: str, force: bool = False, state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    orphaned = 0
    if state.db_manager:
        orphaned = state.db_manager.count_images_by_station(station_id)
    if orphaned > 0 and not force:
        return {"ok": False, "orphaned_detections": orphaned}
    state.station_manager.delete_station(station_id)
    return {"ok": True, "orphaned_detections": orphaned}


@router.get("/deployments")
def list_deployments(state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    df = state.station_manager.get_deployments()
    if not hasattr(df, "to_dict"):
        return []
    records = df.fillna("").to_dict(orient="records")
    mapped = []
    for r in records:
        mapped.append({
            "id":               r.get("id"),
            "station_id":       r.get("station_id"),
            "start_date":       r.get("deployment_date"),
            "end_date":         r.get("retrieval_date"),
            "camera_id":        r.get("sd_card_id"),
            "camera_down_days": r.get("camera_down_days", 0),
            "notes":            r.get("notes"),
        })
    return mapped


@router.post("/deployments")
def add_deployment(body: DeploymentCreate, state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.station_manager.add_deployment(
        station_id=body.station_id,
        deployment_date=body.start_date,
        retrieval_date=body.end_date or None,
        sd_card_id=body.camera_id or "",
        camera_down_days=body.camera_down_days or 0,
        notes=body.notes or "",
    )
    return {"ok": True}


@router.delete("/deployments/{deployment_id}")
def delete_deployment(deployment_id: int, state: AppState = Depends(get_state)):
    if not state.station_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.station_manager.delete_deployment(deployment_id)
    return {"ok": True}



@router.get("/orphans")
def get_orphan_stations(state: AppState = Depends(get_state)):
    """Return station_ids used in images but not registered in the stations table."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    return state.db_manager.get_unregistered_station_ids()


@router.post("/reassign")
def reassign_station(
    from_station: str = Query(...),
    to_station: str = Query(...),
    state: AppState = Depends(get_state),
):
    """Reassign all images from an unregistered station_id to a registered one."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    if from_station == to_station:
        raise HTTPException(status_code=400, detail="from_station and to_station must differ")
    updated = state.db_manager.reassign_images_station(from_station, to_station)
    return {"ok": True, "updated": updated}


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
    current_cameras = state.station_manager.get_current_camera_ids()
    features = []
    for s in stations:
        lat = s.get("gps_lat") or s.get("latitude") or s.get("lat")
        lon = s.get("gps_lon") or s.get("longitude") or s.get("lon")
        if lat and lon:
            # Add mapped keys to properties for frontend compatibility
            props = {
                **s,
                "latitude": lat,
                "longitude": lon,
                "stratum": s.get("habitat_stratum", ""),
                "current_camera_id": current_cameras.get(s.get("station_id")) or "",
            }
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": props,
            })
    return {"type": "FeatureCollection", "features": features}
