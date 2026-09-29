"""Tab 15 — ArcGIS Sync & Spatial Exports."""

import io
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from backend.models.state import AppState
from backend.models.schemas import ArcGISPushRequest
from backend.routers.deps import get_state

router = APIRouter(prefix="/arcgis", tags=["arcgis"])


def _stations_df(state: AppState):
    return state.station_manager.get_stations() if state.station_manager else None


@router.post("/push")
def push_to_arcgis(body: ArcGISPushRequest, state: AppState = Depends(get_state)):
    if not state.arcgis_sync or not state.station_manager:
        raise HTTPException(status_code=503, detail="Services not ready")
    stations_df = _stations_df(state)
    if stations_df is None or stations_df.empty:
        raise HTTPException(status_code=404, detail="No station data to push")
    try:
        state.arcgis_sync.authenticate_with_token(body.token)
        result = state.arcgis_sync.push_stations(stations_df, layer_url=body.url)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def sync_status(state: AppState = Depends(get_state)):
    if not state.arcgis_sync:
        raise HTTPException(status_code=503, detail="Service not ready")
    status = getattr(state.arcgis_sync, "last_sync_status", None)
    return {"status": status}


@router.get("/export/geojson")
def export_geojson(state: AppState = Depends(get_state)):
    if not state.db_manager or not state.spatial_exporter:
        raise HTTPException(status_code=503, detail="Services not ready")
    df = state.db_manager.get_history_df()
    stations_df = _stations_df(state)
    geojson = state.spatial_exporter.to_geojson(df, stations_df)
    data = json.dumps(geojson if isinstance(geojson, dict) else json.loads(geojson), indent=2)
    return StreamingResponse(
        io.BytesIO(data.encode()),
        media_type="application/geo+json",
        headers={"Content-Disposition": "attachment; filename=arcgis_export.geojson"},
    )


@router.get("/export/shapefile")
def export_shapefile(state: AppState = Depends(get_state)):
    if not state.db_manager or not state.spatial_exporter:
        raise HTTPException(status_code=503, detail="Services not ready")
    df = state.db_manager.get_history_df()
    stations_df = _stations_df(state)
    try:
        shp_bytes = state.spatial_exporter.to_shapefile_bytes(df, stations_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StreamingResponse(
        io.BytesIO(shp_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=arcgis_shapefile.zip"},
    )


@router.get("/export/kml")
def export_kml(state: AppState = Depends(get_state)):
    if not state.db_manager or not state.spatial_exporter:
        raise HTTPException(status_code=503, detail="Services not ready")
    df = state.db_manager.get_history_df()
    stations_df = _stations_df(state)
    kml_str = state.spatial_exporter.to_kml(df, stations_df)
    return StreamingResponse(
        io.BytesIO(kml_str.encode() if isinstance(kml_str, str) else kml_str),
        media_type="application/vnd.google-earth.kml+xml",
        headers={"Content-Disposition": "attachment; filename=arcgis_export.kml"},
    )
