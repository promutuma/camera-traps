"""Tab 11 — Spatial Outputs & Map Viewer."""

import io
import json
import zipfile
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from backend.models.state import AppState
from backend.routers.deps import get_state

router = APIRouter(prefix="/spatial", tags=["spatial"])


def _get_df(state):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    df = state.db_manager.get_history_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No data")
    return df


@router.get("/geojson")
def get_geojson(state: AppState = Depends(get_state)):
    df = _get_df(state)
    if not state.spatial_exporter:
        raise HTTPException(status_code=503, detail="Service not ready")
    stations_df = state.station_manager.get_stations() if state.station_manager else None
    geojson = state.spatial_exporter.to_geojson(df, stations_df)
    return geojson if isinstance(geojson, dict) else json.loads(geojson)


@router.get("/export/geojson")
def export_geojson(state: AppState = Depends(get_state)):
    df = _get_df(state)
    if not state.spatial_exporter:
        raise HTTPException(status_code=503, detail="Service not ready")
    stations_df = state.station_manager.get_stations() if state.station_manager else None
    geojson = state.spatial_exporter.to_geojson(df, stations_df)
    data = json.dumps(geojson if isinstance(geojson, dict) else json.loads(geojson), indent=2)
    return StreamingResponse(
        io.BytesIO(data.encode()),
        media_type="application/geo+json",
        headers={"Content-Disposition": "attachment; filename=wildlife_detections.geojson"},
    )


@router.get("/export/csv")
def export_csv(state: AppState = Depends(get_state)):
    df = _get_df(state)
    if not state.spatial_exporter:
        raise HTTPException(status_code=503, detail="Service not ready")
    stations_df = state.station_manager.get_stations() if state.station_manager else None
    csv_str = state.spatial_exporter.detections_to_csv(df, stations_df)
    return StreamingResponse(
        io.BytesIO(csv_str.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=wildlife_georeferenced.csv"},
    )


@router.get("/export/shapefile")
def export_shapefile(state: AppState = Depends(get_state)):
    df = _get_df(state)
    if not state.spatial_exporter:
        raise HTTPException(status_code=503, detail="Service not ready")
    stations_df = state.station_manager.get_stations() if state.station_manager else None
    try:
        shp_bytes = state.spatial_exporter.to_shapefile_bytes(df, stations_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StreamingResponse(
        io.BytesIO(shp_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=wildlife_shapefile.zip"},
    )


@router.get("/export/kml")
def export_kml(state: AppState = Depends(get_state)):
    df = _get_df(state)
    if not state.spatial_exporter:
        raise HTTPException(status_code=503, detail="Service not ready")
    stations_df = state.station_manager.get_stations() if state.station_manager else None
    kml_str = state.spatial_exporter.to_kml(df, stations_df)
    return StreamingResponse(
        io.BytesIO(kml_str.encode() if isinstance(kml_str, str) else kml_str),
        media_type="application/vnd.google-earth.kml+xml",
        headers={"Content-Disposition": "attachment; filename=wildlife_detections.kml"},
    )
