"""Tab 6 — Ecological Analytics: IDE, RAI, richness, accumulation, activity."""

import io
import json
import pandas as pd
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from backend.models.state import AppState
from backend.routers.deps import get_state

router = APIRouter(prefix="/ecological", tags=["ecological"])


def _get_data(state: AppState):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
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
    
    real_trap_nights = {}
    if state.station_manager:
        real_trap_nights = state.station_manager.compute_trap_nights()
        
    trap_map = {}
    for s in stations:
        tn = real_trap_nights.get(s, 0)
        trap_map[s] = tn if tn > 0 else trap_nights
        
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
    
    real_trap_nights = {}
    if state.station_manager:
        real_trap_nights = state.station_manager.compute_trap_nights()
        
    trap_map = {}
    for s in stations:
        tn = real_trap_nights.get(s, 0)
        trap_map[s] = tn if tn > 0 else trap_nights
        
    visit_df, heatmap_df = engine.compute_visitation_rate(summary, trap_map)
    return {
        "visitation": visit_df.fillna("").to_dict(orient="records"),
        "heatmap": heatmap_df.fillna("").to_dict(orient="records"),
    }


@router.get("/export")
def export_ecological(
    state: AppState = Depends(get_state),
    trap_nights: int = Query(30),
    metric: str = Query("ide"),
    format: Optional[str] = Query(None),
    export_format: Optional[str] = Query(None),
):
    fmt = format or export_format or "csv"
    fmt = fmt.lower().strip()
    metric = metric.lower().strip()

    # 1. Fetch history data
    df = _get_data(state)
    engine = _get_engine(state)
    enriched = engine.compute_ides(df, default_station=state.config.default_station_id)
    summary = engine.get_ide_summary(enriched)

    # Determine which DataFrame to export
    export_df = None
    filename_prefix = f"ecological_{metric}"

    if metric == "ide":
        export_df = summary
    elif metric == "rai":
        stations = summary["station_id"].unique().tolist() if not summary.empty else []
        real_trap_nights = {}
        if state.station_manager:
            real_trap_nights = state.station_manager.compute_trap_nights()
        trap_map = {s: real_trap_nights.get(s, 0) or trap_nights for s in stations}
        export_df = engine.compute_rai(summary, trap_map)
    elif metric in ("group-size", "group_size"):
        export_df = engine.compute_mean_group_size(enriched)
    elif metric == "visitation":
        stations = summary["station_id"].unique().tolist() if not summary.empty else []
        real_trap_nights = {}
        if state.station_manager:
            real_trap_nights = state.station_manager.compute_trap_nights()
        trap_map = {s: real_trap_nights.get(s, 0) or trap_nights for s in stations}
        visit_df, _ = engine.compute_visitation_rate(summary, trap_map)
        export_df = visit_df
    elif metric == "richness":
        export_df = engine.compute_species_richness(summary)
    elif metric == "accumulation":
        export_df = engine.compute_species_accumulation(summary)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported metric: {metric}")

    # Handle standard spatial/standardized formats if metric is 'ide'
    if metric == "ide":
        stations_df = state.station_manager.get_stations() if state.station_manager else None

        if fmt == "geojson":
            if not state.spatial_exporter:
                raise HTTPException(status_code=503, detail="Spatial exporter not ready")
            geojson_str = state.spatial_exporter.detections_to_geojson(df=None, stations_df=stations_df, ide_summary=summary)
            return StreamingResponse(
                io.BytesIO(geojson_str.encode()),
                media_type="application/geo+json",
                headers={"Content-Disposition": f"attachment; filename={filename_prefix}.geojson"},
            )
        elif fmt == "shapefile":
            if not state.spatial_exporter:
                raise HTTPException(status_code=503, detail="Spatial exporter not ready")
            shp_bytes = state.spatial_exporter.detections_to_shapefile(ide_summary=summary, stations_df=stations_df)
            return StreamingResponse(
                io.BytesIO(shp_bytes),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={filename_prefix}_shapefile.zip"},
            )
        elif fmt == "kml":
            if not state.spatial_exporter:
                raise HTTPException(status_code=503, detail="Spatial exporter not ready")
            kml_str = state.spatial_exporter.detections_to_kml(ide_summary=summary, stations_df=stations_df)
            return StreamingResponse(
                io.BytesIO(kml_str.encode()),
                media_type="application/vnd.google-earth.kml+xml",
                headers={"Content-Disposition": f"attachment; filename={filename_prefix}.kml"},
            )
        elif fmt in ("darwin-core", "darwin_core", "dwc"):
            # Build Darwin Core DataFrame from representative IDE rows
            ide_rows = enriched.sort_values("datetime_parsed", na_position="last").groupby("ide_id", as_index=False).first()
            if stations_df is not None and not stations_df.empty:
                coord_df = stations_df[["station_id", "gps_lat", "gps_lon"]].copy()
                coord_df.columns = ["station_id", "latitude", "longitude"]
                ide_rows = ide_rows.merge(coord_df, on="station_id", how="left")
            else:
                ide_rows["latitude"] = None
                ide_rows["longitude"] = None

            dwc_df = pd.DataFrame()
            dwc_df["occurrenceID"] = ide_rows["ide_id"]
            dwc_df["basisOfRecord"] = "MachineObservation"
            dwc_df["eventDate"] = ide_rows["datetime_parsed"].dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
            dwc_df["vernacularName"] = ide_rows["detected_animal"]
            dwc_df["scientificNameConfidence"] = ide_rows["detection_confidence"]
            dwc_df["decimalLatitude"] = ide_rows.get("latitude")
            dwc_df["decimalLongitude"] = ide_rows.get("longitude")
            dwc_df["geodeticDatum"] = "WGS84"
            dwc_df["associatedMedia"] = ide_rows["filename"]
            dwc_df["locality"] = ide_rows["station_id"]
            dwc_df["identificationVerificationStatus"] = ide_rows.get("agreement", None)
            
            dwc_df["scientificName"] = dwc_df["vernacularName"].apply(
                lambda x: "Animalia" if x == "Animal" else (x or "Biota")
            )
            dwc_df["individualCount"] = 1
            dwc_df["countryCode"] = "ET"
            dwc_df["georeferenceProtocol"] = "Camera Trap GPS"

            buf = io.StringIO()
            dwc_df.to_csv(buf, index=False)
            buf.seek(0)
            return StreamingResponse(
                io.BytesIO(buf.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename_prefix}_darwin_core.csv"},
            )
        elif fmt in ("wildlife-insights", "wildlife_insights", "wi"):
            # Build Wildlife Insights JSON
            project_details = {}
            if state.project_config:
                active_proj = state.project_config.get_active_project()
                if active_proj:
                    project_details = {
                        "project_id": active_proj.get("id"),
                        "project_name": active_proj.get("name"),
                        "survey_area": active_proj.get("survey_area"),
                        "notes": active_proj.get("notes"),
                    }

            # Retrieve deployments
            conn = state.db_manager.get_connection()
            try:
                deployments_df = pd.read_sql_query("SELECT * FROM deployments", conn) \
                    if "deployments" in pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].values \
                    else pd.DataFrame()
            finally:
                conn.close()

            insights_package = {
                "version": "1.0",
                "provider": "ViumbeLens AI",
                "project": project_details or {
                    "project_name": "ViumbeLens Default Survey",
                    "notes": "Generated from default local SQLite database"
                },
                "deployments": deployments_df.to_dict(orient="records") if not deployments_df.empty else [],
                "images": []
            }

            ide_rows = enriched.sort_values("datetime_parsed", na_position="last").groupby("ide_id", as_index=False).first()
            for _, row in ide_rows.iterrows():
                bbox_data = None
                if row.get("bbox"):
                    try:
                        bbox_data = json.loads(row["bbox"])
                    except Exception:
                        pass

                insights_package["images"].append({
                    "image_id": int(row["id"]),
                    "filename": row["filename"],
                    "station_id": row["station_id"],
                    "timestamp": row["datetime_parsed"].strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(row["datetime_parsed"]) else None,
                    "observations": [
                        {
                            "species_common_name": row["detected_animal"],
                            "confidence": float(row["detection_confidence"]) if pd.notna(row["detection_confidence"]) else None,
                            "bounding_box": bbox_data,
                            "ide_id": row["ide_id"]
                        }
                    ] if row["detected_animal"] else []
                })

            return JSONResponse(
                content=insights_package,
                headers={"Content-Disposition": f"attachment; filename={filename_prefix}_wildlife_insights.json"}
            )

    # Default CSV/Excel handling for all metrics (including 'ide')
    if export_df is None or export_df.empty:
        raise HTTPException(status_code=404, detail="No data available for export")

    if fmt in ("excel", "xlsx"):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name=metric.capitalize()[:30])
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename_prefix}.xlsx"},
        )
    elif fmt == "csv":
        buf = io.StringIO()
        export_df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename_prefix}.csv"},
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt} for metric: {metric}")
