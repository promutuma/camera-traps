"""Tab 13 — Corridor Movement Analysis."""

from fastapi import APIRouter, Depends, HTTPException, Query
from backend.models.state import AppState
from backend.routers.deps import get_state

router = APIRouter(prefix="/corridor", tags=["corridor"])


def _get_data(state):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    df = state.db_manager.get_history_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No data")
    return df


@router.get("/pairs")
def get_pairs(state: AppState = Depends(get_state), max_km: float = Query(50.0)):
    if not state.corridor_analyzer:
        raise HTTPException(status_code=503, detail="Service not ready")
    stations = state.station_manager.get_stations().fillna("").to_dict(orient="records") if state.station_manager else []
    pairs = state.corridor_analyzer.get_station_pairs(stations, max_distance_km=max_km)
    return pairs if isinstance(pairs, list) else (
        pairs.fillna("").to_dict(orient="records") if hasattr(pairs, "to_dict") else []
    )


@router.get("/movements")
def get_movements(state: AppState = Depends(get_state), max_km: float = Query(50.0)):
    df = _get_data(state)
    if not state.corridor_analyzer:
        raise HTTPException(status_code=503, detail="Service not ready")
    stations = state.station_manager.get_stations().fillna("").to_dict(orient="records") if state.station_manager else []
    movements = state.corridor_analyzer.get_movements(df, stations, max_distance_km=max_km)
    return movements if isinstance(movements, list) else (
        movements.fillna("").to_dict(orient="records") if hasattr(movements, "to_dict") else []
    )


@router.get("/bottlenecks")
def get_bottlenecks(state: AppState = Depends(get_state), max_km: float = Query(50.0)):
    df = _get_data(state)
    if not state.corridor_analyzer:
        raise HTTPException(status_code=503, detail="Service not ready")
    stations = state.station_manager.get_stations().fillna("").to_dict(orient="records") if state.station_manager else []
    result = state.corridor_analyzer.get_bottlenecks(df, stations, max_distance_km=max_km)
    return result if isinstance(result, list) else (
        result.fillna("").to_dict(orient="records") if hasattr(result, "to_dict") else []
    )


@router.get("/utilisation")
def get_utilisation(state: AppState = Depends(get_state), max_km: float = Query(50.0)):
    df = _get_data(state)
    if not state.corridor_analyzer:
        raise HTTPException(status_code=503, detail="Service not ready")
    stations = state.station_manager.get_stations().fillna("").to_dict(orient="records") if state.station_manager else []
    result = state.corridor_analyzer.get_utilisation(df, stations, max_distance_km=max_km)
    return result if isinstance(result, list) else (
        result.fillna("").to_dict(orient="records") if hasattr(result, "to_dict") else []
    )
