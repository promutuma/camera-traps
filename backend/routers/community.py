"""Tab 10 — Community Observer."""

from fastapi import APIRouter, Depends, HTTPException
from backend.models.state import AppState
from backend.models.schemas import ObservationCreate
from backend.routers.deps import get_state

router = APIRouter(prefix="/community", tags=["community"])


@router.get("/observations")
def list_observations(state: AppState = Depends(get_state)):
    if not state.community_observer:
        raise HTTPException(status_code=503, detail="Service not ready")
    obs = state.community_observer.get_observations()
    return obs if isinstance(obs, list) else (
        obs.fillna("").to_dict(orient="records") if hasattr(obs, "to_dict") else []
    )


@router.post("/observations")
def add_observation(body: ObservationCreate, state: AppState = Depends(get_state)):
    if not state.community_observer:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.community_observer.add_observation(**body.model_dump())
    return {"ok": True}


@router.delete("/observations/{obs_id}")
def delete_observation(obs_id: int, state: AppState = Depends(get_state)):
    if not state.community_observer:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.community_observer.delete_observation(obs_id)
    return {"ok": True}


@router.get("/crosscheck")
def crosscheck(state: AppState = Depends(get_state)):
    if not state.community_observer or not state.db_manager:
        raise HTTPException(status_code=503, detail="Services not ready")
    df = state.db_manager.get_history_df()
    camera_species = set(df["detected_animal"].dropna().unique()) if not df.empty else set()
    result = state.community_observer.get_species_comparison(camera_species)
    return result if isinstance(result, (list, dict)) else []
