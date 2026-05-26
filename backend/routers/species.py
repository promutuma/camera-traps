"""Tab 12 — Species Reference Library."""

from fastapi import APIRouter, Depends, HTTPException, Query
from backend.models.state import AppState
from backend.routers.deps import get_state

router = APIRouter(prefix="/species", tags=["species"])


@router.get("")
def list_species(state: AppState = Depends(get_state)):
    if not state.species_library:
        raise HTTPException(status_code=503, detail="Service not ready")
    data = state.species_library.get_all()
    return data if isinstance(data, list) else (
        data.fillna("").to_dict(orient="records") if hasattr(data, "to_dict") else []
    )


@router.get("/lookup/{name}")
def lookup_species(name: str, state: AppState = Depends(get_state)):
    if not state.species_library:
        raise HTTPException(status_code=503, detail="Service not ready")
    result = state.species_library.lookup(name)
    if not result:
        raise HTTPException(status_code=404, detail=f"Species '{name}' not found")
    return result


@router.get("/synonyms/{name}")
def resolve_synonym(name: str, state: AppState = Depends(get_state)):
    if not state.species_library:
        raise HTTPException(status_code=503, detail="Service not ready")
    canonical = state.species_library.resolve_synonym(name)
    return {"input": name, "canonical": canonical}


@router.post("/add")
def add_species(
    name: str = Query(...),
    common_name: str = Query(""),
    state: AppState = Depends(get_state),
):
    if not state.species_library:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.species_library.add_species(name, common_name=common_name)
    return {"ok": True}
