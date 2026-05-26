"""Tab 7 — QC Dashboard."""

from fastapi import APIRouter, Depends, HTTPException
from backend.models.state import AppState
from backend.routers.deps import get_state

router = APIRouter(prefix="/qc", tags=["qc"])


@router.get("/flags")
def get_flags(state: AppState = Depends(get_state)):
    if not state.db_manager or not state.qc_engine:
        raise HTTPException(status_code=503, detail="Services not ready")
    df = state.db_manager.get_history_df()
    if df.empty:
        return []
    trap_nights = state.station_manager.compute_trap_nights() if state.station_manager else {}
    flags = state.qc_engine.run_all_checks(df, trap_nights)
    return flags.fillna("").to_dict(orient="records") if hasattr(flags, "to_dict") else flags


@router.get("/summary")
def get_summary(state: AppState = Depends(get_state)):
    if not state.db_manager or not state.qc_engine:
        raise HTTPException(status_code=503, detail="Services not ready")
    df = state.db_manager.get_history_df()
    if df.empty:
        return {"total_flags": 0, "by_type": []}
    trap_nights = state.station_manager.compute_trap_nights() if state.station_manager else {}
    flags = state.qc_engine.run_all_checks(df, trap_nights)
    if hasattr(flags, "empty") and flags.empty:
        return {"total_flags": 0, "by_type": []}
    by_type = (
        flags.groupby("flag_type").size().reset_index(name="count")
        .to_dict(orient="records")
    ) if hasattr(flags, "groupby") and "flag_type" in flags.columns else []
    return {"total_flags": len(flags), "by_type": by_type}
