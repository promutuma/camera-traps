"""Tab 4 — Analysis History."""

import io
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from backend.models.state import AppState
from backend.routers.deps import get_state

router = APIRouter(prefix="/history", tags=["history"])


@router.get("")
def get_history(state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    df = state.db_manager.get_history_df()
    return df.fillna("").to_dict(orient="records")


@router.delete("")
def clear_history(state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    state.db_manager.clear_history()
    return {"ok": True}


@router.get("/export/csv")
def export_csv(state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    df = state.db_manager.get_history_df()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=wildlife_history.csv"},
    )
