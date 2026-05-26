"""Tab 9 — Review Queue."""

from fastapi import APIRouter, Depends, HTTPException
from backend.models.state import AppState
from backend.models.schemas import ReviewAction
from backend.routers.deps import get_state

router = APIRouter(prefix="/review", tags=["review"])


@router.get("/queue")
def get_queue(state: AppState = Depends(get_state)):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    df = state.db_manager.get_history_df()
    if df.empty:
        return []
    threshold = state.config.review_confidence_threshold
    if "detection_confidence" in df.columns:
        queue = df[df["detection_confidence"] < threshold]
    else:
        queue = df
    return queue.fillna("").to_dict(orient="records")


@router.post("/confirm/{image_id}")
def confirm(image_id: int, action: ReviewAction, state: AppState = Depends(get_state)):
    if not state.review_engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.review_engine.accept(
        image_id=str(image_id),
        reviewer_id=action.reviewer_id,
        notes=action.notes or "",
    )
    return {"ok": True}


@router.post("/correct/{image_id}")
def correct(image_id: int, action: ReviewAction, state: AppState = Depends(get_state)):
    if not state.review_engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    if not action.corrected_label:
        raise HTTPException(status_code=400, detail="corrected_label required")
    state.review_engine.correct(
        image_id=str(image_id),
        original_species="",
        corrected_species=action.corrected_label,
        corrected_label=action.corrected_label,
        reviewer_id=action.reviewer_id,
        notes=action.notes or "",
    )
    return {"ok": True}


@router.post("/flag/{image_id}")
def flag(image_id: int, action: ReviewAction, state: AppState = Depends(get_state)):
    if not state.review_engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.review_engine.reject(
        image_id=str(image_id),
        reviewer_id=action.reviewer_id,
        notes=action.notes or "",
    )
    return {"ok": True}


@router.get("/log")
def get_log(state: AppState = Depends(get_state)):
    if not state.review_engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    log = state.review_engine.get_actions_df()
    return log.fillna("").to_dict(orient="records") if hasattr(log, "to_dict") else []


@router.get("/privacy-audit")
def privacy_audit(state: AppState = Depends(get_state)):
    if not state.scrubber:
        raise HTTPException(status_code=503, detail="Service not ready")
    audit = getattr(state.scrubber, "audit_log", [])
    return audit
