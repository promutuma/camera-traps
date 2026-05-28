"""Tab 9 — Review Queue."""

from fastapi import APIRouter, Depends, HTTPException, Query
from backend.models.state import AppState
from backend.models.schemas import ReviewAction, BulkFlagRequest
from backend.routers.deps import get_state

router = APIRouter(prefix="/review", tags=["review"])


@router.get("/queue")
def get_queue(
    state: AppState = Depends(get_state),
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    threshold = state.config.review_confidence_threshold
    df = state.db_manager.get_pending_review(
        confidence_threshold=threshold, limit=limit, offset=offset
    )
    if df.empty:
        return []
    return df.fillna("").to_dict(orient="records")


@router.post("/confirm/{image_id}")
def confirm(image_id: int, action: ReviewAction, state: AppState = Depends(get_state)):
    if not state.review_engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    state.review_engine.accept(
        image_id=str(image_id),
        reviewer_id=action.reviewer_id,
        notes=action.notes or "",
        bbox=action.bbox,
    )
    return {"ok": True}


@router.post("/correct/{image_id}")
def correct(image_id: int, action: ReviewAction, state: AppState = Depends(get_state)):
    if not state.review_engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    if not action.corrected_label:
        raise HTTPException(status_code=400, detail="corrected_label required")
    original_species = ""
    if state.db_manager:
        original_species = state.db_manager.get_detected_animal_for_image(image_id)
    state.review_engine.correct(
        image_id=str(image_id),
        original_species=original_species,
        corrected_species=action.corrected_label,
        corrected_label=action.corrected_label,
        reviewer_id=action.reviewer_id,
        notes=action.notes or "",
        bbox=action.bbox,
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


@router.post("/flag-by-filenames")
def flag_by_filenames(req: BulkFlagRequest, state: AppState = Depends(get_state)):
    """Flag detections by filename — used to persist Upload-page flags to the review log."""
    if not state.review_engine or not state.db_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    found = state.db_manager.get_image_ids_by_filenames(req.filenames)
    for image_id, filename in found:
        state.review_engine.reject(
            image_id=str(image_id),
            filename=filename,
            reviewer_id=req.reviewer_id,
            notes=req.notes or "Flagged during upload review",
        )
    return {"flagged": len(found), "requested": len(req.filenames)}


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
