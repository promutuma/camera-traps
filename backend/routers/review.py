"""Tab 9 — Review Queue."""

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from backend.models.state import AppState
from backend.models.schemas import ReviewAction, BulkFlagByImageIdsRequest
from backend.routers.deps import get_state, get_current_user
from core.review_engine import ReviewConflictError

router = APIRouter(prefix="/review", tags=["review"])


def _reviewer_id(request: Request, action: ReviewAction) -> str:
    if action.reviewer_id and action.reviewer_id.strip():
        return action.reviewer_id.strip()
    return get_current_user(request)


@router.get("/queue")
def get_queue(
    state: AppState = Depends(get_state),
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")
    threshold = state.config.review_confidence_threshold
    total = state.db_manager.count_pending_review(confidence_threshold=threshold)
    df = state.db_manager.get_pending_review(
        confidence_threshold=threshold, limit=limit, offset=offset
    )
    items = df.fillna("").to_dict(orient="records") if not df.empty else []
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.post("/confirm/{detection_id}")
def confirm(
    detection_id: int,
    action: ReviewAction,
    request: Request,
    state: AppState = Depends(get_state),
):
    if not state.review_engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        state.review_engine.accept(
            detection_id=str(detection_id),
            reviewer_id=_reviewer_id(request, action),
            notes=action.notes or "",
            bbox=action.bbox,
        )
    except ReviewConflictError:
        raise HTTPException(status_code=409, detail="Detection already reviewed")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}


@router.post("/correct/{detection_id}")
def correct(
    detection_id: int,
    action: ReviewAction,
    request: Request,
    state: AppState = Depends(get_state),
):
    if not state.review_engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    if not action.corrected_label:
        raise HTTPException(status_code=400, detail="corrected_label required")
    original_species = ""
    if state.db_manager:
        conn = state.db_manager.get_connection()
        try:
            row = conn.execute(
                "SELECT detected_animal FROM detections WHERE id = ?",
                [detection_id],
            ).fetchone()
            original_species = row[0] if row else ""
        finally:
            conn.close()
    try:
        state.review_engine.correct(
            detection_id=str(detection_id),
            original_species=original_species,
            corrected_species=action.corrected_label,
            corrected_label=action.corrected_label,
            reviewer_id=_reviewer_id(request, action),
            notes=action.notes or "",
            bbox=action.bbox,
        )
    except ReviewConflictError:
        raise HTTPException(status_code=409, detail="Detection already reviewed")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}


@router.post("/flag/{detection_id}")
def flag(
    detection_id: int,
    action: ReviewAction,
    request: Request,
    state: AppState = Depends(get_state),
):
    if not state.review_engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        state.review_engine.reject(
            detection_id=str(detection_id),
            reviewer_id=_reviewer_id(request, action),
            notes=action.notes or "",
        )
    except ReviewConflictError:
        raise HTTPException(status_code=409, detail="Detection already reviewed")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}


def _flag_detections_for_images(
    review_engine,
    db_manager,
    image_ids: list,
    reviewer_id: str,
    notes: str,
) -> int:
    flagged = 0
    for image_id in image_ids:
        conn = db_manager.get_connection()
        try:
            filename_row = conn.execute(
                "SELECT filename FROM images WHERE id = ?", [image_id]
            ).fetchone()
            filename = filename_row[0] if filename_row else str(image_id)
            rows = conn.execute(
                "SELECT id FROM detections WHERE image_id = ? ORDER BY confidence DESC",
                [image_id],
            ).fetchall()
        finally:
            conn.close()
        for (det_id,) in rows:
            try:
                review_engine.reject(
                    detection_id=str(det_id),
                    filename=filename,
                    reviewer_id=reviewer_id,
                    notes=notes,
                )
                flagged += 1
            except ReviewConflictError:
                continue
    return flagged


@router.post("/flag-by-image-ids")
def flag_by_image_ids(
    req: BulkFlagByImageIdsRequest,
    request: Request,
    state: AppState = Depends(get_state),
):
    """Flag detections by image PK — preferred over filename when duplicates exist."""
    if not state.review_engine or not state.db_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    reviewer_id = req.reviewer_id.strip() if req.reviewer_id else get_current_user(request)
    flagged = _flag_detections_for_images(
        state.review_engine,
        state.db_manager,
        req.image_ids,
        reviewer_id,
        req.notes or "Flagged during upload review",
    )
    return {"flagged": flagged, "requested": len(req.image_ids)}


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
    return list(reversed(state.scrubber.audit_log))


@router.post("/rescrub")
def rescrub_existing_images(state: AppState = Depends(get_state)):
    """
    Re-run privacy scrub on all images in history that contain Person/Vehicle
    detections. Useful for backfilling scrubbed copies for images processed
    before scrubbing was wired up.
    """
    if not state.scrubber:
        raise HTTPException(status_code=503, detail="Scrubber not ready")
    if not state.config.enable_scrubbing:
        raise HTTPException(status_code=400, detail="Privacy scrubbing is disabled in config")
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="DB not ready")

    df = state.db_manager.get_history_df()
    if df.empty:
        return {"scrubbed": 0, "attempted": 0, "audit": []}

    db_path = os.environ.get("DB_PATH", "wildlife_data.db")
    uploads_dir = Path(db_path).parent / "uploads"
    df = df.copy()
    id_col = "image_id" if "image_id" in df.columns else "id"
    df["filepath"] = df["filename"].apply(
        lambda f: str(uploads_dir / Path(str(f)).name)
    )
    if "primary_label" not in df.columns and "detected_animal" in df.columns:
        df["primary_label"] = df["detected_animal"]

    audit = state.scrubber.scrub_batch(df)
    scrubbed_count = sum(1 for a in audit if a.get("scrubbed") == "Yes")
    return {
        "scrubbed": scrubbed_count,
        "attempted": len(audit),
        "audit": audit,
    }
