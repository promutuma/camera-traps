"""Storage management: downloads, cleanup, quotas."""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List

from backend.models.state import AppState
from backend.routers.deps import get_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/storage", tags=["storage"])


@router.get("/status")
def get_storage_status(state: AppState = Depends(get_state)):
    """Get storage breakdown by tier."""
    if not state.file_manager:
        raise HTTPException(status_code=503, detail="File manager not ready")

    return state.file_manager.get_storage_status()


@router.get("/warnings")
def get_deletion_warnings(state: AppState = Depends(get_state)):
    """Get images pending deletion (for UI warnings)."""
    if not state.file_manager:
        raise HTTPException(status_code=503, detail="File manager not ready")

    return state.file_manager.get_deletion_warnings()


@router.get("/deletion-preview")
def get_deletion_preview(
    state: AppState = Depends(get_state),
    tier: str = Query(...),
    days_old: int = Query(7),
):
    """Preview what would be deleted."""
    if tier not in ("empty", "low_conf", "valid"):
        raise HTTPException(status_code=400, detail="Invalid tier")

    if not state.file_manager:
        raise HTTPException(status_code=503, detail="File manager not ready")

    return state.file_manager.get_deletion_preview(tier, days_old)


@router.post("/cleanup")
def cleanup_images(
    state: AppState = Depends(get_state),
    action: str = Query(...),  # delete_empty, delete_marked, archive
    dry_run: bool = Query(True),
    days_old: int = Query(7),
):
    """Execute cleanup action."""
    if action not in ("delete_empty", "delete_marked"):
        raise HTTPException(status_code=400, detail="Invalid action")

    if not state.file_manager:
        raise HTTPException(status_code=503, detail="File manager not ready")

    if action == "delete_empty":
        return state.file_manager.cleanup_empty_images(dry_run=dry_run)
    else:
        return state.file_manager.cleanup_marked_for_deletion(days_grace=days_old, dry_run=dry_run)


# ──────────────────────────────────────────────────────────────────
# Download Endpoints
# ──────────────────────────────────────────────────────────────────


@router.post("/downloads/batch")
def create_batch_download(
    state: AppState = Depends(get_state),
    tier: Optional[str] = Query(None),  # empty, low_conf, valid
    image_ids: Optional[str] = Query(None),  # JSON array
    station_id: Optional[str] = Query(None),
    include_metadata: bool = Query(True),
):
    """Create a batch download package."""
    if not state.file_manager or not state.db_manager:
        raise HTTPException(status_code=503, detail="Services not ready")

    try:
        # If specific image IDs provided, use those
        if image_ids:
            import json
            ids = json.loads(image_ids)
        elif tier:
            # Otherwise query by tier
            images = state.db_manager.get_images_by_tier(tier, limit=10000)
            ids = [img[0] for img in images]
        else:
            raise HTTPException(status_code=400, detail="Provide tier or image_ids")

        if not ids:
            raise HTTPException(status_code=400, detail="No images to download")

        download_info = state.file_manager.create_batch_download(
            ids, download_type=f"batch_{tier or 'custom'}", include_metadata=include_metadata
        )
        return download_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/downloads/{download_id}/stream")
def download_batch(
    download_id: str,
    state: AppState = Depends(get_state),
):
    """Download batch ZIP file as streaming response."""
    if not state.file_manager or not state.db_manager:
        raise HTTPException(status_code=503, detail="Services not ready")

    try:
        # Get download info from DB
        conn = state.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT image_id FROM download_images WHERE download_id = ? LIMIT 10000",
            (download_id,),
        )
        results = cursor.fetchall()
        conn.close()

        if not results:
            raise HTTPException(status_code=404, detail="Download not found")

        image_ids = [r[0] for r in results]

        if not image_ids:
            raise HTTPException(status_code=400, detail="No images in download")

        # Generate ZIP and stream it
        zip_buffer, file_hash = state.file_manager.get_batch_zip_stream(
            download_id, image_ids, include_metadata=True
        )

        def iterfile(file_obj, chunk_size=8192):
            """Yield file in chunks for streaming."""
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        return StreamingResponse(
            iterfile(zip_buffer),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="viumbelens_batch_{download_id}.zip"',
                "X-File-Hash": file_hash,  # Optional: client can verify integrity
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Download error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mark-for-deletion/{image_id}")
def mark_for_deletion(
    image_id: int,
    state: AppState = Depends(get_state),
):
    """Mark image for deletion (starts grace period)."""
    if not state.db_manager:
        raise HTTPException(status_code=503, detail="Database not ready")

    try:
        state.db_manager.mark_for_deletion(image_id)
        return {"ok": True, "message": "Image marked for deletion (7-day grace period)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────
# Hash Management (Performance Optimization)
# ──────────────────────────────────────────────────────────────────


@router.get("/hash-stats")
def get_hash_stats(state: AppState = Depends(get_state)):
    """Get hash statistics and deduplication opportunities."""
    if not state.file_manager:
        raise HTTPException(status_code=503, detail="File manager not ready")

    try:
        return state.file_manager.get_hash_optimization_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-hashes")
def clear_hashes(
    state: AppState = Depends(get_state),
    strategy: str = Query(...),  # empty, archived, old_30d, duplicates, all
):
    """
    Clear file hashes to reduce processing/storage overhead.

    Strategies:
    - 'empty': Clear hashes for empty frames
    - 'archived': Clear hashes for archived images
    - 'old_30d': Clear hashes older than 30 days
    - 'duplicates': Keep only first occurrence of duplicate hashes
    - 'all': Clear all hashes (maximum performance)
    """
    if strategy not in ("empty", "archived", "old_30d", "duplicates", "all"):
        raise HTTPException(status_code=400, detail="Invalid hash clear strategy")

    if not state.file_manager:
        raise HTTPException(status_code=503, detail="File manager not ready")

    try:
        result = state.file_manager.clear_hashes_for_optimization(strategy)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
