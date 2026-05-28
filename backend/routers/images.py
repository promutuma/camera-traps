"""
Tab 1 — Upload & Process
POST /api/images/upload            save files immediately, return job_id
POST /api/images/process/{id}      start background AI pipeline
GET  /api/images/job/{id}          poll progress (JSON)
GET  /api/images/job/{id}/stream   real-time SSE (progress + per-model events)
GET  /api/images/results/{id}      get finished results
GET  /api/images/stored/{filename} serve persistent image file
GET  /api/images/file/{id}/{name}  serve temp image file
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
import tempfile
import traceback
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Magic bytes for accepted image formats
_ALLOWED_MAGIC: List[tuple] = [
    (b"\xff\xd8\xff",),          # JPEG
    (b"\x89PNG\r\n\x1a\n",),    # PNG
    (b"II\x2a\x00", b"MM\x00\x2a"),  # TIFF (LE and BE)
    (b"BM",),                    # BMP
    (b"GIF87a", b"GIF89a"),      # GIF
    (b"RIFF",),                  # WebP (RIFF container; checked further below)
]

def _is_allowed_image(data: bytes) -> bool:
    """Return True if data starts with a known image magic-byte sequence."""
    for signatures in _ALLOWED_MAGIC:
        if any(data.startswith(sig) for sig in signatures):
            # WebP: RIFF....WEBP
            if data.startswith(b"RIFF") and data[8:12] != b"WEBP":
                continue
            return True
    return False

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

from backend.models.state import AppState
from backend.models.schemas import JobStatus
from backend.routers.deps import get_state
from backend.services.job_manager import job_manager

router = APIRouter(prefix="/images", tags=["images"])

db_path = os.environ.get("DB_PATH", "wildlife_data.db")
UPLOADS_DIR = Path(db_path).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

_MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024


def _safe_filename(name: str) -> str:
    name = Path(name or "upload").name
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "upload"


# ---------------------------------------------------------------------------
# Upload — called immediately on file selection (before "Start" is clicked)
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """Save uploaded files; return job_id immediately so SSE can attach."""
    job = job_manager.create()
    temp_dir = tempfile.mkdtemp()
    job.temp_dir = temp_dir
    job.total = len(files)

    for upload in files:
        contents = await upload.read()

        if len(contents) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"'{upload.filename}' exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
            )

        if not _is_allowed_image(contents):
            raise HTTPException(
                status_code=415,
                detail=f"'{upload.filename}' is not a supported image file (JPEG, PNG, TIFF, BMP, WebP)",
            )

        safe_name = _safe_filename(upload.filename or "upload")

        dest = os.path.join(temp_dir, safe_name)
        with open(dest, "wb") as f:
            f.write(contents)
        job.image_paths.append(dest)

        persistent = UPLOADS_DIR / safe_name
        with open(persistent, "wb") as f:
            f.write(contents)

    return {"job_id": job.job_id, "file_count": len(files)}


# ---------------------------------------------------------------------------
# Serve images
# ---------------------------------------------------------------------------

@router.get("/stored/{filename}")
def serve_stored_image(filename: str):
    safe_name = _safe_filename(filename)
    file_path = UPLOADS_DIR / safe_name
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(file_path))


@router.get("/file/{job_id}/{filename}")
def serve_image(job_id: str, filename: str):
    job = job_manager.get(job_id)
    if not job or not job.temp_dir:
        raise HTTPException(status_code=404, detail="Job not found")
    safe_name = _safe_filename(filename)
    file_path = os.path.join(job.temp_dir, safe_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


# ---------------------------------------------------------------------------
# Processing worker (blocking — runs in thread)
# ---------------------------------------------------------------------------

def _run_processing(job_id: str, state: AppState) -> None:
    job = job_manager.get(job_id)
    if not job:
        return

    job.status = "running"
    cfg = state.config

    try:
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from core.animal_detector import AnimalDetector
        from core.image_processor import ImageProcessor

        if state.md_model:
            state.md_model.set_confidence_threshold(cfg.detection_confidence)
        if state.md_v1000_model:
            state.md_v1000_model.set_confidence_threshold(cfg.detection_confidence)
        if state.dn_model:
            state.dn_model.brightness_threshold = cfg.brightness_threshold

        animal_detector = AnimalDetector(
            megadetector=state.md_model,
            bioclip=state.bio_model,
            confidence_threshold=cfg.detection_confidence,
            megadetector_v1000=state.md_v1000_model,
            speciesnet=state.speciesnet_model,
        )

        processor = ImageProcessor(
            ocr_processor=state.ocr_model,
            animal_detector=animal_detector,
            day_night_classifier=state.dn_model,
            ocr_enabled=cfg.enable_ocr,
            detection_enabled=cfg.enable_detection,
            day_night_enabled=cfg.enable_day_night,
            ocr_strip_percent=cfg.ocr_strip_height,
        )

        results = []
        for idx, image_path in enumerate(job.image_paths):
            result = processor.process_single_image(image_path)
            result_list = result if isinstance(result, list) else [result]

            # Extract per-image model events from the first result
            image_events = result_list[0].pop("_model_events", []) if result_list else []
            image_name = os.path.basename(image_path)
            for ev in image_events:
                job.model_events.append({
                    "type": "model_event",
                    "image": image_name,
                    "image_index": idx,
                    **ev,
                })

            results.extend(result_list)
            job.completed = idx + 1

        for r in results:
            if not r.get("station_id"):
                r["station_id"] = cfg.default_station_id

        job.results = results

        if cfg.enable_scrubbing and state.scrubber:
            import pandas as pd
            df = pd.DataFrame(results)
            job.scrub_audit = state.scrubber.scrub_batch(df)

        if state.db_manager:
            import pandas as pd
            df = pd.DataFrame(results)
            try:
                state.db_manager.save_results(df)
            except Exception as db_exc:
                logger.error("Failed to persist results to DB for job %s: %s", job_id, db_exc)

        job.status = "done"

    except Exception as exc:
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    finally:
        job.finished_at = time.time()
        if state.db_manager:
            try:
                state.db_manager.save_job(job)
            except Exception as persist_exc:
                logger.error("Could not persist job metadata for %s: %s", job_id, persist_exc)


@router.post("/process/{job_id}")
def start_processing(
    job_id: str,
    background_tasks: BackgroundTasks,
    state: AppState = Depends(get_state),
):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("queued",):
        raise HTTPException(status_code=409, detail=f"Job already {job.status}")
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="AI models not loaded yet")

    background_tasks.add_task(_run_processing, job_id, state)
    return {"job_id": job_id, "status": "started"}


# ---------------------------------------------------------------------------
# Poll progress (JSON)
# ---------------------------------------------------------------------------

@router.get("/job/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        total=job.total,
        completed=job.completed,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# SSE stream — progress + per-model events
# ---------------------------------------------------------------------------

@router.get("/job/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """
    Server-Sent Events stream.

    Yields two event types:
      {"type": "model_event", "image": "...", "model": "...", ...}
      {"type": "progress",    "status": "...", "total": N, "completed": N}
    """
    async def event_gen():
        event_cursor = 0
        while True:
            job = job_manager.get(job_id)
            if not job:
                yield _sse({"type": "progress", "status": "error",
                             "error": "Job not found", "total": 0, "completed": 0})
                return

            # Flush any new model events accumulated since last tick
            new_events = job.model_events[event_cursor:]
            for ev in new_events:
                yield _sse(ev)
            event_cursor += len(new_events)

            # Always send a progress heartbeat
            yield _sse({
                "type": "progress",
                "job_id": job.job_id,
                "status": job.status,
                "total": job.total,
                "completed": job.completed,
                "error": job.error,
            })

            if job.status in ("done", "error"):
                return

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Get results when done
# ---------------------------------------------------------------------------

@router.get("/results/{job_id}")
def get_job_results(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=202, detail=f"Job status: {job.status}")
    return {
        "job_id": job_id,
        "results": job.results,
        "scrub_audit": job.scrub_audit,
        "total": job.total,
    }
