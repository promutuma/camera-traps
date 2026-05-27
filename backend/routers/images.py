"""
Tab 1 — Upload & Process
POST /api/images/upload        save files, return job_id
POST /api/images/process/{id}  start background job
GET  /api/images/job/{id}      poll progress
GET  /api/images/results/{id}  get finished results
GET  /api/images/file/{filename} serve image file
"""

from __future__ import annotations
import os
import re
import sys
import time
import tempfile
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from backend.models.state import AppState
from backend.models.schemas import JobStatus
from backend.routers.deps import get_state
from backend.services.job_manager import job_manager

router = APIRouter(prefix="/images", tags=["images"])

_executor = ThreadPoolExecutor(max_workers=2)

# Persistent uploads directory — survives server restarts
UPLOADS_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

_MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024


def _safe_filename(name: str) -> str:
    """Strip path components and replace characters unsafe on any OS."""
    name = Path(name or "upload").name
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "upload"


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """Save uploaded files to temp dir AND persistent uploads/, return job_id."""
    job = job_manager.create()
    temp_dir = tempfile.mkdtemp()
    job.temp_dir = temp_dir
    job.total = len(files)

    for upload in files:
        contents = await upload.read()

        if len(contents) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File '{upload.filename}' exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
            )

        safe_name = _safe_filename(upload.filename or "upload")

        # Temp copy for processing
        dest = os.path.join(temp_dir, safe_name)
        with open(dest, "wb") as f:
            f.write(contents)
        job.image_paths.append(dest)

        # Persistent copy so Results page can always serve the image
        persistent = UPLOADS_DIR / safe_name
        with open(persistent, "wb") as f:
            f.write(contents)

    return {"job_id": job.job_id, "file_count": len(files)}


# ---------------------------------------------------------------------------
# Serve stored images (persistent — works after server restart)
# ---------------------------------------------------------------------------

@router.get("/stored/{filename}")
def serve_stored_image(filename: str):
    """Serve an image from the persistent uploads directory."""
    safe_name = _safe_filename(filename)
    file_path = UPLOADS_DIR / safe_name
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(file_path))


# ---------------------------------------------------------------------------
# Start processing (background)
# ---------------------------------------------------------------------------

def _run_processing(job_id: str, state: AppState) -> None:
    """Blocking worker — runs in ThreadPoolExecutor."""
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

        # Apply current config to the already-loaded models
        if state.md_model:
            state.md_model.set_confidence_threshold(cfg.detection_confidence)
        if state.dn_model:
            state.dn_model.brightness_threshold = cfg.brightness_threshold

        animal_detector = AnimalDetector(
            state.md_model,
            state.bio_model,
            confidence_threshold=cfg.detection_confidence,
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
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
            job.completed = idx + 1

        # Stamp default station_id where missing
        for r in results:
            if not r.get("station_id"):
                r["station_id"] = cfg.default_station_id

        job.results = results

        # Auto-scrub privacy if enabled
        if cfg.enable_scrubbing and state.scrubber:
            import pandas as pd
            df = pd.DataFrame(results)
            audit = state.scrubber.scrub_batch(df)
            job.scrub_audit = audit

        # Persist to DB
        if state.db_manager:
            import pandas as pd
            df = pd.DataFrame(results)
            try:
                state.db_manager.save_results(df)
            except Exception:
                pass

        job.status = "done"

    except Exception as exc:
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    finally:
        job.finished_at = time.time()


@router.post("/process/{job_id}")
def start_processing(job_id: str, background_tasks: BackgroundTasks, state: AppState = Depends(get_state)):
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
# Poll job progress
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


# ---------------------------------------------------------------------------
# Serve image files
# ---------------------------------------------------------------------------

@router.get("/file/{job_id}/{filename}")
def serve_image(job_id: str, filename: str):
    job = job_manager.get(job_id)
    if not job or not job.temp_dir:
        raise HTTPException(status_code=404, detail="Job/temp dir not found")
    safe_name = _safe_filename(filename)
    file_path = os.path.join(job.temp_dir, safe_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
