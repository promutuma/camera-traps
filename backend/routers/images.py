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
import threading
import time
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

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

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse

from backend.models.state import AppState
from backend.models.schemas import JobStatus
from backend.routers.deps import get_state
from backend.services.job_manager import job_manager

router = APIRouter(prefix="/images", tags=["images"])

# Allow at most 1 job to run inference at a time. Each job loads all four
# models into active memory (~4.5 GB); concurrent jobs would OOM most machines.
# The job stays "queued" while waiting and transitions to "running" on acquire.
_job_semaphore = threading.Semaphore(1)

# Number of images to process in parallel within a single job.
# Model weights (MegaDetector ~270 MB, SpeciesNet ~500 MB, OCR ~200 MB) are
# loaded once and shared. Each additional parallel image adds ~200 MB of
# inference buffer. Floor at 1, cap at half the CPU count so we leave cores
# free for the OS and the OCR/day-night sub-tasks per image.
_PARALLEL_IMAGES = max(1, (os.cpu_count() or 4) // 2)

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
async def upload_images(files: List[UploadFile] = File(...), state: AppState = Depends(get_state)):
    """Save uploaded files; return job_id immediately so SSE can attach."""
    import hashlib as _hashlib
    job = job_manager.create()
    temp_dir = tempfile.mkdtemp()
    job.temp_dir = temp_dir
    duplicates = []
    rejected = []   # unsupported file type — skip, don't abort

    for upload in files:
        contents = await upload.read()
        fname = upload.filename or "upload"

        if len(contents) > _MAX_UPLOAD_BYTES:
            rejected.append(f"{fname} (too large)")
            continue

        if not _is_allowed_image(contents):
            rejected.append(fname)
            continue

        # Hash from in-memory bytes — no disk I/O needed
        file_hash = _hashlib.sha256(contents).hexdigest()
        safe_name = _safe_filename(fname)

        # Early duplicate rejection before any disk write or AI processing
        if state.db_manager and state.db_manager.image_hash_exists(file_hash):
            duplicates.append(fname)
            continue

        job.file_hashes[safe_name] = file_hash

        dest = os.path.join(temp_dir, safe_name)
        with open(dest, "wb") as f:
            f.write(contents)
        job.image_paths.append(dest)

        persistent = UPLOADS_DIR / safe_name
        with open(persistent, "wb") as f:
            f.write(contents)

    job.total = len(job.image_paths)

    if not job.image_paths:
        # Return 200 with file_count=0 so the frontend can skip processing this chunk
        # rather than treating it as a fatal error.
        response: dict = {"file_count": 0}
        if duplicates:
            response["duplicates_skipped"] = duplicates
        if rejected:
            response["rejected"] = rejected
        return response

    response = {"job_id": job.job_id, "file_count": len(job.image_paths)}
    if duplicates:
        response["duplicates_skipped"] = duplicates
    if rejected:
        response["rejected"] = rejected
    return response


# ---------------------------------------------------------------------------
# Serve images
# ---------------------------------------------------------------------------

_CACHE_HEADERS = {"Cache-Control": "public, max-age=86400, immutable"}


@router.get("/stored/{filename}")
def serve_stored_image(filename: str):
    safe_name = _safe_filename(filename)
    file_path = UPLOADS_DIR / safe_name
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(file_path), headers=_CACHE_HEADERS)


@router.get("/thumb/{filename}")
def serve_thumbnail(filename: str, w: int = Query(800, ge=64, le=2560)):
    """
    Return a cached JPEG thumbnail scaled to fit within w×w pixels.
    Original is never modified. Cache lives at uploads/thumbs/{w}/{safe_name}.jpg.
    Falls back to the original if PIL is unavailable or resizing fails.
    """
    from PIL import Image as _PILImage
    safe_name = _safe_filename(filename)
    original = UPLOADS_DIR / safe_name
    if not original.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    thumb_dir = UPLOADS_DIR / "thumbs" / str(w)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    # Use full sanitized filename (not just stem) to avoid collisions between
    # IMG_001.jpg and IMG_001.png mapping to the same cache entry.
    thumb_path = thumb_dir / f"{safe_name}.jpg"

    if not thumb_path.is_file():
        try:
            img = _PILImage.open(original)
            img.thumbnail((w, w), _PILImage.LANCZOS)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(thumb_path, "JPEG", quality=82, optimize=True)
        except Exception:
            return FileResponse(str(original), headers=_CACHE_HEADERS)

    return FileResponse(str(thumb_path), media_type="image/jpeg", headers=_CACHE_HEADERS)


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

def _run_processing(job_id: str, state: AppState, station_id: Optional[str] = None) -> None:
    job = job_manager.get(job_id)
    if not job:
        return

    # Block here if another job is already running. The job stays "queued"
    # while waiting so the SSE stream and UI reflect the correct state.
    with _job_semaphore:
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
            if state.dn_model:
                state.dn_model.brightness_threshold = cfg.brightness_threshold

            animal_detector = AnimalDetector(
                megadetector=state.md_model,
                confidence_threshold=cfg.detection_confidence,
                speciesnet=state.speciesnet_model,
                speciesnet_bypass_threshold=cfg.speciesnet_bypass_threshold,
                use_speciesnet_first=True,
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

            import pandas as pd

            # Process images in parallel. Results are saved to the DB as each
            # image completes so data is durable even if the job is interrupted.
            results_by_idx: dict = {}
            image_errors: list = []
            _write_lock = threading.Lock()

            def _process_one(args: tuple) -> tuple:
                idx, path = args
                result = processor.process_single_image(path)
                return idx, path, result if isinstance(result, list) else [result]

            def _save_image_result(image_name: str, result_list: list) -> None:
                """Persist one image's results immediately after processing."""
                if not state.db_manager or not result_list:
                    return
                try:
                    state.db_manager.save_results(pd.DataFrame(result_list))

                    if state.file_manager:
                        best = max(
                            result_list,
                            key=lambda r: r.get("detection_confidence", 0.0) or 0.0,
                        )
                        detected_animal = best.get("detected_animal", "")
                        confidence = best.get("detection_confidence", 0.0) or 0.0
                        is_person_or_vehicle = detected_animal and (
                            detected_animal.lower() in ("person", "vehicle", "human")
                            or "person" in detected_animal.lower()
                            or "vehicle" in detected_animal.lower()
                        )
                        has_animal = (
                            False
                            if (is_person_or_vehicle or not detected_animal or confidence == 0)
                            else confidence > 0.4
                        )
                        try:
                            conn = state.db_manager.get_connection()
                            cursor = conn.cursor()
                            cursor.execute(
                                "SELECT id FROM images WHERE filename = ? ORDER BY id DESC LIMIT 1",
                                (image_name,),
                            )
                            img_row = cursor.fetchone()
                            conn.close()
                            if img_row:
                                state.file_manager.update_image_with_file_info(
                                    img_row[0],
                                    str(UPLOADS_DIR / image_name),
                                    has_animal,
                                    precomputed_hash=job.file_hashes.get(image_name),
                                )
                        except Exception as info_exc:
                            logger.warning("Could not update file info for %s: %s", image_name, info_exc)

                except Exception as db_exc:
                    logger.error("Failed to persist result for %s (job %s): %s", image_name, job_id, db_exc)

            with ThreadPoolExecutor(max_workers=_PARALLEL_IMAGES) as image_pool:
                futures = {
                    image_pool.submit(_process_one, (idx, path)): (idx, path)
                    for idx, path in enumerate(job.image_paths)
                }
                for future in as_completed(futures):
                    idx, image_path = futures[future]
                    try:
                        _, _, result_list = future.result()
                    except Exception as exc:
                        logger.error("Image %s failed: %s", image_path, exc)
                        with _write_lock:
                            image_errors.append(f"{os.path.basename(image_path)}: {exc}")
                            job.completed += 1
                        continue

                    image_events = result_list[0].pop("_model_events", []) if result_list else []
                    image_name = os.path.basename(image_path)

                    for r in result_list:
                        if not r.get("station_id"):
                            r["station_id"] = station_id or cfg.default_station_id

                    # Serialise DB writes through the same lock used for job
                    # state — SQLite tolerates concurrent readers but serialises
                    # writers, so doing this under the lock avoids contention.
                    with _write_lock:
                        _save_image_result(image_name, result_list)
                        for ev in image_events:
                            job.model_events.append({
                                "type": "model_event",
                                "image": image_name,
                                "image_index": idx,
                                **ev,
                            })
                        results_by_idx[idx] = result_list
                        job.completed += 1

            # Reconstruct in upload order for job.results (used by /results/{job_id})
            results = []
            for idx in range(len(job.image_paths)):
                results.extend(results_by_idx.get(idx, []))

            if image_errors:
                job.error = f"{len(image_errors)} image(s) failed: " + "; ".join(image_errors)

            job.results = results

            if cfg.enable_scrubbing and state.scrubber:
                df = pd.DataFrame(results)
                job.scrub_audit = state.scrubber.scrub_batch(df)

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
    station_id: Optional[str] = None,
):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("queued",):
        raise HTTPException(status_code=409, detail=f"Job already {job.status}")
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="AI models not loaded yet")

    background_tasks.add_task(_run_processing, job_id, state, station_id or None)
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
async def stream_job_progress(job_id: str, cursor: int = 0):
    """
    Server-Sent Events stream.

    Yields two event types:
      {"type": "model_event", "image": "...", "model": "...", ...}
      {"type": "progress",    "status": "...", "total": N, "completed": N}

    Pass ?cursor=N to resume from a known position (e.g. after a reconnect)
    so already-delivered model events are not re-sent.
    """
    async def event_gen():
        event_cursor = max(0, cursor)
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
                # Free model_events from RAM — they've all been delivered.
                job.model_events.clear()
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
